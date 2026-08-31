import ezpz
import ezpz.distributed
import json
import os
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Literal

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
# 79th sync: upstream #4172 deleted components/lr_scheduler.py (it had become
# a re-export shim when the optimizer components were grouped into a package
# by #4140). LRSchedulersContainer now lives in components.optimizer.
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, OptimizersContainer
from torchtitan.experiments.ezpz.optimizer.containers import (
    default_mano,
    default_muon,
    default_sophiag,
)
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    SingleDatasetConfig,
)
from torchtitan.components.data.sources import (
    HuggingFaceRandomAccessSource,
    HuggingFaceStreamingSource,
)
from torchtitan.hf_datasets.text_datasets import TextProcessor
from torchtitan.experiments.ezpz.validator import EzpzValidator
from torchtitan.config import CommConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.config.configs import CompileConfig
from torchtitan.experiments.ezpz.blendcorpus.blendcorpus_builder import (
    BlendCorpusDataLoader,
)
from torchtitan.experiments.ezpz.blendcorpus.build_tokenizer import EZPZTokenizer
from torchtitan.experiments.torchft.config.job_config import FaultTolerance
from torchtitan.experiments.ezpz.trainer import FaultTolerantTrainer

from . import model_registry

TT_CONFIG_JSON_ENV = "TT_CONFIG_JSON"


def agpt_debugmodel() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_debugmodel()


def agpt_2b() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_2b()


def agpt_2b_hf() -> FaultTolerantTrainer.Config:
    cfg = ezpz_agpt_2b()
    cfg.dataloader.dataset_path = None
    return cfg


def _set_rope_backend(
    cfg: FaultTolerantTrainer.Config,
    backend: Literal["complex", "cos_sin"],
) -> FaultTolerantTrainer.Config:
    """Switch every RoPE callsite in the model spec to ``backend``.

    PR #3458 (RoPE refactor) split ``RoPE.Config`` into
    ``ComplexRoPE.Config`` / ``CosSinRoPE.Config`` and dropped the
    ``backend`` string field — backend is now encoded in the type.
    The top-level ``Model.Config.rope`` field is also gone; each
    layer's ``Attention.Config`` owns its own rope. So flipping the
    backend means rebuilding each layer's ``attention.rope`` as a
    fresh instance of the target subclass, copying over all other
    fields (dim / max_context_length / theta / scaling / yarn params).
    """
    from dataclasses import fields

    from torchtitan.models.common import ComplexRoPE, CosSinRoPE

    target_cls = ComplexRoPE.Config if backend == "complex" else CosSinRoPE.Config
    model = cfg.model_spec.model
    for layer in model.layers:
        old_rope = layer.attention.rope
        if old_rope is None:
            continue
        kwargs = {f.name: getattr(old_rope, f.name) for f in fields(old_rope)}
        layer.attention.rope = target_cls(**kwargs)
    return cfg


def _set_fp32_residual(
    cfg: FaultTolerantTrainer.Config,
) -> FaultTolerantTrainer.Config:
    """Swap every transformer block to the fp32-residual variant.

    Root cause of the 80B production NaN (task #21): the bf16 residual stream
    overflows across the 84-layer depth. AgptFp32ResidualBlock does the two
    residual adds in fp32 while keeping the attention/FFN GEMMs in bf16 (see
    fp32_residual.py). Rebuilds each layer's block Config as the subclass,
    preserving every other field -- mirrors _set_rope_backend.
    """
    import copy
    from dataclasses import fields

    from torchtitan.experiments.ezpz.agpt.fp32_residual import (
        AgptFp32ResidualBlock,
    )

    # Deep-copy first: ezpz_agpt_*() can hand back a config whose model spec is
    # shared with the agpt_configs template, so mutating .layers in place would
    # poison the plain flavor for the rest of the process. (Verified: without
    # this, a later agpt_80b() returned fp32res blocks.)
    cfg = copy.deepcopy(cfg)
    model = cfg.model_spec.model
    model.layers = [
        AgptFp32ResidualBlock.Config(
            **{f.name: getattr(layer, f.name) for f in fields(layer)}
        )
        for layer in model.layers
    ]
    return cfg


def _set_fp32_residual_depth(
    cfg: FaultTolerantTrainer.Config,
) -> FaultTolerantTrainer.Config:
    """Full-depth fp32 residual stream (tasks #24/#26/#27).

    The per-block _set_fp32_residual casts back to bf16 at each block boundary,
    so the 84-deep cross-layer accumulation is still bf16 -- and it still NaN'd
    at dp=192 (job 8671243). This variant carries the residual in fp32 across
    ALL layers: swaps both the block (AgptFp32ResidualDepthBlock, emits fp32)
    and the model (AgptFp32ResidualModel, forward keeps fp32 through the layer
    loop then casts to the lm_head dtype before the head). GEMMs stay bf16.
    Deep-copies first (same shared-template caveat as _set_fp32_residual).
    """
    import copy
    from dataclasses import fields

    from torchtitan.experiments.ezpz.agpt.fp32_residual import (
        AgptFp32ResidualDepthBlock,
        AgptFp32ResidualModel,
    )

    cfg = copy.deepcopy(cfg)
    model = cfg.model_spec.model
    model.layers = [
        AgptFp32ResidualDepthBlock.Config(
            **{f.name: getattr(layer, f.name) for f in fields(layer)}
        )
        for layer in model.layers
    ]
    # Rebuild the model config itself as the fp32-residual model subclass,
    # preserving every field (incl. the freshly-swapped layers).
    cfg.model_spec.model = AgptFp32ResidualModel.Config(
        **{f.name: getattr(model, f.name) for f in fields(model)}
    )
    return cfg


def agpt_2b_real() -> FaultTolerantTrainer.Config:
    """agpt_2b with real-valued (cos_sin) RoPE instead of complex.

    The default `RoPE.Config(backend="complex")` uses torch.complex64
    ops that torch.compile inductor refuses to lower:

        UserWarning: Torchinductor does not support code generation
        for complex operators. Performance may be worse than eager.

    The `cos_sin` backend uses real-valued sin/cos rotations that
    inductor can compile, so this flavor exists to A/B test whether
    eliminating the eager fallback inside the compiled graph
    improves XPU throughput.
    """
    return _set_rope_backend(ezpz_agpt_2b(), "cos_sin")


def agpt_2b_tied() -> FaultTolerantTrainer.Config:
    # agpt_2b_real + tied input/output embeddings. At vocab 256128/dim 2048 the
    # untied embed+lm_head are ~53%% of a 2B; tying frees that budget. Arch bet #5
    # validation (tied vs untied loss@fixed-tokens). state_dict_adapter already
    # handles the tied case (adapter lines 82,103).
    cfg = agpt_2b_real()
    cfg.model_spec.model.enable_weight_tying = True
    return cfg


def agpt_2b_flex_attn() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_2b_flex_attn()


def agpt_7b() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_7b()


def agpt_7b_hf() -> FaultTolerantTrainer.Config:
    cfg = ezpz_agpt_7b()
    cfg.dataloader.dataset_path = None
    return cfg


def ezpz_agpt_8b() -> FaultTolerantTrainer.Config:
    return _base_config("8B")


def agpt_8b() -> FaultTolerantTrainer.Config:
    return ezpz_agpt_8b()


def agpt(
    flavor: str,
    local_batch_size: int = 1,
    activation_checkpoint_mode: Literal["none", "full", "selective"] = "full",
    seq_len: int = 8192,
    # IMPORTANT: bfloat16 master weights silently freeze RMSNorm.weight.
    # Norm weights init to 1.0 (bf16 ulp = 7.8e-3); per-step updates
    # ~1.6e-5 round to zero forever, so the model's normalization layers
    # never train. FSDP MixedPrecisionPolicy keeps the bf16 cast for
    # forward/backward; reduce stays fp32 — the fp32 master copy is
    # what enables sub-ulp accumulation.
    # See docs/guides/known-bugs/training-dtype-bf16-norm-freeze.md.
    dtype: Literal["bfloat16", "float32"] = "float32",
    compile: bool = True,
    fsdp_reshard_after_forward: Literal["default", "always", "never"] = "default",
    tensor_parallel_degree: int = 1,
    checkpoint_interval: int = 50,
    hf_assets_path: str = "./assets/hf/gemma-7b",
    dataset_path: str | None = None,
) -> FaultTolerantTrainer.Config:
    cfg = _base_config(flavor)
    cfg.hf_assets_path = hf_assets_path
    # 79th sync (#4085): upstream flipped the DEFAULT spmd_backend to
    # "spmd_types", under which every ezpz config dies with
    #   ValueError: When dp_mesh_dims is provided, all parameters must be
    #   DTensors on the full SPMD mesh ... Got plain tensor for parameter
    # That is an UPSTREAM gap, not ours: core llama3 through core's own
    # trainer fails identically (job 12473444), while partial_dtensor passes
    # in the same job. resolve_fsdp_mesh already guards this shape but only
    # when the WHOLE storage mesh is size 1; at TP=1 with FSDP>1 a param whose
    # only non-Replicate axis is tp still loses its annotation. See
    # docs/guides/known-bugs/spmd-types-plain-tensor.md.
    #
    # This pin was "full_dtensor" until 2026-08-20. Two reasons it moved:
    #   1. upstream is REMOVING full_dtensor (601cf4d23, #4217) -- it is a
    #      dead end, and the next sync deletes the file it depends on
    #   2. the original pin cited job 12473350 as evidence full_dtensor
    #      works, but that probe ran --compile.no-enable; compiled agpt on
    #      full_dtensor hits the vc_check DeviceMesh assertion
    # partial_dtensor is the supported fallback and what upstream itself
    # pins for its rl+hf CI suites (b64d3f6a9, #4228).
    cfg.parallelism.spmd_backend = "partial_dtensor"

    # spmd_types loss-parallel CE needs the full vocab size, and it is the
    # caller's job to supply it: CrossEntropyLoss.Config declares
    #   global_vocab_size: int | None = None
    #   """Full vocabulary size, needed for spmd_types loss-parallel CE."""
    # The partial_dtensor/DTensor branch derives it from pred.shape[-1] and
    # never reads the field, which is why leaving it unset has been harmless
    # so far. Under spmd_types at TP>1 the unset None reaches
    #   chunk_size = (global_vocab_size + tp_world_size - 1) // tp_world_size
    # and the run dies with "unsupported operand type(s) for +: NoneType and
    # int" (components/loss.py:134) before step 1.
    #
    # Read it off the model spec rather than hardcoding, so the flavors that
    # differ (gemma 256128, Llama-3 128256, OLMo-2 100352) stay correct and
    # cannot drift from the model. Guarded because not every loss Config has
    # the field -- ChunkedLossWrapper, set by some configs below, does not.
    _vocab = getattr(getattr(cfg.model_spec, "model", None), "vocab_size", None)
    if _vocab is not None and hasattr(cfg.loss, "global_vocab_size"):
        cfg.loss.global_vocab_size = int(_vocab)
    cfg.debug.print_config = True
    # 80th sync (#4121): training batch fields are counted in TOKENS now.
    # The agpt() signature deliberately KEEPS sequence units -- every caller
    # and every doc says "LBS=5", and rewriting ~40 callsites to pass tokens
    # would make each one carry the seq_len multiplication independently. One
    # conversion here is the whole change for anything built through agpt().
    cfg.training.num_tokens_per_microbatch_per_dp_rank = local_batch_size * seq_len
    # 57th sync: PR #3674 replaced the `mode` string with a policy class
    # hierarchy. `None` disables AC (was mode="none"); FullAC.Config()
    # is the agpt default (was mode="full").
    if activation_checkpoint_mode == "none":
        cfg.activation_checkpoint = None
    elif activation_checkpoint_mode == "selective":
        # Plain upstream SelectiveAC, not the MoE subclass: MoeSelectiveAC
        # exists only to drop all_to_all_single from the save list, which is
        # an EP concern agpt does not have.
        cfg.activation_checkpoint = SelectiveAC.Config()
    else:
        cfg.activation_checkpoint = FullAC.Config()
    cfg.training.max_context_length = seq_len
    cfg.training.dtype = dtype
    cfg.dataloader.dataset = "blendcorpus"
    if dataset_path is None:
        dataset_path = f"torchtitan/experiments/ezpz/data-lists/{ezpz.distributed.get_machine().lower()}/books.txt"
    cfg.dataloader.dataset_path = dataset_path
    # Validator reads from the same blendcorpus corpus, but it pulls from
    # the validation split (see BlendCorpusDataLoader.Config.serve_validation).
    # Also default the validator's data_cache_path to the trainer's so it does
    # not keep the bare ".cache/blendcorpus" default and cold-build the
    # validation index at full scale on the first validate() call -- the race
    # that crashed job 12469584 ("mmap length > file size" at TP>1, mistaken
    # for a validator collective deadlock; see docs/guides/known-bugs/).
    # NOTE: this only aligns the in-config DEFAULT. In production the submit
    # scripts pass the warm path explicitly via
    # --validator.dataloader.data-cache-path (applied by tyro AFTER this
    # builder), which is the operative fix; this copy is defense-in-depth for
    # interactive / non-script callers. Either way the validation-split index
    # must be prewarmed (prewarm_blendcorpus_cache.sh builds it).
    if isinstance(cfg.validator.dataloader, BlendCorpusDataLoader.Config):
        cfg.validator.dataloader.dataset_path = dataset_path
        cfg.validator.dataloader.data_cache_path = cfg.dataloader.data_cache_path
    cfg.metrics.log_freq = 1
    cfg.metrics.enable_wandb = True
    if compile:
        cfg.compile = CompileConfig(enable=True)
    cfg.parallelism.fsdp_reshard_after_forward = fsdp_reshard_after_forward
    cfg.parallelism.tensor_parallel_degree = tensor_parallel_degree
    cfg.checkpoint.enable = True
    cfg.checkpoint.interval = checkpoint_interval
    return cfg


def _base_config(flavor: str) -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
        hf_assets_path="./tests/assets/hf/gemma-7b",
        model_spec=model_registry(flavor),
        tokenizer=EZPZTokenizer.Config(backend="hf"),
        loss=CrossEntropyLoss.Config(),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=200,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            # #4121: tokens, not sequences. 8 seqs x 2048 = 16384.
            num_tokens_per_microbatch_per_dp_rank=8 * 2048,
            max_context_length=2048,
            steps=10000,
        ),
        dataloader=BlendCorpusDataLoader.Config(dataset="c4_test"),
        metrics=MetricsProcessor.Config(log_freq=10),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=FullAC.Config(),
        comm=CommConfig(train_timeout_seconds=100),
        fault_tolerance=FaultTolerance(enable=False),
        # Walltime-aware checkpointing: guarantee a save before the PBS walltime
        # runs out (otherwise a short job can save nothing -- see the train loop
        # in trainer.py). The failover submit scripts export
        # $WALLTIME_DEADLINE_EPOCH (absolute job_start+walltime timestamp,
        # survives failover retries -- PREFERRED) and $WALLTIME_SECONDS (relative
        # fallback). 0/unset disables it (interactive runs, non-PBS). Overridable
        # on the CLI via --walltime-deadline-epoch / --walltime-seconds.
        walltime_deadline_epoch=int(
            os.environ.get("WALLTIME_DEADLINE_EPOCH", "0") or "0"
        ),
        walltime_seconds=int(os.environ.get("WALLTIME_SECONDS", "0") or "0"),
        # Validator runs on the blendcorpus validation split (5% of the
        # corpus by default — see BlendCorpusDataLoader.Config.split). The
        # validator builds its own dataloader from this Config every time
        # validate() is called, so serve_validation=True ensures it gets
        # held-out samples instead of the train split.
        # Default enable=False keeps prior behavior; flip per config or via
        # --validator.enable on the CLI.
        # Uses EzpzValidator (subclass of Validator) which fixes loss
        # reporting on TP > 1 — see torchtitan/experiments/ezpz/validator.py.
        validator=EzpzValidator.Config(
            enable=False,
            freq=200,
            steps=10,
            dataloader=BlendCorpusDataLoader.Config(
                dataset="blendcorpus",
                serve_validation=True,
                infinite=False,
            ),
        ),
    )


def ezpz_agpt_debugmodel() -> FaultTolerantTrainer.Config:
    return agpt("debugmodel", local_batch_size=2)


def agpt_debugmodel_local() -> FaultTolerantTrainer.Config:
    """Fully offline debug config for local single-device dev (e.g. macOS/MPS).

    Same debug model as ``agpt_debugmodel`` but with every dependency that needs
    the cluster (Megatron-preprocessed BlendCorpus data, the gemma-7b HF assets,
    wandb, checkpointing) swapped for something that exists in the repo:

    - dataloader: the bundled ``c4_test`` split (``tests/assets/c4_test``) via
      ``HuggingFaceTextDataLoader`` instead of ``BlendCorpusDataLoader`` (which
      reads ``data-lists/<machine>/books.txt`` + preprocessed .bin/.idx).
    - tokenizer: the checked-in fast tokenizer at ``tests/assets/tokenizer``.
    - wandb and checkpointing off, few steps: a self-contained smoke run.

    Nothing here is production-relevant; it exists so ``--config
    agpt_debugmodel_local`` runs end to end with no network and no cluster data.
    """
    cfg = agpt_debugmodel()
    cfg.hf_assets_path = "./tests/assets/tokenizer"
    cfg.tokenizer = EZPZTokenizer.Config(backend="hf")
    # Grain (#4088) DELETED HuggingFaceTextDataLoader with no drop-in
    # replacement -- the equivalent is a GrainDataLoader built from a
    # SingleDatasetConfig(source=HuggingFaceStreamingSource, processor=
    # TextProcessor), a different object graph rather than a rename. Ported
    # lazily: raise where the config is USED so importing the registry (and
    # therefore every blendcorpus production config) still works.
    raise NotImplementedError(
        "c4_test needs porting to GrainDataLoader after the 80th sync; see "
        "torchtitan/components/data/dataset.py SingleDatasetConfig"
    )
    cfg.validator.enable = False
    cfg.metrics.enable_wandb = False
    cfg.checkpoint.enable = False
    cfg.training.steps = 10
    cfg.training.max_context_length = 512
    # 2 seqs x 512 = 1024 tokens (#4121 unit change)
    cfg.training.num_tokens_per_microbatch_per_dp_rank = 2 * 512
    return cfg


def agpt_debugmodel_qknorm_local() -> FaultTolerantTrainer.Config:
    """``agpt_debugmodel_local`` plus QK-Norm.

    QK-Norm adds two more RMSNorms per layer (on head_dim), also initialized
    at 1.0, so it has the same bf16-master freeze exposure as the pre/post
    block norms. Used by the master-weight-dtype ablation
    (scripts/oneoff/fp32_norms_ablation.py) to test the "this recurs for any
    parameter initialized near 1.0, QK-norm gains being exactly that" claim
    in docs/production/agpt/30b-exp/README.md Section 6.
    """
    cfg = agpt_debugmodel_local()
    cfg.model_spec = model_registry("debugmodel_qknorm")
    return cfg


def ezpz_agpt_2b() -> FaultTolerantTrainer.Config:
    return agpt("2b", activation_checkpoint_mode="none")


def agpt_2b_chunkedce() -> FaultTolerantTrainer.Config:
    """agpt_2b with ChunkedLossWrapper to keep peak memory low.

    With vocab=256128 the unchunked logits are ~16 GB at LBS=2 / seq=8192,
    which OOMs on Aurora's 64 GB tiles. ChunkedLossWrapper(num_chunks=8) caps
    the peak slice at ~2 GB. Set lm_head module reference at trainer init
    via the set_lm_head plumbing in `experiments/ezpz/trainer.py`.
    """
    cfg = ezpz_agpt_2b()
    cfg.loss = ChunkedLossWrapper.Config(num_chunks=8)
    return cfg


# ---------------------------------------------------------------------------
# MDS mid-training anneal A/B (fork the Megatron-DeepSpeed AuroraGPT-2B base)
# ---------------------------------------------------------------------------
#
# Base: the MDS stage-3-end checkpoint `global_step138650` (val ~2.05),
# converted to a torchtitan DCP at
#   outputs/checkpoints/agpt-2b-mds-gs138650/step-0/
# by a sibling job (referenced by path -- must exist before these run).
#
# SCHEDULE FRAMING -- MDS stage-3 was ALREADY a constant-LR phase, not an
# anneal. The production script train_aGPT_2B_sophiag_stage3.sh sets
# LR_DECAY_STYLE=constant at LR=2.17e-5; the Megatron scheduler
# (optimizer_param_scheduler.py get_lr) returns max_lr at every post-warmup
# step when lr_constant_plus_cooldown is False (confirmed False in the MDS
# training-config dump). So the val 2.40->2.05 drop over the final 0.706T
# tokens was a DATA-MIX shift (dolmino stage-2 -> nvidia-math1/code2 stage-3),
# NOT a learning-rate decay. This A/B is therefore the FIRST true LR anneal on
# a base that never annealed -- WSD decay-to-0 (ARM B) has never been applied.
#
# Both arms fork model weights only (fresh optimizer + LR schedule + step
# counter) via --checkpoint.initial-load-path, mirroring the CPT recipe
# (docs/production/cpt/README.md). Per the CPT re-warm-shock lesson, LR is
# GENTLE (2e-6 constant, warmup 20) -- NOT re-warmed to the 2.17e-5 peak,
# which disrupted the converged base in the first CPT pilot.
#
# Data mix: on-the-fly gemma tokenization of a raw-text HF math dataset via
# the auto-registering HF dataloader (datasets.py). The pre-tokenized Sunspot
# math lists are Llama2-vocab and unusable here; streaming a raw-text dataset
# lets the model's own gemma tokenizer (EZPZTokenizer, vocab 256000) encode it
# at runtime. open-web-math/open-web-math is gemma-safe: a single default
# config, a "text" column (the auto-register default), no config_name needed.
#
# Everything except the LR schedule is IDENTICAL between the two arms so the
# A/B isolates the schedule.

# Fork target produced by the sibling DCP-conversion job.
#
# MUST be absolute. CheckpointManager.Config.__post_init__ (checkpoint.py:399)
# rejects a relative initial_load_path -- but ONLY when set at construction
# time. We assign it AFTER the Config is built (below), so that guard never
# re-runs and a relative value would slip through. At load time it is then
# resolved against the process CWD; if that is not the repo root the path does
# not exist and CheckpointManager.load() takes the "No checkpoint was provided,
# this is a fresh start." branch (checkpoint.py:879) -- a SILENT no-load that
# leaves the model random-init. Anchoring to the repo root (this file is at
# <repo>/torchtitan/experiments/ezpz/agpt/config_registry.py, i.e. parents[4])
# makes it CWD-independent. An explicit env override is honored for relocated
# checkpoints.
_MDS_ANNEAL_BASE = os.environ.get(
    "MDS_ANNEAL_BASE",
    str(
        Path(__file__).resolve().parents[4]
        / "outputs/checkpoints/agpt-2b-mds-gs138650/step-0"
    ),
)
# Raw-text HF math dataset, streamed + gemma-tokenized on the fly. Default HF
# config, "text" column -> works through the auto-register fallback with no
# extra wiring. (HuggingFaceTB/finemath would need an explicit config_name --
# finemath-4plus -- so open-web-math is the drop-in choice.)
_MDS_ANNEAL_DATASET = "open-web-math/open-web-math"
# Gentle constant LR shared by both arms (well below the 2.17e-5 MDS peak).
_MDS_ANNEAL_LR = 2e-6
# ~50B-token anneal at GBS=6144 x seq 8192 (GBS*seq ~= 50.3M tok/step).
_MDS_ANNEAL_STEPS = 1000


def _agpt_2b_mds_anneal_base() -> FaultTolerantTrainer.Config:
    """Shared fork config for the MDS anneal A/B (schedule set by callers)."""
    # vocab-256000 flavor, seq_len 8192; no AC (2B fits), matches the 2b path.
    cfg = agpt("2b-mds", activation_checkpoint_mode="none", seq_len=8192)
    # Fork the converted MDS base: model weights only, fresh optimizer + step
    # counter (initial_load_model_only defaults True). Same mechanism the CPT
    # sweep used to fork the plateaued v2 base.
    #
    # Fail loudly at config-build time if the DCP is missing. Otherwise a bad
    # path is only "discovered" as a SILENT no-load at load time (the model
    # stays random-init and training starts at loss ~12 instead of ~2), which
    # is exactly the failure this A/B is meant to avoid. A missing .metadata
    # means the dir is not a valid DCP (dcp.load would also silently no-op).
    if not (Path(_MDS_ANNEAL_BASE) / ".metadata").is_file():
        raise ValueError(
            f"MDS anneal base DCP not found or invalid at {_MDS_ANNEAL_BASE!r} "
            "(expected a <dir>/.metadata). Set $MDS_ANNEAL_BASE to the absolute "
            "path of the converted step-0 DCP, or run the HF->DCP converter "
            "first. A missing base would otherwise silently load nothing and "
            "train from random init."
        )
    cfg.checkpoint.initial_load_path = _MDS_ANNEAL_BASE
    cfg.checkpoint.initial_load_model_only = True
    # Stream a raw-text HF math dataset -> gemma tokenization at runtime. Drop
    # the blendcorpus data_file_list path so the HF hub path is used.
    cfg.dataloader.dataset = _MDS_ANNEAL_DATASET
    cfg.dataloader.dataset_path = None
    # Fresh optimizer at the gentle anneal LR (SophiaG was the MDS optimizer,
    # but the fork discards optimizer state; AdamW at a low LR is the safe,
    # batch-robust choice for a short anneal -- consistent with the CPT gentle
    # retry, which also used a low constant LR on the same base family).
    cfg.optimizer = default_adamw(lr=_MDS_ANNEAL_LR)
    cfg.training.steps = _MDS_ANNEAL_STEPS
    cfg.metrics.enable_wandb = True
    return cfg


def agpt_2b_mds_anneal_flat() -> FaultTolerantTrainer.Config:
    """ARM A (control): fork MDS gs138650, CONSTANT low LR (no decay).

    decay_ratio=0.0 -> decay phase is zero steps -> the multiplier is 1.0 at
    every post-warmup step (see the DECAY_RATIO=0 constant-LR precedent in
    docs/journal.md). min_lr_factor=1.0 pins the floor at the full LR so even
    if any decay were computed it would be a no-op. This continues the base at
    a flat gentle LR -- the null hypothesis for the anneal.
    """
    cfg = _agpt_2b_mds_anneal_base()
    cfg.lr_scheduler.warmup_steps = 20
    cfg.lr_scheduler.decay_ratio = 0.0
    cfg.lr_scheduler.decay_type = "linear"
    cfg.lr_scheduler.min_lr_factor = 1.0
    cfg.checkpoint.folder = "checkpoints/agpt-2b-mds-anneal-flat"
    return cfg


def agpt_2b_mds_anneal_wsd() -> FaultTolerantTrainer.Config:
    """ARM B (treatment): fork MDS gs138650, WSD decay-to-0 anneal.

    Same base, same gentle peak LR (2e-6), same data + budget as ARM A -- only
    the schedule differs. decay_ratio=1.0 makes the whole post-warmup run the
    decay phase; min_lr_factor=0.0 + decay_type="linear" drives LR linearly to
    0 by the final step (classic Warmup-Stable-Decay with a zero stable
    window). This is the first true LR anneal applied to the MDS base.
    """
    cfg = _agpt_2b_mds_anneal_base()
    cfg.lr_scheduler.warmup_steps = 20
    cfg.lr_scheduler.decay_ratio = 1.0
    cfg.lr_scheduler.decay_type = "linear"
    cfg.lr_scheduler.min_lr_factor = 0.0
    cfg.checkpoint.folder = "checkpoints/agpt-2b-mds-anneal-wsd"
    return cfg


# --- data-mix A/B (stage-2 mid-training): flat won the anneal, so DATA is the
# lever, not the schedule. These arms fork the SAME MDS base at the SAME winning
# CONSTANT LR 2e-6 and differ ONLY in the training data mix. The MDS base is
# already math+code-saturated (stage-3), so the highest-value axis is DIVERSITY /
# anti-forgetting: does swapping math-web for general/edu web forget math or
# improve general ability? Eval reserves FineMath-4+ + wikitext as FROZEN
# holdouts that NO arm trains on (disjoint by construction). Wave 1 = these two
# single-corpus arms (zero new dataloader code); weighted blends are a phase 2.


def _agpt_2b_mds_mix_base() -> FaultTolerantTrainer.Config:
    """Shared fork config for the data-mix arms: MDS base, constant LR 2e-6
    (the anneal winner), same budget -- caller sets the dataset + folder."""
    cfg = _agpt_2b_mds_anneal_base()
    # Flat / constant-LR schedule (the anneal-proven winner).
    cfg.lr_scheduler.warmup_steps = 20
    cfg.lr_scheduler.decay_ratio = 0.0
    cfg.lr_scheduler.decay_type = "linear"
    cfg.lr_scheduler.min_lr_factor = 1.0
    return cfg


def agpt_2b_mds_mix_owm() -> FaultTolerantTrainer.Config:
    """CONTROL arm: open-web-math 100% (== the anneal flat winner's data).

    Identical to agpt_2b_mds_anneal_flat by construction -- kept as its own name
    so the data-mix matrix reads uniformly and its checkpoints/eval land in the
    mix output tree. Anchors the mix experiment to the anneal result.
    """
    cfg = _agpt_2b_mds_mix_base()
    cfg.dataloader.dataset = _MDS_ANNEAL_DATASET  # open-web-math/open-web-math
    cfg.dataloader.dataset_path = None
    cfg.checkpoint.folder = "checkpoints/agpt-2b-mds-mix-owm"
    return cfg


def agpt_2b_mds_mix_edu() -> FaultTolerantTrainer.Config:
    """DIVERSITY-extreme arm: fineweb-edu 100% (general/edu web, no math).

    Tests the core anti-forgetting question: does replacing math-web with edu-web
    forget math (FineMath holdout rises) or improve general ability (wikitext
    holdout falls)? fineweb_edu_local is a registered LOCAL parquet dir (140
    files, ~280GB, verified 'text' column) -- no HF hub, no 429 at 384 ranks.
    """
    cfg = _agpt_2b_mds_mix_base()
    cfg.dataloader.dataset = "fineweb_edu_local"
    cfg.dataloader.dataset_path = None
    cfg.checkpoint.folder = "checkpoints/agpt-2b-mds-mix-edu"
    return cfg


# --- Phase 2: weighted math/edu BLENDS. Wave 1 showed the extremes bracket the
# space: owm-100 holds math (FineMath 1.804); edu-100 CATASTROPHICALLY forgets it
# (+0.308) for a tiny general gain (-0.024). The math-loss curve is steep, the
# general-gain curve flat -> the optimal mix is math-HEAVY. These arms find where.
# They REPLACE cfg.dataloader wholesale with an InterleavedHuggingFaceTextDataLoader
# (setting .sources on the inherited BlendCorpusDataLoader.Config silently no-ops);
# all sources infinite=True (post_init guard requires uniform infinite). Weights
# are token-mixture ratios.


def _agpt_2b_mds_mix_blend(
    owm_weight: float, edu_weight: float, folder: str
) -> FaultTolerantTrainer.Config:
    """Shared builder for owm/edu weighted-blend mix arms."""
    from torchtitan.hf_datasets.text_datasets import (
        HFDataSource,
        InterleavedHuggingFaceTextDataLoader,
    )

    cfg = _agpt_2b_mds_mix_base()
    # Replace the whole dataloader Config -- the base is a BlendCorpusDataLoader
    # .Config whose .sources field does not exist, so setting it would no-op.
    cfg.dataloader = InterleavedHuggingFaceTextDataLoader.Config(
        sources=[
            HFDataSource(
                dataset="open-web-math/open-web-math", weight=owm_weight, infinite=True
            ),
            HFDataSource(
                dataset="fineweb_edu_local", weight=edu_weight, infinite=True
            ),
        ],
        seed=42,
        stopping_strategy="all_exhausted",
    )
    cfg.checkpoint.folder = folder
    return cfg


def agpt_2b_mds_mix_owm_edu_7525() -> FaultTolerantTrainer.Config:
    """BLEND math-heavy: 75% open-web-math / 25% fineweb-edu. The predicted
    winner -- keeps most math (steep loss) while adding a little general."""
    return _agpt_2b_mds_mix_blend(0.75, 0.25, "checkpoints/agpt-2b-mds-mix-owm75-edu25")


def agpt_2b_mds_mix_owm_edu_5050() -> FaultTolerantTrainer.Config:
    """BLEND balanced: 50% open-web-math / 50% fineweb-edu. Brackets the ratio
    axis on the more-general side of the math-heavy arm."""
    return _agpt_2b_mds_mix_blend(0.50, 0.50, "checkpoints/agpt-2b-mds-mix-owm50-edu50")


def agpt_2b_mds_mix_owm_edu_9010() -> FaultTolerantTrainer.Config:
    """BLEND math-heaviest: 90% open-web-math / 10% fineweb-edu. Probes the
    math-heavy EDGE past the 75/25 winner -- 75/25 already captured ~all of
    edu's wikitext (general) gain at zero FineMath (math) cost, so this arm
    asks whether an even smaller edu slot retains that general gain while
    giving back more of the math corpus. Brackets the ratio axis on the
    more-math side of 75/25."""
    return _agpt_2b_mds_mix_blend(0.90, 0.10, "checkpoints/agpt-2b-mds-mix-owm90-edu10")


# --- Wave 3: SCIENCE-corpus 25% blends. The 75/25 owm/edu winner showed the
# 25% slot is the lever; these swap generic edu for SCIENCE-dense corpora (the
# DOE-mission version). Same recipe (MDS fork, constant LR, 10B tok, 75/25),
# only the 25% source changes. Decided (arm-vs-arm) on a DISJOINT science judge:
# held-out common-pile/peS2o NLL (no arm trains on it) + MMLU-STEM confirmatory;
# FineMath + wikitext stay as retention guards.


def _agpt_2b_mds_mix_blend_src(
    owm_weight: float, src_dataset: str, src_weight: float, folder: str
) -> FaultTolerantTrainer.Config:
    """Generalized owm/<science-src> weighted-blend builder (mirrors
    _agpt_2b_mds_mix_blend; src_dataset is any registered local-parquet name)."""
    from torchtitan.hf_datasets.text_datasets import (
        HFDataSource,
        InterleavedHuggingFaceTextDataLoader,
    )

    cfg = _agpt_2b_mds_mix_base()
    cfg.dataloader = InterleavedHuggingFaceTextDataLoader.Config(
        sources=[
            HFDataSource(
                dataset="open-web-math/open-web-math", weight=owm_weight, infinite=True
            ),
            HFDataSource(dataset=src_dataset, weight=src_weight, infinite=True),
        ],
        seed=42,
        stopping_strategy="all_exhausted",
    )
    cfg.checkpoint.folder = folder
    return cfg


def agpt_2b_mds_mix_owm_cosmo_7525() -> FaultTolerantTrainer.Config:
    """SCIENCE arm: 75% open-web-math / 25% cosmopedia-science (synthetic STEM
    textbooks: auto_math_text+khanacademy+openstax+stanford+wikihow). Tests
    whether synthetic-science textbooks in the 25% slot beat generic edu."""
    return _agpt_2b_mds_mix_blend_src(
        0.75, "cosmopedia_science_local", 0.25,
        "checkpoints/agpt-2b-mds-mix-owm75-cosmo25",
    )


def agpt_2b_mds_mix_owm_nemotron_7525() -> FaultTolerantTrainer.Config:
    """SCIENCE arm: 75% open-web-math / 25% Nemotron-CC-Math-4+ (layout-aware
    CC math+science web). The memo's #1 science lever. GATED corpus -- requires
    HF access granted for nvidia/Nemotron-CC-Math-v1 + precache of config 4plus
    registered as nemotron_cc_math_4plus_local."""
    return _agpt_2b_mds_mix_blend_src(
        0.75, "nemotron_cc_math_4plus_local", 0.25,
        "checkpoints/agpt-2b-mds-mix-owm75-nemotron25",
    )


# --- olmo-mix anneal A/B (second base for the "both bases" anneal experiment) ---
# The olmo-mix step-92859 base (v2 256N chain, val ~2.65, fp32 DCP) is the WEAKER
# but apples-to-apples base (the CPT pilot forked it). vocab 256128 (stock 2b, not
# 256000 like MDS). It uses the plain "2b" flavor via ezpz_agpt_2b (COMPLEX RoPE),
# which is how production 2B was trained -- agpt_2b_real (cos_sin) is a
# compile-throughput TEST flavor only, NEVER production, and forking a complex-
# trained base with a cos_sin config applies the wrong Q/K rotation (different
# channel pairing) and corrupts the model. Same anneal mechanism/LR/data as MDS.
_OLMO_ANNEAL_BASE = os.environ.get(
    "OLMO_ANNEAL_BASE",
    str(
        Path(__file__).resolve().parents[4]
        / "outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-92859"
    ),
)


def _agpt_2b_olmo_anneal_base() -> FaultTolerantTrainer.Config:
    """Shared fork config for the olmo-mix anneal A/B (schedule set by callers)."""
    # ezpz_agpt_2b = stock vocab-256128 flavor + COMPLEX RoPE, matching how the
    # olmo base was actually trained. (Do NOT use agpt_2b_real here: cos_sin RoPE
    # is a compile-throughput test flavor and mismatches the complex-trained base
    # -- it rotates a different Q/K channel pairing and corrupts the fork.)
    cfg = ezpz_agpt_2b()
    # #4121 HAZARD: num_tokens_per_microbatch_per_dp_rank was already computed
    # by agpt() as local_batch_size * seq_len. Overriding the sequence length
    # AFTER that does NOT update the token count, so the effective batch would
    # silently change. agpt()'s default seq_len is 8192, so this assignment is
    # currently a no-op -- but only by coincidence. Recompute explicitly so it
    # stays correct if that default ever moves.
    _lbs = cfg.training.num_tokens_per_microbatch_per_dp_rank // 8192
    cfg.training.max_context_length = 8192
    cfg.training.num_tokens_per_microbatch_per_dp_rank = _lbs * 8192
    cfg.activation_checkpoint = None
    if not (Path(_OLMO_ANNEAL_BASE) / ".metadata").is_file():
        raise ValueError(
            f"olmo anneal base DCP not found or invalid at {_OLMO_ANNEAL_BASE!r} "
            "(expected a <dir>/.metadata). Set $OLMO_ANNEAL_BASE to the absolute "
            "path of the step-92859 DCP. A missing base would silently load "
            "nothing and train from random init."
        )
    cfg.checkpoint.initial_load_path = _OLMO_ANNEAL_BASE
    cfg.checkpoint.initial_load_model_only = True
    cfg.dataloader.dataset = _MDS_ANNEAL_DATASET
    cfg.dataloader.dataset_path = None
    cfg.optimizer = default_adamw(lr=_MDS_ANNEAL_LR)
    cfg.training.steps = _MDS_ANNEAL_STEPS
    cfg.metrics.enable_wandb = True
    return cfg


def agpt_2b_olmo_anneal_flat() -> FaultTolerantTrainer.Config:
    """ARM A (control): fork olmo-mix step-92859, CONSTANT low LR (no decay)."""
    cfg = _agpt_2b_olmo_anneal_base()
    cfg.lr_scheduler.warmup_steps = 20
    cfg.lr_scheduler.decay_ratio = 0.0
    cfg.lr_scheduler.decay_type = "linear"
    cfg.lr_scheduler.min_lr_factor = 1.0
    cfg.checkpoint.folder = "checkpoints/agpt-2b-olmo-anneal-flat"
    return cfg


def agpt_2b_olmo_anneal_wsd() -> FaultTolerantTrainer.Config:
    """ARM B (treatment): fork olmo-mix step-92859, WSD decay-to-0 anneal."""
    cfg = _agpt_2b_olmo_anneal_base()
    cfg.lr_scheduler.warmup_steps = 20
    cfg.lr_scheduler.decay_ratio = 1.0
    cfg.lr_scheduler.decay_type = "linear"
    cfg.lr_scheduler.min_lr_factor = 0.0
    cfg.checkpoint.folder = "checkpoints/agpt-2b-olmo-anneal-wsd"
    return cfg


def ezpz_agpt_2b_flex_attn() -> FaultTolerantTrainer.Config:
    return agpt("2b_flex_attn", local_batch_size=2)


def ezpz_agpt_20b_flex_attn() -> FaultTolerantTrainer.Config:
    return agpt("20b_flex_attn")


def agpt_20b_flex_attn() -> FaultTolerantTrainer.Config:
    return agpt("20b_flex_attn")


def ezpz_agpt_7b() -> FaultTolerantTrainer.Config:
    return agpt("7b", local_batch_size=2, seq_len=4096, hf_assets_path="./assets/hf/llama-2-7b-hf")


def _load_json_overrides() -> dict[str, Any]:
    path = os.environ.get(TT_CONFIG_JSON_ENV, "").strip()
    if not path:
        raise ValueError(
            f"{TT_CONFIG_JSON_ENV} must point to a JSON file when using *_from_json configs."
        )

    with open(path, encoding="utf-8") as f:
        overrides = json.load(f)

    if not isinstance(overrides, dict):
        raise ValueError(
            f"Expected top-level JSON object in {path!r}, got {type(overrides).__name__}."
        )

    return overrides


def _apply_config_overrides(
    target: Any,
    overrides: dict[str, Any],
    path: str = "",
) -> None:
    for key, value in overrides.items():
        if not hasattr(target, key):
            raise KeyError(f"Unknown config field {key!r} at path {path or '<root>'}.")

        current_value = getattr(target, key)
        field_path = f"{path}.{key}" if path else key

        if isinstance(value, dict):
            if not is_dataclass(current_value):
                raise TypeError(
                    f"Expected dataclass at {field_path!r} for nested override, "
                    f"got {type(current_value).__name__}."
                )
            _apply_config_overrides(current_value, value, field_path)
            continue

        setattr(target, key, value)


def _config_from_json(flavor: str) -> FaultTolerantTrainer.Config:
    cfg = _base_config(flavor)
    _apply_config_overrides(cfg, _load_json_overrides())
    return cfg


def ezpz_agpt_debugmodel_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("debugmodel")


def ezpz_agpt_2b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("2b")


def ezpz_agpt_7b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("7b")


def ezpz_agpt_8b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("8B")


def ezpz_agpt_blendcorpus_debugmodel() -> FaultTolerantTrainer.Config:
    cfg = _base_config("debugmodel")
    cfg.dataloader.dataset = "blendcorpus"
    if isinstance(cfg.tokenizer, EZPZTokenizer.Config):
        cfg.tokenizer.backend = "sptoken"
    return cfg


def ezpz_agpt_20b() -> FaultTolerantTrainer.Config:
    return agpt("20b")


def agpt_20b() -> FaultTolerantTrainer.Config:
    return agpt("20b")


def agpt_20b_noac() -> FaultTolerantTrainer.Config:
    """agpt_20b with activation checkpointing OFF.

    Diagnostic for the vc_check/DeviceMesh assertion. The backend matrix
    (job 12473420) showed the failure correlates with whether
    model.parallelize() ran, not with the spmd backend:

                        TP=1                  TP=2
        partial         PASS (skipped)        vc_check (ran)
        full_dtensor    vc_check (ran)        vc_check (ran)

    CLAUDE.md describes the bug as "compile + AC + TP". compile and
    parallelize are both confirmed necessary; AC is the untested leg and the
    one that decides the remedy. If AC is required, selective AC may dodge it
    (as it did for the MoE router recompute bug) and we keep compile AND TP.
    If not, the only lever is avoiding model.parallelize().
    """
    return agpt("20b", activation_checkpoint_mode="none")


def agpt_20b_selac() -> FaultTolerantTrainer.Config:
    """agpt_20b with selective AC instead of FullAC.

    The vc_check assertion needs all three of compile + AC + model.parallelize
    (job 12473421: AC=none does not fire it, no-compile does not fire it).
    Selective AC saves a chosen op set instead of recomputing whole blocks, so
    it may avoid whatever DeviceMesh-bearing value FullAC stashes -- the same
    move that fixed the MoE router recompute bug earlier today. If it works,
    we keep compile AND TP instead of surrendering one of them.
    """
    return agpt("20b", activation_checkpoint_mode="selective")


def agpt_20b_chunkedce() -> FaultTolerantTrainer.Config:
    """agpt_20b with ChunkedLossWrapper. See agpt_2b_chunkedce for rationale."""
    cfg = ezpz_agpt_20b()
    cfg.loss = ChunkedLossWrapper.Config(num_chunks=8)
    return cfg


def agpt_20b_real() -> FaultTolerantTrainer.Config:
    """agpt_20b with real-valued (cos_sin) RoPE. See agpt_2b_real."""
    return _set_rope_backend(ezpz_agpt_20b(), "cos_sin")


def ezpz_agpt_30b() -> FaultTolerantTrainer.Config:
    return agpt("30b")


def agpt_30b() -> FaultTolerantTrainer.Config:
    """The proposed next flagship. See docs/production/agpt/30b-exp/.

    28.1B params: dim=6144, 64 layers, 48 heads (head_dim 128), 8 KV heads,
    ffn 16384, gemma 256,128 vocab. Geometry is interpolated between 20B and
    80B -- the proposal fixes only dim=6144.
    """
    return agpt("30b")


def agpt_30b_real() -> FaultTolerantTrainer.Config:
    """agpt_30b with real-valued (cos_sin) RoPE. See agpt_2b_real."""
    return _set_rope_backend(ezpz_agpt_30b(), "cos_sin")


def agpt_30b_llama3tok() -> FaultTolerantTrainer.Config:
    """30B with the Llama-3 128k vocab -- see docs/production/agpt/30b-exp/.

    26.5B params vs 28.1B for the gemma-vocab variant. Halves the embedding
    (3.15B -> 1.58B) using a tokenizer we already vendor, and the proposal's
    own fertility table prefers Llama on code.

    The Llama-3 assets are set HERE rather than left to the caller. This
    function previously inherited the family default (gemma-7b, vocab 256,128)
    while its model declares vocab_size=128,256, which is an inconsistent
    config: the tokenizer can emit ids the embedding cannot index. Its
    docstring also told callers to pass ``--tokenizer.path``, which is not a
    real flag -- the field is top-level ``hf_assets_path`` (``--hf-assets-path``).
    Every run of this config had in fact died at argument parsing with
    "Unrecognized options: --tokenizer.path", which is why it never produced a
    single step (jobs 12473195, 12473200).
    """
    return agpt("30b_llama3tok", hf_assets_path="./assets/hf/Llama-3.1-8B")


def agpt_30b_olmo2tok() -> FaultTolerantTrainer.Config:
    """30B with OLMo-2's 100,352 vocab -- see docs/production/agpt/30b-exp/.

    exp07's nine-tokenizer bake-off measured OLMo-2 tied with Llama-3.1 on
    fertility (225,749 vs 225,539 tok/MB on held-out olmo-mix-1124 text) while
    using a 22% smaller vocab. At dim=6144 that is 1.23B of embedding against
    Llama-3's 1.58B -- 0.34B freed for the same token cost, and exp05 showed
    freed HBM converts into batch size, the dominant throughput lever here.

    OLMo-2's tokenizer is also the only one tested that was fit on our own
    corpus family (dolma/olmo-mix) at production scale.

    26.2B params. Sets the OLMo-2 assets explicitly, as agpt_30b_llama3tok
    does, so the tokenizer and the embedding cannot disagree.
    """
    return agpt("30b_olmo2tok", hf_assets_path="./assets/hf/OLMo-2-1124-7B")


# Local fineweb-edu shards for the optimizer comparison. The 80th upstream sync
# (#4088) deleted HuggingFaceTextDataLoader, so `dataset="fineweb_edu_local"` on
# a BlendCorpusDataLoader.Config now raises -- the replacement is a Grain
# dataset graph, not a renamed class.
#
# HOW MANY SHARDS. Each parquet holds ~726k rows / ~1.4B tokens, and
# HuggingFaceRandomAccessSource materializes what it is given (streaming=False),
# so pointing at all 140 files would try to hold 267 GB. 16 shards is ~22B
# tokens: comfortably more than the 10B-per-arm budget, so no arm repeats data,
# while staying small enough to materialize. Sorted + sliced, never sampled, so
# every arm reads byte-identical input.
_FINEWEB_EDU_DIR = "/lus/tegu/projects/datasets/datasets/fineweb-edu-100BT/sample/100BT"
_FINEWEB_EDU_NUM_SHARDS = 16


def _fineweb_edu_shards() -> list[str]:
    """The exact shard list every comparison arm reads.

    Sorted so the selection is deterministic across arms and across reruns:
    two arms trained on different shards would not be comparable, and that
    difference would be invisible in the loss curve.
    """
    import glob

    files = sorted(glob.glob(f"{_FINEWEB_EDU_DIR}/*.parquet"))
    if len(files) < _FINEWEB_EDU_NUM_SHARDS:
        raise ValueError(
            f"expected >= {_FINEWEB_EDU_NUM_SHARDS} parquet shards in "
            f"{_FINEWEB_EDU_DIR}, found {len(files)}"
        )
    return files[:_FINEWEB_EDU_NUM_SHARDS]


def _use_fineweb_edu(cfg: FaultTolerantTrainer.Config) -> FaultTolerantTrainer.Config:
    """Point a config at the LOCAL fineweb-edu parquet shards via Grain.

    agpt() defaults dataset_path to data-lists/<machine>/books.txt, which on
    Sunspot is THREE shards totalling ~11 GB -- about 5.8B tokens. A 10B-token
    comparison arm would therefore loop that corpus 1.7x, and repeated data
    bends the loss curve in ways that need not be the same for every optimizer.
    That is precisely the confound a fixed-batch optimizer comparison exists to
    exclude, so the arms read a corpus larger than their budget instead.

    books.txt is also the ONLY blendcorpus list that resolves on Sunspot -- every
    other list under data-lists/sunspot/ points at /gila, which is not mounted
    here. So there is no "use blendcorpus with a bigger corpus" option; reading
    real data at this scale requires the Grain path.

    Streaming allenai/olmo-mix-1124 from the hub was the other candidate and is
    rejected: it 429-storms at this rank count (datasets.py module docstring
    documents the failure at 384 ranks; these arms run 192), and the on-disk
    olmo-mix cache holds only the wiki slice (6.1 GB), smaller AND narrower than
    the books list it would replace.

    ConcatThenSplitPackingConfig matches how the blendcorpus path feeds the
    model: documents concatenated and split at max_context_length, so every
    sequence is full rather than padded, and tokens-per-step means what the
    batch arithmetic assumes.
    """
    cfg.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(
            dataset=SingleDatasetConfig(
                source=HuggingFaceRandomAccessSource.Config(
                    path="parquet",
                    split="train",
                    load_dataset_kwargs={"data_files": _fineweb_edu_shards()},
                ),
                processor=TextProcessor.Config(),
                post_filters=(lambda sample: sample is not None,),
            )
        )
    )
    return cfg

def _use_hf_streaming(
    cfg: FaultTolerantTrainer.Config,
    *,
    path: str,
    name: str | None = None,
    split: str = "train",
) -> FaultTolerantTrainer.Config:
    """Read an arbitrary Hugging Face dataset by STREAMING it.

    Same object graph as :func:`_use_fineweb_edu` -- ConcatThenSplitPacking
    over a SingleDatasetConfig -- but sourced from
    HuggingFaceStreamingSource instead of local parquet shards, so it needs
    no preprocessed corpus on disk.

    This exists so the model/dataloader path can be exercised on a machine
    that has GPUs but none of our pretokenized data (Perlmutter, a laptop,
    CI). It deliberately does NOT go through the ezpz BlendCorpus wrapper,
    which raises NotImplementedError for every dataset except
    "blendcorpus" after the 80th sync deleted the HF delegate -- the
    replacement is exactly this Grain object graph.

    Not for production runs: streaming throughput is network-bound and the
    shard order is not the deterministic, sorted selection the comparison
    arms rely on.
    """
    cfg.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(
            dataset=SingleDatasetConfig(
                source=HuggingFaceStreamingSource.Config(
                    path=path,
                    name=name,
                    split=split,
                ),
                processor=TextProcessor.Config(),
                post_filters=(lambda sample: sample is not None,),
            )
        )
    )
    return cfg


def agpt_2b_real_stream_c4() -> FaultTolerantTrainer.Config:
    """agpt_2b_real reading streamed C4 -- an integration smoke config.

    Purpose is to exercise the full dataloader -> model -> attention path
    (the #4121 token-unit flags and the [B, L*N, H] attention unflatten) on
    hardware that has no local corpus. Loss values are not meaningful; the
    question is whether real batches flow and the shapes are right.
    """
    cfg = agpt("2b_real")
    # wikitext rather than C4: it is small enough to pre-cache on a login
    # node, which matters because NERSC COMPUTE nodes have no egress -- a
    # live hub call there dies with errno 524 / "not cached in None".
    return _use_hf_streaming(
        cfg, path="Salesforce/wikitext", name="wikitext-103-raw-v1"
    )


def agpt_30b_olmo2tok_optcmp_adamw() -> FaultTolerantTrainer.Config:
    """AdamW arm of the fixed-batch optimizer comparison, on fineweb-edu.

    Same model and data as the mano/sophiag arms; only the optimizer differs.
    See docs/experiments/optimizer-comparison/README.md.
    """
    cfg = agpt("30b_olmo2tok", hf_assets_path="./assets/hf/OLMo-2-1124-7B")
    return _use_fineweb_edu(cfg)


def agpt_30b_olmo2tok_mano() -> FaultTolerantTrainer.Config:
    """agpt_30b_olmo2tok with the Mano optimizer instead of AdamW.

    Mano is manifold-normalized; the 2B competition configs
    (competition/configs.py) run it at lr=3e-4, the same peak the 30B AdamW
    baseline uses, so the LR is held constant across the two and the optimizer
    is the only variable.

    NOT a resume target. Mano's optimizer state has a different shape from
    AdamW's, so this cannot load an agpt-30b-olmo2tok-converge checkpoint --
    it is a fresh run with its own checkpoint folder, and comparisons against
    the AdamW baseline are per-token, not per-wallclock.
    """
    cfg = agpt("30b_olmo2tok", hf_assets_path="./assets/hf/OLMo-2-1124-7B")
    cfg.optimizer = default_mano(lr=3.0e-4)
    return _use_fineweb_edu(cfg)


def agpt_30b_olmo2tok_sophiag() -> FaultTolerantTrainer.Config:
    """agpt_30b_olmo2tok with the SophiaG optimizer instead of AdamW.

    Third arm of the fixed-batch optimizer comparison (AdamW / Mano / SophiaG,
    all at GBS=960). SophiaG is a second-order method: it estimates a diagonal
    Hessian and clips the per-coordinate update at rho, so its useful LR range
    does not have to resemble either first-order optimizer's.

    The lr here is a PLACEHOLDER. Do not trust it -- the comparison runs pass
    --optimizer.lr explicitly from the LR-finder result measured at THIS batch
    size. Batch dependence is not a small effect for these optimizers: the 2B
    finder put Mano at 4.79e-03 while the 80B at GBS=6144 wanted ~3e-6, three
    orders of magnitude apart, so an inherited LR says nothing.

    SophiaG has form here: the 2026-07-03 80B run NaN'd at step 14 and burned
    ~12h. That is what --nan-abort-consecutive and the finder's blow-up
    detection are for; expect this arm to be the one that finds the ceiling.

    NOT a resume target for an AdamW or Mano checkpoint -- the optimizer state
    shapes differ. Fresh run, own checkpoint folder, per-token comparisons.
    """
    cfg = agpt("30b_olmo2tok", hf_assets_path="./assets/hf/OLMo-2-1124-7B")
    cfg.optimizer = default_sophiag(lr=3.0e-4)
    return _use_fineweb_edu(cfg)


def agpt_30b_olmo2tok_muon() -> FaultTolerantTrainer.Config:
    """agpt_30b_olmo2tok with the Muon optimizer instead of AdamW.

    Fourth arm of the fixed-batch optimizer comparison (AdamW / Mano /
    SophiaG all ran at GBS=960). Muon won the 2B competition
    (competition/README.md: agpt2b-n2-1000steps, 3.557) and has never been
    run at 30B.

    The lr here is a PLACEHOLDER, as it was for the mano and sophiag arms.
    Do not trust it: the comparison runs pass --optimizer.lr explicitly from
    an LR-finder result measured at THIS batch size. The 2B competition value
    was 2.4e-3, and the three measured 30B/GBS=960 suggestions all landed near
    3-6e-05, so the inherited constant is roughly 40-80x too high.

    MUON PARTITIONS BY SHAPE, NOT BY NAME. optimizer/muon.py:85-89 gates on
    ``p.ndim == 2 and max(p.shape) <= 10000``; everything else silently falls
    through to an internal AdamW branch. Measured on this exact config, only
    21.5% of the 26.2B parameters are on the Muon path:

      MUON  : 256 tensors,  5.637B  -- every attention projection, x64 layers:
                                       wq/wo (6144,6144), wk/wv (1024,6144)
      AdamW : 323 tensors, 20.561B  -- w1/w2/w3 (16384 > 10000), tok_embeddings
                                       and lm_head (100352 > 10000), and all
                                       129 rank-1 norm weights

    So this arm is a Muon/AdamW hybrid, not a Muon run: attention runs on Muon,
    the FFN and embeddings on AdamW. That boundary is a coincidence of
    dim=6144 and hidden_dim=16384 straddling the hardcoded 10000, not a
    designed split -- at a flavor with dim > 10000 the attention projections
    would fall through too and this would be a pure-AdamW run wearing a Muon
    label. Recorded so the comparison against the other three arms is read
    correctly.

    For the Muon-path tensors the LR is then rescaled by
    ``0.2 * sqrt(max(A, B))`` (muon.py:130-139). All four Muon shapes here have
    max dim 6144, so the factor is a UNIFORM 15.677x -- the effective LR on
    every Muon tensor is 15.677x the number the finder reports, while the
    AdamW-path majority sees the base LR unscaled.

    NOT a resume target for an AdamW, Mano, or SophiaG checkpoint -- the
    optimizer state shapes differ. Fresh run, own checkpoint folder,
    per-token comparisons.
    """
    cfg = agpt("30b_olmo2tok", hf_assets_path="./assets/hf/OLMo-2-1124-7B")
    cfg.optimizer = default_muon(lr=3.0e-4)
    return _use_fineweb_edu(cfg)


def agpt_30b_olmo2tok_muon_norescale() -> FaultTolerantTrainer.Config:
    """agpt_30b_olmo2tok_muon with Muon's 15.677x LR rescale DISABLED.

    Exists to answer one question the first Muon sweep raised and could not
    settle. That sweep suggested a base LR of 5.68e-04 -- 18.6x AdamW's and
    10.1x Mano's -- but only 21.5% of parameters are on the Muon path, and
    only those get the 0.2*sqrt(max(A,B)) = 15.677x rescale. The other 78.5%
    (w1/w2/w3, embeddings, lm_head, norms) run on an internal AdamW branch at
    the base LR unscaled.

    So the reported base could mean either of two things and the first sweep
    cannot tell them apart:

      (a) Muon genuinely wants a high base, and the 8.90e-03 effective rate on
          its own tensors is what it is asking for; or
      (b) the AdamW-path MAJORITY is anchoring the curve, and 5.68e-04 mostly
          describes what those parameters tolerate -- in which case the Muon
          tensors are along for the ride at 15.677x whatever the majority
          picks.

    With ``adjuster_lr_ref=False`` every parameter sees the same base LR and
    the two populations are no longer coupled by a constant. If the suggestion
    lands near 5.68e-04 again, the rescale was not what set it and (b) is the
    better reading. If it lands near 8.90e-03 -- i.e. the previous effective
    Muon-path rate -- then the Muon tensors were the binding constraint and
    (a) is. Anything else means the two populations interact in a way neither
    reading captures.

    Note ``adjuster_lr_ref`` must be passed EXPLICITLY. ``default_muon`` omits
    the key entirely, so it falls through to ``muon.py:56``, which defaults it
    True -- the opposite of what this config is for. Verified reaching
    optimizer_kwargs before this config was registered.

    Equivalences worth knowing when reading this against upstream: ezpz's
    ``adjuster_lr_ref=True`` is upstream ``dist_muon.py``'s
    ``match_rms_adamw``, and ``False`` is its ``original``. Neither is the
    Bernstein/muP-correct ``spectral_unclamped`` = sqrt(d_out/d_in), which
    ezpz's Muon does not expose.

    NOT a resume target for any other arm. Fresh run, own checkpoint folder.
    """
    cfg = agpt("30b_olmo2tok", hf_assets_path="./assets/hf/OLMo-2-1124-7B")
    cfg.optimizer = default_muon(lr=3.0e-4, adjuster_lr_ref=False)
    return _use_fineweb_edu(cfg)


# ---------------------------------------------------------------------------
# muP ladder entry points
# ---------------------------------------------------------------------------
# agpt/mup.py registers the muP widths into `agpt_configs`, which is the MODEL
# FLAVOR registry. `--config` does not read that -- it resolves the callables
# in THIS module (see the "Config function not found" error, which lists them).
# Without these wrappers the muP flavors can be inspected but never trained,
# which is how a stage-4 rehearsal failed 9/9 with rc=1 before the first real
# allocation was spent.
#
# Each pairs the muP model flavor with `default_mup_adamw` at the SAME base_dim.
# That pairing is not optional: the readout init and the per-group LRs are two
# halves of one parametrization, and mixing a muP model with a plain AdamW
# config silently produces neither (see mup.py's module docstring).


def _mup_cfg(flavor: str, *, dim: int, base_dim: int) -> FaultTolerantTrainer.Config:
    """agpt under muP at one rung of the ladder.

    ``eta`` comes from the MUP_ETA environment variable when set, defaulting to
    8e-4. It is an env var rather than a CLI flag because muP's four groups
    hold a RATIO -- hidden at eta/m, the rest at eta -- and tyro can only
    override one group at a time. Sweeping eta via
    ``--optimizer.param-groups.0...lr`` silently moves only the embedding
    group's LR and leaves the 42-parameter hidden group at the default, which
    is exactly the bug that made a stage-4 rehearsal report flat curves and a
    spurious NO TRANSFER verdict.
    """
    import os

    from torchtitan.experiments.ezpz.agpt.mup import default_mup_adamw

    eta = float(os.environ.get("MUP_ETA", "8e-4"))
    # The muP ladder is built at OLMo-2 vocab (100,352). agpt()'s default
    # hf_assets_path is gemma-7b (vocab 256,128), so omitting this loads the
    # WRONG tokenizer -- the model trains and reports a plausible loss while
    # the token ids mean something else, which is exactly the Polaris 20B
    # eval-gibberish failure. mup.py's own _mup_trainer_config already passes
    # this; these config_registry copies dropped it.
    cfg = agpt(flavor, hf_assets_path="./assets/hf/OLMo-2-1124-7B")
    cfg.optimizer = default_mup_adamw(eta, dim=dim, base_dim=base_dim)
    return cfg


def agpt_mup_1536() -> FaultTolerantTrainer.Config:
    """muP base rung. m=1, so this is the width eta is tuned at."""
    return _mup_cfg("mup_1536", dim=1536, base_dim=1536)


def agpt_mup_3072() -> FaultTolerantTrainer.Config:
    """muP ladder, m=2."""
    return _mup_cfg("mup_3072", dim=3072, base_dim=1536)


def agpt_mup_6144() -> FaultTolerantTrainer.Config:
    """muP ladder top rung, m=4. Production 30B geometry."""
    return _mup_cfg("mup_6144", dim=6144, base_dim=1536)


def agpt_30b_olmo2tok_muon_ffn() -> FaultTolerantTrainer.Config:
    """Muon at 30B with the FFN on the Muon path, not just attention.

    agpt_30b_olmo2tok_muon is a Muon/AdamW hybrid by accident: the shape gate
    (muon.py, `max(p.shape) <= muon_max_dim`) is meant to exclude embeddings
    and the LM head, but at dim=6144 / hidden_dim=16384 it also excludes
    w1/w2/w3. Only the attention projections -- 21.5% of parameters -- run on
    Muon, and the no-rescale LR sweep showed the AdamW-path majority is what
    sets the measured optimum.

    Raising muon_max_dim to 20000 puts w1/w2/w3 on Muon while leaving the
    embedding and head (100352) off, which is what the gate was always trying
    to express. Verified on the real 30B shapes: 2 of 5 distinct shapes on
    Muon at the default, 4 of 5 at 20000, embedding excluded in both.

    That takes the Muon path from 21.5% to roughly 95% of parameters, so this
    is the first agpt config where a Muon result is a statement about Muon.

    BLOCKED: THIS CONFIG DOES NOT TRAIN. Muon raises at construction because
    the FFN tensors exceed the bf16 Newton-Schulz overflow ceiling
    (muon.py:_NS_BF16_SAFE_DIM). Kept in the registry because the partition
    measurement it produced is worth having and because it documents a real
    constraint, not as a runnable arm.

    HOW THAT WAS LEARNED. Job 12474361 ran this config through the LR finder:
    step 1 loss 11.9940 (correct for a fresh 30B init), step 2 NaN, at
    lr=1e-6, and every one of the 90 sweep points thereafter NaN. The finder
    then reports no suggestion, so the run reads as "inconclusive" rather than
    "broken" -- the failure mode this codebase keeps producing.

    The cause was already documented in zeropower_via_newtonschulz5's own
    docstring: Newton-Schulz casts to bf16 and A @ A overflows at large
    dimensions. The historical 10000 cutoff was therefore doing TWO jobs, not
    one -- excluding the embedding/head AND keeping the 16384-dim FFN below
    that threshold. The comment in an earlier revision of this file, that
    20000 is "what the gate was always trying to express", was wrong about
    half its purpose.

    TO UNBLOCK, in order: make zeropower_via_newtonschulz5 accumulate A @ A
    and A @ A @ A in fp32, verify no NaN on a 2N smoke, raise
    _NS_BF16_SAFE_DIM, and only then measure an LR. Do NOT inherit 5.68e-04 --
    that was measured on the 21.5% hybrid and the no-rescale arm showed it
    tracks the AdamW-path majority this config removes.

    NOT a resume target for any other arm: the Muon/AdamW parameter split
    differs, so the optimizer state shapes do not match.
    """
    cfg = agpt("30b_olmo2tok", hf_assets_path="./assets/hf/OLMo-2-1124-7B")
    cfg.optimizer = default_muon(lr=3.0e-4, muon_max_dim=20000)
    return _use_fineweb_edu(cfg)


def agpt_30b_olmo2tok_mup() -> FaultTolerantTrainer.Config:
    """muP stage 5: the 30B arm, comparable to the optcmp AdamW baseline.

    agpt_mup_6144 has the right GEOMETRY but not the right data. The optimizer
    comparison's arms all read fineweb-edu through _use_fineweb_edu() and use
    the OLMo-2 tokenizer assets; a muP arm on different data could not be
    compared against AdamW's measured 2.51357 at 23.59B tokens, which is the
    entire point of stage 5.

    So this is agpt_mup_6144's parametrization on agpt_30b_olmo2tok's data
    path. The only intended difference from agpt_30b_olmo2tok_optcmp_adamw is
    muP itself: the readout init, the muP attention scale, and four LR groups
    with hidden at eta/m instead of one catch-all.

    LR comes from stage 4, which is the whole argument for doing this. The
    transfer measurement put the optimum at eta=6.4e-5 at BOTH dim 3072 and
    6144, with interior minima on each. Passed via MUP_ETA (see _mup_cfg for
    why it is an env var and not a CLI flag).

    NOT a resume target for any optcmp checkpoint -- the optimizer state has
    four param groups where those have one. Fresh run, own checkpoint folder.
    """
    from torchtitan.experiments.ezpz.agpt.mup import default_mup_adamw

    cfg = agpt("mup_6144", hf_assets_path="./assets/hf/OLMo-2-1124-7B")
    cfg.optimizer = default_mup_adamw(6.4e-5, dim=6144, base_dim=1536)
    return _use_fineweb_edu(cfg)


def agpt_mup_tiny_256() -> FaultTolerantTrainer.Config:
    """CPU-sized muP base rung -- 6 layers, vocab 32000. Rehearsal only."""
    return _mup_cfg("mup_tiny_256", dim=256, base_dim=256)


def agpt_mup_tiny_512() -> FaultTolerantTrainer.Config:
    """CPU-sized muP ladder, m=2."""
    return _mup_cfg("mup_tiny_512", dim=512, base_dim=256)


def agpt_mup_tiny_1024() -> FaultTolerantTrainer.Config:
    """CPU-sized muP ladder, m=4."""
    return _mup_cfg("mup_tiny_1024", dim=1024, base_dim=256)


def ezpz_agpt_50b() -> FaultTolerantTrainer.Config:
    return agpt("50b")


def agpt_50b() -> FaultTolerantTrainer.Config:
    return agpt("50b")


def ezpz_agpt_50b_wide() -> FaultTolerantTrainer.Config:
    return agpt("50B_wide", tensor_parallel_degree=2)


def agpt_50b_wide() -> FaultTolerantTrainer.Config:
    return agpt("50B_wide", tensor_parallel_degree=2)


def ezpz_agpt_70b_wide() -> FaultTolerantTrainer.Config:
    return agpt("70B_wide", tensor_parallel_degree=2)


def agpt_70b_wide() -> FaultTolerantTrainer.Config:
    return agpt("70B_wide", tensor_parallel_degree=2)


def ezpz_agpt_80b() -> FaultTolerantTrainer.Config:
    return agpt("80B", tensor_parallel_degree=2)


def agpt_80b() -> FaultTolerantTrainer.Config:
    return agpt("80B", tensor_parallel_degree=2)


def agpt_80b_chunkedce() -> FaultTolerantTrainer.Config:
    """agpt_80b with ChunkedLossWrapper. See agpt_2b_chunkedce for rationale."""
    cfg = ezpz_agpt_80b()
    cfg.loss = ChunkedLossWrapper.Config(num_chunks=8)
    return cfg


def agpt_80b_real() -> FaultTolerantTrainer.Config:
    """agpt_80b with real-valued (cos_sin) RoPE. See agpt_2b_real."""
    return _set_rope_backend(ezpz_agpt_80b(), "cos_sin")


def _set_z_loss(
    cfg: FaultTolerantTrainer.Config, coef: float = 1e-4
) -> FaultTolerantTrainer.Config:
    """Swap the loss for cross entropy plus an output z-loss penalty.

    Adds ``coef * (log Z)^2``, which removes cross entropy's invariance to a
    constant logit shift and keeps the output logits at a bounded scale.
    PaLM's coefficient is 1e-4 (arXiv:2204.02311 Sec 5); OLMo-2 cites z-loss
    among its stability changes.

    Carries over ``global_vocab_size`` if the incoming loss config had it --
    it is set from the model spec further up and is needed by the
    spmd_types loss-parallel path.

    Refuses to silently drop a ChunkedLossWrapper: chunked CE never
    materializes the full logits, so there is nothing to take a logsumexp
    over, and quietly replacing it would change memory behaviour as well as
    the loss. Use a non-chunked flavor as the base.
    """
    from torchtitan.experiments.ezpz.zloss import CrossEntropyWithZLoss

    # __qualname__, NOT __name__. Every nested loss config here is literally
    # named "Config" (ChunkedLossWrapper.Config, CrossEntropyLoss.Config...),
    # so a __name__ check matches nothing and the guard never fires. Verified
    # the hard way: the first version silently swapped a chunked config.
    if "ChunkedLossWrapper" in type(cfg.loss).__qualname__:
        raise ValueError(
            "z-loss cannot wrap ChunkedLossWrapper: chunked cross entropy "
            "never materializes full logits, so log Z is not available. "
            "Start from a non-chunked flavor."
        )
    vocab = getattr(cfg.loss, "global_vocab_size", None)
    cfg.loss = CrossEntropyWithZLoss.Config(
        z_loss_coef=coef, global_vocab_size=vocab
    )
    return cfg


def agpt_2b_zloss() -> FaultTolerantTrainer.Config:
    """agpt_2b with output z-loss at PaLM's 1e-4. THE SMOKE TARGET.

    2B is where z-loss should be validated before anything larger: it trains
    in minutes at 2N, and the thing to confirm is mechanical -- that the
    penalty is reported in metrics, that loss stays finite, and that
    throughput does not collapse. 2B has never had a logit-overflow problem,
    so this flavor is NOT expected to improve anything. It exists to prove the
    plumbing before the 80B arm below is worth running.
    """
    return _set_z_loss(agpt_2b(), coef=1e-4)


def agpt_80b_zloss() -> FaultTolerantTrainer.Config:
    """agpt_80b with output z-loss at PaLM's 1e-4. THE ACTUAL TARGET.

    The 80B NaN is a bf16 overflow, and the two score-bounding remedies
    already tried both bound ATTENTION scores and are both blocked on this
    stack: softcap hard-codes compiled flex_attention (broken on XPU) and
    QK-Norm crashes in backward at 4N/TP=4 where plain agpt_80b is 10/10
    clean. Nothing bounds the OUTPUT logits, and z-loss needs no
    flex_attention, no compile and no new backward kernel.

    NOT A PREDICTED FIX. The fp32-residual result (clean at 4N, still NaN at
    dp=192) narrowed the overflow to a bf16 SUBLAYER GEMM -- attention QK^T or
    the FFN SwiGLU intermediate -- and bounding the output logits may not
    reach either. This is the cheapest untried item on the list, not a
    prediction. Run it against the CONFIRMED-STABLE corner (TP=4, LBS=1,
    bf16, batch via GAS, >=4N) so a NaN means something.

    Smoke agpt_2b_zloss first.
    """
    return _set_z_loss(ezpz_agpt_80b(), coef=1e-4)


def agpt_80b_fp32res() -> FaultTolerantTrainer.Config:
    """agpt_80b with the fp32 residual-stream fix (task #21/#24).

    Targets the bf16 residual overflow that NaNs 80B at production dp. Does the
    two residual adds in fp32 while keeping attention/FFN GEMMs bf16. Much
    cheaper than mixed-precision-param=float32 (which fp32s ALL activations).
    Validate NaN-free + throughput before production use.
    """
    return _set_fp32_residual(ezpz_agpt_80b())


def agpt_80b_real_fp32res() -> FaultTolerantTrainer.Config:
    """agpt_80b_real (cos_sin RoPE) + the fp32 residual-stream fix."""
    return _set_fp32_residual(_set_rope_backend(ezpz_agpt_80b(), "cos_sin"))


def agpt_80b_fp32res_depth() -> FaultTolerantTrainer.Config:
    """agpt_80b with the FULL-DEPTH fp32 residual stream (task #27).

    Carries the residual in fp32 across all 84 layers (vs the per-block
    agpt_80b_fp32res which re-truncates to bf16 at each boundary and still
    NaN'd at dp=192). GEMMs stay bf16. The dp>186 wall test for this variant is
    the open question.
    """
    return _set_fp32_residual_depth(ezpz_agpt_80b())


def agpt_80b_qknorm() -> FaultTolerantTrainer.Config:
    """agpt_80b + QK-Norm (task: 80B dp>186 NaN, score-bounding fix).

    RMSNorm on Q,K per head bounds the attention-score magnitude directly --
    the prime remaining overflow suspect after full-depth fp32 residual proved
    necessary-but-insufficient (agpt_80b_fp32res_depth still NaN'd at dp=192).
    Near-free at train time; no ckpt-compat cost (no surviving 80B prod ckpt).
    Wall test: 4N smoke for numeric sanity, then 64N/dp=192 (the wall that
    killed the residual protos).
    """
    return agpt("80B_qknorm", tensor_parallel_degree=2)


def agpt_80b_softcap() -> FaultTolerantTrainer.Config:
    """agpt_80b + logit softcap (Gemma-2 tanh score_mod, cap +/-30).

    Alternative attention-score-bounding lever to QK-Norm. Same wall-test plan.
    """
    return agpt("80B_softcap", tensor_parallel_degree=2)


def agpt_80b_qknorm_softcap() -> FaultTolerantTrainer.Config:
    """agpt_80b + QK-Norm AND logit softcap -- both score-bounding levers, for
    the dp=192 wall test if either alone is insufficient."""
    return agpt("80B_qknorm_softcap", tensor_parallel_degree=2)


def ezpz_agpt_80b_alt() -> FaultTolerantTrainer.Config:
    return agpt("80B_alt", tensor_parallel_degree=2)


def agpt_80b_alt() -> FaultTolerantTrainer.Config:
    return agpt("80B_alt", tensor_parallel_degree=2)


def ezpz_agpt_80b_wide() -> FaultTolerantTrainer.Config:
    return agpt("80B_wide", tensor_parallel_degree=2)


def agpt_80b_wide() -> FaultTolerantTrainer.Config:
    return agpt("80B_wide", tensor_parallel_degree=2)


def ezpz_agpt_80b_deep() -> FaultTolerantTrainer.Config:
    return agpt("80B_deep", tensor_parallel_degree=2)


def agpt_80b_deep() -> FaultTolerantTrainer.Config:
    return agpt("80B_deep", tensor_parallel_degree=2)


def ezpz_agpt_80b_deep_alt() -> FaultTolerantTrainer.Config:
    return agpt("80B_deep_alt", tensor_parallel_degree=2)


def agpt_80b_deep_alt() -> FaultTolerantTrainer.Config:
    return agpt("80B_deep_alt", tensor_parallel_degree=2)


def ezpz_agpt_80b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("80B")


def ezpz_agpt_80b_wide_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("80B_wide")


def ezpz_agpt_80b_deep_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("80B_deep")


# Competition speedrun configs — makes them discoverable via --config
from torchtitan.experiments.ezpz.competition.configs import (  # noqa: E402, F401
    speedrun_2b_adamw,
    speedrun_2b_adamw_cosine,
    speedrun_2b_adamw_fast_warmup,
    speedrun_2b_adamw_high_lr,
    speedrun_2b_adamw_qknorm,
    speedrun_2b_adamw_short_decay,
    speedrun_2b_mano,
    speedrun_2b_mano_1e3,
    speedrun_2b_mano_cosine,
    speedrun_2b_mano_high_lr,
    speedrun_2b_mano_qknorm,
    speedrun_2b_muon,
    speedrun_2b_muon_aggressive,
    speedrun_2b_muon_cosine,
    speedrun_2b_muon_fast_warmup,
    speedrun_2b_muon_qknorm,
    speedrun_2b_muon_short_decay,
    speedrun_2b_sophiag,
    speedrun_2b_spam,
    speedrun_2b_torchmuon,
    speedrun_2b_torchmuon_cosine,
)
from torchtitan.experiments.ezpz.competition.configs import (  # noqa: E402, F401
    full_2b_adamw,
    full_2b_adamw_qknorm,
    full_2b_mano,
    full_2b_mano_qknorm,
    full_2b_muon,
    r4_adamw,
    r4_adamw_higher_lr,
    r4_adamw_qknorm,
    r4_adamw_qknorm_softcap,
    r4_adamw_softcap,
    r4_mano,
    r4_mano_higher_lr,
    r4_mano_qknorm,
    r5_adamw_constant_lr,
    r5_mano_constant_lr,
    r5_mano_lr1e3,
    r5_mano_lr2e3,
    r5_mano_lr3e4,
    r5_mano_lr6e4,
    r5_schedulefree,
    smoke_2b_50steps,
    smoke_2b_async_ckpt,
    smoke_2b_async_ckpt_pinned,
    speedrun_2b_kitchen_sink,
    speedrun_2b_mano_kitchen_sink,
    speedrun_2b_relu2,
    speedrun_2b_softcap,
)
