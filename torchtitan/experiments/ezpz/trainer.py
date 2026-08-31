# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from typing import cast

import ezpz

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.components.data.loader import DataloaderExhaustedError
from torchtitan.components.loss import ChunkedLossWrapper, IGNORE_INDEX
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.ezpz.lr_finder import LRFinderConfig
from torchtitan.experiments.ezpz import signal_stop
from torchtitan.experiments.ezpz.xpu_graph import maybe_wrap_with_xpu_graph
from torchtitan.experiments.ezpz.ckpt_key_compat import (
    maybe_install_flat_attention_compat,
)
from torchtitan.experiments.ezpz.ckpt_owner_claim import check_and_claim
from torchtitan.experiments.torchft.config.job_config import FaultTolerance
from torchtitan.experiments.torchft.manager import (
    TorchFTManager as FTManager,
    maybe_semi_sync_training,
)
from torchtitan.experiments.torchft.optimizer import (
    TorchFTOptimizersContainer as FTOptimizersContainer,
)
from torchtitan.protocols import BaseModel
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.tools.profiler import Profiler
from torchtitan.trainer import Trainer


_CLIP_FOREACH_LOGGED = False


def _clip_foreach() -> bool:
    """Whether to use the fused multi-tensor path in ``clip_grad_norm_``.

    Defaults to True, matching upstream. ``EZPZ_CLIP_NO_FOREACH=1`` selects the
    unfused loop, which exists to test whether the fused path is what exhausts
    level_zero resources under HSDP on XPU (30B dies in
    ``clip_grad.py:106 torch.stack([norm.to(first_device) ...])`` with
    UR_RESULT_ERROR_OUT_OF_RESOURCES at 71.68% memory -- resource exhaustion,
    not an OOM). Logged once so a run's own output proves which path it took,
    rather than the caller assuming the env var reached the ranks.
    """
    global _CLIP_FOREACH_LOGGED
    foreach = os.environ.get("EZPZ_CLIP_NO_FOREACH", "0") != "1"
    if not _CLIP_FOREACH_LOGGED:
        _CLIP_FOREACH_LOGGED = True
        logger.info("EZPZ_CLIP_NO_FOREACH active: clip_grad foreach=%s", foreach)
    return foreach



def _set_pg_timeouts_xpu_aware(
    timeout: timedelta,
    parallel_dims: ParallelDims,
) -> None:
    """Apply ``timeout`` to every PG in the mesh, with explicit XPU support.

    Upstream ``dist_utils.set_pg_timeouts`` delegates to
    ``torch.distributed.distributed_c10d._set_pg_timeout``, whose device
    dispatch only knows about cpu (gloo) and cuda (nccl / gloo /
    torchcomms). On XPU the loop adds no backends, emits the warning
    ``"Set timeout is now only supported for either nccl or gloo."``,
    and never calls ``_set_default_timeout``. Result: ``train_timeout_seconds``
    silently no-ops and a hung collective burns the full PBS walltime
    instead of aborting (see
    ``docs/upstream-issues/train_timeout_xpu_silent_noop.md``).

    Workaround: set the timeout on every mesh PG ourselves (with the
    safety barrier the upstream helper uses), then additionally call
    ``ProcessGroupXCCL.set_timeout`` on any xccl-backed groups.

    61st sync: we no longer delegate to ``dist_utils.set_pg_timeouts``.
    The spmd_types series switched it from
    ``distributed_c10d._set_pg_timeout`` to ``torch.distributed.set_timeout``,
    which does NOT exist in our pinned torch 2.13 (AttributeError ->
    every collective unguarded). So we inline the pre-sync behavior
    against ``_set_pg_timeout`` (present in torch 2.13's
    ``distributed_c10d``), keeping this shim independent of upstream's
    timeout-API churn. Remove once PyTorch's timeout dispatch learns
    about XPU and our torch exposes the matching API.
    """
    from torch.distributed import distributed_c10d as c10d

    device_module = dist_utils.device_module
    # Flush in-flight work under the old timeout before lowering it
    # (mirrors upstream set_pg_timeouts' safety barrier). The gloo (CPU)
    # backend does not take device_ids, and any integer index there is
    # resolved against the default accelerator (e.g. MPS on a Mac), which has
    # no c10d::barrier -- so only pass device_ids for real accelerators.
    if dist_utils.device_type == "cpu":
        torch.distributed.barrier()
    else:
        torch.distributed.barrier(device_ids=[device_module.current_device()])
    device_module.synchronize()

    timeout_groups: list[torch.distributed.ProcessGroup | None] = [
        mesh.get_group()
        for mesh in parallel_dims.get_all_one_dimensional_meshes().values()
    ] + [None]
    for group in timeout_groups:
        c10d._set_pg_timeout(timeout, group)

    xpu_device = torch.device("xpu")
    if not (
        torch.distributed.is_xccl_available() and torch.xpu.is_available()
    ):
        return

    from torch._C._distributed_c10d import ProcessGroupXCCL

    groups: list[torch.distributed.ProcessGroup | None] = [
        mesh.get_group()
        for mesh in parallel_dims.get_all_one_dimensional_meshes().values()
    ] + [None]
    patched = 0
    for group in groups:
        if group is None:
            group = torch.distributed.distributed_c10d._get_default_group()
        if xpu_device not in group._device_types:
            continue
        backend = group._get_backend(xpu_device)
        if isinstance(backend, ProcessGroupXCCL):
            backend.set_timeout(timeout)
            patched += 1
    if patched:
        logger.info(
            f"Applied train timeout {timeout} to {patched} xccl ProcessGroup(s) "
            "(upstream _set_pg_timeout has no xpu branch)."
        )


class FaultTolerantTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        fault_tolerance: FaultTolerance = field(default_factory=FaultTolerance)
        lr_finder: LRFinderConfig = field(default_factory=LRFinderConfig)

        # Batch-size ramp (analogous to LR warmup, but for global batch
        # size). Ramps the effective gradient-accumulation count -- and
        # thus effective GBS = LBS * dp_degree * GAS -- linearly from
        # `batch_ramp_start_gas` up to the full `gradient_accumulation_steps`
        # over the first `batch_ramp_steps` optimizer steps, then holds at
        # full GBS. LBS and DP degree are unchanged, so this needs no
        # dataloader/parallelism changes; loss stays correct because the
        # ezpz train_step normalizes by global_valid_tokens (token count),
        # not by microbatch count.
        #
        # Motivation: 80B NaNs at GBS>=372 (grad_norm NaNs at step 2) but
        # is stable at GBS=168. A ramp lets training stabilize at small GBS
        # before reaching the target GBS. `batch_ramp_steps=0` disables it
        # (default), preserving exact current behavior.
        batch_ramp_steps: int = 0
        """Number of steps to linearly ramp GAS from batch_ramp_start_gas to
        the full gradient_accumulation_steps. 0 disables the ramp."""
        batch_ramp_start_gas: int = 1
        """GAS to start the ramp from (effective GBS at step 0 =
        LBS * dp_degree * batch_ramp_start_gas). Must be >= 1 and <= the
        full gradient_accumulation_steps."""

        # Walltime-aware checkpointing. A short job can otherwise run its whole
        # PBS window and save NOTHING (e.g. 20B 512N at ~48s/step never reaches
        # the next interval=100 boundary inside a 2h window), wasting all the
        # compute. The train loop watches the clock and, once it is within
        # walltime_checkpoint_margin_seconds of the deadline, forces a final
        # checkpoint and stops cleanly -- guaranteeing a save before walltime
        # regardless of step rate, model size, or startup cost.
        #
        # TWO ways to set the deadline (deadline_epoch wins if both set):
        #
        # * walltime_deadline_epoch (PREFERRED) -- an ABSOLUTE Unix timestamp
        #   (job_start + PBS_walltime), exported once as $WALLTIME_DEADLINE_EPOCH
        #   by the failover submit scripts. Because it is absolute, it SURVIVES
        #   failover relaunches: a mid-job bad-node swap restarts the trainer but
        #   the deadline is unchanged, so the backstop still fires before the real
        #   PBS walltime. This fixes the 2026-06-27 failure where the relative
        #   budget reset on retry and the job was SIGTERM'd mid-save (see
        #   memory project_walltime_ckpt_resets_on_failover_retry).
        #
        # * walltime_seconds (FALLBACK) -- a RELATIVE budget in seconds, measured
        #   from loop entry. Resets on each relaunch, so it is unreliable across
        #   failover retries; kept only for backward compatibility / non-PBS use.
        #
        # Both 0/unset disables the feature (exact prior behavior).
        walltime_deadline_epoch: int = 0
        """Absolute Unix timestamp (job_start + walltime) past which a final
        checkpoint is forced. From $WALLTIME_DEADLINE_EPOCH. Survives failover
        retries. Takes precedence over walltime_seconds. 0 disables."""
        walltime_seconds: int = 0
        """RELATIVE walltime budget in seconds from loop entry (fallback when
        walltime_deadline_epoch is unset). Resets on failover relaunch -- prefer
        walltime_deadline_epoch. 0 disables."""
        walltime_checkpoint_margin_seconds: int = 600
        """Force a final checkpoint + stop once within this many seconds of the
        deadline. Must cover one checkpoint save + async flush at the target
        scale (20B/512N+ may want ~900; 600 is safe for <=2B/256N)."""

        # Signal-triggered final checkpoint. The walltime guard above only
        # fires when the deadline is CONFIGURED and the clock is read between
        # steps. It cannot help when the stop arrives from outside as a signal:
        #   * `timeout N ezpz launch ...` sends SIGTERM at N (every walltime-
        #     bound PBS script here wraps the launch in one),
        #   * PBS sends SIGTERM at walltime before SIGKILL,
        #   * an operator sends SIGINT/SIGTERM to stop a run early.
        # In all three the process dies wherever it happens to be and every
        # step since the last interval boundary is discarded. Job 12473683
        # (30B/Mano/64N) was projected to stop at ~step 147 with its last save
        # at 125 -- ~22 steps, ~1.4B tokens, ~1h of 64-node time thrown away.
        #
        # The handler CANNOT save: dcp.save is a collective and every rank must
        # call it from the same place in the loop, so saving from a handler
        # that fires at an arbitrary instruction would hang or corrupt. It only
        # sets a flag; the loop checks it between steps and takes the same
        # forced-save path the walltime guard uses.
        save_on_signal: bool = True
        """Catch SIGTERM/SIGINT and force a final checkpoint before exiting,
        instead of losing every step since the last interval boundary. The
        handler only sets a flag -- the save happens in the train loop, where
        all ranks participate. Set False to restore the default OS behavior."""

        # NaN-abort guard. A diverged run (e.g. an 80B optimizer NaN) otherwise
        # keeps "training" on non-finite losses for the ENTIRE walltime -- the
        # 2026-07-03 80B SophiaG run NaN'd at step 14 and burned ~12h / ~6,100
        # node-h producing garbage. When enabled, the loop aborts after this
        # many CONSECUTIVE non-finite (NaN/inf) reported losses, reclaiming the
        # rest of the window. Counts reset on any finite loss, so a lone
        # transient never trips it. 0 disables (exact prior behavior); set a
        # small value (e.g. 5) for optimizer-stability-risky runs.
        grad_norm_abort: float = 0.0
        """Stop when grad_norm exceeds this, in units of its own recent median.

        The nan_abort_consecutive guard below only catches NON-FINITE values,
        which is useless for the failure actually observed: SophiaG at 30B went
        grad_norm 0.39 -> 2.41 -> 89.9 -> 1702 -> 100611 over nine steps
        (job 12473783, step 1048). Every one of those is finite, so nothing
        fired, and the run burned ~50 steps and 2.5 nats before recovering.

        A ratio against the running median rather than an absolute threshold:
        grad_norm's healthy scale differs by an order of magnitude across
        optimizers and shrinks as training proceeds, so any fixed number is
        either too loose early or too tight late. 0 disables."""
        grad_norm_abort_window: int = 50
        """Steps of history used for the median grad_norm baseline."""

        nan_abort_consecutive: int = 0
        """Abort training after this many consecutive non-finite reported
        losses (NaN/inf). 0 disables. Set ~5 for NaN-prone runs (e.g. 80B
        optimizer probes) so a divergence does not burn the full walltime."""

        # Diagnostics. Everything torchtitan logs is a scalar summary of the
        # WHOLE model, which says THAT a run is sick and not WHERE. These add
        # per-layer grad norms, update ratios, clipping, optimizer state and
        # QK statistics. Default OFF: opt-in instrumentation, not a tax.
        diagnostics: bool = False
        """Enable extended training diagnostics (per-layer grad norms, update
        ratios, clip pre/post, optimizer state). See diagnostics/__init__.py
        for what each metric catches and which incident motivated it."""

        diagnostics_interval: int = 50
        """Steps between diagnostic samples. Independent of metrics.log_freq
        because these cost O(params) rather than O(1)."""

        diagnostics_per_layer: bool = False
        """Also emit per-layer grad-norm skew and the top-k worst layers.
        Separate flag: this is the expensive half."""

        diagnostics_attention: bool = False
        """Sample QK magnitude statistics from attention. NOT entropy --
        SDPA is fused and never materializes scores; see
        diagnostics/attention.py."""

        history_bridge: bool = False
        """Feed per-rank scalars through ezpz.History for CROSS-RANK spread
        (loss/std, loss/max). Our metrics are pre-reduced, so today we log the
        mean and cannot see whether one rank is pathological. Auto-aggregation
        is unavailable above world_size 384 -- the bridge warns rather than
        silently reporting zeros."""

        wandb_watch: bool = False
        """wandb.watch(log="all") for true weight/grad HISTOGRAMS. Costly at
        26B params; throttled to every 500 steps."""

    ft_manager: FTManager

    @record
    def __init__(self, config: Config):
        torch._C._log_api_usage_once("torchtitan.train")

        self.config = config
        assert config.model_spec is not None, (
            "model_spec must be set before creating Trainer"
        )
        model_spec = config.model_spec

        device_module, device_type = utils.device_module, utils.device_type
        # pyrefly: ignore [read-only]
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed and build meshes (FT override handles ft_manager creation)
        self.parallel_dims = parallel_dims = self.init_distributed()

        # Logging needs to happen after distributed initialized
        config.maybe_log()

        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            batch_degree, batch_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            batch_degree, batch_rank = 1, 0

        # FT addition: adjust dp info via ft_manager
        batch_degree, batch_rank = self.ft_manager.get_dp_info(batch_degree, batch_rank)

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(
            gc_freq=config.training.gc_freq, debug=config.training.gc_debug
        )

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["pp"],
        )

        # build tokenizer
        self.tokenizer = (
            config.tokenizer.build(tokenizer_path=config.hf_assets_path)
            if config.tokenizer is not None
            else None
        )

        # build dataloader
        # Under PP the dataloader must serve MICROBATCHES, not the full local
        # batch -- upstream #3856 ("Always Pre-Split Microbatches for PP")
        # moved the split to the dataloader. Mirror the base Trainer
        # (torchtitan/trainer.py): feed pipeline_parallel_microbatch_size when
        # PP is on, else the plain local batch size. Without this, a PP run
        # would receive full-size batches and mis-shape every microbatch.
        # (The validator build below intentionally keeps local_batch_size --
        # upstream did not change the validator path.)
        # 80th sync (#4121): pipeline_parallel_microbatch_size (a SIZE) became
        # num_pp_microbatches (a COUNT). The old code divided to get the count;
        # now it is read directly, and the per-microbatch size is what gets
        # derived. Semantic inversion, not a rename -- inverting it the wrong
        # way silently mis-shapes every PP microbatch.
        _num_pp_microbatches = (
            config.parallelism.num_pp_microbatches if parallel_dims.pp_enabled else 1
        )
        # #4121 counts batches in TOKEN SLOTS, but BlendCorpus (this
        # experiment's dataloader) counts SEQUENCES -- its micro_batch_size /
        # global_batch_size feed a Megatron-style sampler, not a token packer.
        # Passing the token count straight through made the embedding try to
        # allocate 8192 seq x 8192 tok x 2048 dim x 2 B = exactly 256 GiB on a
        # 40 GiB card (smoke 7554426), with 32 GiB still free -- a shape bug
        # wearing an OOM costume.
        #
        # Convert back to sequences for the dataloader only. Core's Grain path
        # keeps the token units it expects; nothing else here changes.
        _seq_len = config.training.max_context_length
        dataloader_batch_size = (
            config.training.num_tokens_per_microbatch_per_dp_rank // _seq_len
        )
        if dataloader_batch_size < 1:
            raise ValueError(
                "training.num_tokens_per_microbatch_per_dp_rank "
                f"({config.training.num_tokens_per_microbatch_per_dp_rank}) "
                f"must be at least one full sequence of "
                f"max_context_length ({_seq_len})."
            )
        self.dataloader = config.dataloader.build(
            dp_world_size=batch_degree,
            dp_rank=batch_rank,
            tokenizer=self.tokenizer,
            # BOTH naming conventions, deliberately. #4121 renamed the
            # dataloader kwargs, and this tree runs two dataloaders whose
            # signatures did NOT converge:
            #   GrainDataLoader   wants max_context_length / num_tokens_per_batch
            #   BlendCorpusDataLoader wants seq_len / local_batch_size
            # Both take **kwargs and ignore what they do not name, so sending
            # both pairs satisfies either one. Sending only the new pair breaks
            # blendcorpus -- which is every production config -- and sending
            # only the old pair breaks Grain, as it did in job 12473732
            # ("missing 2 required keyword-only arguments").
            # NOT the same value: BlendCorpus counts SEQUENCES (its
            # micro_batch_size feeds a Megatron-style sampler) while Grain
            # counts TOKENS. Send each the unit it actually means --
            # collapsing them mis-sizes whichever loader disagrees, and the
            # blendcorpus side of that is a 256 GiB embedding allocation
            # (smoke 7554426), not a clean error.
            seq_len=config.training.max_context_length,
            local_batch_size=dataloader_batch_size,
            max_context_length=config.training.max_context_length,
            num_tokens_per_batch=(
                config.training.num_tokens_per_microbatch_per_dp_rank
            ),
            # train_step pulls gas * num_pipeline_parallel_microbatches batches
            # per optimizer step, so the dataloader must be sized for that many
            # -- not the raw step count. Without the PP factor the iterator runs
            # dry mid-run ("Ran out of data") and a later microbatch group comes
            # up short, which the pipeline schedule reports as
            # "Expecting N arg_mbs but got M". The factor is 1 when PP is off,
            # so this is unchanged for every non-PP run. (Upstream applies the
            # same product to snapshot_every_n_steps, torchtitan/trainer.py.)
            # (computed locally: self.num_pipeline_parallel_microbatches is not
            # assigned until later in __init__, after the dataloader is built.)
            training_steps=config.training.steps * _num_pp_microbatches,
            # Sequences, not tokens -- see the conversion above.
            global_batch_size=(
                config.training.num_tokens_per_train_step // _seq_len
            ),
            parallel_dims=parallel_dims,
        )

        # The SDPA wrapper needs max_context_length to unflatten #4121's flat
        # [T, N, H] batches back to [B, L, N, H] for scaled_dot_product_attention.
        # It cannot take it as an argument: the forward's positional-arg names
        # are contract-checked under TP>1 (set_gqa_inner_attention_local_map
        # matches in_dst_shardings by name), so a module-level setter is used.
        # Set here -- after the dataloader, before the model is built -- so it
        # is in place well before the first forward.
        from torchtitan.experiments.ezpz.agpt import set_ezpz_max_context_length

        set_ezpz_max_context_length(config.training.max_context_length)

        # build model (using meta init)
        model_config = model_spec.model
        # set the model args from training job configs
        model_config.update_from_config(
            config=config,
        )
        self.model_config = model_config

        # logger.info(
        #     f"Building {model_spec.name} {model_spec.flavor} "
        #     f"with {json.dumps(model_config.to_dict(), indent=2, ensure_ascii=False)}"
        # )
        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
        ):
            model = model_config.build()

        # if ezpz.dist
        # if ezpz.distributed.asni
        if ezpz.distributed.verify_wandb():
            import wandb
            if wandb.run is not None:
                wandb.run.watch(model, log="all")

        # Quantization is now applied to the config at model_registry time
        # (#3127). The runtime model_converters layer is gone.

        # Verify all submodules satisfy the Module protocol
        model.verify_module_protocol()

        # Check if any quantization converter is on the model_config
        from torchtitan.components.quantization.utils import has_quantization as _has_quantization
        has_quantization = _has_quantization(model_config)

        # metrics logging (FT addition: ft_enable, ft_replica_id)
        self.metrics_processor = config.metrics.build(
            parallel_dims=parallel_dims,
            dump_folder=config.dump_folder,
            pp_schedule=config.parallelism.pipeline_parallel_schedule,
            ft_enable=config.fault_tolerance.enable,
            ft_replica_id=config.fault_tolerance.replica_id,
            config_dict=config.to_dict(),
            has_quantization=has_quantization,
        )
        color = self.metrics_processor.color

        # calculate model size and flops per token
        (
            model_param_count,
            self.metrics_processor.num_flops_per_token,
        ) = model_config.get_nparams_and_flops(model, config.training.max_context_length)

        heading = 80 * "="
        logger.info(
            "\n".join([
                "\n",
                f"{heading}",
                f"{color.blue}Model: {model_spec.name} {model_spec.flavor} ",
                f"{color.red}config: {model_param_count:,} total parameters{color.reset}",
                f"{heading}",
                "\n",
            ])
        )
        # logger.info(
        #     "\n" + 80 * "="
        #     f"{color.blue}Model {model_spec.name} {model_spec.flavor} "
        #     f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        #
        # )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        buffer_device: torch.device | None
        if config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = torch.device(device_type)
        else:
            init_device = device_type
            buffer_device = None

        # Loss is now built from the JobConfig.loss field (upstream #2937 /
        # ChunkedLossWrapper). The FT integration no longer wraps the loss
        # function — FTOptimizersContainer below still passes ft_manager
        # for gradient sync.
        self.loss_fn = config.loss.build(compile_config=config.compile)

        # 80th sync (#4121): batch sizes are counted in TOKENS, not sequences.
        #   num_tokens_per_microbatch_per_dp_rank == old local_batch_size * seq_len
        #   num_tokens_per_train_step            == old global_batch_size * seq_len
        # This is upstream's arithmetic (trainer.py:409-428) copied deliberately
        # rather than re-derived: it owns the divisibility contract, and a
        # subtly different GAS here would change the effective batch of every
        # run without failing anything.
        num_tokens_per_dp_rank = (
            config.training.num_tokens_per_microbatch_per_dp_rank
            * _num_pp_microbatches
        )
        num_tokens_per_train_step = config.training.num_tokens_per_train_step
        if num_tokens_per_train_step < 0:
            num_tokens_per_train_step = num_tokens_per_dp_rank * batch_degree
        if num_tokens_per_train_step % (num_tokens_per_dp_rank * batch_degree) != 0:
            raise ValueError(
                "training.num_tokens_per_train_step "
                f"({num_tokens_per_train_step}) must be divisible by the number "
                "of tokens processed globally in one gradient accumulation "
                f"iteration ({num_tokens_per_dp_rank * batch_degree})."
            )
        self.gradient_accumulation_steps = num_tokens_per_train_step // (
            num_tokens_per_dp_rank * batch_degree
        )
        assert self.gradient_accumulation_steps > 0

        # How many pipeline microbatches make up one local batch. This is 1
        # whenever PP is off, so the extra loop it drives in train_step is a
        # no-op for every non-PP run. Mirrors the base Trainer
        # (torchtitan/trainer.py); FaultTolerantTrainer does not call
        # super().__init__(), so it must be set here explicitly or the PP
        # branch would AttributeError.
        self.num_pipeline_parallel_microbatches = _num_pp_microbatches

        # Same reason again (third instance of this in this file): core's
        # Trainer.__init__ calls dist_utils.set_spmd_backend(
        # config.parallelism.spmd_backend) at trainer.py:319, and we never
        # reach it. Without this the module-level default -- currently
        # "spmd_types" -- stays live no matter what the config or CLI says,
        # and components/loss.py:43 then fires a bare
        #   assert get_spmd_backend() == "partial_dtensor"
        # on any TP>1 run whose pred is a DTensor. Observed as an
        # unexplained AssertionError with no message in job 12473496, on the
        # partial_dtensor CONTROL arm, i.e. a config that should trivially
        # satisfy the assert.
        dist_utils.set_spmd_backend(config.parallelism.spmd_backend)

        # 78th sync (#3559 CUDA-graph capture + #4146 in-place loss accum):
        # the base Trainer.__init__ now builds a `fwd_bwd_fn` indirection and
        # forward_backward_step dispatches through it. Same reason as above --
        # we do not call super().__init__() -- so without this every run dies
        # with "'FaultTolerantTrainer' object has no attribute 'fwd_bwd_fn'"
        # at step 1 (job 12473170: 3/3 arms rc=143, 0 steps).
        #
        # Core's wrap_with_cuda_graph is a CUDA-only path (it hard-gates on
        # device_type=="cuda"), so it is never applied here. Instead, opt in to
        # the XPU twin via EZPZ_XPU_GRAPHS=1 -- the frameworks RC exposes the
        # full torch.xpu graph API. Default OFF returns the plain body, so
        # behaviour is identical to pre-merge unless explicitly enabled.
        # See experiments/ezpz/xpu_graph.py.
        self.fwd_bwd_fn = maybe_wrap_with_xpu_graph(self._forward_backward_body)

        # Batch-size ramp config validation (see Config docstrings).
        self.batch_ramp_steps = config.batch_ramp_steps
        self.batch_ramp_start_gas = config.batch_ramp_start_gas
        if self.batch_ramp_steps < 0:
            raise ValueError(
                f"batch_ramp_steps must be >= 0, got {self.batch_ramp_steps}"
            )
        if self.batch_ramp_steps > 0:
            if not 1 <= self.batch_ramp_start_gas <= self.gradient_accumulation_steps:
                raise ValueError(
                    "batch_ramp_start_gas must be in "
                    f"[1, {self.gradient_accumulation_steps}], got "
                    f"{self.batch_ramp_start_gas}"
                )
            logger.info(
                "Batch-size ramp ENABLED: GAS %d -> %d over %d steps "
                "(effective GBS %d -> %d)",
                self.batch_ramp_start_gas,
                self.gradient_accumulation_steps,
                self.batch_ramp_steps,
                self.batch_ramp_start_gas
                * config.training.num_tokens_per_microbatch_per_dp_rank
                * batch_degree,
                config.training.num_tokens_per_train_step,
            )

        # apply parallelisms and initialization
        if parallel_dims.pp_enabled:
            from torchtitan.components.metrics import ensure_pp_loss_visible

            if not model_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {model_spec.name} "
                    f"does not support pipelining"
                )

            # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = model_spec.pipelining_fn(
                model,
                parallel_dims=parallel_dims,
                training=config.training,
                parallelism=config.parallelism,
                compile_config=config.compile,
                ac_config=config.activation_checkpoint,
                dump_folder=config.dump_folder,
                device=self.device,
                model_config=model_config,
                parallelize_fn=model_spec.parallelize_fn,
                loss_fn=self.loss_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point,
            # model_parts is used instead
            del model

            for m in self.model_parts:
                m.to_empty(device=init_device)
                with torch.no_grad():
                    cast(BaseModel, m).init_states(buffer_device=buffer_device)
                m.train()

            # confirm that user will be able to view loss metrics on the console
            ensure_pp_loss_visible(
                parallel_dims=parallel_dims,
                pp_schedule=config.parallelism.pipeline_parallel_schedule,
                color=color,
            )
        else:
            # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
            model = model_spec.parallelize_fn(
                model,
                parallel_dims=parallel_dims,
                training=config.training,
                parallelism=config.parallelism,
                compile_config=config.compile,
                ac_config=config.activation_checkpoint,
                dump_folder=config.dump_folder,
            )

            model.to_empty(device=init_device)
            with torch.no_grad():
                cast(BaseModel, model).init_states(buffer_device=buffer_device)
            model.train()

            self.model_parts = [model]

        # Set lm_head reference for ChunkedLossWrapper after model construction.
        # Replayed from upstream torchtitan/trainer.py (lines 391-411). Required
        # whenever loss=ChunkedLossWrapper.Config(...) — the loss object computes
        # logits from hidden states in chunks, so it needs a handle to lm_head
        # and signals the model to skip its own lm_head pass via _skip_lm_head.
        # Non-PP: single model part always has lm_head.
        # PP: only the last stage has lm_head; non-last stages skip this.
        if isinstance(self.loss_fn, ChunkedLossWrapper):
            if parallel_dims.pp_enabled:
                if self.pp_has_last_stage:
                    lm_head = self.model_parts[-1].lm_head
                    assert (
                        lm_head is not None
                    ), "Last PP stage must have lm_head for ChunkedLossWrapper"
                    self.loss_fn.set_lm_head(lm_head)
                    self.model_parts[-1]._skip_lm_head = True
            else:
                assert len(self.model_parts) == 1
                lm_head = self.model_parts[0].lm_head
                assert lm_head is not None, "Model must have lm_head for ChunkedLossWrapper"
                self.loss_fn.set_lm_head(lm_head)
                self.model_parts[0]._skip_lm_head = True

        # FT addition: set all reduce hook
        self.ft_manager.maybe_set_all_reduce_hook(self.model_parts)

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        # FT addition: pass ft_manager for FTOptimizersContainer
        if isinstance(config.optimizer, FTOptimizersContainer.Config):
            self.optimizers = config.optimizer.build(
                model_parts=self.model_parts, ft_manager=self.ft_manager
            )
        else:
            self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        if model_spec.post_optimizer_build_fn is not None:
            model_spec.post_optimizer_build_fn(
                self.optimizers, self.model_parts, parallel_dims
            )
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )
        # The post-optimizer model_converters hook is gone in #3127 —
        # quantization is applied to the config and runs as part of
        # forward, not via a runtime post-step hook.
        self.metrics_processor.optimizers = self.optimizers
        self.metrics_processor.model_parts = self.model_parts

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0
        self.ntokens_seen = 0
        # Diagnostics state. prev_weight_norms drives update_ratio; it is a
        # dict of scalars (one float per parameter), NOT a copy of the
        # weights -- the whole point of difference-of-norms.
        self._diag_prev_weight_norms: dict[str, float] = {}
        self._history_bridge = None
        if getattr(config, "diagnostics", False) or getattr(
            config, "history_bridge", False
        ):
            from torchtitan.experiments.ezpz.diagnostics.history_bridge import (
                HistoryBridge,
            )
            from torchtitan.experiments.ezpz.diagnostics import attention as _attn

            _attn.configure(
                enabled=getattr(config, "diagnostics_attention", False),
                every=getattr(config, "diagnostics_interval", 50),
            )
            self._history_bridge = HistoryBridge(
                enabled=getattr(config, "history_bridge", False),
                outdir=os.path.join(config.dump_folder, "history"),
                world_size=int(os.environ.get("WORLD_SIZE", "1")),
            )

        # Build checkpoint manager.
        # When fault tolerance is enabled and config.checkpoint uses
        # FTCheckpointManager.Config, ft_manager is passed through.
        # Otherwise the base CheckpointManager is used without it.
        ckpt_kwargs: dict = dict(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            sd_adapter=(
                model_spec.state_dict_adapter(model_config, config.hf_assets_path)
                if model_spec.state_dict_adapter
                else None
            ),
            base_folder=config.dump_folder,
        )
        # FTCheckpointManager accepts ft_manager; base CheckpointManager does not
        from torchtitan.experiments.torchft.checkpoint import (
            TorchFTCheckpointManager as FTCheckpointManager,
        )

        if isinstance(config.checkpoint, FTCheckpointManager.Config):
            ckpt_kwargs["ft_manager"] = self.ft_manager
        self.checkpointer = config.checkpoint.build(**ckpt_kwargs)

        # 57th sync: PR #3694 deleted the --disable_loss_parallel flag.
        # TP-on now always implies LP-on; the context no longer takes
        # enable_loss_parallel (loss-parallel autograd moved into
        # cross_entropy_loss).
        # 61st sync: the spmd_types series renamed get_train_context ->
        # get_spmd_context and added the spmd_typechecking kwarg. Mirror
        # upstream Trainer; we run spmd_backend=default so typechecking is
        # off (the kwarg is inert unless backend == "spmd_types").
        self.train_context = dist_utils.get_spmd_context(
            parallel_dims=parallel_dims,
            spmd_typechecking=(
                config.parallelism.spmd_backend == "spmd_types"
                and config.debug.spmd_typechecking
            ),
        )

        # Build validator if validation is configured
        if config.validator.enable:
            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    self.pp_schedule,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                )
                if parallel_dims.pp_enabled
                else (None, None, None)
            )

            self.validator = config.validator.build(
                parallelism=config.parallelism,
                job_config=config,
                dp_world_size=batch_degree,
                dp_rank=batch_rank,
                tokenizer=self.tokenizer,
                parallel_dims=parallel_dims,
                loss_fn=self.loss_fn,
                validation_context=self.train_context,
                metrics_processor=self.metrics_processor,
                seq_len=config.training.max_context_length,
                # #4121 renamed this kwarg on core's Validator:
                # local_batch_size (SEQUENCES) -> num_tokens_per_batch (TOKENS).
                # It is keyword-only with no default, so passing the old name
                # is a hard TypeError at config.build() -- before step 1, and
                # only when the validator is enabled, which is why it survived
                # the sync smokes (job 7558514).
                num_tokens_per_batch=(
                    config.training.num_tokens_per_microbatch_per_dp_rank
                ),
                pp_schedule=pp_schedule,
                pp_has_first_stage=pp_has_first_stage,
                pp_has_last_stage=pp_has_last_stage,
            )

        logger.info(
            "Trainer is initialized with "
            f"tokens/microbatch/dp-rank "
            f"{config.training.num_tokens_per_microbatch_per_dp_rank}, "
            f"tokens/train-step "
            f"{config.training.num_tokens_per_train_step}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {config.training.max_context_length}, "
            f"total steps {config.training.steps} "
            f"(warmup {config.lr_scheduler.warmup_steps})"
        )

    def init_distributed(self) -> ParallelDims:
        config = self.config

        # determine the global ranks when fault tolerance is enabled
        global_ranks = []
        ft_config = config.fault_tolerance
        if ft_config.enable:
            group_size = ft_config.group_size
            replica_id = ft_config.replica_id
            first_rank = replica_id * group_size
            last_rank = first_rank + group_size - 1
            global_ranks = list(range(first_rank, last_rank + 1))

        # init distributed and build meshes
        dist_utils.init_distributed(
            config.comm,
            enable_cpu_backend=config.training.enable_cpu_offload,
            base_folder=config.dump_folder,
            ranks=global_ranks,
        )

        # On XPU, ProcessGroupXCCL inherits Backend::supportsSplitting() ==
        # false, but DeviceMesh._init_one_process_group still routes nested
        # mesh PG creation through ``split_group`` whenever
        # ``bound_device_id`` is set on the default group. That always blows
        # up with "No backend for the parent process group or its backend
        # does not support splitting", which kills every nested mesh
        # construction the EP sparse mesh requires. Steer the gate to the
        # ``new_group`` fallback for xccl until upstream lands the
        # supportsSplitting override + split implementation.
        from torchtitan.experiments.ezpz.xccl_split_group_workaround import (
            maybe_install_xccl_split_group_workaround,
        )

        maybe_install_xccl_split_group_workaround()
        # Async-checkpoint CPU staging calls dist.new_group(backend="gloo"),
        # which crashes on XPU because the gloo subgroup inherits
        # bound_device_id=xpu and torch's eager-connect then calls
        # _get_backend(xpu) on the gloo-only group. Install before the
        # checkpointer builds. See gloo_new_group_workaround.py.
        from torchtitan.experiments.ezpz.gloo_new_group_workaround import (
            maybe_install_gloo_new_group_workaround,
        )
        maybe_install_gloo_new_group_workaround()

        # FT addition: build FTManager
        self.ft_manager = config.fault_tolerance.build()

        world_size = int(os.environ["WORLD_SIZE"])

        return ParallelDims.from_config(config.parallelism, world_size)

    def _effective_gas(self) -> int:
        """Gradient-accumulation steps for the current step under the ramp.

        Linearly interpolates GAS from ``batch_ramp_start_gas`` (at step 0)
        to the full ``gradient_accumulation_steps`` (at ``batch_ramp_steps``),
        holding at full after. Returns the full GAS when the ramp is
        disabled (``batch_ramp_steps == 0``). ``self.step`` is 0-indexed at
        the point train_step runs.
        """
        if self.batch_ramp_steps <= 0:
            return self.gradient_accumulation_steps
        if self.step >= self.batch_ramp_steps:
            return self.gradient_accumulation_steps
        start = self.batch_ramp_start_gas
        full = self.gradient_accumulation_steps
        # Linear interpolation; round to nearest int, clamp to [start, full].
        frac = self.step / self.batch_ramp_steps
        gas = round(start + (full - start) * frac)
        return max(start, min(full, gas))

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()
        # Save the current step learning rate for logging
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        # Effective grad-accum count for this step (batch-size ramp).
        # Equals self.gradient_accumulation_steps unless the ramp is on.
        gas = self._effective_gas()

        # Collect all microbatches on CPU and count total valid tokens.
        # Two nested levels, mirroring the base Trainer: the OUTER level is
        # gradient accumulation (one optimizer step per `gas` groups), the
        # INNER level is pipeline microbatches (the PP schedule consumes a
        # whole group at once). `num_pipeline_parallel_microbatches` is 1
        # whenever PP is off, so with PP disabled this is exactly the old
        # flat `gas` loop -- one (input_dict, labels) pair per group.
        microbatch_groups: list[list[tuple[dict[str, torch.Tensor], torch.Tensor]]] = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _microbatch in range(gas):
            microbatches = []
            for _pp_microbatch in range(self.num_pipeline_parallel_microbatches):
                input_dict, labels = next(data_iterator)
                local_valid_tokens += (labels != IGNORE_INDEX).sum()
                microbatches.append((input_dict, labels))
            microbatch_groups.append(microbatches)

        # All-reduce to get global token count across DP ranks
        # Move to GPU for distributed communication
        local_valid_tokens = local_valid_tokens.to(self.device)
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            global_valid_tokens = dist_utils.dist_sum(local_valid_tokens, batch_mesh)
        else:
            # Upstream PR #3586 (2026-06-09) retyped global_valid_tokens
            # as `float | None` and switched the no-DP branch to
            # `float(local_valid_tokens.item())`. Mirror that here so the
            # annotation contract holds. DP branch keeps returning a
            # tensor from dist_sum — upstream itself does the same; the
            # consumer (BaseLoss.__call__) accepts either at runtime.
            global_valid_tokens = float(local_valid_tokens.item())

        # Process each group: move to GPU, forward/backward, then free.
        # Under PP the WHOLE group (the microbatch list) goes to
        # forward_backward_step in one call -- the pipeline schedule drives the
        # microbatches through the stages itself, so handing it one dict at a
        # time trips `assert isinstance(input_dict, list)` in the base Trainer.
        # Without PP each group holds exactly one pair and this unwraps to the
        # original per-microbatch call.
        accumulated_losses = []
        for microbatches in microbatch_groups:
            input_dict_mbs = []
            label_mbs = []
            for input_dict, labels in microbatches:
                # Move tensors to GPU
                for k, v in input_dict.items():
                    if isinstance(v, torch.Tensor):
                        input_dict[k] = v.to(self.device)
                input_dict_mbs.append(input_dict)
                label_mbs.append(labels.to(self.device))

            if parallel_dims.pp_enabled:
                fwd_bwd_input_dict = input_dict_mbs
                fwd_bwd_labels = label_mbs
            else:
                assert len(input_dict_mbs) == len(label_mbs) == 1
                fwd_bwd_input_dict = input_dict_mbs[0]
                fwd_bwd_labels = label_mbs[0]

            loss = self.forward_backward_step(
                input_dict=fwd_bwd_input_dict,
                labels=fwd_bwd_labels,
                # pyrefly: ignore [bad-argument-type]
                global_valid_tokens=global_valid_tokens,
            )
            accumulated_losses.append(loss.detach())

        # foreach=True fuses the per-parameter norms into one multi-tensor op.
        # Under HSDP on XPU that path exhausts level_zero resources -- the 30B
        # dies in clip_grad.py:106 `torch.stack([norm.to(first_device) ...])`
        # with UR_RESULT_ERROR_OUT_OF_RESOURCES at only 71.68% memory, so it is
        # device-resource exhaustion (events/command-lists), not an OOM.
        # EZPZ_CLIP_NO_FOREACH=1 falls back to the unfused loop to test that.
        # Default is unchanged (foreach=True) -- this is opt-in diagnosis.
        # Arm the attention sampler for THIS step. Must happen before the
        # next forward; the SDPA wrapper checks it to decide whether to
        # sample. Without this call observe() can never fire -- which is
        # exactly the bug the first smoke exposed.
        if getattr(self.config, "diagnostics_attention", False):
            from torchtitan.experiments.ezpz.diagnostics import attention as _attn
            _attn.set_step(self.step)

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=_clip_foreach(),
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
        # Refuse to apply a non-finite update.
        #
        # The nan_abort_consecutive guard below runs AFTER this step and only
        # inspects the loss, so a NaN gradient is written into the weights
        # before anything notices. Job 12473142 shows why that ordering
        # matters: grad_norm went nan at step 30 and loss only at step 31, so
        # the earliest available signal was one full step ahead of the one we
        # were watching -- and five more updates landed before the abort.
        #
        # grad_norm is already reduced across ranks by clip_grad_norm_ (and
        # across pp when pp_mesh is passed), so every rank sees the same value
        # and this branches identically everywhere. No extra collective.
        #
        # Deliberately NOT torch._assert_async, which upstream uses in #4226:
        # on CUDA a failed device-side assert invalidates the process, which
        # our failover machinery would see as a crash rather than a clean
        # stop, and its XPU behavior is undocumented. A host-side check costs
        # one already-materialized .item() -- grad_norm is read for logging at
        # the metrics call below regardless.
        # Staging is a checkpoint concern, not an optimizer one -- it must run
        # whether or not we take the step, or a skipped step would leave an
        # async save un-awaited.
        self.checkpointer.maybe_wait_for_staging()
        if not math.isfinite(float(grad_norm.item())):
            # CAPTURE BEFORE ZEROING. zero_grad() below destroys the gradients,
            # and the diagnostics that would characterize this step do not run
            # until ~90 lines later (collect_param_stats), by which point there
            # is nothing left to measure. So the ONE step that matters -- the
            # non-finite one -- was the only step with no gradient data.
            #
            # Found on job 12474403 (80B depth bisect): the L72 arm hit a NaN
            # grad_norm at step 55 and recovered at 56, and step 55 is absent
            # from W&B entirely while 51-54 and 56-57 logged normally. Every
            # neighbouring step looks ordinary (qk_q_absmax ~61, grad_absmax
            # ~0.02), so the event is invisible from both sides.
            #
            # This runs only on non-finite steps -- rare by construction, 1 in
            # 60 in that arm -- so the O(params) scan is affordable here even
            # though it is gated behind an interval in the normal path.
            try:
                from torchtitan.experiments.ezpz import diagnostics as _diag_nan

                _nan_stats = _diag_nan.collect_param_stats(
                    self.model_parts, per_layer=True
                )
                _bad = [
                    k for k, v in _nan_stats.items()
                    if isinstance(v, float) and not math.isfinite(v)
                ]
                logger.error(
                    "NON-FINITE GRADIENT CAPTURE step %s: %s",
                    self.step,
                    ", ".join(
                        f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in sorted(_nan_stats.items())
                    ),
                )
                if _bad:
                    logger.error(
                        "NON-FINITE GRADIENT CAPTURE step %s: metrics that are "
                        "THEMSELVES non-finite (these name the affected "
                        "tensors): %s",
                        self.step,
                        ", ".join(sorted(_bad)),
                    )
                # NOT pushed to W&B here. This trainer's processor takes
                # log(step, avg_loss, max_loss, grad_norm, extra_metrics=...)
                # -- the losses are positional and are not yet reduced at this
                # point in the step (that happens ~30 lines below). Calling it
                # early would either need fabricated loss values or a second
                # partial row at the same step. The console capture above is
                # the record; W&B still shows the hole, and the log names the
                # tensors, which is what the hole was hiding.
            except Exception as _e:  # instrumentation must not kill the run
                logger.error(
                    "NON-FINITE GRADIENT CAPTURE step %s FAILED: %s: %s",
                    self.step, type(_e).__name__, _e,
                )

            self.optimizers.zero_grad()
            logger.error(
                f"non-finite grad_norm ({grad_norm}) at step {self.step}: "
                "SKIPPING the optimizer step to avoid writing NaN into the "
                "weights. The model is unchanged; the loss guard decides "
                "whether to abort."
            )
        else:
            self.optimizers.step()
        self.lr_schedulers.step()

        # Reduce the data collected over gradient accumulation steps.
        loss = torch.sum(torch.stack(accumulated_losses))

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return float(loss.detach().item())

        if parallel_dims.dp_cp_enabled:
            loss = loss.detach()
            # FT addition: use ft_manager.loss_sync_pg for extra process group
            ft_pg = self.ft_manager.loss_sync_pg
            loss_mesh = parallel_dims.get_optional_mesh("loss")

            # For global_avg_loss, we want the average loss across all ranks:
            # loss = local_loss_sum / global_valid_tokens
            # global_avg_loss = sum(local_loss_sum) / global_valid_tokens
            #                 = sum(loss)
            #
            # For global_max_loss, we want the max of local average losses across ranks:
            # local_avg_loss = local_loss_sum / local_valid_tokens
            #                = (loss * global_valid_tokens) / local_valid_tokens
            # global_max_loss = max(local_avg_loss)
            local_avg_loss = loss * global_valid_tokens / local_valid_tokens
            global_avg_loss, global_max_loss, global_ntokens_seen = (
                dist_utils.dist_sum(loss, loss_mesh, ft_pg),
                dist_utils.dist_max(local_avg_loss, loss_mesh, ft_pg),
                dist_utils.dist_sum(
                    torch.tensor(
                        self.ntokens_seen, dtype=torch.int64, device=self.device
                    ),
                    loss_mesh,
                    ft_pg,
                ),
            )
        else:
            global_avg_loss = global_max_loss = float(loss.detach().item())
            global_ntokens_seen = self.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            "lr": lr,
        }

        # z-loss penalty, if the configured loss computes one. UNGATED, unlike
        # the diagnostics block below: this is O(1), and a penalty term whose
        # magnitude you cannot see is untunable -- the whole point of z-loss is
        # choosing a coefficient, which needs the number every step.
        #
        # Polled rather than read off the loss's return value because every
        # core call site discards the metrics dict a loss returns
        # (components/validate.py:258, distributed/pipeline_parallel.py:319).
        # Measured on job 12474386: the penalty was applied and completely
        # invisible in the logs.
        try:
            from torchtitan.experiments.ezpz import zloss as _zloss

            extra_metrics.update(_zloss.drain())
        except Exception:
            # Reporting must never take down a run; drain() is already
            # exception-safe, this covers the import on stacks without it.
            pass

        # Extended diagnostics. Gated on its own interval, not log_freq: these
        # are O(params) rather than O(1), so they must not run every step.
        # Collected AFTER the optimizer step, so update_ratio compares the
        # weights this step produced against the previous sample.
        if getattr(self.config, "diagnostics", False) and (
            self.step % max(1, getattr(self.config, "diagnostics_interval", 50)) == 0
        ):
            try:
                from torchtitan.experiments.ezpz import diagnostics as _diag
                from torchtitan.experiments.ezpz.diagnostics import (
                    attention as _attn,
                )

                extra_metrics.update(
                    _diag.clipping_metrics(grad_norm, self.config.training.max_norm)
                )
                extra_metrics.update(
                    _diag.collect_param_stats(
                        self.model_parts,
                        per_layer=getattr(
                            self.config, "diagnostics_per_layer", False
                        ),
                    )
                )
                ratios, self._diag_prev_weight_norms = _diag.collect_update_ratios(
                    self.model_parts, self._diag_prev_weight_norms
                )
                extra_metrics.update(ratios)
                extra_metrics.update(_diag.collect_optimizer_stats(self.optimizers))
                extra_metrics.update(_attn.drain())
            except Exception as e:
                # Instrumentation must never take down a training run. A
                # broken probe costs a metric; an exception here costs the
                # allocation.
                logger.warning(f"diagnostics failed at step {self.step}: {e}")

        # Cross-rank spread. Feeds the PER-RANK loss in (not the reduced one)
        # so std/max mean what they say.
        if self._history_bridge is not None and self._history_bridge.enabled:
            extra_metrics.update(
                self._history_bridge.update(
                    {
                        "loss": float(loss.detach().item()),
                        "grad_norm": float(grad_norm.item()),
                    },
                    step=self.step,
                )
            )
        # Echo the FULL metric dict when asked. The smoke's verification
        # cannot see these keys any other way: they go to W&B, not stdout, and
        # W&B's terminal summary silently omits constant-valued keys.
        if os.environ.get("EZPZ_DIAG_ECHO") == "1" and extra_metrics:
            logger.info(
                "DIAG_ECHO "
                + ", ".join(f"{k}={v}" for k, v in sorted(extra_metrics.items()))
            )
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )

        # Publish the step grad_norm for the training loop. train_step returns
        # only the loss (several callsites depend on that), but the grad_norm
        # runaway guard in train() needs this value and grad_norm is local to
        # this method. Reuses the .item() already materialized just above.
        self._last_grad_norm = float(grad_norm.item())

        if isinstance(global_avg_loss, torch.Tensor):
            return float(global_avg_loss.item())
        return float(global_avg_loss)

    @record
    def train(self):
        config = self.config

        # Checkpoints written before the attention QKV wrapper refactor store
        # layers.N.attention.{wq,wk,wv} flat, while current code asks for
        # layers.N.attention.qkv_linear.{wq,wk,wv} and dcp.load matches by
        # exact key. Install the remap ONLY for such a checkpoint -- the
        # detector reads the on-disk metadata, so a current-format checkpoint
        # is left completely untouched. Without this, resuming an old ckpt
        # dies with "Missing key in checkpoint state_dict" before step 1
        # (it burned two umbrella slots three dispatches running).
        # initial_load_path matters here: when checkpoint.folder holds no
        # resumable step the checkpointer loads the SEED instead, and a seed
        # can just as easily be pre-refactor. Probing only `folder` in that
        # case probes a directory that does not exist yet (job 8771774).
        maybe_install_flat_attention_compat(
            self.checkpointer,
            config.checkpoint.folder,
            config.checkpoint.load_step,
            dump_folder=config.dump_folder,
            initial_load_path=getattr(
                config.checkpoint, "initial_load_path", ""
            )
            or "",
        )

        # Two concurrent jobs writing one checkpoint.folder is silent and it
        # physically mixes their shards -- it left two 453 GB dirs holding
        # 3,072 files where 192 belong. Record a claim and say so loudly if
        # someone else already holds one. Advisory only: a crashed predecessor
        # leaves a stale claim behind, and refusing to start on one would turn
        # every crash into a failed resume.
        #
        # Only claim if this job will actually WRITE. A --checkpoint.no-enable
        # run (smoke tests, config sweeps, bisects) never creates a file, so
        # claiming would be a pure false positive: it leaves a claim on the
        # default folder that then warns the next job -- which may be the one
        # legitimately using that directory. Observed 2026-08-19, where a
        # throwaway sweep spooked a live 12h run into a shard-mixing warning
        # about a collision that could not happen.
        if getattr(config.checkpoint, "enable", True):
            check_and_claim(
                config.checkpoint.folder,
                dump_folder=config.dump_folder,
                world_size=int(os.environ.get("WORLD_SIZE", -1)),
                is_rank_zero=(
                    not torch.distributed.is_initialized()
                    or torch.distributed.get_rank() == 0
                ),
            )

        self.checkpointer.load(step=config.checkpoint.load_step)
        logger.info(f"Training starts at step {self.step + 1}")

        # FT addition: per-replica profiling leaf folder
        leaf_folder = (
            ""
            if not self.ft_manager.enabled
            else f"replica_{self.ft_manager.replica_id}"
        )
        with (
            config.profiler.build(
                global_step=self.step,
                base_folder=config.dump_folder,
                leaf_folder=leaf_folder,
            ) as profiler,
            # FT addition: maybe_semi_sync_training context manager
            maybe_semi_sync_training(
                config.fault_tolerance,
                ft_manager=self.ft_manager,
                model=self.model_parts[0],
                n_layers=(
                    len(self.model_config.layers)
                    if hasattr(self.model_config, "layers")
                    else 0
                ),
                optimizer=self.optimizers,
                fragment_fn=(
                    config.model_spec.fragment_fn
                    if hasattr(config.model_spec, "fragment_fn")
                    else None
                ),
            ),
        ):
            # Walltime-aware checkpointing: resolve an ABSOLUTE wall-clock
            # deadline (time.time() epoch) so the backstop survives failover
            # relaunches. Prefer walltime_deadline_epoch (set once per job);
            # fall back to the relative walltime_seconds measured from now.
            wall_margin = config.walltime_checkpoint_margin_seconds
            if config.walltime_deadline_epoch > 0:
                wall_deadline = float(config.walltime_deadline_epoch)
                logger.info(
                    f"walltime-ckpt: absolute deadline {wall_deadline:.0f} "
                    f"(~{(wall_deadline - time.time()) / 60:.0f} min from now), "
                    f"margin {wall_margin}s"
                )
            elif config.walltime_seconds > 0:
                wall_deadline = time.time() + config.walltime_seconds
                logger.info(
                    f"walltime-ckpt: relative budget {config.walltime_seconds}s "
                    f"from loop entry (NOTE: resets on failover retry -- prefer "
                    f"walltime_deadline_epoch), margin {wall_margin}s"
                )
            else:
                wall_deadline = None

            # Cooperative stop on SIGTERM/SIGINT. Complements the walltime
            # deadline above rather than replacing it: the deadline needs to be
            # configured and only helps when it is set correctly, while the
            # signal path covers `timeout N` in the PBS scripts, PBS's own
            # pre-walltime SIGTERM, and an operator stopping a run by hand --
            # none of which the clock can see. Installed here, just before the
            # loop, so the handler cannot fire during setup (dataloader build,
            # checkpoint load) where there is no step worth saving.
            if config.save_on_signal:
                if signal_stop.install():
                    logger.info(
                        "signal-ckpt: SIGTERM/SIGINT will force a final "
                        "checkpoint and stop cleanly after the current step"
                    )
                else:
                    logger.warning(
                        "signal-ckpt: could not install signal handlers "
                        "(not the main thread?); a SIGTERM will lose every "
                        "step since the last checkpoint interval"
                    )

            # NaN-abort: count consecutive non-finite reported losses so a
            # diverged run does not "train" on NaN for the whole walltime.
            # Field lives on the top-level trainer Config (sibling of
            # walltime_deadline_epoch/batch_ramp_steps), so read config.<field>
            # -- NOT config.training.<field> (that is core TrainingConfig,
            # which has no such attribute -> AttributeError at train() start).
            nan_abort_n = config.nan_abort_consecutive
            consecutive_nonfinite = 0
            # Grad-norm runaway detector. Keeps a short history so the
            # threshold tracks the run's own scale instead of a guessed
            # constant (see grad_norm_abort in the Config).
            import collections
            import statistics as _stats

            gn_hist: collections.deque = collections.deque(
                maxlen=max(2, config.grad_norm_abort_window)
            )

            # --- Opt-in per-op numerics capture (80B overflow localization) ---
            # Inert unless EZPZ_DUMP_NUMERICS=1. Captures one step of per-op
            # activation stats via DebugMode into {dump_folder}/numerics/ to
            # name the op that first goes non-finite. An UNFILTERED fp64 capture
            # over 84L x dim9216 OOMs / exceeds the walltime window at 80B (why
            # the first attempt was reverted), so restrict to the overflow
            # suspects via EZPZ_NUMERICS_OPS (comma-separated op-name
            # substrings, default "mm,bmm,softmax" = attention scores + FFN
            # gate). EZPZ_NUMERICS_MIN_NUMEL raises the small-tensor cutoff.
            # See agent_tooling/numerics_debugging/. Revert after diagnosis.
            _numerics_capture = None
            if os.environ.get("EZPZ_DUMP_NUMERICS", "0") == "1":
                from agent_tooling.numerics_debugging.activation_tracer import (
                    ActivationCaptureProfiler,
                )

                _ops_env = os.environ.get("EZPZ_NUMERICS_OPS", "mm,bmm,softmax")
                _op_filter = {o for o in _ops_env.split(",") if o} or None
                _numerics_capture = ActivationCaptureProfiler(
                    enabled=True,
                    model=self.model_parts[0],
                    dump_dir=os.path.join(config.dump_folder, "numerics"),
                    capture_step=int(os.environ.get("EZPZ_NUMERICS_STEP", "6")),
                    op_filter=_op_filter,
                    min_numel=int(os.environ.get("EZPZ_NUMERICS_MIN_NUMEL", "1000")),
                )
                _numerics_capture.__enter__()

            data_iterator = self.batch_generator(self.dataloader)
            while self.should_continue_training():
                self.step += 1
                self.gc_handler.run(self.step)
                try:
                    loss_val = self.train_step(data_iterator)
                except DataloaderExhaustedError:
                    logger.warning("Ran out of data; last step was canceled.")
                    break

                # NaN-abort guard (opt-in via --nan-abort-consecutive>0).
                # A diverged optimizer keeps emitting NaN/inf loss every step;
                # without this the job burns its full window (see the 2026-07-03
                # 80B SophiaG NaN, ~12h wasted). Reset on any finite loss so a
                # lone transient never trips it.
                # Grad-norm runaway. Checked BEFORE the nan guard because
                # this failure never produces a non-finite value: the observed
                # SophiaG blow-up (job 12473783) ran 0.39 -> 2.41 -> 89.9 ->
                # 1702 -> 100611 with every value finite, so nan_abort never
                # fired and ~50 steps were spent diverging and recovering.
                #
                # Compares against the run's OWN recent median rather than a
                # constant: healthy grad_norm differs ~10x across these three
                # optimizers and drifts down as training proceeds, so a fixed
                # threshold is wrong for someone. Requires a full window before
                # arming, so early-training transients (a warmup spike is
                # normal and self-corrects) cannot trip it.
                # Skip warmup entirely. Measured on the healthy arms: EVERY
                # grad_norm above 20x sits at step <= 19 (adamw steps 5,6,7,11
                # up to 82.3; mano 6..19 up to 74.3), i.e. the documented
                # warmup transient that self-corrects. Without this skip the
                # guard aborts two perfectly good runs -- worse than the
                # failure it exists to catch. Loosening the threshold instead
                # would have blinded it to the real event (89.9 at step 1049).
                _gn = getattr(self, "_last_grad_norm", None)
                if (
                    config.grad_norm_abort > 0
                    and _gn is not None
                    and self.step > config.lr_scheduler.warmup_steps
                ):
                    gn = float(_gn)
                    if math.isfinite(gn):
                        if len(gn_hist) == gn_hist.maxlen:
                            med = _stats.median(gn_hist)
                            if med > 0 and gn > config.grad_norm_abort * med:
                                logger.error(
                                    "grad_norm runaway at step %d: %.4g is "
                                    "%.1fx the trailing median (%.4g) over "
                                    "%d steps; aborting before it burns the "
                                    "window (threshold %.1fx)",
                                    self.step, gn, gn / med, med,
                                    gn_hist.maxlen, config.grad_norm_abort,
                                )
                                break
                        gn_hist.append(gn)

                if nan_abort_n > 0:
                    if loss_val is None or not math.isfinite(loss_val):
                        consecutive_nonfinite += 1
                        logger.warning(
                            f"non-finite loss at step {self.step} "
                            f"({consecutive_nonfinite}/{nan_abort_n} consecutive)"
                        )
                        if consecutive_nonfinite >= nan_abort_n:
                            logger.error(
                                f"aborting: {consecutive_nonfinite} consecutive "
                                f"non-finite losses (nan_abort_consecutive="
                                f"{nan_abort_n}); run has diverged, stopping to "
                                "reclaim walltime"
                            )
                            break
                    else:
                        consecutive_nonfinite = 0

                saved_this_step = self.checkpointer.save(
                    self.step, last_step=(self.step == config.training.steps)
                )

                # Walltime guard: once within margin of the (absolute) deadline,
                # force a final checkpoint now and stop cleanly. Reuses the
                # interval-bypassing last_step path, then flushes any async save
                # so it actually lands on disk before the process exits. Without
                # this, a short job can run its whole window and save nothing.
                if wall_deadline is not None:
                    remaining = wall_deadline - time.time()
                    if remaining <= wall_margin:
                        logger.info(
                            f"walltime deadline near at step {self.step} "
                            f"({remaining:.0f}s left <= margin {wall_margin}s); "
                            "forcing final checkpoint and stopping"
                        )
                        if not saved_this_step:
                            self.checkpointer.save(self.step, last_step=True)
                        # Block until any async save is fully on disk before we
                        # break to teardown (close() does NOT wait for pending
                        # saves).
                        self.checkpointer.maybe_wait_for_saving()
                        break

                # Cooperative stop: a SIGTERM/SIGINT arrived (inner `timeout`,
                # PBS pre-walltime kill, or an operator). Same forced-save path
                # as the walltime guard -- last_step=True bypasses the interval,
                # then block until the async save is actually on disk, because
                # the process is about to be killed and close() does NOT wait.
                # Checked here, between steps, because every rank is
                # synchronized at this point; dcp.save is collective and cannot
                # be called from the handler itself.
                if signal_stop.stop_requested():
                    logger.warning(
                        f"{signal_stop.stop_signal_name()} received at step "
                        f"{self.step}; forcing final checkpoint and stopping"
                    )
                    if not saved_this_step:
                        self.checkpointer.save(self.step, last_step=True)
                    self.checkpointer.maybe_wait_for_saving()
                    logger.info(
                        f"signal-ckpt: checkpoint for step {self.step} is on "
                        "disk; exiting cleanly"
                    )
                    break

                # Run validation if validator is available
                if self.config.validator.enable and self.validator.should_validate(
                    self.step
                ):
                    self.validator.validate(self.model_parts, self.step)

                # signal the profiler that the next profiling step has started
                profiler.step()

                if _numerics_capture is not None:
                    _numerics_capture.step()

                # reduce timeout after first train step for faster signal
                # (assuming lazy init and compilation are finished)
                if self.step == 1:
                    _set_pg_timeouts_xpu_aware(
                        timeout=timedelta(seconds=config.comm.train_timeout_seconds),
                        parallel_dims=self.parallel_dims,
                    )

            if _numerics_capture is not None:
                _numerics_capture.__exit__(None, None, None)

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")
