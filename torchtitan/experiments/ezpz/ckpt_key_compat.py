"""Load checkpoints written before two upstream key renames.

TWO renames are handled, and an old enough checkpoint needs both in the same
load:

1. the attention QKV wrapper refactor (below), and
2. ``output.weight`` -> ``lm_head.weight``, when upstream renamed the LM head
   (``self.lm_head = config.lm_head.build()`` in
   ``torchtitan/models/common/decoder.py``).

The head rename was found on 2026-08-17 the same way as the first: the 2B-256
constant-LR fork died on all 3,072 ranks with ``Missing key in checkpoint
state_dict: lm_head.weight`` and sat dead for ~9 hours holding a 256-node seat
of umbrella 8756957. Its seed (``...-constlr-from9500/step-9500``) predates
BOTH renames; the same fork's own step-20600, written by current code, needs
neither -- which is why one fork could half-work. Note that EVERY production
2B/20B checkpoint still on disk spells the head ``output.weight``; the live
chains only avoid this because their clones are pinned to older code.

Original motivation follows.

Load checkpoints written before the attention QKV wrapper refactor.

Older AuroraGPT checkpoints store the attention projections flat::

    layers.0.attention.wq.weight
    layers.0.attention.wk.weight
    layers.0.attention.wv.weight

Current model code wraps them in a ``qkv_linear`` submodule
(``self.qkv_linear = config.qkv_linear.build()`` in
``torchtitan/models/common/attention.py``), so the state dict it asks for is::

    layers.0.attention.qkv_linear.wq.weight
    ...

`dcp.load` matches by exact key, so loading an old checkpoint with new code
dies on the first mismatch::

    RuntimeError: Missing key in checkpoint state_dict:
      layers.0.attention.qkv_linear.wk.weight

That is what killed the 2B-256 constant-LR fork (trainer 4) in umbrella
8714502: it never reached step 1 and burned its slot of a 2098-node job.
Its seed, ``...-constlr-from9500/step-9500``, is the highest surviving
PRE-DECAY 256N checkpoint, so reseeding from anything newer would defeat the
experiment.

This module renames the requested keys down to the on-disk spelling before
`dcp.load`, then renames them back so the model sees what it expects. It is a
pure key remap -- no tensor is read, reshaped, or reinterpreted -- so it cannot
change numerics. The wrapper only moved where the parameters live in the module
tree; the tensors themselves are unchanged.

Optimizer state is remapped too: those keys embed the parameter FQN
(``optimizer.param_groups.layers.0.attention.wk.weight.betas``), so a
model-only remap would leave the optimizer half-translated.

Usage -- call once before the trainer loads, and only for a run whose seed
predates the refactor::

    from torchtitan.experiments.ezpz.ckpt_key_compat import (
        install_flat_attention_compat,
    )
    install_flat_attention_compat(checkpointer)

`needs_flat_attention_compat(path)` reads a checkpoint's DCP metadata and
reports whether the shim is required, so callers can install it conditionally
rather than guessing.
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

import torch

# layers.<N>.attention.<wq|wk|wv>.<rest>  ->  ...attention.qkv_linear.<w?>.<rest>
# Anchored on the attention prefix so nothing else in the tree can match, and
# scoped to the three projections the wrapper actually absorbed (wo stayed put).
_NEW_TO_OLD = re.compile(r"(layers\.\d+\.attention\.)qkv_linear\.(w[qkv]\.)")
_OLD_TO_NEW = re.compile(r"(layers\.\d+\.attention\.)(w[qkv]\.)")

# output.<rest>  <->  lm_head.<rest>  (the second rename, see module docstring).
#
# The head is a TOP-LEVEL module, so the FQN is either `output.weight` or an
# optimizer key that embeds it after a known optimizer prefix
# (`optimizer.param_groups.output.weight.betas`). Anchoring on "start of
# string, or after any dot" is too loose: it would also rewrite
# `layers.0.moe.output.weight`, a different module that legitimately contains
# `output`. MoE configs have exactly such a key, so this is not hypothetical --
# the unit test caught it.
#
# So: match at the start of the string, or immediately after an optimizer
# prefix segment, and nowhere else.
# The composite spelling `optimizer.state.<fqn>...` is what DCP produces AFTER
# flattening. But the OptimizersContainer's own state_dict() emits
# CONTAINER-level keys with no `optimizer.` prefix -- `state.<fqn>.<name>`
# and `param_groups.<fqn>.<key>` (torchtitan/components/checkpoint_utils.py).
# The optimizer swap below sees that spelling, so the head remap has to match
# it too. Purely additive: every previously-matching and previously-
# non-matching shape (composite keys, bare model keys, the
# `layers.0.moe.output.weight` trap) is unchanged.
_OPT_PREFIX = r"(?:^|^(?:optimizer|optimizers)\.(?:[A-Za-z_]+\.)*?|^(?:state|param_groups)\.)"
_HEAD_NEW_TO_OLD = re.compile(_OPT_PREFIX + r"lm_head\.")
_HEAD_OLD_TO_NEW = re.compile(_OPT_PREFIX + r"output\.")


def to_flat(key: str) -> str:
    """Rewrite a current-code key to its pre-refactor spelling.

    A key that is already flat is returned unchanged. Without that guard a
    second application would strip ``qkv_linear`` off a key that legitimately
    carries it, silently corrupting a load against a current-format
    checkpoint.
    """
    if "qkv_linear." not in key:
        return key
    return _NEW_TO_OLD.sub(r"\1\2", key)


def to_nested(key: str) -> str:
    """Rewrite a pre-refactor key to its current-code spelling."""
    # Guard against double-application: a key already carrying qkv_linear is
    # left alone.
    if "qkv_linear." in key:
        return key
    return _OLD_TO_NEW.sub(r"\1qkv_linear.\2", key)


def head_to_old(key: str) -> str:
    """Rewrite ``lm_head.*`` to the pre-rename ``output.*``."""
    return _HEAD_NEW_TO_OLD.sub(lambda m: m.group(0)[: -len("lm_head.")] + "output.", key)


def head_to_new(key: str) -> str:
    """Rewrite ``output.*`` to the current ``lm_head.*``."""
    return _HEAD_OLD_TO_NEW.sub(lambda m: m.group(0)[: -len("output.")] + "lm_head.", key)


def needs_output_head_compat(checkpoint_dir: str) -> bool:
    """True if `checkpoint_dir` spells the head ``output.weight``.

    Independent of the attention check: a checkpoint can need either rename,
    both, or neither. The 2B-256 constant-LR seed (step-9500) needs both --
    it predates the qkv wrapper AND the head rename -- while the same fork's
    own step-20600, written by current code, needs neither.
    """
    from torch.distributed.checkpoint import FileSystemReader  # noqa: PLC0415

    try:
        md = FileSystemReader(checkpoint_dir).read_metadata()
    except Exception:
        return False
    keys = md.state_dict_metadata.keys()
    return "output.weight" in keys and "lm_head.weight" not in keys


def needs_flat_attention_compat(checkpoint_dir: str) -> bool:
    """True if `checkpoint_dir` holds pre-refactor flat attention keys.

    Reads only the DCP metadata, so it is cheap and safe to call on a path
    that may not exist (returns False rather than raising).
    """
    from torch.distributed.checkpoint import FileSystemReader  # noqa: PLC0415

    try:
        md = FileSystemReader(checkpoint_dir).read_metadata()
    except Exception:
        return False
    keys = md.state_dict_metadata.keys()
    has_nested = any("attention.qkv_linear." in k for k in keys)
    has_flat = any(_OLD_TO_NEW.search(k) for k in keys)
    return has_flat and not has_nested


def needs_split_qkv_compat(checkpoint_dir: str) -> bool:
    """True when a native DCP stores logical Q/K/V instead of packed QKV."""
    from torch.distributed.checkpoint import FileSystemReader  # noqa: PLC0415

    try:
        keys = FileSystemReader(checkpoint_dir).read_metadata().state_dict_metadata.keys()
    except Exception:
        return False
    has_split = any("attention.qkv_linear.wq.weight" in key for key in keys)
    has_fused = any("attention.qkv_linear.wqkv.weight" in key for key in keys)
    return has_split and not has_fused


def split_qkv_to_fused(
    wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, *, head_dim: int
) -> torch.Tensor:
    """Pack logical Q/K/V weights using QKVLinear's KV-group ordering."""
    num_kv_heads = wk.shape[0] // head_dim
    heads_per_kv = wq.shape[0] // (num_kv_heads * head_dim)
    tail = wq.shape[1:]
    q = wq.reshape(num_kv_heads, heads_per_kv, head_dim, *tail)
    k = wk.reshape(num_kv_heads, 1, head_dim, *tail)
    v = wv.reshape(num_kv_heads, 1, head_dim, *tail)
    return torch.cat([q, k, v], dim=1).reshape(-1, *tail)


def _install_split_qkv_model_compat(checkpointer: Any) -> None:
    """Restore the former native-DCP hooks for a legacy split-QKV load."""
    model_wrapper = getattr(checkpointer, "states", {}).get("model")
    models = getattr(model_wrapper, "model", ())
    from torchtitan.models.common.attention import QKVLinear  # noqa: PLC0415
    from torch.distributed.tensor import DTensor, Replicate  # noqa: PLC0415

    def split_on_save(module, state_dict, prefix, local_metadata) -> None:
        del local_metadata
        for param in ("weight", "bias"):
            key = f"{prefix}wqkv.{param}"
            if key not in state_dict:
                continue
            tensor = state_dict.pop(key)
            if isinstance(tensor, DTensor):
                tensor = tensor.redistribute(
                    tensor.device_mesh, [Replicate()] * tensor.device_mesh.ndim
                )
            num_kv_heads = tensor.shape[0] // (module.r_dim * module.head_dim)
            tail = tensor.shape[1:]
            packed = tensor.reshape(
                num_kv_heads, module.r_dim, module.head_dim, *tail
            )
            state_dict[f"{prefix}wq.{param}"] = packed[
                :, : module.heads_per_kv
            ].reshape(-1, *tail).contiguous()
            state_dict[f"{prefix}wk.{param}"] = packed[
                :, module.heads_per_kv
            ].reshape(-1, *tail).contiguous()
            state_dict[f"{prefix}wv.{param}"] = packed[
                :, module.heads_per_kv + 1
            ].reshape(-1, *tail).contiguous()

    def merge_on_load(module, state_dict, prefix, *args) -> None:
        del args
        for param in ("weight", "bias"):
            keys = tuple(f"{prefix}{name}.{param}" for name in ("wq", "wk", "wv"))
            if not all(key in state_dict for key in keys):
                continue
            wq, wk, wv = (state_dict.pop(key) for key in keys)
            if isinstance(wq, DTensor):
                wq, wk, wv = (
                    tensor.redistribute(
                        tensor.device_mesh, [Replicate()] * tensor.device_mesh.ndim
                    )
                    for tensor in (wq, wk, wv)
                )
            state_dict[f"{prefix}wqkv.{param}"] = split_qkv_to_fused(
                wq, wk, wv, head_dim=module.head_dim
            )

    installed = False
    for model in models:
        for module in model.modules():
            if isinstance(module, QKVLinear) and not getattr(
                module, "_legacy_split_qkv_compat", False
            ):
                module.register_state_dict_post_hook(split_on_save)
                module.register_load_state_dict_pre_hook(merge_on_load)
                module._legacy_split_qkv_compat = True
                installed = True

    # ModelWrapper builds its stable-storage cache before this compatibility
    # detector runs. Rebuild it after registering the hooks so DCP requests the
    # split keys that exist on disk rather than retaining the stale fused key.
    if installed and model_wrapper is not None:
        model_wrapper.cached_state_dict = model_wrapper._get_state_dict()


def install_flat_attention_compat(
    checkpointer: Any, *, attention: bool = True, head: bool = False
) -> None:
    """Wrap `checkpointer.dcp_load` so pre-rename checkpoints load.

    ``attention`` handles the qkv_linear wrapper; ``head`` handles
    ``output`` -> ``lm_head``. They compose: a checkpoint old enough to
    predate both (the 2B-256 constant-LR seed) needs both passes in the same
    load, so this is one wrapper applying whichever renames were requested
    rather than two shims fighting over ``dcp_load``.

    Idempotent: installing twice is a no-op.
    """
    if getattr(checkpointer, "_flat_attention_compat", False):
        return

    original = checkpointer.dcp_load

    def _down(key: str) -> str:
        """Current-code spelling -> on-disk spelling."""
        if attention:
            key = to_flat(key)
        if head:
            key = head_to_old(key)
        return key

    def _up(key: str) -> str:
        """On-disk spelling -> current-code spelling."""
        if attention:
            key = to_nested(key)
        if head:
            key = head_to_new(key)
        return key

    @contextlib.contextmanager
    def _optimizer_keys_remapped(state_dict: dict[str, Any]):
        """Temporarily make the optimizer container emit on-disk key spellings.

        The wrapper below can only rename what it is HANDED, and for the
        optimizer it is handed the live container object under the single key
        "optimizer" -- not FQN strings. torchtitan only expands the MODEL to
        flat keys (`_flattened_model_states_sd`); the optimizer's real keys
        (`state.<fqn>.<name>`, `param_groups.<fqn>.<key>`) are produced INSIDE
        `dcp.load`, when it calls `state_dict()` on that container. By then the
        wrapper has already run, which is why a checkpoint whose model half
        loaded fine still died on
        `optimizer.state.layers.0.attention.qkv_linear.wq.weight.step`.

        So intercept one level lower: swap the container's `state_dict` for the
        duration of the load, renaming its output down to the on-disk spelling.

        SCOPED, never permanent. The SAVE path also calls `state_dict()` on
        every Stateful, so a permanently-installed wrapper would write NEW
        checkpoints in the OLD spelling -- turning a read-compat shim into a
        corruption source. The finally is load-bearing.

        `load_state_dict` is deliberately NOT wrapped: it must receive
        current-code keys, because `_unflatten_optim_state_dict` looks each one
        up against the LIVE optimizer's param_names.
        """
        opt = state_dict.get("optimizer")
        real = getattr(opt, "state_dict", None)
        if opt is None or real is None or not callable(real):
            yield
            return

        def _renamed_state_dict(*a: Any, **kw: Any) -> dict[str, Any]:
            return {_down(k): v for k, v in real(*a, **kw).items()}

        try:
            opt.state_dict = _renamed_state_dict  # type: ignore[method-assign]
            yield
        finally:
            # Restore by DELETING the instance attribute so the class method is
            # visible again; assigning `real` back would leave a bound-method
            # shadow that survives into the save path.
            try:
                del opt.state_dict  # type: ignore[attr-defined]
            except (AttributeError, TypeError):
                opt.state_dict = real  # type: ignore[method-assign]

    def dcp_load(state_dict: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        renamed = {_down(k): v for k, v in state_dict.items()}

        # The swap must wrap BOTH exits. The early return below fires whenever
        # `_down` is a no-op on every TOP-LEVEL key -- and the top-level keys
        # are model FQNs plus the bare literals "optimizer"/"lr_scheduler"/
        # "dataloader"/"train_state". A checkpoint needing ONLY the
        # optimizer-side rename takes that path, so installing the swap after
        # it would silently skip exactly the case this exists for.
        with _optimizer_keys_remapped(state_dict):
            if renamed.keys() == state_dict.keys():
                return original(state_dict, *args, **kwargs)

            original(renamed, *args, **kwargs)

            # dcp.load fills the dict in place, so copy the loaded values back
            # under the keys the caller (and the model) expects.
            state_dict.clear()
            state_dict.update({_up(k): v for k, v in renamed.items()})

    checkpointer.dcp_load = dcp_load
    checkpointer._flat_attention_compat = True


def maybe_install_flat_attention_compat(
    checkpointer: Any,
    folder: str,
    load_step: int = -1,
    dump_folder: str = "",
    initial_load_path: str = "",
) -> bool:
    """Install the remap only if the checkpoint about to be loaded needs it.

    `folder` is `checkpoint.folder` (dump_folder-relative, as configured);
    `dump_folder` is `config.dump_folder`, which the checkpointer PREPENDS to
    `folder` -- pass it or the detector looks in the wrong place. It is a FLAT
    field on `Trainer.Config`; there is no `config.job` namespace, and reaching
    for one raises AttributeError at the top of `train()`, before step 1.
    `load_step` is the step being resumed, or -1 for "latest". Resolves the step
    directory
    the same way the checkpointer will, inspects its metadata, and installs
    the shim only for a pre-refactor flat-attention checkpoint.

    Returns whether the shim was installed. Never raises: a resume that is
    going to fail should fail in the checkpointer with its own error, not
    here in the detector.

    `initial_load_path` is the SEED path (`checkpoint.initial_load_path`). It
    is consulted only when `folder` holds no resumable step, because that is
    the only condition under which the checkpointer honors it -- with a
    resumable checkpoint present it logs "initial_load_path is provided but
    the checkpoint.folder exists" and loads from `folder` instead. Passing it
    matters because a seeded FRESH chain is exactly the case where `folder` is
    empty: probing only `folder` there means probing a directory that does not
    exist yet, so the shim never installs and the load dies with the very
    "Missing key in checkpoint state_dict" this exists to prevent (job
    8771774, seeding t4 from a converted pre-refactor checkpoint).

    The `dump_folder` argument is NOT optional in practice. Omitting it in
    job 8744247 made the detector stat `<cwd>/checkpoints/...` while the real
    tree is `<cwd>/outputs/checkpoints/...`; `read_metadata()` raised, the
    blanket `except` returned False, and trainer 4 died on all 3,072 ranks
    with `Missing key in checkpoint state_dict: layers.0.attention.qkv_linear...`
    -- the exact failure this shim exists to prevent. A wrong path must be
    loud, so a non-existent resolved directory now warns.
    """
    import logging  # noqa: PLC0415
    import os  # noqa: PLC0415

    log = logging.getLogger(__name__)
    try:
        rel = os.path.join(dump_folder, folder) if dump_folder else folder
        base = rel if os.path.isabs(rel) else os.path.join(os.getcwd(), rel)

        # Resolve the step dir the checkpointer will ACTUALLY load, which is
        # not always under `folder`: a resumable step there wins, otherwise the
        # seed at initial_load_path is used (and `folder` may not even exist).
        step_dir = ""
        if os.path.isdir(base):
            if load_step is not None and load_step >= 0:
                cand = os.path.join(base, f"step-{load_step}")
                if os.path.isdir(cand):
                    step_dir = cand
            else:
                steps = [
                    int(d.split("-", 1)[1])
                    for d in os.listdir(base)
                    if d.startswith("step-") and d.split("-", 1)[1].isdigit()
                ]
                if steps:
                    step_dir = os.path.join(base, f"step-{max(steps)}")

        if not step_dir:
            if initial_load_path and os.path.isdir(initial_load_path):
                # initial_load_path names the step dir itself, not its parent.
                step_dir = initial_load_path
                log.info(
                    "flat-attention compat: %r holds no resumable step, so the "
                    "checkpointer will load the seed at %r. Checking that "
                    "instead.",
                    base,
                    initial_load_path,
                )
            else:
                if not os.path.isdir(base):
                    log.warning(
                        "flat-attention compat: checkpoint folder %r does not "
                        "exist and no usable initial_load_path was given, so "
                        "the pre-refactor key check CANNOT run. If this resume "
                        "is from an old checkpoint it will fail with 'Missing "
                        "key in checkpoint state_dict'. Check that "
                        "dump_folder=%r + folder=%r is correct.",
                        base,
                        dump_folder,
                        folder,
                    )
                return False

        want_attention = needs_flat_attention_compat(step_dir)
        want_head = needs_output_head_compat(step_dir)
        want_split_qkv = needs_split_qkv_compat(step_dir)
        if not (want_attention or want_head or want_split_qkv):
            return False

        if want_split_qkv:
            _install_split_qkv_model_compat(checkpointer)

        install_flat_attention_compat(
            checkpointer, attention=want_attention, head=want_head
        )
        needed = []
        if want_attention:
            needed.append(
                "flat attention keys (layers.N.attention.{wq,wk,wv} -> "
                "...qkv_linear.{wq,wk,wv})"
            )
        if want_head:
            needed.append("the pre-rename head (output.weight -> lm_head.weight)")
        if want_split_qkv:
            needed.append("split Q/K/V weights packed into native fused wqkv")
        log.warning(
            "%s holds PRE-REFACTOR keys: %s. Installing the remap so it can be "
            "loaded by current code. This is a key rename only -- no tensor is "
            "altered.",
            step_dir,
            " and ".join(needed),
        )
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("flat-attention compat detection skipped: %r", e)
        return False
