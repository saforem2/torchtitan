"""Load checkpoints written before the attention QKV wrapper refactor.

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

import re
from typing import Any

# layers.<N>.attention.<wq|wk|wv>.<rest>  ->  ...attention.qkv_linear.<w?>.<rest>
# Anchored on the attention prefix so nothing else in the tree can match, and
# scoped to the three projections the wrapper actually absorbed (wo stayed put).
_NEW_TO_OLD = re.compile(r"(layers\.\d+\.attention\.)qkv_linear\.(w[qkv]\.)")
_OLD_TO_NEW = re.compile(r"(layers\.\d+\.attention\.)(w[qkv]\.)")


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


def install_flat_attention_compat(checkpointer: Any) -> None:
    """Wrap `checkpointer.dcp_load` so flat-attention checkpoints load.

    Idempotent: installing twice is a no-op.
    """
    if getattr(checkpointer, "_flat_attention_compat", False):
        return

    original = checkpointer.dcp_load

    def dcp_load(state_dict: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        renamed = {to_flat(k): v for k, v in state_dict.items()}
        # Nothing to do if the model's keys already match the old spelling.
        if renamed.keys() == state_dict.keys():
            return original(state_dict, *args, **kwargs)

        original(renamed, *args, **kwargs)

        # dcp.load fills the dict in place, so copy the loaded values back
        # under the keys the caller (and the model) expects.
        state_dict.clear()
        state_dict.update({to_nested(k): v for k, v in renamed.items()})

    checkpointer.dcp_load = dcp_load
    checkpointer._flat_attention_compat = True


def maybe_install_flat_attention_compat(
    checkpointer: Any, folder: str, load_step: int = -1, dump_folder: str = ""
) -> bool:
    """Install the remap only if the checkpoint about to be loaded needs it.

    `folder` is `checkpoint.folder` (dump_folder-relative, as configured);
    `dump_folder` is `job.dump_folder`, which the checkpointer PREPENDS to
    `folder` -- pass it or the detector looks in the wrong place. `load_step`
    is the step being resumed, or -1 for "latest". Resolves the step directory
    the same way the checkpointer will, inspects its metadata, and installs
    the shim only for a pre-refactor flat-attention checkpoint.

    Returns whether the shim was installed. Never raises: a resume that is
    going to fail should fail in the checkpointer with its own error, not
    here in the detector.

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
        if not os.path.isdir(base):
            log.warning(
                "flat-attention compat: checkpoint folder %r does not exist, so "
                "the pre-refactor key check CANNOT run. If this resume is from an "
                "old checkpoint it will fail with 'Missing key in checkpoint "
                "state_dict'. Check that dump_folder=%r + folder=%r is correct.",
                base,
                dump_folder,
                folder,
            )
            return False
        if load_step is not None and load_step >= 0:
            step_dir = os.path.join(base, f"step-{load_step}")
        else:
            steps = [
                int(d.split("-", 1)[1])
                for d in os.listdir(base)
                if d.startswith("step-") and d.split("-", 1)[1].isdigit()
            ]
            if not steps:
                return False
            step_dir = os.path.join(base, f"step-{max(steps)}")

        if not needs_flat_attention_compat(step_dir):
            return False

        install_flat_attention_compat(checkpointer)
        log.warning(
            "%s holds PRE-REFACTOR flat attention keys "
            "(layers.N.attention.{wq,wk,wv}); installing the qkv_linear remap "
            "so it can be loaded by current code. This is a key rename only -- "
            "no tensor is altered.",
            step_dir,
        )
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("flat-attention compat detection skipped: %r", e)
        return False
