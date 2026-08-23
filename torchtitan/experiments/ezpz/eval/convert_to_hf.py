# Adapted from scripts/checkpoint_conversion/convert_to_hf.py
# Modified to support experiment model paths (torchtitan.experiments.*)

import argparse
import importlib
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import TORCH_DTYPE_MAP


@torch.inference_mode()
def convert_to_hf(
    input_dir,
    output_dir,
    model_name,
    model_flavor,
    hf_assets_path,
    export_dtype,
):
    # load model and model args so that we can get the state dict shape
    # Support both core models (torchtitan.models.*) and experiment models
    if "." in model_name:
        model_module = importlib.import_module(f"torchtitan.{model_name}")
    else:
        model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    model_spec = model_module.model_registry(model_flavor)
    model_config = model_spec.model

    with torch.device("cpu"):
        model = model_config.build()
    model = ModelWrapper(model)

    sd_adapter = model_spec.state_dict_adapter(model_config, hf_assets_path)
    assert (
        sd_adapter is not None
    ), "trying to convert checkpoint from DCP to HF safetensors format, but sd_adapter is not provided."

    # RoPE convention is NOT recoverable from the checkpoint: both rope caches
    # are registered with persistent=False, so nothing on disk says which one
    # the run used. The adapter reads it from the flavor passed on the command
    # line, and picking the wrong one applies (or skips) the Q/K permute --
    # which loads without error and only shows up as gibberish at generation
    # time.
    #
    # There is NO safe per-chain default: the chains changed convention
    # MID-FLIGHT when 5ffb850a1 (2026-06-25) flipped CONFIG_SUFFIX to _real and
    # the running chains picked it up on their next resume. Per W&B run
    # metadata (the authoritative record of what each run executed):
    #
    #   20b_v2_256  agpt_20b -> agpt_20b_real  2026-07-10  (loss 2.69 -> 6.14)
    #   20b_v2_512  agpt_20b -> agpt_20b_real  2026-07-05  (loss 2.57 -> 6.03)
    #   2b_v2_512   agpt_2b  -> agpt_2b_real   2026-08-05
    #   2b_v2_256   agpt_2b  throughout
    #
    # So the right flavor depends on WHICH STEP is being converted. Resolve it
    # with scripts/eval/rope_flavor_for_step.py; do not guess, and do not infer
    # from a submit script's default.
    #
    # Announce the convention loudly and write it next to the weights, so a
    # mismatch is visible in the log and auditable afterwards.
    rope_is_cos_sin = getattr(sd_adapter, "_is_cos_sin", None)
    rope_name = {True: "cos_sin", False: "complex", None: "unknown"}[rope_is_cos_sin]
    print(
        f"[convert_to_hf] flavor={model_flavor!r} -> RoPE={rope_name} "
        f"(Q/K permute {'SKIPPED' if rope_is_cos_sin else 'APPLIED'}). "
        "The checkpoint does not record its convention -- if this does not "
        "match how it was TRAINED, the export is silently corrupt. Resolve "
        "with scripts/eval/rope_flavor_for_step.py; see "
        "docs/guides/known-bugs/rope-flavor-mismatch.md"
    )

    # allocate state dict memory with empty weights to load checkpoint
    state_dict = model._get_state_dict()
    dcp.load(
        state_dict,
        checkpoint_id=input_dir,
    )

    # convert state dict tt->hf
    hf_state_dict = sd_adapter.to_hf(state_dict)

    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    # map and apply export dtype if needed
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    if target_dtype != torch.float32:
        hf_state_dict = {k: v.to(target_dtype) for k, v in hf_state_dict.items()}

    dcp.save(
        hf_state_dict,
        storage_writer=storage_writer,
    )

    # Provenance for the export: which flavor produced it, and therefore which
    # RoPE convention the weights are in. Without this there is no way to audit
    # an existing HF dir after the fact -- the weights look identical either way.
    try:
        import json

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / "ezpz_export.json", "w") as fh:
            json.dump(
                {
                    "source_dcp": str(input_dir),
                    "model_name": model_name,
                    "model_flavor": model_flavor,
                    "rope": rope_name,
                    "export_dtype": export_dtype,
                },
                fh,
                indent=2,
            )
    except OSError as exc:
        # Provenance is a nicety; never fail a good conversion over it.
        print(f"[convert_to_hf] WARNING: could not write ezpz_export.json: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP weights to HF format.")
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with DCP weights."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for HF checkpoint."
    )
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        help="Path to HF assets directory.",
        default="./assets/hf/gemma-7b",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        nargs="?",
        default="experiments.ezpz.agpt",
    )
    parser.add_argument(
        # REQUIRED, deliberately no default. This selects the RoPE convention,
        # the chains switched convention mid-flight, and a wrong value produces
        # an export that loads cleanly and generates gibberish. The old default
        # of "2b" was silently wrong for every post-2026-07 20B checkpoint.
        # Resolve with scripts/eval/rope_flavor_for_step.py.
        "--model_flavor",
        type=str,
        required=True,
        help=(
            "Model flavor, e.g. 2b / 2b_real / 20b / 20b_real / 80b. REQUIRED: "
            "this picks the RoPE convention and there is no safe default -- "
            "the chains switched mid-flight. Resolve a specific step with "
            "scripts/eval/rope_flavor_for_step.py --chain <k> --step <N>."
        ),
    )
    parser.add_argument(
        "--export_dtype",
        type=str,
        nargs="?",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Export dtype for HF checkpoint (default: bfloat16)",
    )
    args = parser.parse_args()

    convert_to_hf(
        args.input_dir,
        args.output_dir,
        args.model_name,
        args.model_flavor,
        args.hf_assets_path,
        args.export_dtype,
    )
