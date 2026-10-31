#!/usr/bin/env python3
"""Attach a replicated-comparison job's evidence to its online W&B run."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import wandb


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, action="append", default=[])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args()
    files = []
    checkpoint_files = []
    for path in sorted(args.output.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(args.output)
        if "wandb" in relative.parts:
            continue
        entry = {"path": str(relative), "bytes": path.stat().st_size, "sha256": digest(path)}
        if "checkpoints" in relative.parts:
            checkpoint_files.append(entry)
        else:
            files.append((path, "output/" + str(relative), entry))
    for index, path in enumerate(args.log):
        if path.is_file():
            entry = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
            files.append((path, f"logs/{index}-{path.name}", entry))
    source_sha = subprocess.check_output(["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True).strip()
    manifest = {"job_id": args.job_id, "source_sha": source_sha,
                "scheduler_exit_code": args.exit_code,
                "files": [entry for _, _, entry in files],
                "checkpoint_files": checkpoint_files,
                "checkpoint_note": "Complete checkpoint files remain on project storage; this artifact includes their size/hash manifest."}
    manifest_path = args.output / "archive-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    run = wandb.init(entity="aurora_gpt", project="torchtitan.ezpz.train",
                     id=args.run_id, resume="allow", mode="online",
                     group="agpt-replicated-20261031", job_type="evidence-archive",
                     dir=str(args.output / "wandb"), settings=wandb.Settings(console="off"))
    artifact = wandb.Artifact("agpt-replicated-job-" + args.job_id, type="campaign-evidence",
                             metadata={"source_sha": source_sha, "exit_code": args.exit_code})
    for path, name, _ in files:
        artifact.add_file(str(path), name=name)
    artifact.add_file(str(manifest_path), name="archive-manifest.json")
    source_paths = subprocess.check_output(["git", "-C", str(args.root), "diff", "--name-only", "5be47f7c", "HEAD"], text=True).splitlines()
    for relative in source_paths:
        path = args.root / relative
        if path.is_file():
            artifact.add_file(str(path), name="source/" + relative)
    run.log_artifact(artifact).wait()
    run.summary["evidence/artifact"] = artifact.qualified_name
    run.summary["evidence/source_sha"] = source_sha
    run.summary["evidence/file_count"] = len(files)
    run.summary["evidence/scheduler_exit_code"] = args.exit_code
    run.summary.update()
    print("ARCHIVE_COMPLETE " + json.dumps({"url": run.url, "artifact": artifact.qualified_name, "files": len(files)}), flush=True)
    run.finish(exit_code=args.exit_code)


if __name__ == "__main__":
    main()
