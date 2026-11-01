#!/usr/bin/env python3
"""Attach a replicated-comparison job's evidence to its online W&B run."""

import argparse
import hashlib
import gzip
import json
import re
from pathlib import Path
import shutil
import subprocess
import tempfile

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
    parser.add_argument("--project", default="torchtitan.ezpz.train")
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--application-exit-code", type=int)
    args = parser.parse_args()
    application_exit = args.exit_code if args.application_exit_code is None else args.application_exit_code
    files = []
    checkpoint_files = []
    for path in sorted(args.output.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(args.output)
        if "wandb" in relative.parts or relative.name == "archive-manifest.json":
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
    source_sha = subprocess.check_output(
        ["git", "-C", str(args.root), "rev-parse", args.source_sha + "^{commit}"],
        text=True,
    ).strip()
    recorded_path = args.output / "provenance" / "tt-head.txt"
    if recorded_path.exists() and recorded_path.read_text().strip() != source_sha:
        raise ValueError("Source SHA does not match this run's recorded provenance")
    # Preserve large text logs losslessly outside the training output tree.
    cache = args.root / ".cache" / "agpt-evidence-compression"
    cache.mkdir(parents=True, exist_ok=True)
    stored = []
    for path, name, entry in files:
        entry["artifact_path"] = name
        if entry["bytes"] >= 64 * 1024 * 1024 and (path.suffix in (".log", ".txt", ".out") or re.search(r"\.[oe]\d+$", path.name)):
            target = cache / (entry["sha256"] + ".log.gz")
            if not target.exists():
                temporary = target.with_suffix(".partial")
                with path.open("rb") as source, temporary.open("wb") as raw:
                    with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=3, mtime=0) as compressed:
                        shutil.copyfileobj(source, compressed, length=8 * 1024 * 1024)
                temporary.rename(target)
            restored = hashlib.sha256()
            restored_bytes = 0
            with gzip.open(target, "rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    restored.update(block)
                    restored_bytes += len(block)
            if restored.hexdigest() != entry["sha256"] or restored_bytes != entry["bytes"]:
                raise ValueError("Lossless compression verification failed: " + str(path))
            name += ".gz"
            entry.update(artifact_path=name, codec="gzip", stored_bytes=target.stat().st_size,
                         stored_sha256=digest(target), decompression_verified=True)
            path = target
        stored.append((path, name, entry))
    files = stored
    manifest = {"job_id": args.job_id, "source_sha": source_sha,
                "scheduler_exit_code": args.exit_code,
                "application_exit_code": application_exit,
                "files": [entry for _, _, entry in files],
                "checkpoint_files": checkpoint_files,
                "checkpoint_note": "Complete checkpoint files remain on project storage; this artifact includes their size/hash manifest."}
    manifest_path = args.output / "archive-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    run = wandb.init(entity="aurora_gpt", project=args.project,
                     id="agpt-replicated-archive-20261101", resume="allow", mode="online", job_type="evidence-archive", config={"archival_only": True},
                     group="agpt-replicated-20261031",
                     dir=str(args.output / "wandb"), settings=wandb.Settings(console="off", x_disable_meta=True, x_save_requirements=False, x_disable_stats=True, disable_git=True, save_code=False, disable_code=True))
    artifact = wandb.Artifact("agpt-replicated-evidence-" + args.run_id, type="campaign-evidence",
                             metadata={"source_sha": source_sha, "exit_code": args.exit_code, "archival_environment": {"wandb_version": wandb.__version__, "purpose": "CPU evidence upload only; not the training runtime"}, "training_runtime_authority": "output provenance and rank diagnostics"})
    for path, name, _ in files:
        artifact.add_file(str(path), name=name)
    artifact.add_file(str(manifest_path), name="archive-manifest.json")
    source_paths = subprocess.check_output(
        ["git", "-C", str(args.root), "diff", "--name-only", "--diff-filter=ACMR",
         "5be47f7c", source_sha, "--", "torchtitan/experiments/ezpz"], text=True,
    ).splitlines()
    # Read historical blobs, since later jobs may have advanced the checkout.
    with tempfile.TemporaryDirectory(prefix="agpt-evidence-source-") as source_dir:
        for relative in source_paths:
            path = Path(source_dir) / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(subprocess.check_output(
                ["git", "-C", str(args.root), "show", source_sha + ":" + relative]
            ))
            artifact.add_file(str(path), name="source/" + relative)
        run.log_artifact(artifact).wait()
    target = wandb.Api().run("aurora_gpt/" + args.project + "/" + args.run_id)
    target.log_artifact(wandb.Api().artifact(artifact.qualified_name))
    target.summary["evidence/artifact"] = artifact.qualified_name
    target.summary["evidence/source_sha"] = source_sha
    target.summary["evidence/file_count"] = len(files)
    target.summary["evidence/scheduler_exit_code"] = args.exit_code
    target.summary["evidence/application_exit_code"] = application_exit
    target.summary.update()
    print("ARCHIVE_COMPLETE " + json.dumps({"url": target.url, "archive_run_url": run.url, "artifact": artifact.qualified_name, "files": len(files)}), flush=True)
    run.finish(exit_code=0)


if __name__ == "__main__":
    main()
