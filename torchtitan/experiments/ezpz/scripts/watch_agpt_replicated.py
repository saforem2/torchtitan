#!/usr/bin/env python3
"""Monitor one explicit Sunspot job set, or display its local cached state."""

import argparse
import json
from pathlib import Path
import subprocess
import time

from rich.console import Console
from rich.table import Table


def probe(jobs):
    script = """
import json, pathlib, re, subprocess
jobs = json.loads(JOBS_JSON)
results = {}
for jobid, spec in jobs.items():
    p = subprocess.run(['/opt/pbs/bin/qstat', '-xf', jobid], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=20)
    attrs = dict(re.findall(r'^    ([A-Za-z_]+) = (.*)$', p.stdout, re.M))
    path = pathlib.Path(spec['log'])
    tail = ''
    if path.exists():
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size - 12000))
            tail = f.read().decode('utf-8', errors='replace')
    progress = [line for line in tail.splitlines() if re.search(r'step:|iteration +[0-9]|VERDICT:|Error|Traceback|BENCHMARK_DIAGNOSTICS', line)]
    results[jobid] = dict(label=spec['label'], scheduler=attrs.get('job_state','unknown'), exit_status=attrs.get('Exit_status'), progress=progress[-3:], tail=tail[-2500:], scheduler_error=p.stderr.strip(), log=str(path))
print(json.dumps(results))
""".replace("JOBS_JSON", repr(json.dumps(jobs)))
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "sunspot", "python3 -"],
        input=script, text=True, capture_output=True, timeout=110,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    previous = None
    while True:
        try:
            if args.display:
                state = json.loads(args.state.read_text()) if args.state.exists() else {}
            else:
                jobs = json.loads(args.jobs.read_text())
                state = {"checked_at": time.time(), "jobs": probe(jobs)}
                args.state.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.state.with_suffix(".tmp")
                temporary.write_text(json.dumps(state, indent=2) + "\n")
                temporary.replace(args.state)
            compact = json.dumps(state.get("jobs", {}), sort_keys=True)
            if compact != previous:
                table = Table(title="AGPT replicated comparison " + time.strftime("%H:%M:%S"))
                for heading in ("Job", "Arm", "PBS", "Exit", "Latest evidence"):
                    table.add_column(heading)
                for jobid, job in state.get("jobs", {}).items():
                    table.add_row(jobid, job["label"], job["scheduler"], str(job["exit_status"]), "\n".join(job["progress"]))
                Console().print(table)
                previous = compact
            if args.once:
                return
            if state.get("campaign_terminal"):
                return
            if not args.display and state.get("jobs") and all(
                job["scheduler"] == "F" and job["exit_status"] is not None
                for job in state["jobs"].values()
            ):
                print("ALL_JOBS_TERMINAL: inspect complete logs and artifacts", flush=True)
                return
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            print(f"MONITOR_ERROR {error}", flush=True)
        time.sleep(5 if args.display else 75)


if __name__ == "__main__":
    main()
