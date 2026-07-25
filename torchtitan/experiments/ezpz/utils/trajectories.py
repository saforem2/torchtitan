#!/usr/bin/env python3
"""Single source of truth for every AuroraGPT production trajectory.

Historically the trajectory metadata was duplicated across three
hand-maintained registries that drifted apart:

  - ``plot_production_wandb.py::PRODUCTION_RUNS``  (W&B run-id lists)
  - ``scripts/check_stale_docs.sh::CHAIN_TO_README`` (ckpt-dir -> README)
  - the eval scripts' ``TRAJECTORIES`` / ``V2_TRAJECTORIES`` maps
    (eval-output-subdir + GBS)

They drifted: the 20B 256N chain was relocated to its own clone on
2026-06-12 but ``check_stale_docs.sh`` never learned the new path, and
the ``2b_v2_256`` run-id list fell six dispatches behind. This module
is now the ONE place a trajectory is described; every consumer derives
its view from here so they cannot diverge again.

Each record in ``TRAJECTORIES`` carries:

  key            stable id, also the W&B-view key (e.g. "2b_v2_256")
  model          "2b" | "20b" | "80b"
  version        "v1" | "v2"
  num_nodes      node count of the dispatch
  ckpt_dir       absolute ckpt directory (may live in a sibling clone)
  readme         repo-relative path to the trajectory's README
  gbs            global batch size
  seq_len        sequence length (8192 everywhere today)
  token_target   total-tokens goal for the % column (NOT hardcoded 4.67T)
  wandb_run_ids  ordered list of W&B run-ids (empty for pure smokes)
  olog_fallbacks {run_id: .o-log path} for runs with empty W&B history
  eval_subdir    subdir under outputs/evals/ (or None)
  cls            "live" | "wandb_only" | "smoke" | "placeholder" | "historical"

Derived views (so existing consumers are drop-in):
  PRODUCTION_RUNS         -> exact shape plot_production_wandb.py expects
  chain_to_readme()       -> {ckpt_dir: readme} for existing dirs only
  emit_stale_map_bash()   -> `declare -A CHAIN_TO_README=( ... )` text
  eval_trajectories()     -> records with a non-None eval_subdir

CLI:
  python -m torchtitan.experiments.ezpz.utils.trajectories --emit stale-map
  python -m torchtitan.experiments.ezpz.utils.trajectories --emit json
"""

from __future__ import annotations

from pathlib import Path

# Repo root = five parents up from this file
# (torchtitan/experiments/ezpz/utils/trajectories.py).
REPO_ROOT = Path(__file__).resolve().parents[4]

# Sibling production clones (some trajectories live outside the main repo).
RUNS = Path("/flare/AuroraGPT/foremans/runs")
_2B_V2 = RUNS / "agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints"
_20B_V2 = RUNS / "agpt-20b-v2/torchtitan-ezpz/outputs/checkpoints"
_20B_N256 = RUNS / "agpt-20b-n256/torchtitan-ezpz/outputs/checkpoints"
_80B_V2 = RUNS / "agpt-80b-v2/torchtitan-ezpz/checkpoints"

# 4.67T is the olmo-mix-1124 token budget shared by every current
# production trajectory. Kept as a named constant so a future model with
# a different corpus carries its own target in its record.
OLMO_MIX_1124_TOKENS = 4_673_780_159_710

SEQ_LEN = 8192

# Production-tracking docs live under here (repo-relative paths in records).
_DOCS = "torchtitan/experiments/ezpz/docs/production/agpt"


TRAJECTORIES: list[dict] = [
    # ---- v1 (bf16-tainted, historical reference) ----
    {
        "key": "2b_v1_256",
        "model": "2b",
        "version": "v1",
        "num_nodes": 256,
        "ckpt_dir": None,  # v1 ckpts archived; charts come from W&B only
        "readme": f"{_DOCS}/historical/v1-bf16/README.md",
        "gbs": 3072,
        "seq_len": SEQ_LEN,
        "token_target": OLMO_MIX_1124_TOKENS,
        "wandb_run_ids": [
            "v5ytgu0o", "pjanidnw", "4u9w23p9", "tahlsmy9", "iy1xbv0t",
            "11jzfnno", "hqwaw075", "6ictshbs", "wviyqysc",
        ],
        "olog_fallbacks": None,
        "eval_subdir": None,
        "cls": "historical",
    },
    {
        "key": "20b_v1_256",
        "model": "20b",
        "version": "v1",
        "num_nodes": 256,
        "ckpt_dir": None,
        "readme": f"{_DOCS}/historical/v1-bf16/README.md",
        "gbs": 3072,
        "seq_len": SEQ_LEN,
        "token_target": OLMO_MIX_1124_TOKENS,
        "wandb_run_ids": [
            "q9oq5huj", "pnkaurba", "lrlv3xsc", "pigwfqkg", "lvyzlocg",
            "e2anhgt2", "he01jr7f", "t0ja3dl4",
        ],
        "olog_fallbacks": None,
        "eval_subdir": None,
        "cls": "historical",
    },
    # ---- v2 (current production, fp32-master) ----
    {
        "key": "2b_v2_256",
        "model": "2b",
        "version": "v2",
        "num_nodes": 256,
        "ckpt_dir": str(_2B_V2 / "agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144"),
        "readme": f"{_DOCS}/2b/n256/README.md",
        "gbs": 6144,
        "seq_len": SEQ_LEN,
        "token_target": OLMO_MIX_1124_TOKENS,
        # lytjeegk=8459818 0t4h0kuw=8470100 j7bz39tj=8470101 0qpf3hnc=8481320
        # iekiq5rq=8503506 ni0etxx7=8505118 0fk1bvtt=8505119 3n22a69q=8505175
        # 8vmrcxqr=8505252 56lkkkh1=8507195 24wfvoje=8507198 bs6tay8l=8508020
        # zrqx75x7=8508977 ied4spbx=8513544 yklnyjd5=8516364 8ujhblrp=8519833
        # a52q40kx=8521626 jkde9zdg=8521630(74301->80402) zcmlqbd8=8534293(->step86259)
        #   [jkde9zdg CORRECTED 2026-07-24 from okyt09kv: okyt09kv was the
        #    preflight-smoke run-id (0 rows in torchtitan.ezpz.train), which left
        #    a 74319->80301 gap the plotter bridged with a straight line; the
        #    real training run for job 8521630 is jkde9zdg -- same swallow as the
        #    completion trio below.]
        # --- chain COMPLETION (86201->92859), added 2026-07-06, run-ids
        #     CORRECTED 2026-07-24 (see note below): ---
        # 9itxu3pt=8572612(sneak2h,86201-86674) ew4pqb51=8573619(sneak2h,86675-87145)
        # fm3gzdxt=8558531(cont12,87126->92859 DONE)
        "wandb_run_ids": [
            "lytjeegk", "0t4h0kuw", "j7bz39tj", "0qpf3hnc", "iekiq5rq",
            "ni0etxx7", "0fk1bvtt", "3n22a69q", "8vmrcxqr",
            "56lkkkh1", "24wfvoje", "bs6tay8l",
            "zrqx75x7", "ied4spbx", "yklnyjd5",
            "8ujhblrp", "a52q40kx", "jkde9zdg", "zcmlqbd8",
            "9itxu3pt", "ew4pqb51", "fm3gzdxt",
        ],
        # NOTE (2026-07-24): these 3 completion jobs each run an
        # `ezpz.examples.test` preflight smoke FIRST, which opens a wandb run in
        # the `aurora_gpt/ezpz.examples.test` project. With reinit='default',
        # the real training `wandb.init(project=torchtitan.ezpz.train)` returns
        # that already-active smoke run instead of a new one -- so the smoke
        # run-ids (a5h4aaf7/gx7ph91w/mqi69lx2) hold only the 5-step toy curve.
        # The REAL training tail (86201->92859) is logged to the correct
        # project under 9itxu3pt/ew4pqb51/fm3gzdxt (used above); no .o-log
        # fallback is needed.
        "olog_fallbacks": None,
        "eval_subdir": "agpt-2b-v2-256n",
        "cls": "live",
    },
    {
        "key": "2b_v2_512",
        "model": "2b",
        "version": "v2",
        "num_nodes": 512,
        "ckpt_dir": str(_2B_V2 / "agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288"),
        "readme": f"{_DOCS}/2b/n512/README.md",
        "gbs": 12288,
        "seq_len": SEQ_LEN,
        "token_target": OLMO_MIX_1124_TOKENS,
        # i252kps9=8460301 d4hlr8qe=8463626 1va7zfki=8463627 6op7ozfh=8466847
        # y70rh76h=8479989 logai2xn=8485509 2qqhpcrm=8485511 w78n1akt=8506221
        # i0ayskft=8507196 21grc6o7=8507199 nv4qwxc8=8508753
        #   (i0ayskft W&B history was empty when first recorded 2026-07-06 and
        #   used an .o-log fallback; the run synced later -- 4389 rows,
        #   steps 16601->20988 -- so the fallback was dropped 2026-07-24.)
        "wandb_run_ids": [
            "i252kps9", "d4hlr8qe", "1va7zfki", "6op7ozfh",
            "y70rh76h", "logai2xn", "2qqhpcrm", "w78n1akt",
            "i0ayskft", "21grc6o7", "nv4qwxc8",
        ],
        "olog_fallbacks": None,
        "eval_subdir": "agpt-2b-v2-512n",
        "cls": "live",
    },
    {
        "key": "20b_v2_512",
        "model": "20b",
        "version": "v2",
        "num_nodes": 512,
        "ckpt_dir": str(_20B_V2 / "agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288"),
        "readme": f"{_DOCS}/20b/n512/README.md",
        "gbs": 12288,
        "seq_len": SEQ_LEN,
        "token_target": OLMO_MIX_1124_TOKENS,
        # 9tsyx5us=8460302 ej3zy5cq=8463628 s6b159xk=8479579 gkzl19dg=8481645
        # 10vf1mqr=8481647 wjy5pvxm=8505258 cv3wii8x=8505259
        #   [qttj3l3p=8505124 DROPPED 2026-07-24: crashed dud, 0 steps in W&B and
        #    no step lines in its .o log; its 801->1413 range is fully covered by
        #    the retry wjy5pvxm=8505258. Not a swallowed-smoke -- nothing to
        #    recover, so the empty run-id is removed rather than remapped.]
        # tu77pzu7=8507197 8vixdfg2=8507200 0pmsn01c=8509393(->step4418)
        # tu1iseu1=8638793 native-autoretry relaunch (4401->5109), added 2026-07-06
        # 8o2xakm3=8638795 cont (5109->6000), added 2026-07-23
        # --- 128N debug-scaling sneaks (GAS=4 -> GBS=12288, bit-identical to
        #     the 512N chain) run to unstick the queue-starved chain, 2026-07-23:
        # g59v83go=8688010 (6001->6008, walltime-killed pre-save)
        # jyq4w87d=8689162 (6001->6012, saved step-6005/6010)
        "wandb_run_ids": [
            "9tsyx5us", "ej3zy5cq", "s6b159xk", "gkzl19dg", "10vf1mqr",
            "wjy5pvxm", "cv3wii8x",
            "tu77pzu7", "8vixdfg2", "0pmsn01c",
            "tu1iseu1", "8o2xakm3", "g59v83go", "jyq4w87d",
        ],
        "olog_fallbacks": None,
        "eval_subdir": "agpt-20b-v2-512n",
        "cls": "live",
    },
    {
        "key": "20b_v2_256",
        "model": "20b",
        "version": "v2",
        "num_nodes": 256,
        # RELOCATED 2026-06-12 from agpt-20b-v2 to its own clone.
        "ckpt_dir": str(_20B_N256 / "agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144"),
        "readme": f"{_DOCS}/20b/n256/README.md",
        "gbs": 6144,
        "seq_len": SEQ_LEN,
        "token_target": OLMO_MIX_1124_TOKENS,
        # r1yyxbmt=8463659 72airpph=8470102 m9c5wx2e=8470103 6eocrnxs=8479581
        # 5481v99b=8479582 yrq1s1ac=8481646 xt03uvp6=8481648 f1p8nyxh=8505122
        # g6ekeu4j=8505123(->~step500)
        # --- gap 500->3136 filled 2026-07-06: ---
        # kk4h0i7m=8505255(301-1125, in torchtitan.ezpz.train)
        # 17sfemjj=8558548(1101-2107) rugscgjs=8558549(2101->3135)
        #   run-ids CORRECTED 2026-07-24: the .o-log-noted dpiog1q7/auy8wohg
        #   were the ezpz.examples.test PREFLIGHT-SMOKE run-ids (reinit-swallow,
        #   see the 2b_v2_256 note); the real training runs are in
        #   torchtitan.ezpz.train under 17sfemjj/rugscgjs (no fallback needed).
        # --- 2026-07 continuations, added 2026-07-24: ---
        # 6yr6ivh4=8647385(3101->3603) uvgmafv9=8661054(4201->4375)
        # 5rvusq43=8681340(5101->5860+, live full-throughput resume)
        "wandb_run_ids": [
            "r1yyxbmt", "72airpph", "m9c5wx2e", "6eocrnxs",
            "5481v99b", "yrq1s1ac", "xt03uvp6",
            "f1p8nyxh", "g6ekeu4j",
            "kk4h0i7m", "17sfemjj", "rugscgjs",
            "6yr6ivh4", "uvgmafv9", "5rvusq43",
        ],
        "olog_fallbacks": None,
        "eval_subdir": None,
        "cls": "live",
    },
    {
        "key": "2b_v2_512_lr3.22e-5",
        "model": "2b",
        "version": "v2",
        "num_nodes": 512,
        # sqrt(2)-LR fork ckpt dir was never persisted to disk; W&B only.
        "ckpt_dir": None,
        "readme": f"{_DOCS}/2b/n512/README.md",
        "gbs": 12288,
        "seq_len": SEQ_LEN,
        "token_target": OLMO_MIX_1124_TOKENS,
        # 8edrii5e=8467141 oujzdxri=8467142 (sqrt(2)-LR fork)
        "wandb_run_ids": ["8edrii5e", "oujzdxri"],
        "olog_fallbacks": None,
        "eval_subdir": None,
        "cls": "wandb_only",
    },
    {
        "key": "80b_v2_4_smoke",
        "model": "80b",
        "version": "v2",
        "num_nodes": 4,
        "ckpt_dir": str(_80B_V2 / "agpt-80b-adamw-olmo-mix-1124-n4-gbs24"),
        "readme": f"{_DOCS}/80b/n4/README.md",
        "gbs": 24,
        "seq_len": SEQ_LEN,
        "token_target": OLMO_MIX_1124_TOKENS,
        # 4N end-to-end validation smoke (<=20 steps); not a production
        # chain. No W&B run-list tracked here; the page is hand-narrated.
        "wandb_run_ids": [],
        "olog_fallbacks": None,
        "eval_subdir": None,
        "cls": "smoke",
    },
]

# Note: the 80B smoke is intentionally EXCLUDED from PRODUCTION_RUNS
# (it has no run-id list and is not a chart trajectory). The derived
# view below filters to records that carry W&B run-ids.


def by_key(key: str) -> dict:
    for t in TRAJECTORIES:
        if t["key"] == key:
            return t
    raise KeyError(f"no trajectory with key {key!r}")


def _production_runs_view() -> dict[str, dict]:
    """Reconstruct the legacy ``PRODUCTION_RUNS`` dict EXACTLY.

    Shape per key: {"run_ids", "num_nodes", "model"} plus an optional
    "olog_fallbacks". Key order and value order match the original
    literal so the migration is byte-for-byte (asserted in tests).
    """
    out: dict[str, dict] = {}
    for t in TRAJECTORIES:
        # Records without W&B run-ids (pure smokes) are not chart
        # trajectories and were never in the legacy PRODUCTION_RUNS.
        if not t["wandb_run_ids"]:
            continue
        entry: dict = {
            "run_ids": list(t["wandb_run_ids"]),
            "num_nodes": t["num_nodes"],
            "model": t["model"],
        }
        if t.get("olog_fallbacks"):
            entry["olog_fallbacks"] = dict(t["olog_fallbacks"])
        out[t["key"]] = entry
    return out


# Drop-in replacement for the old literal. plot_production_wandb.py does
# `from ...trajectories import PRODUCTION_RUNS`.
PRODUCTION_RUNS: dict[str, dict] = _production_runs_view()


def chain_to_readme(existing_only: bool = True) -> dict[str, str]:
    """{ckpt_dir: readme_repo_rel} for trajectories that have a ckpt_dir.

    With ``existing_only`` (default) only directories present on disk are
    returned, matching check_stale_docs.sh's `[[ -d ]]` guard but now
    pointed at the correct (post-relocation) paths.
    """
    out: dict[str, str] = {}
    for t in TRAJECTORIES:
        ckpt = t.get("ckpt_dir")
        if not ckpt:
            continue
        if existing_only and not Path(ckpt).is_dir():
            continue
        out[ckpt] = t["readme"]
    return out


def emit_stale_map_bash() -> str:
    """Emit a bash `declare -A CHAIN_TO_README=( ... )` block for
    check_stale_docs.sh to `eval`. Only on-disk dirs are emitted.
    """
    lines = ["declare -A CHAIN_TO_README=("]
    for ckpt, readme in chain_to_readme(existing_only=True).items():
        lines.append(f'    ["{ckpt}"]="{readme}"')
    lines.append(")")
    return "\n".join(lines)


def eval_trajectories() -> list[dict]:
    """Records that have an eval-output subdir, for the eval plotters."""
    return [t for t in TRAJECTORIES if t.get("eval_subdir")]


def live_trajectories() -> list[dict]:
    """Records whose ckpt dir is on disk and should have disk fields
    auto-filled (class 'live')."""
    return [
        t for t in TRAJECTORIES
        if t["cls"] == "live" and t.get("ckpt_dir") and Path(t["ckpt_dir"]).is_dir()
    ]


def _main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Trajectory manifest emitter")
    ap.add_argument(
        "--emit",
        choices=["stale-map", "json", "keys"],
        default="keys",
        help="stale-map: bash CHAIN_TO_README; json: full records; keys: one key per line",
    )
    args = ap.parse_args()

    if args.emit == "stale-map":
        print(emit_stale_map_bash())
    elif args.emit == "json":
        print(json.dumps(TRAJECTORIES, indent=2))
    else:
        for t in TRAJECTORIES:
            print(t["key"])
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
