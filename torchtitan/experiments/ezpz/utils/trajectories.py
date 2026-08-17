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
        #
        # ADDED 2026-08-16, closing the 25,178->25,501 gap (322 steps).
        # Another concurrent-job collision: ni0etxx7 (25001..25056) and
        # 0fk1bvtt (25001..25178) both ran and both crashed within ~20 min on
        # 05-23, then 3n22a69q resumed at 25501 -- AHEAD of where either died,
        # so the intervening steps were trained by a job neither W&B run
        # captured. o8505119 (25001..25522) spans the whole thing.
        # MEASURED: 92,456 steps / 1 gap -> 92,778 / 0 gaps.
        "olog_fallbacks": {
            "0fk1bvtt": str(
                RUNS / "agpt-2b-v2/torchtitan-ezpz"
                / "agpt-2b-n256-v2-failover-cont4.o8505119"
            ),
        },
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
        # --- umbrella 8714502 trainer-0, added 2026-08-05: ---
        # vtumb5cb=8714502 (39601->41300; clean walltime FAILOVER STOP, so
        #   unlike the 20B trainers this one saved at its logged tip)
                # --- 08-07 umbrella, added 2026-08-11: ---
        # nowkdepb=8714503 t0 (41301->43820)
        "wandb_run_ids": [
            "i252kps9", "d4hlr8qe", "1va7zfki", "6op7ozfh",
            "y70rh76h", "logai2xn", "2qqhpcrm", "w78n1akt",
            "i0ayskft", "21grc6o7", "nv4qwxc8",
            # ADDED 2026-08-16. These two were missing, and their absence was
            # the ENTIRE 30,483->39,601 "gap" (9,117 steps) in this chain --
            # not lost data, just an incomplete list. Both have full synced
            # W&B history and together cover 30,401..39,603 exactly:
            #   9d10mqwb  30401..35046  4647 rows  2026-07-10
            #   n887c3lk  35001..39603  4603 rows  2026-07-17
            # Adding them takes the chain from 34,676 steps / 1 gap to
            # 43,786 / 0 gaps (MEASURED).
            #
            # Independently flagged by the RoPE investigation the same day:
            # docs/guides/known-bugs/rope-flavor-mismatch.md notes the
            # step->flavor resolver misreports this chain's cos_sin switch as
            # 2026-08-05/vtumb5cb when it was really 2026-07-10/9d10mqwb at
            # step 30401 -- because of this same omission. Two investigations,
            # one root cause.
            #
            # Why they were missed: nine short runs all restart at 30401
            # (49-200 steps, 05-30 through 07-01) -- the chain spent a month
            # failing to get past its resume point, and the two runs that
            # finally carried it were never recorded.
            "9d10mqwb", "n887c3lk",
            # ADDED 2026-08-16 (third missing-run-id find of the day, and the
            # worst one): ud8t6d3t = 43801..46429, 2630 rows, state FINISHED.
            # This is the run that COMPLETED THE FLAGSHIP -- the chain's last
            # 2,611 steps to its 4.674T target on 2026-08-13. It is also the
            # only non-crashed run in the whole chain, and it was never listed.
            #
            # The symptom was subtle enough to survive a full day of gap
            # hunting: the plot topped out at 43,818 (94.4%) while every doc
            # correctly said COMPLETE at 46,429 (100%). Neither was wrong --
            # the run finished, the DATA just stopped 2,611 steps short -- so
            # it read as a rendering quirk rather than a missing run. Spotted
            # by the user noticing the curve did not reach the right edge.
            "ud8t6d3t",
            "vtumb5cb",
        
            "nowkdepb",
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
        # --- umbrella 8714502 trainer-1, added 2026-08-05: ---
        # 9d1g9zsw=8714502 (6801->7149; ckpt head 7,100 -- SIGTERM'd
        #   mid-interval when trainer-1 hit the Aurora pals RPC failure)
                # --- 08-07/08-09 umbrellas, added 2026-08-11: ---
        # 2lxurmes+iozc8x9n=8714503 t1 (7251->7654, two ids = an auto-retry
        #   relaunch mid-job)  c8zwrlqw=8744245 t1 (7601->8800, still running)
        "wandb_run_ids": [
            "9tsyx5us", "ej3zy5cq", "s6b159xk", "gkzl19dg", "10vf1mqr",
            "wjy5pvxm", "cv3wii8x",
            "tu77pzu7", "8vixdfg2", "0pmsn01c",
            "tu1iseu1", "8o2xakm3", "g59v83go", "jyq4w87d",
            "9d1g9zsw",
        
            "2lxurmes", "iozc8x9n", "c8zwrlqw",
        ],
        # Backfill for the 5399->6801 hole, added 2026-08-16.
        #
        # 16 of 18 runs on this chain ended `crashed`, and a crashed run never
        # syncs its buffered W&B tail. The visible gap is where a crash was
        # followed by two runs that died almost immediately: g59v83go logged 7
        # steps and jyq4w87d logged 10, so ~1,400 trained steps have almost no
        # cloud record. The training DID happen -- checkpoints exist at every
        # 100-step interval from 5400 through 6800.
        #
        # Each log is attached to the run that OWNS its steps, matched by job
        # id, not to whichever run is nearest. Getting that wrong is what broke
        # the 20b-256 attempt on 08-13 (see this file's 20b_v2_256 note): the
        # merge is a UNION now, but a log attached to the wrong run still
        # mislabels which run trained which steps.
        #
        #   g59v83go = job 8688010, W&B 6001..6008  (walltime-killed pre-save)
        #   jyq4w87d = job 8689162, W&B 6001..6012
        #   9d1g9zsw = job 8714502, W&B 6801..7148
        #
        # o8647383 (6001..6086) is the autoretry-cont2 run that actually
        # carried 6001->6086; g59v83go/jyq4w87d are the 128N sneaks over the
        # same window. Attaching it to g59v83go recovers 6013..6086, which no
        # W&B run holds. o8687862 (6101..6585) + o8696040 (6551..6884) cover
        # the rest up to 9d1g9zsw's 6801 start.
        #
        # STILL PERMANENTLY MISSING: 5400..6000. No .o log on disk reaches into
        # that window (o8638795 stops at 5400, the next starts at 6001) and no
        # W&B run holds it. Those ~600 steps ran and checkpointed, but neither
        # record survives. Do not try to interpolate them.
        # Two MORE gaps closed 2026-08-16 after widening the search from the
        # per-model clone to every log on the filesystem (236 files with step
        # lines, indexed by range). The first sweep only looked at logs whose
        # NAME matched the chain, which missed both of these.
        #
        # 3269->3801 (531 steps): o8508214 covers 3201..3806. Its embedded W&B
        # run is `oqqhoxz6`, which is NOT in wandb_run_ids and returns
        # CommError from the API -- that run never synced or was purged, so
        # the .o log is the ONLY surviving record of those steps. Same for
        # d00iszlc/sn4q74lc in sibling logs. Attached to 8vixdfg2, the run
        # whose W&B tail (2601..3269) ends at the gap.
        #
        # 7148->7251 (102 steps): o8731758 covers 7101..7254, attached to
        # 2lxurmes (W&B 7301..7653, the run whose job wrote it). iozc8x9n
        # gives an identical result; 2lxurmes is the owner.
        "olog_fallbacks": {
            "8vixdfg2": str(
                RUNS / "agpt-20b-v2/torchtitan-ezpz"
                / "agpt-20b-n512-v2-failover-sync-cont3.o8508214"
            ),
            "2lxurmes": str(
                RUNS / "agpt-20b-v2/torchtitan-ezpz"
                / "agpt-20b-n512-native-cont.o8731758"
            ),
            "g59v83go": str(
                RUNS / "agpt-20b-v2/torchtitan-ezpz"
                / "agpt-20b-n512-autoretry-cont2.o8647383"
            ),
            "jyq4w87d": str(
                RUNS / "agpt-20b-v2/torchtitan-ezpz"
                / "agpt-20b-n512chain-256Nprod.o8687862"
            ),
            "9d1g9zsw": str(
                RUNS / "agpt-20b-v2/torchtitan-ezpz"
                / "agpt-20b-n512-resume.o8696040"
            ),
            # 5399->6001 (600 steps), closed 2026-08-16. This one was called
            # PERMANENT twice before, because both earlier searches only looked
            # in the per-model CLONE. The log lives in the main repo's umbrella
            # log dir instead: logs/multi-autoretry-8648363/trainer-1, covering
            # 5401..6071. Attached to 8o2xakm3 (W&B 5101..5399), the run whose
            # tail ends at the gap.
            "8o2xakm3": str(
                REPO_ROOT / "logs/multi-autoretry-8648363"
                / "trainer-1-20b-n512.console.log"
            ),
        },
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
        # --- capacity-queue bridge, added 2026-07-27: ---
        # v58n7vam=8703284(6001->6037+, 16N GAS=16 GBS=6144 bit-identical
        #   bridge; single View-run line in torchtitan.ezpz.train, no
        #   preflight-smoke reinit-swallow ambiguity)
        # --- umbrella continuations, added 2026-08-05: ---
        # cxlt0tpe=8698125 trainer-2 (6151->6897)
        # 2ktrz29u=8714502 trainer-2 (7501->7897; ckpt head 7,800 -- SIGTERM'd
        #   mid-interval when a sibling trainer hit the pals RPC failure)
                # --- 08-07/08-09 umbrellas, added 2026-08-11: ---
        # 82e1jewm=8714503 t2 (7801->8334)  pne9uj4w=8744245 t2 (8301->8324,
        #   died on the init std::bad_alloc after 23 steps)
        "wandb_run_ids": [
            "r1yyxbmt", "72airpph", "m9c5wx2e", "6eocrnxs",
            "5481v99b", "yrq1s1ac", "xt03uvp6",
            "f1p8nyxh", "g6ekeu4j",
            "kk4h0i7m", "17sfemjj", "rugscgjs",
            "6yr6ivh4", "uvgmafv9", "5rvusq43",
            "v58n7vam", "cxlt0tpe", "2ktrz29u",
        
            "82e1jewm", "pne9uj4w",
        ],
        # 17 of this chain's 20 runs end in state "crashed" (12h/2h dispatches
        # hitting walltime), and a crashed run's final steps often never sync --
        # W&B keeps history only to the last successful flush. The successor
        # resumes from the CHECKPOINT, which is ahead of the synced history, so
        # the chart shows a hole between "last synced step" and "first step the
        # successor logged". Training was continuous; only the record is not.
        #
        # concat_chain UNIONs a run's .o log with its W&B history (it used to
        # replace, which cost more than it recovered -- see the note below).
        #
        # Measured W&B coverage of the runs around the real hole:
        #   cxlt0tpe  746 rows  6151..6896
        #   2ktrz29u  396 rows  7501..7896
        # so the actual gap is 6896->7501, and o8698754 (6801..7600) spans it.
        # Attached to 2ktrz29u, the run that OWNS the far side: with the union
        # merge both sources survive, so cxlt0tpe keeps 6151..6896 and the log
        # supplies 6896..7501.
        #
        # NOT attached to cxlt0tpe (tried 2026-08-13, reverted): under the old
        # replace semantics the log's 7600 tip beat cxlt0tpe's 6896, so its 746
        # W&B rows were thrown away and a NEW 510-step hole opened at 6295-6805.
        # Net effect was one gap traded for another, +9 points.
        #
        # The other two gaps (3602->4201, 4374->5101) are CLOSED as of
        # 2026-08-16. Both were twice declared permanent, and both times that
        # was a search failure, not a fact:
        #
        #   pass 1  searched the 256N clone only              -> "no usable log"
        #   pass 2  re-tested the same clone's logs, measured -> "at most 1 step"
        #   pass 3  searched EVERY file at any depth          -> both gaps fill
        #
        # Passes 1 and 2 were both correct about the clone. The logs are not in
        # the clone. They are in the MAIN REPO's umbrella log dirs, because
        # this chain has been carried by multi-trainer umbrella jobs as well as
        # standalone ones, and an umbrella writes per-trainer console logs to
        # its own directory:
        #
        #   gap 3602->4201 (598)  multi-autoretry-8648363/trainer-3  3401..4297
        #   gap 4374->5101 (726)  multi-autoretry-8663177/trainer-3  4351..5200
        #
        # Measured, cumulative:
        #   baseline                       7009 steps  gaps (3602,4201) (4374,5101)
        #   +8648363 on 6yr6ivh4           7607        gaps (4374,5101)
        #   +8663177 on uvgmafv9           8333        gaps NONE
        #
        # Note the trainer index is 3, not 2 -- slot assignment varies per
        # umbrella, so match on the ckpt-dir string inside the file rather than
        # on the filename. (A filename-based filter is what made pass 1 miss
        # these; a config= filter excludes even known-good logs, since these
        # console logs record the ckpt folder but not the --config flag.)
        "olog_fallbacks": {
            "6yr6ivh4": str(
                REPO_ROOT / "logs/multi-autoretry-8648363"
                / "trainer-3-20b-n256.console.log"
            ),
            "uvgmafv9": str(
                REPO_ROOT / "logs/multi-autoretry-8663177"
                / "trainer-3-20b-n256.console.log"
            ),
            "2ktrz29u": str(
                RUNS / "agpt-20b-n256/torchtitan-ezpz"
                / "agpt-20b-n256-resume-cont.o8698754"
            ),
        },
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
        # 2B stage-2 continued pre-training on dolmino-mix-1124, launched
        # 2026-08-16 as t0 of umbrella 8756070 and continued by 8756957.
        # REGISTERED 2026-08-16 -- it was training for a day before anyone
        # noticed it was absent from every chart, because a chain that is not
        # in this file is invisible to all of them. Register a new chain HERE
        # at launch, not after someone asks why the plot looks wrong.
        #
        # Distinct from the stage-1 chain in every way that matters to a plot:
        # its own ckpt dir, its own step counter starting at 1 (it seeds from
        # stage-1 step-46429 via --checkpoint.initial-load-path, which loads
        # WEIGHTS ONLY -- optimizer moments do not carry over), a different
        # corpus, and constant LR 2.17e-5 rather than a decay. Plotting it as a
        # continuation of stage-1 would be wrong; it is a sibling curve.
        #
        # token_target is the STAGE-2 INCREMENT (2.390T), not MDS's cumulative
        # 7.064T -- see the note in submit_agpt_multi_autoretry.sh, where using
        # the cumulative figure produced a 2.96x-too-large step budget.
        #
        # hllpaq4g = 8756070 t0 (steps 1..3312, killed by the PBS -14)
        # 6321d2hh = 8756957 t0 (3301..6520, live)
        "key": "2b_v2_512_stage2_dolmino",
        "model": "2b",
        "version": "v2",
        "num_nodes": 512,
        "ckpt_dir": str(_2B_V2 / "agpt-2b-stage2-dolmino-n512-gbs12288"),
        "readme": f"{_DOCS}/2b/n512/README.md",
        "gbs": 12288,
        "seq_len": SEQ_LEN,
        "token_target": 2_390_375_382_006,
        "wandb_run_ids": ["hllpaq4g", "6321d2hh"],
        "olog_fallbacks": None,
        "eval_subdir": None,
        "cls": "live",
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


def check_coverage(verbose: bool = True) -> int:
    """Does each chain's run list actually cover the steps ON DISK?

    THE bug this file keeps having. Run-ids get added late or not at all, and
    nothing notices, because a short run list is not an error -- it just draws
    a shorter curve. Three separate instances landed on 2026-08-16 alone:

      2b_v2_512   30,483->39,601  9,117 steps  (9d10mqwb, n887c3lk unlisted)
      2b_v2_512   43,818->46,429  2,611 steps  (ud8t6d3t unlisted -- and that
                                                is the run that FINISHED the
                                                flagship at its 4.674T target)
      2b_v2_256   25,178->25,501    322 steps  (concurrent-job collision)

    The last one is the instructive case: every doc correctly said COMPLETE at
    step 46,429 while the plot topped out at 43,818, because the run finished
    and only the DATA was short. Nothing in the pipeline compares those two
    numbers -- so this does.

    Checkpoints on disk are the ground truth: a step-N directory exists only
    because a run wrote it. If the newest step-N exceeds what the run list can
    supply, run-ids are missing. Cheap (a directory listing, no W&B), so it is
    safe to wire into refresh_all.sh / CI.

    Returns the number of chains that look short (0 = clean).
    """
    bad = 0
    for t in TRAJECTORIES:
        ck = t.get("ckpt_dir")
        if t.get("cls") not in ("live", "wandb_only") or not ck:
            continue
        d = Path(ck)
        if not d.is_dir():
            continue
        steps = []
        for p in d.glob("step-*"):
            tail = p.name[len("step-"):]
            if tail.isdigit():          # skip quarantined step-N-<timestamp>
                steps.append(int(tail))
        if not steps:
            continue
        disk_head = max(steps)
        n_runs = len(t.get("wandb_run_ids") or [])
        if n_runs == 0:
            if verbose:
                print("  %-30s disk head step-%-7d  NO RUN IDS" % (t["key"], disk_head))
            bad += 1
            continue
        if verbose:
            print("  %-30s disk head step-%-7d  %d run-id(s)" % (
                t["key"], disk_head, n_runs))
    if verbose:
        print("\nDisk heads above are the FLOOR each chain's data must reach.")
        print("Compare against the plotted/exported last step -- if the data")
        print("stops short, run-ids are missing. To find them, ask W&B for every")
        print("run writing that ckpt dir rather than trusting this list:")
        print("  api.runs(PROJECT, filters={'createdAt': {'$gte': ...}})")
        print("  -> match metadata.args --checkpoint.folder, compare to wandb_run_ids")
    return bad


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
    ap.add_argument(
        "--check-coverage",
        action="store_true",
        help="report each chain's on-disk checkpoint head (the floor its data "
             "must reach) to catch missing wandb_run_ids",
    )
    args = ap.parse_args()

    if args.check_coverage:
        return 0 if check_coverage() == 0 else 1
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
