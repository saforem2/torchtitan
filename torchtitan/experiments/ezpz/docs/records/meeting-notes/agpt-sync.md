# AuroraGPT Sync — Meeting Notes

> Sam Foreman. Most recent first.

---

## 2026-08-24

> [!IMPORTANT]
> **Headline:** **the 2B-512 flagship reached its 4.674T target and finished**
> -- 46,429/46,429, loss 2.687 -- and its **stage-2 dolmino continuation is
> already 32.5% done** (7,728 steps, 0.778T of 2.390T) from a standing start
> this window. Both 20B chains advanced (**256N 8,300 -> 10,369**, +2,069;
> **512N 8,300 -> 9,690**, +1,390) and the constant-LR fork moved **16,984 ->
> 21,307** (+4,323). That is **~1.66T tokens added across five chains in two
> weeks**, the most productive window this year, and it came from the umbrella
> launcher finally converting most of its seats: it went from 1-3 of 5 to
> **four chains training simultaneously**, with the seat-level failure modes
> that caused the waste now diagnosed and fixed rather than merely counted.
> Alongside production, the 30B side experiment ran its config to completion
> (2000/2000, 12.028 -> 2.115, zero NaN) -- a useful convergence answer, not
> this window's main effort.

Covers the two weeks since 2026-08-10. **Production pre-training is the
headline and leads below.** The 2B flagship completing is the milestone; the
umbrella becoming reliable is what made the rest of the window's tokens
possible. Then the hardening work that came out of the same runs (a NaN-gradient
hole, blind Polaris failover, two seats on the wrong RoPE flavor), and the 30B
convergence experiment.

**Current status note:** everything is idle as of this writing, but that is
recent and external. A failed prologue on another user's job offlined ~1,025
nodes late last week, and a full-system reservation (`M8769283`, Mon 14:00 ->
Tue 00:30 UTC) now blocks both queued jobs -- PBS will not start a job that
cannot finish before a reservation, so the umbrella's 12 h ask and the 30B's
6 h ask both missed their `14:00 - walltime` deadlines. Eligibility resumes
Tue 00:30.

### 1. Production pre-training -- the flagship landed, and four seats ran at once

| chain | 08-10 | now | delta | loss | tokens |
|---|---|---|---|---|---|
| **2B 512N** | 43,800 | **46,429 DONE** | +2,629 | **2.687** | **4.674T (100%)** |
| **2B 512N stage-2 dolmino** | -- | **7,728** | **+7,728** (new) | 2.518 | 0.778T (32.5%) |
| **20B 256N** | 8,300 | **10,369** | +2,069 | 2.365 | 0.261T |
| **20B 512N** | 8,300 | **9,690** | +1,390 | 2.407 | 0.975T |
| **2B 512N constlr-from9200** | 16,984 | **21,307** | +4,323 | 2.745 | 2.145T |

**~1.66T tokens added across five chains.** Three things stand out.

**The flagship is done.** 2B-512 reached 46,429/46,429 on 2026-08-13 under
umbrella `8744247`, which was also the first umbrella to run nearly its whole
allocation (23h18m of 24h, 97%). The chain that had been the pacing item since
May is now a finished artifact at its full 4.674T budget.

**Stage 2 started and is a third done.** The dolmino continued-pretrain seeded
from the finished step-46429 and reached 7,728 steps across two umbrellas -- an
entirely new chain that did not exist at the last sync, now further along than
either 20B chain is on its own target.

**The constant-LR fork was the quiet mover** at +4,323 steps, more than either
20B chain, and is at 45.9% of its budget.

### 2. The umbrella went from wasting allocation to being the delivery mechanism

At 08-10 this section read "the umbrella is wasting most of its allocation"
with 3/5, 3/5, 1/5 productive slots. That has changed materially, and it is the
reason section 1 has numbers in it.

- **`8744247` (08-13): 23h18m / 24h, 97% of allocation.** Carried 2B-512 to its
  target, 20B-256 to 9,850, and the constlr fork to 16,984.
- **`8756957` (08-16): 12h00m / 12h, the first umbrella to use 100%.** The 12 h
  ask appears both easier to schedule and easier to survive than the 24 h
  dispatches that kept dying young -- worth making the default.
- **`8764675` (08-20): a full 12 h clean walltime finish**, `Exit_status=-29`,
  carrying 20B-512 to 10,699 and 20B-256 to 11,800. During this run `prod_dash`
  showed **four chains training simultaneously** -- and exposed a dashboard bug
  in doing so, since `CKPT_RE.search()` took only the first of the five ckpt
  dirs the shared `.o` names, so it had been reporting exactly one live seat.

The seat-level failures are now diagnosed rather than tallied:

| cause | status |
|---|---|
| `output.weight` -> `lm_head.weight` rename (t4, 3rd occurrence) | **FIXED** `e1320edf9` -- shims compose; step-9500 predates both renames |
| pre-#3623 nested->flat optimizer state | **converted** -- step-9500 round-trips, 1110 subkeys / 111 params |
| idle watchdog killing a healthy 20B-512 mid-DCP-load | **FIXED** -- `IDLE_TIMEOUT=5400`; 6,144 shards load silently for >30 min |
| `Config.job` AttributeError in the constlr clone | **FIXED** in the clone 08-16 |
| `std::bad_alloc` at init | **OPEN** -- fresh 512N reproduction; t1 ran fine at the same 522 nodes in the same job, which argues against pure rank count |

The remaining waste is per-seat, not per-job: seats that die at init never hold
a slot productively, while the surviving seats now run the full window. See the
[dispatch log](../../live/dispatch-log.md) for the per-slot table.

**One process lesson worth repeating:** PBS snapshots the submit script at
`qsub`. The 20B constant-LR fix (`b08fccfd1`, Aug 19 12:23) never reached job
`8764675` (queued Aug 18 20:03), which then trained a full 12 h cycle on the
decaying config. Held successors are submitted alongside their predecessor, so
they are pre-fix too and had to be replaced. **Check `qstat -f <id> | grep
ctime` against the fix commit date whenever a script fix lands with jobs already
queued.**

### 3. Side experiment: the 30B ran its config out -- 2000/2000, zero NaN

An exploratory Sunspot experiment, not a production chain -- it exists to
answer whether a 30B config converges and resumes at all before anyone proposes
scheduling one. It now has that answer. Four jobs, one continuous trajectory,
ending in an actual completion rather than a timeout (`rc=0`, 2 h 58 of a 6 h
allocation):

| job | steps | loss |
|---|---|---|
| 12473304 | 1 -> 482 | 12.028 -> 3.357 |
| 12473476 | 401 -> 871 | -> 2.617 |
| 12473515 | 801 -> 1781 | -> 2.246 |
| 12473545 | 1751 -> 2000 | -> **2.115** |

Steady state to the last step: 496 tps, **28.3% MFU**, memory flat at 59.84%,
grad_norm falling 0.26 -> 0.074. Nine checkpoints, 2.6 T. Zero NaN/inf anywhere
in the four jobs. That closes all three questions exp08 was opened for -- loss
descends, grad_norm stays bounded, checkpoints round-trip -- and **the last two
resumes were compiled**, retiring the "compiled resume is broken at 30B"
caveat (the real culprit was `full_dtensor`, not resume).

Details: [`exp08-convergence.md`](../proposals/30b-exp/exp08-convergence.md).

### 4. Production had no defense against a NaN gradient

Found while reading the 80th upstream sync, not from a failure.

**The guard ran after the optimizer step, and only looked at loss.** Guard at
`trainer.py:1085`, step at `:871` -- so a NaN gradient was written into the
weights before anything noticed. Our own 80B report is the proof:
`grad_norm nan @30, loss nan @31, nan-abort @35`. grad_norm went bad a full
step **before** the loss, so the earliest available signal was one ahead of the
one we watched, and five more updates landed on top.

Worse than "off by default": production omits `--nan-abort-consecutive`
entirely, because the pinned pre-#3623 clones have no such field and passing it
crashes every rank.

**Scope, stated carefully:** no canonical chain was observed doing this. All
five seats of umbrella 8764675 log zero non-finite loss/grad_norm lines. The
two cited jobs are experiments. The defect is that nothing *would* have caught
it, not that it happened in production.

Fixed with a host-side `isfinite` check on grad_norm before the step:
non-finite -> zero_grad, log, skip the step, still advance the LR scheduler. No
new collective, since grad_norm is already rank-reduced inside
`clip_grad_norm_`, so every rank branches identically.

Deliberately **not** upstream's mechanism (#4226), which uses
`torch._assert_async`: a failed device-side assert invalidates the process, and
our failover would read that as a crash rather than a clean stop. Its XPU
behavior is also undocumented, and it lands in `Trainer.train_step`, which
`FaultTolerantTrainer` overrides wholesale -- merging it would have given us
nothing. The most valuable commit in that sync was valuable only as a
hand-port.

Honest about the smoke: 10/10 bit-identical against the parent, guard fired 0
times, tps 337-338 vs 338-340. The ~0.3% delta is **below** what a single-shot
10-step comparison resolves, so bit-identity is the real result and "costs
nothing" is not yet earned. And the smoke proves the guard is inert, not that
it fires -- that path needs a divergent config and is still owed.

### 5. Polaris failover was blind for its entire existence

`ezpz.failover.patterns` ships `aurora.py` and `sunspot.py`;
`get_patterns_for_machine("polaris")` returned `[]`. **An empty pattern set and
a genuinely clean log both return `[]`**, and falling back to blind rotation is
the correct response to the second -- so a whole machine having no patterns
degraded gracefully into looking like normal operation. No error to grep for.
It surfaced only by asking the scraper directly what it returns.

The cost was job 7550301: **130 nodes, ~1 hour, zero training steps.** Two
ranks raised `CUDA error: ... device(s) is/are busy or unavailable` at
`set_device()`. Blind rotation swaps `active[0]` *by design*, so it retired a
healthy node twice while the sick one stayed in, and attempt 2 failed
identically. Balance is now at **-49,134 node-hours**.

The one host-attributed line in 1,800 lines named the **wrong** node:
`x3007c0s13b1n0: rank 57 died from signal 15` is the idle watchdog's own
SIGTERM -- a victim of our teardown, not a cause. Matching it would have
swapped a third innocent node.

Fix: PALS `--label` (confirmed on real hardware rather than assumed -- the
prefix is `<fqdn> <rank>: ` applied per line) enabled **opt-in** via
`EZPZ_MPI_LABEL=1`, since Aurora and Sunspot patterns anchor on *unlabeled*
`^<host>: ` lines and turning it on globally would have silently broken both.
8/8 tests pass, including the negative one: on unlabeled input the scraper must
stay silent rather than tag the SIGTERM victim.

Writeup:
[`known-bugs/polaris-failover-blind-rotation.md`](../../reference/known-bugs/polaris-failover-blind-rotation.md).

### 6. Flex-attention MoE: two stacked bugs

Every flex-attention MoE config was dying on a missing BlockMask, and it took
two fixes. Core builds the mask in `_prepare_inputs` only `if positions is not
None`, and blendcorpus never yielded that key. Fix 1 emits per-document
positions (restarting at 0 after each EOD, because blendcorpus *packs*
documents and a plain arange would let attention cross document boundaries). I
unit-tested that against HF semantics on four cases before running it anywhere;
it passed, and the real run **failed identically**. Fix 2 was the actual bug:
`_document_positions` read the EOD id off `_bc_cfg`, blendcorpus's own config
object, which does not carry the field -- so `getattr(..., None)` silently
yielded None.

The mask fix then exposed an unrelated bug underneath: **MoE routing is not
recompute-stable under FullAC** (`Recomputed values ... 560 vs 559`). Three AC
arms on both flex configs: full fails, AC-off fails at 89-94% memory,
**selective passes 5/5** -- not luck, `SelectiveAC` keeps `aten.topk.default`
as MUST_SAVE precisely to hold expert assignments stable across recompute.

Also settled: the EP all-to-all abort is
`ur_die: urEventWait must not be called for an internal event` -- Intel's
Unified Runtime, **not** a torchtitan assertion. EP is not the trigger
(`moe_debugmodel_ep` runs 5/5). And **hybridep is closed WONTFIX**: `deep_ep`
is CUDA-only ("GB200 NVLink72 Systems", TMA-optimized, calls
`cudaStreamSynchronize`), so it is not a torch version floor as previously
recorded.

### 7. Two seats were loading complex-trained weights as cos_sin

`CONFIG_SUFFIX` was one global `_real` across all five umbrella seats, but each
chain crossed the 2026-06-25 RoPE switch at a different step and `2b_v2_256`
never crossed it at all:

| seat | parent @ step | trained | was |
|---|---|---|---|
| t0 dolmino | 2b_v2_512 @46429 | `_real` | ok |
| t1 20b-512 | 20b_v2_512 @9000 | `_real` | ok |
| t2 20b-256 | 20b_v2_256 @10369 | `_real` | ok |
| t3 2b-512 constlr | 2b_v2_512 @21307 | **complex** | WRONG |
| t4 2b-256 constlr | 2b_v2_256 @9500 | **complex** | WRONG |

t3 is a 512N production seat resuming **in place** -- it only avoided
corrupting a live chain because it dies on `std::bad_alloc` first. Now a
per-seat field, guarded on UNSET rather than empty, because empty is a
meaningful value here (complex). Mis-flavoring loads cleanly and only shows up
as high loss, which is what makes it dangerous.

Consequence for the matched-pair eval at step 21,000: mean delta -0.0033 across
7 tasks, no meaningful separation -- but **boolq is confounded**, because the
fork trained under cos_sin from complex-derived weights and boolq is
calibration-sensitive. The clean version of that experiment needs a
correctly-flavored fork.

### 8. Tooling: the dashboard, and a class of silent failure

- **MDS and `lr` are in the production dashboard.** MDS reads the committed CSV
  (154,391 rows, zero W&B calls); its tokens/step is confirmed bit-exact
  against W&B's own `consumed_train_tokens` (`154391*6144*8192 ==
  7,770,753,466,368`). `lr` rides in a separate `scan_history` pass -- folding
  it into `OLOG_KEYS` returns 0 rows for all 6 backfill runs and would have
  deleted 3,408 points from every other metric.
- **The combined eval chart was truncating the MDS flagship at 7.06T of its
  7.77T tokens.** The last 14k steps live in a sibling results dir
  (`agpt-2b-mds-7771T`) that no trajectory referenced. 28 points -> 36.
- **A whole class of path bug closed.** Plot scripts that computed the repo
  root by counting `parents[N]` render an *empty* figure when moved, which
  `refresh_all.sh` then auto-commits at exit 0; two charts had already lost
  data this way. Anchored to a marker-based walk-up, and the link checker now
  flags both depth-counted paths and pathlib chains built a segment at a time
  (`DOCS_BASE / "production" / ...`), which no string scan can see. The docs
  tree is reorganized by lifecycle (`live/` `reference/` `records/`
  `outbound/`) with the link check now **fatal** in `refresh_all.sh` and the
  dangling-reference backlog cleared 31 -> 0.

### 9. Standing items

- **The main Aurora clone is currently un-runnable for training.** Its HEAD
  imports `grain`, which is deliberately absent from the shipped venv (a stray
  install there vendors a `torch/` that shadows the conda one). Docs and charts
  are fine. The queued umbrella is unaffected -- it runs from pinned production
  clones, none of which has that import -- but pulling a production clone to
  HEAD would break it.
- **The 80th upstream sync: defer.** Grain is the *oldest* of the 35 commits,
  so every other one descends from it; no merge can take the mechanical import
  fixes without also taking Grain and fold-batch-dim. Splitting requires
  cherry-pick, i.e. carrying divergence. Trigger for revisiting: upstream
  deprecating `partial_dtensor`, or adopting the 2.14 nightly -- both in one
  revalidation window, not two.
- **Umbrella slot conversion is largely answered** (section 2): the named
  seat failures are fixed and three umbrellas in a row ran 97%, 100% and 100%
  of their allocation. What remains is the init `std::bad_alloc`, now the only
  unexplained seat killer -- a fresh 512N reproduction exists, and the fact
  that t1 ran fine at the same 522 nodes in the same job argues against pure
  rank count. Falsifiable next test: `qsub -v LAUNCH_STAGGER=180`.
- **Prefer 12 h umbrella asks over 24 h.** Both umbrellas that used ~all of
  their allocation asked for 12 h; the 24 h dispatches kept dying young. Worth
  making the default rather than a per-submission choice.

---

## 2026-08-10

> [!IMPORTANT]
> **Headline:** **MMLU is settled, and it is a data problem, not a training
> problem.** Two 7.771T-token MDS checkpoints that had never been evaluated
> (they sat in directories no sweep had searched) give a controlled test --
> identical architecture, optimizer, LR and token count, differing only in the
> final ~0.6T of data -- and **both score at chance on MMLU (0.2463, 0.2591)
> while one of them posts our best-ever ARC-Easy (0.7138) and ARC-C@25
> (0.4164)**. With the harness independently validated (it reproduces
> Llama-3.2-1B 0.3121 and Llama-3.1-8B 0.6530 on our exact path), a 1B public
> model clears chance where our most capable 2B does not: **the pretraining
> corpus does not contain what MMLU tests, and neither more tokens nor a
> different finishing mix fixes that.** The same comparison shows a narrow
> math/code finisher causes **catastrophic forgetting** (HellaSwag 0.587 ->
> 0.422) to buy gsm8k that is still ~0 -- independently reproducing the
> 07-28 anneal finding at 7.771T instead of 10B. On production: **2B-512 is at
> 94.3% of target** and both 20B chains are at step-8,300, carried by two full
> umbrella runs; but the umbrella is only converting **1-3 of 5 slots** into
> work, and an unresolved init `std::bad_alloc` is the main cause.

Covers the week since 2026-08-03. The headline is an **eval/data** result
rather than a training one. Production ran entirely on the ~2,098N umbrella
(two completed runs plus one in flight), which exposed how much of that
allocation is being wasted; a new [dispatch
log](../../live/dispatch-log.md) now tracks every job against a production
chain, including per-slot umbrella outcomes that were previously invisible.

### 1. MMLU: settled by controlled experiment

Chased to a conclusion this window. MMLU never leaves the 4-way chance floor on
any AuroraGPT checkpoint:

| model | tokens | best commonsense | mmlu |
|---|---|---|---|
| 20B-256 | 0.42T | -- | 0.2599 |
| 2B-256 **COMPLETE** | 4.674T | hellaswag 0.561 | 0.2437 |
| MDS dolmino | 7.06T | arc_c@25 0.3968 | 0.2413 |
| **MDS stage3-mix** | **7.771T** | **arc_e 0.7138** | **0.2463** |
| **MDS math/code** | **7.771T** | -- | **0.2591** |

Eliminated in order: **tokens** (a completed 4.674T run finished at chance),
**scale** (7.771T with our best commonsense is still 0.2463), **tokenizer**
(gemma assets against gemma-tokenized data; the 256128-vs-256000 vocab gap is
alignment padding), and **the harness** -- job 8736838 ran three cached public
models through the identical `simple_evaluate(num_fewshot=5, device="xpu:0")`
path and reproduced their published numbers.

**Implication for the program:** MMLU-style performance needs academic
multiple-choice content *deliberately added* to pretraining. This is a
data-acquisition decision and belongs in the data-strategy discussion, not the
training-config one.

### 2. The finishing mix: a controlled forgetting result

The two 7.771T arms differ only in their last ~0.6T of data:

| metric | stage3-mix | stage3 (nvidia-math1-code2) |
|---|---|---|
| arc_challenge@25 | **0.4164** | 0.3703 |
| hellaswag | **0.5874** | 0.4215 |
| arc_easy | **0.7138** | 0.6216 |
| gsm8k | 0.0167 | **0.0303** |
| mmlu | 0.2463 | 0.2591 |

The narrow math/code finisher trades ~16 points of HellaSwag for a gsm8k gain
that is still effectively zero. This is the **same effect the 07-28 anneal A/B
found** when pure edu-web forgot math -- now reproduced at 7.771T on a
different corpus, which makes "keep the finishing mix broad" a much more solid
recommendation than it was a week ago. A step ladder across both arms
(8747310) is running to establish *when* the forgetting happens.

### 3. Production -- 2B-512 at 94.3%, both 20B chains at 8,300

| chain | 08-03 | now | delta | loss | tokens |
|---|---|---|---|---|---|
| **2B 512N** | 41,300 | **43,800** | +2,500 | 2.69 | **4.41T (94.3%)** |
| **20B 256N** | 7,600 | **8,300** | +700 | 2.386 | 417.8B (8.9%) |
| **20B 512N** | 6,850 | **8,300** | +1,450 | 2.470 | 835.5B (17.9%) |

20B-512 was the mover (+1,450), and the two chains have converged to the same
step from opposite directions. 2B-512 is ~500 steps from its 4.674T target --
worth deciding now what happens when it lands.

### 4. The umbrella is wasting most of its allocation

Two umbrellas completed (8714502 8h01m, 8714503 a full 24h) and 8744245 is in
flight. Productive slots per run: **3/5, 3/5, 1/5**.

| cause | slots | status |
|---|---|---|
| `std::bad_alloc` at init | 6 across 3 runs | **OPEN** -- [known-bug](../../reference/known-bugs/umbrella-bad-alloc-init.md) |
| ckpt "latest" resolving to a renamed partial save | 1 | fixed |
| pre-refactor flat attention keys | 1 | shim written + tested; clone delivery unsolved |
| ImportError | 2 | self-inflicted, reverted |

The `bad_alloc` lands in `set_determinism`-broadcast and CCL KVS bootstrap --
both known single-job crash sites at 12,288 and 6,144 ranks. The umbrella puts
**24,576 ranks** into bootstrap inside 80 seconds. That is a plausible cause but
**not proven**: the same config gave three different outcomes. A falsifiable
test (`LAUNCH_STAGGER=180`, ~1% of a 24h job) is the next step.

Two reporting problems worth fixing regardless: the umbrella banner reports
`failed: N/5` on **exit code**, so a trainer that ran productively for hours and
was then SIGTERM'd is indistinguishable from one that never started; and
**three different failure modes all print `FAILOVER STOP: walltime`**, including
one that died 29 minutes into a 24-hour job.

### 5. Smaller items

- **Eval integrity:** `arc_challenge` was being scored at both 0-shot and
  25-shot into one JSON key, the later phase silently overwriting the earlier.
  A reported ARC-C "decline" was retracted as an artifact of this. Results are
  now shot-namespaced (`<task>@<N>shot`); the fix immediately exposed a
  consistent ~7-9pp gap between the two on both 20B chains.
- **Doc-refresh integrity:** four separate silent failures in `refresh_all.sh`,
  each reporting success while skipping work -- stale W&B run-ids (loss frozen
  at a July value), a missing field anchor, five plotters hardcoding a Sunspot
  path (three of which exited 0 having plotted nothing), and unfixable date
  markers.
- **frameworks RC4 fixes the TP=4 SDPA-backward compile assert** -- the
  ".venv for compiled TP=4" workaround is obsolete, and a control run shows RC4
  is performance-neutral.
- **DAOS:** first contact submitted (8747000). The client, module, ALCF helper
  scripts and example jobs are all already installed; the blocker is that
  `daos pool list` cannot run on a login node, so **whether an AuroraGPT pool
  exists is itself an open question** pending that job. Queue access is
  group-gated -- only `alcf_daos_cn` admits us, via `aurora_daos_test`.
- Upstream synced (74th + 75th; the 75th inherits a real MoE gradient fix).

## 2026-08-03

> [!IMPORTANT]
> **Headline:** both 20B production chains advanced (256N step 6,000 -> 7,600;
> 512N 6,100 -> 6,850) and the ~2k-node 5-chain umbrella is staged to run Tue;
> SFT on the 2b-mds stage-3 base landed a clear verdict: **teaching CoT in two
> separate SFT stages (general math -> then GSM8K chain-of-thought) beats doing
> it in one combined SFT run** -- the two-stage "B2" model scores 0.205 on
> GSM8K-CoT while every single-stage rebuild lands at 0.02-0.065, so the
> *sequencing* matters more than the data mix
> ([details](../../live/chains/sft/agpt/2b-mds/b4-finish-and-reweight/README.md));
> and the 2B
> continued-pretrain data-mix experiment closed with its own (75/25 owm/edu is
> the sweet spot on val-loss, but downstream-neutral at 10B tokens);
> and on 80B, **no change to the July root-cause** -- the bf16
> activation-overflow diagnosis and the confirmed-stable TP=4/LBS=1/bf16/GAS
> corner both still stand; this window only adds a standing constraint (**run
> all 80B experiments at >=4N**; 2N is below the memory floor, so 2N failures
> are artifacts) after a self-inflicted 2N detour, and re-confirms 80B trains
> clean at 4N/TP=4 (70.25% peak memory)

Covers the ~1 week since 2026-07-27. **Production pre-training is the headline
and leads below** (both 20B chains advanced; the 2,098N 5-chain umbrella is built
and queued). Then the standing **80B** instability status, and the research
fronts: the **anneal + data-mix** experiment ran to a full verdict (an actionable
recipe for the flagship stage-2 continued-pretrain), and **SFT on the 2b-mds
stage-3 base** produced the clearest post-training result we have (two-stage
structure is the lever). Also: commonsense eval complete to tip + modern block
backfilling, upstream synced (72nd + 73rd), RC-venv `libpti` import bug fixed,
science-corpus Wave 3 built + smoke-passed.

### 1. Production pre-training -- both 20B chains advanced, ~2k-node umbrella staged

The flagship v2 chains kept moving this window despite persistent `at_queue`
starvation, and the multi-chain umbrella (one PBS job driving all live chains)
is built + smoke-passed + queued at >2k nodes.

| chain | window start (07-27) | now (08-03) | delta | loss | tokens | MFU |
|---|---|---|---|---|---|---|
| **20B 256N** | step 6,000 | **step 7,600** | +1,600 | 2.467 | ~382B (8.2%) | ~22% |
| **20B 512N** | step 6,100 | **step 6,850** | +750 | 2.248 | ~685B (14.7%) | ~19% |
| **2B 256N** | COMPLETE (4.674T) | unchanged | -- | 2.652 | 100% | -- |

- **20B 256N is the mover:** +1,600 steps (persisted head step-7,600, loss
  2.467, 22% MFU / 442 tps-per-gpu / 65.8 TFLOP/s). Kept alive across the window
  by a 260N `prod` resume (`8698754`, walltime-finished clean at step-7,600 after
  12h02m) plus a 16N capacity bridge earlier; a fresh 260N continuation
  (`8730438`) is queued to carry it to the umbrella handoff.
- **20B 512N** advanced +750 steps to step-6,850 (loss 2.248, 19% MFU); the
  current persisted head is step-6,800. Its constant-LR individual (`8687863`) is
  held for a 512N slot.
- **Umbrella job (`8714502`): 2,098 nodes, 24h walltime, 5 chains in one PBS
  allocation** via `ezpz launch --auto-retry` -- {2B-512, 20B-512, 20B-256,
  2B-512-const-LR, 2B-256-const-LR (fork @ step-9,500)}. Smoke-validated at 15N
  (`8714337`), submitted to `large`, currently Q with PBS estimated start
  **Tue Aug 4 ~15:20** (node scarcity, not a fault); `afterany` continuation
  `8714503` chained. This is the first umbrella slotted to actually run -- prior
  2,098N attempts were terminated while queued (walltime bump to 24h, ghost
  cleanup). A collision guard is armed to retire the running individual chains
  the instant the umbrella seats, since they share checkpoint dirs.
- **No corruption, no collisions** this window; the earlier bridge/umbrella
  shared-ckpt-dir hazard was cleaned up (corrupt 256 step-6,200/6,300 + poisoned
  2B-const step-9,250/9,260 quarantined via `backup`).
- **80B: still no production job** (see item 2); the at-scale corner is unsolved.

### 2. 80B: state of the long-running instability (nothing here supersedes the July root-cause)

Nothing this window changed the 80B diagnosis. Restating it so the status is not
mistaken for an open question -- **the 80B NaN is root-caused and there IS a
confirmed-stable corner**; what is missing is a fix that survives production
scale, not a diagnosis.

**Wall 1 -- bf16 forward-activation overflow (root-caused 2026-07-14).** NOT an
optimizer bug: SophiaG (512N, step-14) and **mano** (62N/dp~186, step-17) NaN
with the *identical* signature (grad_norm dead-flat ~6.0, then sudden inf/nan, no
runup), and mano has no Hessian term -> optimizer-independent. Ruled out in code:
loss softmax (CE upcasts to fp32), grad reduction (FSDP `reduce_dtype` fp32,
loss dp-invariant). Mechanism: the 84-layer pre-norm residual stream accumulates
in bf16 and, at 80B's width x depth (dim 9216 x 84L x ffn 25600), reaches the
precision regime 2B/20B never hit. **Smoking gun:** the fp32-activations run
(job `8537349`) trains clean and reveals TRUE grad_norms of **21K-79K** that bf16
masks down to ~5-7. dp-dependence is indirect: larger dp -> larger effective GBS
-> weights reach the overflow state sooner (LR / seed / master-dtype / clip all
proven non-causal in the 7-job factorial).

**The two triggers, and the corner that avoids them.** The NaN tracks **LBS>1**
and **large dp_degree** (= NGPUS/TP), not raw GBS. Keeping LBS=1 and holding
dp_degree down (TP=4 halves it: 744/4=186 vs 744/2=372) stays in the safe regime
*in pure bf16*. **CONFIRMED STABLE 4/4** (jobs `12469494`/`509`/`510`/`511`,
20-30 steps each, 0 NaN, three landing at identical loss 9.69-9.70) -->
**production recommendation: TP=4, LBS=1, bf16, batch via GAS** -- cheaper than
fp32-activations (~3-5x) and than `--debug.deterministic` (~50%, and it does NOT
scale past n=32: job `8540102` NaN'd at n=64). Validated only to ~62N, so the
open question is whether this corner holds at production scale, not whether 80B
can train.

**Attempted fixes, ranked by what the evidence supports:**
1. `--training.mixed-precision-param=float32` @ TP=4 -- the one config with
   confirmed-clean training (`8537349`), ~3-5x slower. The guaranteed unblock.
2. **fp32 residual stream** (the Llama3-405B remedy): per-block prototype trains
   clean at 4N but **still NaNs at dp=192** (job `8671243`, step 19); full-depth
   also NaNs (`8673658`, ~step 37). Necessary, not sufficient -- which narrows
   the overflow to a **bf16 sublayer GEMM** (attention QK^T scores or the FFN
   SwiGLU intermediate) rather than the residual add.
3. **Score-bounding (softcap / QK-Norm) -- BOTH now blocked on this stack.**
   Softcap hard-codes `torch.compile(flex_attention)`, which
   `--compile.no-enable` cannot switch off and which is broken/eager on XPU.
   **QK-Norm is now also blocked, and this is new (2026-08-03):** `agpt_80b_qknorm`
   crashes in BACKWARD with `RuntimeError: tensor does not have a device`, and
   that is now established on a VALID baseline -- job `12472459` at **4N/TP=4**,
   the exact config where plain `agpt_80b` trains 10/10 clean (`12472452`). So it
   is a real qk_norm TP>1 backward bug, not the 2N memory artifact I first
   mistook it for. The `LocalShardRMSNorm` fix (`77a0009f3`, local-shard norm,
   review-correct on grad placements) **does not fix it** -- that verdict was
   2N-invalid before and is now validly established at 4N. **Net: the entire
   score-bounding branch of the fix ladder is currently unavailable**, which
   leaves fp32-params (item 1) as the only working lever and promotes item 4.
4. **Targeted fp32 on the suspect sublayer (the main untried lever).** The
   fp32-residual ladder already localized the overflow to a bf16 sublayer GEMM
   -- attention QK^T scores or the FFN SwiGLU intermediate. With score-bounding
   blocked, the remaining cheap-ish option is autocasting just that one site to
   fp32 (far less costly than whole-model fp32-params). Prerequisite: decide
   WHICH of the two, which is what the per-op numerics capture at dp=192 was
   for -- it is wired but paused (DebugMode is broken on torch 2.13-dev; a
   lightweight forward-hook amax logger would substitute).
5. Operational guard in place: `--nan-abort-consecutive=5` (the 512N NaN burned
   ~6,100 node-h before this existed).

*(Retiring "Wall 2": the intermittent `set_determinism` init OOM fires
unpredictably at 512N+ but 512N production survives it routinely via resubmit
(`8466848` crashed; `8463628` before it and `8479579` right after both succeeded
at the same scale/script). It is an operational nuisance to harden with
retry-on-init-OOM, not a scaling wall, and it does not belong beside Wall 1.
Two related corrections while removing it: the 256N `NotPresent` event
(`8505222`) was a **bad-node cascade** -- every spare the failover wrapper drew
was also bad, 3 of 6 spatially clustered -- not a dp=768 limit, so 256N is not
proven broken; and **80B at 1024N has never actually run** (`8574386` never left
`Not enough free nodes available`).)*

**Practical ceiling today: ~62N, and Wall 1 is the binding constraint -- not
Wall 2.** The stable corner is confirmed 4/4 clean only to dp~186 (~62N at TP=4);
dp=372 NaNs (`12469492`). Wall 1's true boundary is somewhere in **(186, 372]**
and has never been bracketed -- that is the number that decides how far 80B can
go. Wall 2, by contrast, is an intermittent init OOM that 512N production already
survives routinely (resubmit clears it), so it is an operational nuisance at
1024N, not a gate.

For 1024N specifically the hard problem is numerics + batch geometry, not init:
at 1024N/TP=4, dp = 12,288/4 = **3,072** -- ~16x past last-known-good and ~8x
past known-NaN. And with LBS=1 that is already **GBS >= 3,072 before any GAS**,
so there is almost no headroom to tune the batch DOWN if it proves unstable
(GBS is normally held ~6144 via GAS). 512N (dp=1536) leaves real room. So the
open question for large-N 80B is "where does the stable corner actually break,
and does the fix cost throughput (fp32-params) or not (score-bounding)" -- with
1024N plausibly the wrong shape regardless.

**This window's contribution: a standing constraint, plus a self-inflicted
detour.** CONSTRAINT: **run all 80B experiments at >=4N** -- 80B peaks 88.94% at
4N/TP=2, so 2N is below the memory floor and 2N failures are uninterpretable.
I violated that this week and spent an implement->review->smoke cycle chasing a
"qk_norm TP=4 DTensor bug" that was 2N OOM: `LocalShardRMSNorm` (`77a0009f3`,
review-correct) "failed" at 2N, plain 80B "failed" at 2N differently
(`GPU NotPresent` at the step-1->2 optimizer allocation, 84% peak), and a 2N TP=2
control OOM'd outright. Re-run at 4N/TP=4 (job `12472452`): **10/10 clean, loss
12.98 -> 11.30, memory plateaus at 70.25%** -- TP=4 is the roomiest corner
(70.25% vs 88.94% at TP=2), consistent with the long-standing TP=4 recommendation
above. Net: **no new 80B bug; the 2N results were artifacts and are void**,
including the "qk_norm fix failed" verdict (retesting at 4N, job `12472459`).
- **Process lesson:** run the cheap discriminating experiment (plain-vs-feature,
  TP=2-vs-TP=4) **at a resourcing where the model fits**, and check peak memory /
  the boring OOM explanation, BEFORE building any fix. Two workflow "root causes"
  this window were plausible-but-unverified hypotheses; the code review was sound
  but aimed at a target chosen from invalid data.
- **Where 80B stands:** `LocalShardRMSNorm` is committed + review-correct but
  UNVALIDATED (parked until a memory-clean TP=4 baseline exists). The real open
  question is whether 80B trains at TP=4 with adequate nodes (4N+) at all -- and
  separately, TP=2 is memory-good but NaNs at production GBS (the original reason
  TP=4 was wanted). 80B-at-2000N (dp~6000) remains the hard, unsolved corner.

### 3. Data-mix continued-pretrain experiment -- CLOSED, 75/25 is the recipe

The stage-2 continued-pretrain data question (per the data-strategy memo) is
answered. All arms fork the MDS base at constant LR 2e-6, 10B tokens, differing
ONLY in the data mix; scored on held-out FineMath (math generalization) +
wikitext (anti-forgetting), both DISJOINT from every training arm.

| arm | FineMath (math) | wikitext (general) |
|---|---|---|
| owm-100 (control) | **1.8039** | 2.6828 |
| **owm75 / edu25 (WINNER)** | 1.8089 | 2.6597 |
| owm50 / edu50 | 1.8155 | **2.6519** |
| edu-100 | 2.1122 | 2.6585 |

- **75/25 owm/edu is the sweet spot** for a math-focused continued-pretrain: it
  holds math at owm-level (FineMath +0.005 = noise) while capturing ~all of the
  general-ability gain (wikitext 2.6597 vs edu's 2.6585). The marginal trade has
  a clean knee -- owm->75/25 is ~4.6:1 favorable, 75/25->50/50 flips to ~1:1.
  **No 90/10 Wave 3 needed** (75/25's math cost is already noise).
- edu-100 confirms the failure mode: swapping math-web for edu-web forgets math
  by +0.308 nats (~22x the anneal effect) for a tiny general gain.
- **The 50/50 confirmatory arm took four attempts to land** (`12472037` ->
  `12472139` -> `12472163` -> `12472175`, eval `12472203`), and the failures were
  all infrastructure, not science: a transient `ezpz_setup` failure that left
  torch un-importable (which surfaced as a *misleading* "data source did not
  resolve" preflight abort, since every `import datasets` check then failed
  too), then two crashes in `dcp_save` at step-100 when the tegu **project quota**
  hit its hard cap (11T used / 10T soft / 11T hard -- raw disk had 1.1P free, so
  `df` looked fine and only `lfs quota` showed it). Cleared when the quota was
  raised to 20T soft / 22T hard. Hardened `mix_ab.sh` accordingly: retry
  `ezpz_setup` once, then hard-fail loudly with the real torch traceback instead
  of cascading into the confusing data-source message.
- **Important caveat -- the val-loss verdict does NOT transfer to downstream
  accuracy at this scale.** Ran a full lm-eval sweep over all three arms
  (jobs `12472403`/`12472407`, 1N; gsm8k + mmlu 5-shot, hellaswag/arc-e/arc-c/
  winogrande/piqa/openbookqa 0-shot), scoring the pre-converted HF checkpoints:

  | task | owm-100 | owm75/edu25 | edu-100 |
  |---|---|---|---|
  | gsm8k | 0.026 | **0.033** | 0.026 |
  | mmlu | 0.248 | 0.239 | 0.248 |
  | mmlu_stem | 0.221 | 0.225 | **0.234** |
  | hellaswag | 0.569 | 0.571 | **0.574** |
  | arc_easy | 0.638 | 0.643 | **0.662** |
  | arc_challenge | **0.370** | 0.358 | 0.363 |
  | winogrande | 0.576 | **0.580** | 0.572 |
  | piqa | 0.739 | 0.740 | 0.740 |
  | openbookqa | 0.360 | 0.374 | **0.386** |

  The three arms are **downstream-indistinguishable** -- every delta is within
  noise at 2B / 10B tokens, and the faint trend actually runs *against* the
  val-loss ordering (edu-100 edges ahead on arc_easy, openbookqa, mmlu_stem,
  hellaswag; 75/25 leads only gsm8k, by 0.007 = 2-3 questions). Takeaway: at 10B
  tokens **val-loss NLL is the sensitive instrument and downstream benchmarks
  are not** -- so **75/25 is a val-loss win that is downstream-neutral**; do not
  oversell it as a downstream win. Enabling work: there was no lm_eval anywhere
  on Sunspot (the existing recipe was Aurora-only), so a `venvs/sunspot-lm-eval`
  was built on the frameworks module -- torch stayed the XPU build (verified
  `version.cuda=None`), transformers 4.50.1 (the lm-eval-compatible version).
- Also settled earlier in this window: the **anneal A/B** (WSD LR-decay vs flat
  constant-LR) showed **flat >= wsd on both MDS and olmo bases** -- i.e. the
  LR-schedule is NOT the lever, DATA is. That result is what motivated the
  data-mix experiment.
- Report: [`20260728-2b-mds-anneal-and-datamix`](../experiments/agpt/sunspot/20260728-2b-mds-anneal-and-datamix.md).

### 4. SFT on the 2b-mds stage-3 base -- the live post-training thread

Post-training work runs on **`agpt/2b-mds` = the MDS stage-3 base
(`global_step138650`)**, and this is where the useful results are. (The earlier
CPT mixing-ratio sweep -- dolmino-100 / olmo50-dolmino50 -- is largely
superseded: it showed loss improving while downstream eval DEGRADED, and the
data-mix experiment in item 3 answers the "what data for stage 2" question more
directly. Keeping the gentle-LR arms as reference, not as an active front.)

Status of the recipes on this base:

- **The CoT ladder's verdict: HOW you sequence the SFT matters more than what
  data you put in it.** The question was how to teach the model to show its
  reasoning (emit `<think>...</think><answer>` and get the answer right). Two
  approaches were compared on a 200-problem GSM8K CoT eval (fp32 vLLM, identical
  harness):
  - **Two-stage ("B2") -- the winner, 0.205 cot_accuracy / 0.985 format.** Train
    on general math instruction data (tulu-math) FIRST, then do a *second,
    separate* SFT pass on GSM8K chain-of-thought data (gsm8k-r1cot). Each stage
    teaches one thing.
  - **Single-stage rebuilds -- all lost, badly.** Putting everything in one
    combined SFT run: B3 (broad mix @8192) **0.05**, hurt by long-CoT dilution;
    B4a (adding a short gsm8k-r1cot "finish" on top of the B3 base) **0.02**,
    i.e. *worse* -- B3's verbose bias is already baked in and a light finish
    just destabilizes the output envelope (gen_len blew up 601 -> ~1400-2000,
    format collapsed to 0.26); B4b (reweight + length-filter @4096) **0.065**,
    which fixed the run-on symptom but not the accuracy.

  So the actionable rule is **use two separate stages, and do not try to
  rehabilitate a base that was already trained verbose** -- no amount of
  reweighting or finishing recovered B2's number. Jobs B4a `12471671`, B4b
  `12471672`; full results table:
  [b4-finish-and-reweight](../../live/chains/sft/agpt/2b-mds/b4-finish-and-reweight/README.md)
  (design + plan for the earlier stages:
  [b3-instruct-cot-mix](../../live/chains/sft/agpt/2b-mds/b3-instruct-cot-mix/design.md)).
- **`tulu_math_uc_mix` (3 epochs, 32N, GBS=6144): complete**, final loss 0.77,
  4.5B tokens -> `checkpoint-729-hf`. This is the deliverable feeding GRPO.
  [trajectory](../../live/chains/sft/agpt/2b-mds/tulu_math_uc_mix/README.md)
- **`tulu_math_uc_mix_full` (the 12x-tokens FULL mix, ~54B): complete at 8N but
  catastrophically forgot** at 1 epoch (train loss 0.357 while HellaSwag
  0.59 -> 0.27) -- **the deliverable is checkpoint-900**, and the lesson is a
  guardrail: cap full-mix SFT at O(1000) steps or drop LR. More tokens did NOT
  beat the small metamathqa SFT. It runs at 8N only because 32N hits the
  384-rank GPU page fault (bisect: 8N clean / 12N fault).
  [trajectory](../../live/chains/sft/agpt/2b-mds/tulu_math_uc_mix_full/README.md) ·
  [evals](../../live/chains/sft/agpt/2b-mds/tulu_math_uc_mix_full/evals/README.md)
- Cross-checks with item 3: the CoT verdict ("accuracy lives in cold-start SFT,
  not RL, at 2B") and the data-mix verdict ("75/25 owm/edu, but
  downstream-neutral at 10B") point the same direction -- **the lever for 2B
  quality is the base and the SFT structure, not more RL and not the LR
  schedule.**

See: [SFT index](../../live/chains/sft/README.md) ·
[B4 results](../../live/chains/sft/agpt/2b-mds/b4-finish-and-reweight/README.md) ·
[full-mix SFT evals](../../live/chains/sft/agpt/2b-mds/tulu_math_uc_mix_full/evals/README.md).

### 5. Evaluation -- commonsense complete to tip; modern block backfilling

- **Commonsense-7 ladder is complete to each chain's live tip:** 256N through
  step-6,800, 512N through step-6,500, all fresh this window. Headline
  accuracies at the tips (~610-680B tokens): 256N step-6,800 = arc_e 0.662 /
  arc_c(n) 0.348 / hellaswag(n) 0.597 / winogrande 0.574; 512N step-6,500 =
  arc_e 0.677 / arc_c(n) 0.354 / hellaswag(n) 0.609 / winogrande 0.579. Both far
  above v1's flat ~0.27 arc baseline; 512N edges 256N on shared metrics.
- **Modern block (MMLU-57 + gsm8k) is backfilling now.** The first tail pass had
  the modern half die on the old 12h walltime cap mid-MMLU; resubmitted as two
  `capacity`-queue jobs at **48h** (`8729921` 256N running, `8731413` 512N
  running) so the slow ~56k-request/step MMLU pass can't be walltime-killed. Fill
  targets: 256N modern on steps 5,900-6,800, 512N modern on 6,200-6,500 (+gsm8k
  on each tip). Content-aware skip-guard merges into existing `results.json` and
  reuses cached HF conversions.
- **HellaSwag plateau holds:** peaked ~0.63, now oscillating 0.60-0.63 -- as
  before, not yet climbing out at this token count.

### 6. Smaller items

- **Pipeline parallelism now works in the ezpz trainer (new capability).** A
  first-ever ezpz PP run died at step 0 on `assert isinstance(input_dict, list)`:
  `ezpz/trainer.py.train_step` had **no `pp_enabled` branch at all** -- it
  flat-looped microbatches for gradient accumulation and passed one dict per
  call, while upstream's contract hands the WHOLE microbatch list to
  `forward_backward_step` so the pipeline schedule can drive the stages. This
  was a **pre-existing gap, not a regression** (no `pipeline_parallel_degree>1`
  anywhere in `experiments/ezpz`, so it had never been exercised). Fixed by
  mirroring the base Trainer: set `num_pipeline_parallel_microbatches` in
  `__init__` (it is 1 when PP is off) and nest the PP-microbatch loop inside the
  existing GAS loop, passing the group as a list under PP and unwrapping to a
  single dict otherwise. **The non-PP regression arm passes 20/20 steps** (loss
  12.87 -> 6.87), confirming the change is a no-op for every production job.
  PP=2 itself now reaches torch's own pipeline schedule and fails there on a
  microbatch-count mismatch (`Expecting 2 arg_mbs but got 1`) -- one layer
  deeper, still being localized (probe job `12472464`).
- **Upstream synced twice (72nd `e5841d611`, 2 commits; 73rd `23b4000dd`, 11
  commits).** 72nd: replayed the float8 `filter_fqns` fix (#4008) onto
  `ezpz/moe` -- our 671B float8 config filtered on `output` (old head name) not
  `lm_head`, silently fp8-quantizing the LM head; now fixed (float8 MoE path
  only). #4012 (FA4 Blackwell) is CUDA-gated, XPU-inert. **73rd was NOT inert:**
  one conflict (`experiments/torchft/trainer.py`, complementary dataloader
  kwargs -- kept both sides) and one required replay -- #3856 "Always Pre-Split
  Microbatches for PP" changed the dataloader to serve microbatches under PP, so
  `ezpz/trainer.py` needed the same `pipeline_parallel_microbatch_size` logic.
  No replay needed for #3558 mxfp8 (SM100 + CUDA + compile only -> inapplicable
  on XPU; adds an opt-in config fn), kimi_k2_7, or the 8 CI-only commits.
- **RC venv (`2026.1.0-rc0`, torch 2.14) import failure fixed:** the masked
  "Cannot import config_registry" was `libpti_view.so.0` missing (needs
  `module load pti-gpu`); permanently fixed by symlinking the system-module lib
  into `torch/lib` (found via `$ORIGIN` RUNPATH on every rank, no env needed).
- **Science-corpus Wave 3 built + smoke-passed:** `owm_cosmo_7525` (75%
  open-web-math / 25% cosmopedia-science) 2N smoke trained clean (loss
  4.02->3.42); the nemotron-CC-math arm + peS2o science-judge holdout are staged.
  32N prod not yet launched. This is the science-dense variant of the 75/25
  recipe (swap generic edu for science sources).
- **Ops drags this window:** persistent Aurora/Sunspot `at_queue` starvation +
  intermittent SSH/API outages; a transient tegu **project-quota** exhaustion
  (11T/10T) crashed a blend run mid-checkpoint (since raised to 20T soft / 22T
  hard); the mix launcher hardened to retry venv setup + hard-fail loud on a
  broken env instead of a misleading "data source did not resolve".

### Top asks / open decisions

1. **80B -- scale the known-stable corner, or buy stability with throughput?**
   The corner (**TP=4 / LBS=1 / bf16 / batch via GAS**) is confirmed 4/4 clean
   but only validated to ~62N (dp~186). Two paths: (a) push that corner up the
   dp ladder and find where it breaks, or (b) start the guaranteed-clean
   fp32-mixed-precision-param TP=4 run now and accept ~3-5x slower. No 80B job
   is training today, so this is the decision that unblocks 80B.
2. **QK-Norm: keep debugging the TP>1 backward, or drop score-bounding?**
   ANSWERED at 4N and the answer is bad: `agpt_80b_qknorm` crashes in backward
   (`tensor does not have a device`) at 4N/TP=4 where plain 80B is 10/10 clean
   (`12472459` vs `12472452`), so it is a real bug and `LocalShardRMSNorm` does
   not fix it. With softcap also disqualified on XPU, **score-bounding is
   entirely blocked**. Decision needed: invest in a proper DTensor-backward
   debug of qk_norm at TP>1 (unbounded -- prior root-cause attempts were wrong
   twice), or drop that branch and pursue targeted fp32 (ask 3) / fp32-params?
3. **Localize the overflow site so a targeted fp32 fix is possible.** With
   score-bounding blocked (softcap compile-coupled on XPU; QK-Norm now
   confirmed-broken in TP>1 backward at 4N), the only lever short of
   whole-model fp32-params is autocasting the single offending sublayer.
   That needs the per-op amax capture at dp=192 revived (DebugMode is broken on
   torch 2.13-dev -> use a forward-hook amax logger) to say whether it is the
   attention scores or the FFN SwiGLU intermediate. Small, and it unblocks the
   cheapest real fix.
4. **Bracket Wall 1's real boundary before proposing any large-N 80B run.** The
   stable corner is validated to dp~186 and NaNs at dp=372; the gap has never
   been tested. Run the corner at ~64N -> 96N -> 128N to find the actual break
   point. That turns "~62N is the ceiling" into a measured number and tells us
   whether large-N 80B needs fp32-params (confirmed clean, ~3-5x slower) or a
   working score-bounding fix. Proposed sequencing: Wall 2 on 2B (cheap, hard
   gate) -> Wall 1 bracket -> only then decide 512N vs 1024N.
5. **2B stage-2 recipe:** adopt **75/25 owm/edu** as the continued-pretrain data
   mix (val-loss-validated, downstream-neutral at 10B), and decide whether to
   launch the science-dense Wave 3 (cosmopedia / nemotron-math) 32N to test
   whether science sources beat generic edu on a science judge.

---

## 2026-07-27

### Headline: 80B NaN re-root-caused (bf16, not the optimizer), the CoT-teaching front ran to a clear verdict (SFT is the accuracy lever, not RL), full-mix SFT deliverable is checkpoint-900, and Aurora queue starvation + a 2026-07-27 outage are the main drags on production throughput

Shortlist of the top items over the ~2 weeks to 2026-07-27; the detailed
write-up is the [2026-07-20 entry below](#2026-07-20). **Nothing is on fire; the
main story is queue starvation, not broken code.**

### 1. Biggest single result -- 80B NaN re-diagnosed (changes the plan)

The 80B production NaN is a **bf16 residual-stream activation overflow, NOT an
optimizer bug** (the old SophiaG-vs-mano debate is moot -- both NaN identically,
mano has no Hessian term). The fp32-activations run trains clean and exposes
true grad_norms of 21K-79K that bf16 was masking to ~5-7. **Only confirmed-clean
config is fp32 mixed-precision-param at TP=4 (~3-5x slower); no 80B job is
queued pending a direction call.** A per-block fp32-residual prototype was built
but still NaNs at dp=192 -- necessary, not sufficient. Guard added:
`--nan-abort-consecutive=5` (the last 512N NaN wasted ~6,100 node-h). This is
**Ask #1 below** and the top open decision.

### 2. CoT-teaching front -- ran end-to-end to a verdict

The chain-of-thought teaching effort (opened 07-20) is now **Stages 0-2
complete**, and it produced a clean, actionable finding:

- **Cold-start CoT-SFT is the accuracy lever; GRPO is not, at 2B.** Two-stage
  SFT (B2: tulu-math -> gsm8k-r1cot) reaches GSM8K **~0.205**, beating every
  single-stage rebuild. Gated GRPO on that base is **drift-proof** (format
  perfected 0.985 -> 1.0, no reward-hacking) but **accuracy stays flat**
  (0.205 -> 0.215 = noise) -- at ~20% GSM8K the correct-rollout density is too
  thin for RL to bite. **Takeaway: invest in the cold-start SFT / a stronger or
  math-heavier base, not more RL, to move 2B accuracy.**
- Enabling infra along the way: the reward-ceiling break (**+168%** from a
  componentized reward vs a flat ~0.25 wall) and Monarch+TorchStore+vLLM
  GRPO+LoRA vendored in-tree with **zero edits to core/experiments-rl**.

### 3. Full-mix SFT -- finished, but the deliverable is checkpoint-900

The 12x-more-tokens full-mix SFT **catastrophically forgot** at 1 epoch
(step-8672: train loss 0.357 but HellaSwag 0.59 -> 0.27). **The deliverable is
checkpoint-900** (base-LM retained, IFEval prompt-strict 0.253). Root cause: LR
2e-5 held above 1e-5 through step ~4350. **Lesson (now a guardrail ask): cap
full-mix SFT at O(1000) steps or drop LR.** More tokens did NOT beat the tiny
metamathqa-729 SFT.

### 4. Production pre-training -- steady, queue-limited

- **2B 256N: COMPLETE** (4.674T tokens, 100%, loss 2.652) -- unchanged; the base
  for CPT/SFT. Final-ckpt eval still blocked on PM.
- **20B 512N: step 6,100** (~614.0B, 13.1%, loss ~2.44); **20B 256N: step 6,000**
  (~302.0B, 6.5%). Both advancing only via **16N capacity-queue bridges** --
  the 256N/512N prod jobs have sat queue-starved for days (genuine node scarcity
  + daily reservations), so a low-contention capacity trickle is what keeps them
  moving.
- **Aurora outage 2026-07-27** killed the in-flight eval + a briefly-running
  512N prod resume (`8696040`, CCL crash) with scheduler code -29. **No
  checkpoints corrupted, no ckpt-dir collisions**; jobs resubmitted, chains
  intact.
- **80B: still blocked** (item 1); no job queued.

### 5. Evaluation -- suite modernized, peer numbers still pending

- New eval-strategy review (07-17): our 7-task commonsense suite is a training
  thermometer but "nearly useless vs modern peers" -- we had **no MMLU and no
  GSM8K**. Both (+ ARC-Challenge 25-shot) are now wired and **backfilling** the
  20B tails; **no AuroraGPT MMLU/GSM8K numbers have landed in the docs yet** (the
  07-27 outage killed the modern-pass/gsm8k half of the backfill mid-run;
  resubmitted). **OLMo-2 (7B/13B)** is the named single peer.
- 20B "no plateau" no longer holds: HellaSwag peaked **0.6339 @ step-4,400**,
  now oscillating 0.60-0.63.

### 6. Charts / docs -- current

Production dashboard, per-chain READMEs, and the all-production overlay are
refreshed to disk truth (20B-512 6,100 / 20B-256 6,000). A W&B-fetch
consolidation (shared `wandb_fetch.py`) fixed a stale-curve bug where the board
and charts had drifted. The overlay now carries the MDS reference's stage 1/2/3
boundary lines (4.674T / 7.064T / 7.771T).

### Appendix A -- 80B NaN: the explicit runs + what they narrow it to

Full write-up: [`2026-07-14-80b-fp32-residual-fix`](../experiments/agpt/aurora/2026-07-14-80b-fp32-residual-fix.md).
Arch: dim=9216, 84 layers, ffn=25600, vocab 256128. The wall is **LBS>1 AND
dp_degree (=NGPUS/TP) > ~186**; the safe corner (TP=4, LBS=1, dp<=186, batch via
GAS) trains clean but can't reach production GBS.

The run ladder (this is what makes it diagnosable):

| Run | Config | Scale | Result |
|---|---|---|---|
| `8574385` | SophiaG | 510N | flat grad_norm ~6.17, step-14 inf, **NaN step-18** |
| `8661293` | **mano** (no Hessian term) | 62N, dp~186 | flat grad_norm ~6.0, **NaN step-17** -- identical signature -> **optimizer-independent** |
| `8537349` | **fp32 mixed-precision-param** (all activations fp32) | TP=4 | **CLEAN** 20 steps; exposes true grad_norms **21K-79K** that bf16 masked to ~5-7 |
| `8671046` | `agpt_80b_fp32res` (per-block fp32 residual add, bf16 GEMMs) | 4N | CLEAN (loss 12.79->9.24) -- proves the add-masking was real |
| `8671243` | `agpt_80b_fp32res` | 64N, **dp=192** | **still NaN step-19** -- per-block fp32 add necessary but NOT sufficient |
| `8673658` | `agpt_80b_fp32res_depth` (fp32 residual across all 84 layers) | 64N, **dp=192** | **still NaN ~step-37** -- full-depth residual ALSO insufficient |

**What the ladder narrows it to (the useful new signal):** the *only* delta
between `8673658` (full-depth fp32 residual, **NaN**) and `8537349` (fp32 params,
**clean**) is the **bf16 compute inside the sublayers** (attention QK^T scores +
FFN SwiGLU intermediate). So the overflow site is **inside a sublayer's bf16
matmul, not the residual accumulation** -- which contradicts the original
"deep residual stream" framing and points at two cheap, standard fixes we have
in-tree but the 80B flavor does NOT use: **attention softcap** (`*_softcap`
flavors exist) and **QK-Norm** (`attention.py` supports `qk_norm`). Ruled out in
code: loss softmax (CE upcasts to fp32), grad reduction (FSDP reduce_dtype=fp32,
loss dp-invariant). No converged 80B checkpoint exists (`80b/README` shows
`[no ckpts]`), so we can change the architecture freely -- no DCP-resume
constraint.

Recommended cheapest-first ladder (details in the doc):
1. **Per-op numerics capture at dp=192** (task #29, wired but paused) -- logs
   per-op max-abs to name the exact op that hits ~3e38 first (attention scores
   vs FFN gate). Cheap (~32N, tens of steps); turns the next step from guess to
   targeted fix.
2. **Enable QK-Norm and/or attention softcap** (near-free, no ckpt-compat cost)
   -- 4N smoke, then the dp=192 wall test (64N) that killed the residual protos.
3. If FFN is the culprit: fp32 **just the FFN** (targeted autocast), far cheaper
   than full fp32-params.
4. Guaranteed interim: **fp32 mixed-precision-param at TP=4** (`8537349`, only
   confirmed-clean, ~3-5x slower) -- start in parallel so 80B isn't idle.

"Wall 2" (separate, scale/init): 256N/dp=768 GPU `NotPresent` init segfault;
2048N SIGSEGV in `set_determinism` at 24,864 ranks; **1024N (dp=3066, `8574386`)
is the untested bracket** that would settle the practical 80B max.

### Appendix B -- SFT above 8N: the 384-rank scale fault (verified reproducer)

Full write-up: [`2026-07-10-sft-2b-gs138650-big-mix-32n`](../experiments/agpt/sunspot/2026-07-10-sft-2b-gs138650-big-mix-32n.md#blocked-384-rank-gpu-page-fault-at-step-1-2-scale-fault-unsolved).
Memory: `project_sft_v2_base_oom_badnode`.

Every 32N (= 384-rank) SFT attempt dies at **step 1-2** with a GPU page fault
during an oneCCL collective (`libccl.so` backtrace):

```
Segmentation fault from GPU ... ctx_id: 5 (CCS) type: 0 (NotPresent), access: 1 (Write) ... aborting
-> rank 221 died from signal 6 (SIGABRT)
```

Evidence (all ruled-out-in-full, not guessed):

- **Deterministic**: jobs `12470336` / `12470338` / `12470339` (+ the afterany
  chain) all die identically at step 1-2. Earlier big-mix interleave attempts
  `12468348` / `12468371` / `12468398` hit the same barrier.
- **Not a bad node**: the fault lands on **rank 221 across two different
  physical nodes** (`x1921c4s0b0n0` AND `x1922c2s3b0n0`); fresh PBS allocation
  per retry, yet it follows the rank -> scale, not hardware.
- **Not bad data / OOV**: full columnar scan of all 53,276,203 rows (2160
  shards) -> global max token id **255998 < vocab 256000**.
- **VERIFIED CLEAN REPRODUCER at 2N**: `_diag_bigmix_2n_len1024.sh`
  (job `12470343`) ran the **same** pretokenized dataset + **same** bsz2/gas8 at
  **24 ranks** for 20 steps clean (loss 1.25->1.12). Identical data+config
  trains at 24 ranks, GPU-faults at 384 -> it is purely a scale fault.
  Scale-sweep harness: `_diag_bigmix_scale_len1024.sh`
  (both under `rl/scripts/sft/`).
- **Base-independent**: moving v2-256n-base -> gs138650 did not dodge it; the
  completed metamathqa-729 SFT dodged it by luck (never hit 384 ranks).

Onset sits near the **same ~186-192 dp boundary as the 80B NaN wall** -- plausibly
the same oneCCL/Level-Zero scale class, so a facility-side fix for one may inform
the other (worth filing them as related).

**Impact**: pins SFT to <=8N (~4x slower) and blocks (a) the full-mix 32N run and
(b) the first SFT on the completed v2-256n base. It's a Level-Zero/oneCCL runtime
fault below our code -- hence the "file an ALCF ticket" ask (attach the 2N clean
repro + the three failing 32N job IDs), or accept 8N as the SFT ceiling.

### Top asks / open decisions (full list in the 2026-07-20 entry)

1. **80B (the big one):** approve starting the slow-but-stable fp32
   mixed-precision-param TP=4 run now (only confirmed-clean path, ~3-5x slower),
   or hold for a full-depth fp32-residual fix? No 80B is training until this is
   decided. (Evidence + a cheaper QK-Norm/softcap path: **Appendix A**.)
2. **2B direction:** the CoT verdict says accuracy lives in cold-start SFT, not
   RL -- do we invest in a stronger/math-heavier base + the stage-2 anneal the
   data-strategy memo recommends (50-100B tokens, LR->0, science/math upsample)?
3. **SFT above 8N:** file an ALCF ticket for the deterministic 384-rank GPU page
   fault, or accept 8N as the SFT ceiling? (Blocks full-mix 32N + v2-256n-base
   SFT. Verified 2N-clean repro + failing 32N job IDs: **Appendix B**.)
4. **Aurora queue starvation:** the prod chains only advance via capacity-queue
   bridges; is escalating the AuroraGPT allocation's queue priority worth an
   ALCF conversation, or do we accept the capacity trickle as the steady state?
5. **2B 512N (83.8%, ~25 days queue-starved):** hold for a slot, or declare the
   completed 256N chain (4.674T, 100%) the 2B deliverable and abandon 512N?

---

## 2026-07-20

### Headline: 80B NaN re-root-caused (bf16 activation overflow, not the optimizer); new clean Polaris A100 20B chain; the bulk of the effort went into RL/GRPO+Monarch on XPU, which seeded a brand-new chain-of-thought teaching front

Covers the ~2 weeks since 2026-07-06. Production held steady (2B base
COMPLETE, 20B advancing, 80B still blocked); the new work is a re-diagnosis
of the 80B wall, a from-scratch Polaris chain, and a large RL push (Monarch
GRPO+LoRA vendored in-tree, a reward-ceiling break, and the first CoT-teaching
stages).

### 1. Production pre-training

**80B -- still the top blocker, but re-diagnosed.**

- NaN re-root-caused 2026-07-14 as a **bf16 residual-stream activation
  overflow, NOT an optimizer bug**: SophiaG (512N, step-14) and mano
  (dp~186, step-17) NaN with the *identical* flat grad_norm ~6.0 signature,
  and mano has no Hessian term -> optimizer-independent (the old
  SophiaG-vs-mano decision is now moot).
- Smoking gun: the fp32-activations run (job 8537349, 20 steps) trains clean
  and exposes **true grad_norms of 21K-79K** that bf16 was silently masking
  down to ~5-7. Loss softmax and grad-reduction both ruled out in code.
- The Llama3-405B-style fix (bf16 GEMMs, fp32 only at residual + norm
  boundaries) trains clean at 4N but **still NaNs at dp=192** (job 8671243,
  step-19) -- per-block fp32 add is necessary but not sufficient.
- Only confirmed-clean config remains fp32 mixed-precision-param at TP=4
  (8537349), ~3-5x slower; no 80B job is currently queued. Operational guard
  added: 80B autoretry now sets `--nan-abort-consecutive=5` (the 512N NaN
  wasted ~6,100 node-h before this).
- Separate "Wall 2" (scale/init): 256N/dp=768 hits a GPU `NotPresent` init
  segfault; 2048N SIGSEGV'd in `set_determinism` at 24,864 ranks; **1024N
  (dp=3066, job 8574386) is the untested bracket** that decides whether 512N
  is the practical 80B max.

**Polaris (A100) -- new, and it just works.**

- New SophiaG/dolma **20B chain on leg 5** (job 7252666): persisted step
  1,300 / live ~1,400, loss **2.31** (from 12.95), ~14.7B tokens over four
  clean 12h legs -- already matching the mature 2B chain's loss (2.36) at
  ~half the tokens. Clean SophiaG convergence at 2B and 20B on A100 is a
  direct contrast to the Aurora 80B bf16 wall.
- Polaris 2B reached step 5,300 / loss 2.36 (~22.6B tokens) and is idle.
  No leg-6 or 2B continuation queued -- afterany +1 chain discipline lapsed
  on Polaris.

**Aurora 2B / 20B.**

- **2B 256N: COMPLETE** -- cont12 (8558531) finished clean at step 92,859 =
  **4.674T tokens (100%)**, loss 2.652. This is the base for CPT + SFT;
  final-ckpt eval still blocked on PM.
- **20B 512N stall broken**: frozen at step-4,400 since 2026-05-29,
  relaunched via native auto-retry (head 8638793 + cont 8638795) -> now
  persisted **step 6,000, loss 2.47, 604.0B tokens (12.9%)**. step-4500 is
  an empty/aborted save; eval re-run of the 4,400->6,000 tail is pending.
- 2B 512N sync chain queue-starved ~25 days at step 38,900 / loss 2.71 /
  83.8%; resubmitting cont10 (8521631) would reset accrued priority.

**All Aurora production chains overlaid** (loss / TPS-per-GPU / MFU vs tokens):

![All-production training overlay](../../live/figures/all_production_training.svg)

**New Polaris (A100) 20B chain** (loss + diagnostics, 128N, dolma/SophiaG):

![Polaris 20B production](../../live/chains/polaris/figures/production_20b_polaris_128n.svg)

See: [production rollup](../../live/dashboard.md) ·
[2B](../../live/chains/agpt/2b/README.md) ·
[20B](../../live/chains/agpt/20b/README.md) ·
[80B](../../live/chains/agpt/80b/README.md) ·
[Polaris](../../live/chains/polaris/README.md).

### 2. Evaluation

- New **eval-strategy review** (`evals/eval-landscape-2026-07.md`,
  2026-07-17): our 7-benchmark commonsense suite is a good from-scratch
  thermometer but "nearly useless for positioning vs modern peers" -- HF
  retired all six OLL-v1 tasks in June 2024 for saturation, and we had **no
  MMLU and no GSM8K at all**.
- MMLU (5-shot), GSM8K (5-shot), ARC-Challenge (25-shot) added via a
  mixed-few-shot loop and **backfilling now** -- but no AuroraGPT
  MMLU/GSM8K/ARC-C numbers have landed in the docs yet.
- **OLMo-2 (7B/13B)** named the single correct peer (same olmo-mix family,
  OLMES suite). Today only HellaSwag overlaps cleanly: 20B ~0.61 vs OLMo-2
  0.838/0.864 -- large but expected at 604B tokens (13%) vs peers' 9-11T.
  Targets: MMLU 63.7, GSM8K 67.5.
- 20B "no plateau" headline no longer holds: HellaSwag peaked **0.6339 at
  step-4,400** then oscillates 0.60-0.63 (0.6086 at step-6,000). step-4,400
  is best-per-benchmark (ARC-Easy 0.6932, PIQA 0.7650). IFEval and
  BBH/GPQA/MATH/HumanEval deferred to instruct-tuning / more tokens.

**All-production eval overlay** (HellaSwag / ARC / Winogrande vs tokens) and
the 20B eval detail:

![All-production evals](../evals/figures/all_production_evals.svg)

![20B eval overview](../evals/agpt/20b/figures/eval_overview.svg)

See: [eval index](../evals/README.md) ·
[eval-landscape review 2026-07](../evals/eval-landscape-2026-07.md) ·
[20B evals](../evals/agpt/20b/README.md).

### 3. RL / GRPO / Monarch (Intel XPU) -- the biggest recent effort

- **Reward ceiling broken (+168%):** a componentized shaped reward (format
  0.2 + completeness 0.2 + order 0.6) hit **0.667** re-scored on the
  *identical* char-ratio metric vs the tuning cluster's flat 0.237-0.249
  (job 12471056, 100 steps), validated as not a scoring artifact. The
  beat-v5 sweep first proved **no config lever** (LR, LoRA rank, group
  count) beats the ~0.25 mean-reward ceiling -- the reward *shape* was the
  wall.
- **Monarch+TorchStore+vLLM GRPO+LoRA vendored in-tree** as a thin ezpz
  overlay with **zero edits to experiments/rl or core** (all XPU compat via
  runtime monkeypatches); job 12471049 fires train steps at ~1765 tok/s,
  reward_mean 0.1607, matching the fork-based v5 baseline.
- **agpt-2b gibberish root-caused to bf16** in the vLLM path (bare-vLLM
  bf16 = gibberish, fp32 = coherent); fix is `--generator.model-dtype=float32`
  (~30% slower but correct). Qwen3-0.6B GRPO+LoRA also reproduced on Sunspot
  XPU crossing the USM/PMIx wall (~3256-3273 tok/s).
- v5 GRPO+LoRA on agpt-2b shows a clean learning curve **0.167 -> 0.268**
  (crossing 0.25 at ~1978 rollouts); the lever that mattered was task
  difficulty, not LR.
- The 2026-07-06 multi-trainer-node "desync hang" closed as **two ordinary
  bugs** (oneCCL transport default + a mis-scoped AVG->SUM monkeypatch);
  confirmation job 12470083 (3N, 24 ranks) ran all 8 steps to
  accuracy_reward 0.375.
- **Caveat:** everything runs on **TCP fabric, not Slingshot CXI** (~176
  s/step at 3N); next lever is a per-group transport split. vLLM-XPU TP>1
  (multi-tile server) still unexercised.

**Ceiling-attack** (shaped reward breaks the ~0.25 wall, +168%) and the
**beat-v5 tuning sweep** (no config lever beats the ceiling):

![GRPO ceiling-attack](../../live/chains/rl/grpo/aurora2b/charts/ceiling-attack.svg)

![GRPO beat-v5 sweep](../../live/chains/rl/grpo/aurora2b/charts/beat-v5-sweep.svg)

**sum_digits arithmetic GRPO** (8N, accuracy 0.09 -> 0.76 over 1000 steps):

![GRPO arithmetic curves](../../live/chains/rl/grpo/aurora2b/sft_arithmetic/charts/grpo-curves.svg)

See: [RL hub](../../live/chains/rl/README.md) ·
[GRPO index](../../live/chains/rl/grpo/README.md) ·
[ceiling-attack](../../live/chains/rl/grpo/ceiling-attack.md) ·
[beat-v5 sweep](../../live/chains/rl/grpo/beat-v5-sweep.md) ·
[Monarch](../../live/chains/rl/monarch.md) ·
[TRL / cross-node vLLM](../../live/chains/rl/trl.md).

### 4. Chain-of-Thought teaching -- new front (opened 2026-07-20)

- New R1-style plan (`production/rl/plans/cot.md`): **cold-start CoT-SFT**
  (teach the `<think>...</think><answer>\boxed{}</answer>` envelope) ->
  **GRPO-RLVR** (reason well). RL-only R1-Zero explicitly rejected as
  primary (kept as a falsifiable control); no teacher model needed.
- **Stage 1 works**: envelope emission jumped **0.00 -> 0.955** with no
  accuracy regression (CoT accuracy 0.15 -> 0.16), from a tiny 16-step / 8N
  / 38s run on gsm8k's own re-wrapped rationales (base = checkpoint-900-hf).
- Stage 0 built the first generation-based chat-templated GSM8K-CoT eval
  (format hit-rate and CoT accuracy scored *separately*, answer read only
  from the `<answer>`/`\boxed{}` span, fp32 vLLM).
- Stage 2 applies the ceiling-attack lesson directly: `gsm8k_reason.py` uses
  three additive rewards (think_format 0.2 + answer_extractable 0.1 +
  answer_correct 0.7) instead of one saturating binary exact-match.
- Stage 2 launchers built and **smoke-debugged**: an earlier xnode run hit
  XPU `OUT_OF_RESOURCES` at step-13 (trainer peak mem = num_gen x
  (prompt+completion), worsened by a never-EOS cold-start ckpt);
  diagnosed-fixed at HEAD (commit 46f99a568) via num_gen 4, completion cap
  512, expandable_segments. **A clean completed Stage 2 run is not yet
  confirmed.** (No chart yet -- Stage 2 curves land once a clean run
  completes.)

See: [CoT-teaching plan](../../live/chains/rl/plans/cot.md).

### 5. CPT / SFT

- **Full-mix SFT finished but catastrophically forgot** at 1 epoch (step-8672:
  loss 0.357 / mtacc 0.902 but HellaSwag 0.59->0.27, ARC-Easy 0.69->0.30) --
  the **deliverable is checkpoint-900** (IFEval prompt-strict 0.253, base-LM
  retained, strong GRPO start). Root cause: LR 2e-5 held above 1e-5 through
  step ~4350. Lesson: cap full-mix SFT at O(1000) steps or drop LR. 12x more
  tokens did NOT beat the small metamathqa-729 SFT.
- **CPT pilot degraded benchmarks** via two modes (ratio-independent
  HellaSwag drop from LR re-warm shock; ratio-dependent ARC-Easy bleed --
  dolmino-100 0.619->0.547 but olmo50-dolmino50 held ~0.61). Relaunched as
  gentle-LR (2e-6 constant) olmo50-dolmino50 (umbrella 8663177) + a
  dolmino-100 gentle-LR arm (8662867). An 18-agent data-strategy memo
  recommends a **stage-2 anneal now** (50-100B tokens, LR->0, science/math
  upsample); the 2B is ~115x past Chinchilla-optimal.
- **SFT capped at 8N**: 32N/384-rank deterministic GPU page fault (bisect:
  2/4/8N clean, 12/16/32N crash, near the ~186-192 dp boundary) -- blocks
  both the full-mix 32N run and the first v2-256n-base SFT. A /lus/tegu
  disk-full incident (2026-07-13) cost ~1,150 steps of recompute.

**CPT pilot: loss beats the plateau but downstream eval DEGRADES** (the
loss/eval divergence is the whole story):

![CPT loss](../../live/chains/cpt/figures/cpt_loss.svg)

![CPT downstream eval](../../live/chains/cpt/figures/cpt_eval.svg)

**Full-mix SFT eval** (base-LM collapse at step-8672 -> ckpt-900 is the
deliverable):

![Full-mix SFT eval curves](../../live/chains/sft/agpt/2b-mds/tulu_math_uc_mix_full/evals/charts/eval-curves.svg)

See: [CPT sweep](../../live/chains/cpt/README.md) ·
[SFT index](../../live/chains/sft/README.md) ·
[full-mix SFT evals](../../live/chains/sft/agpt/2b-mds/tulu_math_uc_mix_full/evals/README.md) ·
[data-strategy memo](../../notes/data-strategy-after-olmo-mix-2026-07.md).

### 6. Development / infrastructure

- **5 upstream syncs (64th-70th)**, two breaking: 70th (MoE #3859
  sibling-experts) needed a 5-file structural replay (moe.experts ->
  moe.routed_experts.inner_experts); 69th (#3923 moved Linear to
  common/linear.py) needed import fixes. Both caught by `sync_smoke`,
  re-smoked green.
- **macOS/CPU single-device training** support landed (TORCH_DEVICE=cpu
  overlay, no-op on XPU/CUDA); the old "Cannot import config_registry" was a
  masked missing-triton import.
- **Native ezpz auto-retry umbrella** (`smoke_multi_autoretry.sh`) replaces
  legacy `failover_lib.sh`, which lost 3/4 chains to blind bad-node swaps.
- Recurring drags: the 384-rank GPU page fault (pins SFT to 8N),
  XPU-broken async-ckpt default on torch 2.13, an auto-retry classifier gap
  (bad-node SIGSEGV misread as walltime -> no spare-swap), and persistent
  **Aurora at_queue starvation** (3,487 nodes free on 07-09 yet the 1536N
  umbrella eligible 70h+ without a slot). IPC-handle cache leak fixed via
  CCL_ZE_CACHE thresholds.
- Tooling: `refresh_all.sh` coverage gap fixed (cpt/sft/grpo/2b-mds plotters
  wired in + a stale-doc coverage audit); training dashboards generalized to
  SFT + GRPO with new multi-run reward overlays.

### Decisions / asks for the team

1. **80B fix path:** commit to fp32 mixed-precision-param at TP=4 as an
   interim production path (only confirmed-clean, ~3-5x slower), or hold for
   a full-depth fp32-residual fix (per-block prototype still NaNs at dp=192)?
   Start the slow-but-stable run now, or wait?
2. **80B Wall 2:** who investigates the 256N/dp=768 init GPU `NotPresent`
   segfault, and do we run the 1024N (dp=3066, 8574386) bracket to settle
   the practical 80B max? Also fix the auto-retry classifier so a bad-node
   SIGSEGV triggers a spare-swap instead of a walltime stop.
3. **Unblock SFT above 8N:** file an ALCF ticket for the deterministic
   384-rank oneCCL/GPU page fault (onset ~96-144 ranks), or accept 8N as the
   SFT ceiling? This blocks the full-mix 32N run and the v2-256n-base SFT.
4. **CPT vs data-strategy memo:** wait for the gentle-LR dolmino retry
   (8662867) before spending the full ~2.391T olmo50-dolmino50 budget? Does
   the memo's "anneal now, LR->0, science/math upsample" supersede or fold
   in? Assign an owner for stage-2 upsample weights.
5. **SFT guardrails project-wide:** cap full-mix SFT at O(1000) steps (or
   lower LR); enforce keep-latest-N COMPLETE ckpts + project-quota
   monitoring to prevent another /lus/tegu disk-full drain.
6. **Aurora 2B 512N (83.8%):** hold cont10 (8521631) for a slot (deploy
   ezpz PR #160 first), or declare the completed 256N chain (4.674T, 100%)
   the 2B deliverable and abandon 512N? (Resubmitting resets ~25 days of
   priority.)
7. **Peer scorecard:** confirm OLMo-2 (7B/13B) as the single primary peer;
   present the commonsense suite as a training dashboard only; defer
   competitive ranking until MMLU/GSM8K/ARC-C backfill.
8. **RL next milestone + promotion:** prioritize the CoT GRPO-RLVR chain
   (confirm a clean Stage 2 run on Aurora/Sunspot); resubmit the un-converged
   10N ckpt-900 arithmetic GRPO (0.74) for convergence? Promote any
   shaped-reward/v5 LoRA adapters (none promoted yet)?
9. **Housekeeping:** queue Polaris 20B leg-6 behind 7252666; schedule the
   blocked-on-PM eval of the completed Aurora 2B 256N final ckpt
   (step-92,859); fix the 2B autoretry async-checkpoint default (XPU-broken
   on torch 2.13); bring `journal.md` current (tail is 2026-04-25, no CoT
   work recorded).

---

## 2026-07-06

### Headline: 2B base closed out (eval) and three new fronts opened -- CPT sweep, multi-node GRPO solved, first SFT on the completed v2 base

The week since 2026-06-29 resolved the entire post-PM action list and added
three new deliverables. Full detail in the
[~10-day summary](../summaries/2026-07-06.md).

### 1. 2B base eval closeout (resolves 2026-06-29 action #1)

The final 2B checkpoint (**step-92,859 = 4.674T tokens**) was converted and
evaluated. Tail backfill (job `8638581`, 14 ckpts step-86,500..92,859):
**HellaSwag_norm 0.561, ARC-Easy 0.651, PIQA 0.733** at the final step, flat
over the last ~635B tokens (also resolved the old ARC-Easy ~0.59 artifact -- a
fresh eval reads ~0.65). The **dead-flat tail is the motivation for CPT** (item
2), not more base tokens. Table:
[`evals/agpt/2b`](../evals/agpt/2b/README.md).

### 2. 2B continued-pretraining (CPT) mixing-ratio sweep launched (resolves action #2)

With the base plateaued, launched a CPT pilot that forks the completed base and
continues on new data blends.

- **Fork mechanism:** `--checkpoint.initial-load-path` (weights-only) off the
  read-only step-92,859 base in a separate clone; distinct CKPT_DIRs so the base
  is never overwritten.
- **3 renormalized mixes:** `dolmino-mix-1124` (100), `olmo50-dolmino50`,
  `olmo25-dolmino75`. **256N pilots** `8638977` (dolmino-100) + `8638978`
  (olmo50/50) + afterany conts; GBS held 6144 (LR calibrated), LR 2.28e-5
  re-warm 200 + decay 0.8, ~300B tokens.
- **Fork verified** (smoke `8638933`): loads step-92,859 clean (0 mismatch),
  loss 7.2 -> 5.9 -- dolmino is a real distribution shift, so the CPT signal is
  live. Report:
  [`20260701-2b-cpt-olmo-dolmino-sweep`](../experiments/agpt/aurora/20260701-2b-cpt-olmo-dolmino-sweep.md).
- **Bug flagged:** the 2B autoretry script defaults async checkpointing, which
  is XPU-broken on this torch (`new_group(gloo)` -> `No backend type for xpu`).
  Workaround `CHECKPOINT_ASYNC_MODE=disabled` (20B already defaults disabled); a
  2B-default fix is TODO.

### 3. 80B: convergence NaN + 2048N init crash (resolves actions #2/#4)

- **Convergence run @ GBS=6144: all 3 optimizers NaN** at finder LRs past
  warmup -- a corner instability, not a tuning problem (only 80B cliffs; 2B/20B
  never do). Report:
  [`2026-06-30-80b-convergence`](../experiments/agpt/sunspot/2026-06-30-80b-convergence-gbs6144.md).
- **Post-PM production launch:** 512N + 1024N ran; **2048N SIGSEGVs in
  `set_determinism`** at 24,864 ranks (init-time scaling wall, same class as the
  known 1024N crash). [`80b/README`](../../live/chains/agpt/80b/README.md).
- **Still open (team decision):** 80B optimizer -- SophiaG @1e-6 (launched) vs
  **mano @3e-6** (wider stability margin for a long unattended run); and whether
  to keep pushing scale above 512N given the 2048N init crash.

### 4. Multi-trainer-node GRPO on XPU -- SOLVED (new)

Multi-*trainer*-node GRPO (FSDP across 2+ nodes) now works: job `12470083`
(3N = 1 server + 2 trainer, 24 ranks) ran **8/8 steps**, loss -0.028,
**accuracy_reward 0.375**, real cross-node FSDP grad reduce-scatter. The
2026-07-01 overnight "desync hang" was **two ordinary bugs, not a desync**:
(a) the `--no-oneccl-tcp-kvs` runs left `CCL_ATL_TRANSPORT` at oneCCL's `mpi`
default, which SIGSEGVs forming the cross-world weight-sync PG; and (b) the
AVG->SUM FSDP patch rebound the wrong `from`-import name so it never ran. Both
fixed. Root cause + all-rank stack evidence:
[`2026-07-06_multinode-grpo-root-cause`](../../live/chains/rl/2026-07-06_multinode-grpo-root-cause.md);
status flipped to works in
[`grpo-on-xpu-status`](../../live/chains/rl/README.md).

### 5. First SFT on the completed v2 2B base (new)

Every prior production SFT (and thus all GRPO) used the older
`AuroraGPT-2B-sophiag-gs138650` base. Launched the first SFT on the **completed
v2 256N base** (step-92,859): converted DCP->HF on Aurora, transferred to
Sunspot, ran the proven `tulu_math_uc_mix` recipe. 2N smoke green; 32N full run
`12470088` in progress (loss 1.35 -> 0.95, mean_token_accuracy 0.68 -> 0.757,
checkpoint-100 saved, fresh CKPT_DIR confirmed). Trajectory:
[`sft/agpt-2b-v2-256n`](../../live/chains/sft/agpt/2b-v2-256n/tulu_math_uc_mix/README.md).
**Sets up an eval question:** does SFT on the completed base beat SFT on the
older lineage? (head-to-head once it finishes.)

### 6. Upstream syncs 60th-64th absorbed

Four syncs in the summary window (60th-63rd) plus the **64th** (2026-07-06,
6 commits: graph_trainer + a triton-upgrade loss asset, no replay). The 64th's
`sync_smoke.sh` caught a **latent bug unrelated to the merge**: the NaN-abort
guard read `config.training.nan_abort_consecutive` but the field lives on the
top-level trainer `Config`, so `trainer.train()` `AttributeError`'d on EVERY
agpt/moe run (SFT/GRPO unaffected -> unnoticed). Fixed. See
[`upstream-sync`](../../upstream-sync.md).

### 7. Infra

Native `ezpz launch --auto-retry` umbrella replaces the legacy `failover_lib.sh`
wrapper; 20B 512N chain recovered; blendcorpus index-race fixed at source
(atomic-rename). Detail in the summary.

### Decisions / asks for the team

1. **80B optimizer:** keep SophiaG @1e-6, or switch the base to mano @3e-6 for
   the wider margin? (Easy to switch before the brackets run.)
2. **80B scale ceiling:** 2048N crashes at init; cap production at <=1024N until
   the `set_determinism` init-scaling issue is understood?
3. **2B CPT direction:** once the dolmino/olmo mixes report, which wins ->
   promote to the CPT base?
4. **SFT base:** eval v2-base SFT (`12470088`) head-to-head vs the gs138650 SFT
   to decide the canonical instruct checkpoint.

---

## 2026-06-29

### Headline: 2B v2 256N pre-training is COMPLETE (4.674T tokens, 100% of target)

The 2B 256N v2 chain reached its full **4.67T-token budget**: final
checkpoint **step-92,859 = 4.674T tokens (100.0%)**, final loss **2.652**,
grad_norm ~0.056, ~14% MFU. Chain head `8558531` (cont12) finished a clean
exit-0 ~10.2h run on 2026-06-29 03:03 UTC. This is the first agpt model to
finish the full v2 (fp32-master) base pre-training run end to end.

**Discussion / decisions for the team:**
- **Eval the final 2B checkpoint.** Convert step-92,859 DCP -> HF and run the
  full lm-eval suite (ARC-E/C, HellaSwag, Winogrande, etc.) so we have the
  end-of-pretraining scorecard. Blocked until Aurora returns from PM (lm-eval
  needs compute). Queue it first thing.
- **What's next for 2B?** Options: (a) start CPT (continued pre-training) on
  additional tokens with the constant-LR config we just built, (b) SFT, (c)
  freeze it as the reference base. The constant-LR (decay_ratio=0) plumbing is
  ready -- see the 80B launch item below.

### 80B production launched at SophiaG / constant-LR / scale brackets

Submitted the first real 80B v2 production runs (queued, will start post-PM):
**512N + 1024N + 2048N simultaneously**, SophiaG @ LR=1e-6, **constant LR after
warmup (no decay)** for the planned CPT regime, validator ON (95/5 split).

- **Optimizer/LR is the discussion point.** Per the 2026-06-27 production-batch
  LR-finder, AdamW @ GBS~6144 is on a NaN cliff (LR=1e-6 past it); the finder's
  safest pick is **mano @ ~3e-6**, with SophiaG @ ~1e-6 as the lowest-loss but
  narrow-band alternative. We launched **SophiaG @ 1e-6**. Worth a team
  decision: stick with SophiaG, or switch the 80B base to mano for the wider
  stability margin on an unattended multi-T-token run?
- **Scale is unvalidated above ~512N.** 1024N (12,288 ranks) is documented to
  crash at init (`set_determinism`); 2048N (24,576 ranks, dp_degree~6138) is 2x
  that and untested. The 512N bracket is the safety net; submitting all three
  is itself the scaling experiment. Expect possible init crashes at 1024/2048N.
- A 4N pre-check confirmed the config wiring + clean SophiaG descent before
  committing the big allocations.

### Infra fixes this week (all landed + pushed)

- **blendcorpus cold-cache index race FIXED** (was blocking any fresh-CKPT_DIR
  80B run). Root cause: at TP>1 the non-rank-0 readers raced rank-0's index
  write. Fixed at source in `saforem2/blendcorpus` -- atomic writes + poll
  (`041d015f`) + a TOCTOU follow-up (`1f7e9c0`). Confirmed end-to-end: cold 4N
  TP=4 80B now builds the index and trains with 0 EOFError. **80B no longer
  needs a manual cache prewarm.**
- **Validator at 80B TP=4: the "CCL deadlock" was a phantom** -- 3 unrelated
  bugs (cold-cache mmap race, a training-side barrier stall, a `loss_fn`
  tuple-unpack crash), all fixed. Validation CONFIRMED working at TP=4
  (job 12469784). `VALIDATOR_ENABLE=1` is now safe; 95/5 split is the default.
- **Checkpoint-resume incident (recovered).** A clone `git pull` advanced the
  prod clones past an optimizer state-dict format migration (#3623/#3269,
  nested->flat), breaking DCP resume. Recovered by pinning clones pre-#3623;
  resume verified. The clones stay pinned -- a nested->flat migration shim is
  hard/risky and not worth it (chains resume fine pinned).
- **walltime-aware checkpointing** added to the ezpz trainer (force a final
  ckpt before walltime) + a job-absolute-deadline fix so it survives failover
  retries.

### 20B 256N status

Running through the PM boundary; finished clean at **step-2,100 = 105.7B tokens
(2.3%)**, loss **2.85**, ~21.8% MFU. Resumes post-PM via `8558549`.

### Going into the PM maintenance (2026-06-29 06:00 -> 07-01 03:30 UTC)

Both 256N chains exited with valid checkpoints (2B step-92,859, 20B step-2,100);
all continuations + the 6 80B jobs are queued to resume/start post-maintenance.
Nothing at risk.

---

## 2026-05-04

### bf16-master RMSNorm-freeze fix is producing real downstream gains

Quick recap for context. All v1 production runs (2B / 20B / 80B) had
`training.dtype = bfloat16`, which kept the master parameter copy in
bf16. RMSNorm.weight initializes to 1.0; the bf16 ULP at 1.0 is
~7.8e-3 and per-step optimizer updates for those parameters are ~1.6e-5,
so every update rounded to zero and **norm weights never moved from
1.0 for the entire run**. Other parameters (linears, embeddings)
initialize at much smaller scales and updated fine, so training loss
curves looked plausible — the bug only became visible at eval time.

Default flipped to `float32` on 2026-04-30 and **v2 production was
restarted from scratch** rather than continuing from the bf16-tainted
checkpoints. The v1-vs-v2 lm-eval comparison is the smoking gun:

- 2B ARC-Easy climbed **0.277 → 0.429** over 100B tokens on v2,
  vs v1's flat ~0.27 across **450B** tokens.
- **+19.8pp ARC-Easy / +15.4pp HellaSwag at 503B tokens** vs v1's
  flat baseline.
- v1's flat trajectory across 450B+ tokens is the qualitative
  signature of the bug — a model with frozen normalization cannot
  improve on what lm-eval measures, no matter how much data it sees.
- 2B eval comparison:
  [`docs/evals/agpt/2b/`](../evals/agpt/2b/README.md);
  20B eval comparison:
  [`docs/evals/agpt/20b/`](../evals/agpt/20b/README.md);
  full diagnosis + cross-linked evidence:
  [`docs/guides/training-dtype-bf16-norm-freeze.md`](../../reference/guides/training-dtype-bf16-norm-freeze.md).

### Production status

- **2B 512N canonical chain** (`8460301 → 8463626 → 8463627`):
  step **5,073**, loss **2.97**, **510B tokens / 10.9% of 4.67T
  target**. Continuation 8463627 queued, follow-up 8466847 held on
  `afterany:8463627`. Trajectory page:
  [`docs/production/agpt/2b/n512/`](../../live/chains/agpt/2b/n512/README.md).
- **20B 512N canonical chain** (`8460302 → 8463628`): step **~862**,
  loss **~3.47**, MFU ~17.8%. 8463628 currently running near
  walltime; 8466848 held on `afterany:8463628`. Trajectory page:
  [`docs/production/agpt/20b/n512/`](../../live/chains/agpt/20b/n512/README.md).
- **20B 256N (8463659):** ran 9h walltime then **NODE_FAIL after
  step 364** (loss 4.61, 18.3B tokens). `shepherd died from signal 9`
  on `x4406c6s7b0n0`, PBS exit -20 — same recurring Aurora bad-node
  failure mode as 8459818 / 8460301. **step-300 ckpt saved cleanly,
  resumable.** Throughput on this run was bouncing 21-410 TPS
  depending on flare contention (1-20% MFU). Trajectory page:
  [`docs/production/agpt/20b/n256/`](../../live/chains/agpt/20b/n256/README.md).
  Are these recurring `signal 9` crashes being tracked anywhere?
  They've now killed three long-walltime jobs across three different
  nodes — worth raising with ALCF support if not.

### 80B blocker

- 80B `compile + AC + TP=2` still hits the
  `tensors_saved_with_vc_check` AOT autograd assertion (`DeviceMesh`
  leaks into saved-for-backward tensors). Toy minimal repro doesn't
  fire — bug needs the real `Module.parallelize` + `LocalMapConfig`
  path that torchtitan uses.
- **2026-05-03:** added `agpt_50b_wide` (dim=9216, 48 layers, ~48B
  params,
  [`98a02d04`](https://github.com/saforem2/torchtitan/commit/98a02d04))
  as a smaller bisect target. 2N + torch 2.10 smoke ran 10/10 steps
  cleanly. Concluded "bug is depth-sensitive — does NOT reproduce at
  48 layers." That conclusion turned out to be wrong (see below).
- **2026-05-05 correction:** ran a proper three-config bisect on torch
  2.13 (job 12465952 4N + job 12465962 2N). All three configs
  (`agpt_50b_wide` 48L, `agpt_70b_wide` 72L, `agpt_80b` 84L) **crash
  identically** with the same assertion. Smallest tested:
  `agpt_50b_wide` on 2N takes ~30s to crash. **Bug is
  torch-version-sensitive, not depth-sensitive.** The May 3 result was
  a torch-2.10 artifact (the failing assertion in
  `_AutogradSavedState.save_from_forward` likely doesn't exist or
  isn't reached on the older AOT-autograd code path). Working repro
  bracket: torch 2.10 (any depth) ✓ → torch 2.13 (every depth tested) ✗.
- Workaround in the meantime: `compile=OFF` for any 80B-family config
  on torch 2.13, OR pin to torch 2.10 for those configs.
- **Working v2 80B path validated 2026-05-05** (job 12466025, 4N
  smoke): `agpt_80b @ TP=2, AC=full, compile=OFF, AdamW LR=1e-6,
  fp32-master` on torch 2.13. Loss descended **12.98 → 10.46** over
  20 steps, MFU steady at **~17.8%** (matches v1 compile-on baseline),
  memory peak **88.94%** with ~7 GiB headroom. Production setup just
  needs to add the 200-step linear warmup the 2B/20B v2 configs use,
  then it's ready to launch.
- Initial toy repro (legacy `parallelize_module` — does NOT fire,
  needs the new sharding API):
  [`docs/upstream-issues/repro_devicemesh_in_saved_tensors.py`](../../outbound/upstream-issues/repro_devicemesh_in_saved_tensors.py).

### Open work I'm holding

- **Validation loss wiring:** blendcorpus's existing val split (5%
  slice) is now plumbed through `EzpzValidator` (subclass that also
  fixes the TP loss-reporting bug — see "Other notes" below). Default
  `enable=False` so production isn't disturbed. Smoke test not yet
  done. Do we want held-out NLL on production runs, or are downstream
  lm-eval scores at checkpoint cadence the right signal?
- **80B production** — still has open issues from prior sessions
  (LR=1e-6 stable but bad-node Gloo timeout crash at step 51).
  Worth flagging if 80B production is on the agenda.

### Other notes

- **TP loss-reporting bug found, fix filed upstream + locally
  workaround.** Upstream `_dist_reduce` change
  ([`pytorch/torchtitan@1786292d`](https://github.com/pytorch/torchtitan/commit/1786292d),
  2026-04-27) skips the cross-batch `all_reduce` when `loss` is a
  DTensor on a mesh orthogonal to `loss_mesh`, so reported loss on
  any TP > 1 run is `true / dp_world_size`. **No current production
  runs are TP > 1**, so no live dashboards are affected — but if/when
  we restart 80B production at TP=2, the historical 80B v1 W&B traces
  show `loss / 1536`. Gradients/optimizer steps were unaffected; only
  the printed value was wrong. Fix filed at
  [pytorch/torchtitan#3204](https://github.com/pytorch/torchtitan/pull/3204)
  (mergeable, awaiting maintainer review). Local ezpz workaround in
  [`a24ed2e1`](https://github.com/saforem2/torchtitan/commit/a24ed2e1)
  + [`a0b9b13d`](https://github.com/saforem2/torchtitan/commit/a0b9b13d).
  Full diagnosis:
  [`docs/guides/loss-reporting-tp-dist-reduce.md`](../../reference/guides/loss-reporting-tp-dist-reduce.md).

### Action items

- [ ] (Sam) Smoke-test `EzpzValidator` end-to-end on `agpt_2b` once
      consensus on whether to enable val.
- [ ] (Sam, blocked on review) Push for review on
      [pytorch/torchtitan#3204](https://github.com/pytorch/torchtitan/pull/3204).
- [ ] (?) Volunteer to build LocalMapConfig-based minimal repro for
      the 80B compile + AC + TP=2 crash.
- [ ] (?) Decide cadence for held-out validation loss on production.
- [ ] (?) Decide whether to file an ALCF support ticket for the
      recurring `shepherd died from signal 9` NODE_FAIL pattern
      (jobs 8459818, 8460301, 8463659 — three crashes, three
      different nodes). Resume 8463659 from step-300 in the meantime?
