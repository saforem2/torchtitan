# Session resume notes -- 2026-08-21

> [!NOTE]
> **Status notes added 2026-08-31. The cluster-state table in section 2 is a
> 2026-08-21 snapshot and is NOT current.** The page is kept as written: the
> `initial_load_path` precedence finding in section 4 is the reusable
> result, and the dashboard restart in section 3 is still the procedure.
>
> - **Section 2 is superseded.** `8769730` is no longer the pending
>   production cycle. Umbrella `8773440` ran on 2026-08-26 (5h13m of a 12h
>   slot, three of five seats trained) and **nothing has trained since**.
>   Its successor `8784460` has been queued roughly 120 h at
>   `score_boost = 0`; Aurora went into maintenance 08-31 14:00 UTC until
>   Tue 04:00. Ticket:
>   [`ops/alcf-ticket-8784460-not-scheduling-20260830.md`](ops/alcf-ticket-8784460-not-scheduling-20260830.md).
>   Current chain heads: 2B-512 **COMPLETE** at step 46,429 (2.68687,
>   4.674T tokens), 2B-256 **COMPLETE** at step 92,859, 20B-512 at step
>   10,600, 20B-256 at step 12,000. See
>   [`production/README.md`](production/README.md).
> - **Section 7 (re-arm monitors) is moot** -- the job it names, `8769730`,
>   is not the one to watch.
> - Section 4 is already marked RESOLVED and needs nothing; its `c2131f3f2`
>   fix and the "grep the trainer console, not the `.o`" lesson both stand.

Written before a Claude session restart. Nothing here requires the old session;
every long-running thing lives on the cluster (PBS) or is a one-line restart.

## 1. Nothing is lost by restarting

All monitors were **local watchers**, not workers -- they polled `qstat` and
tailed logs. Killing them stops the notifications, not the work. Every PBS job
below keeps running regardless.

The only local process worth restarting is the dashboard web server.

## 2. Cluster state (as of 2026-08-21 ~12:45 UTC) -- SUPERSEDED, see the note above

| job | state | what it is |
|---|---|---|
| `8769730` | **Q** | umbrella successor, 2098 nodes. Carries BOTH the 20B constant-LR fix and the per-seat RoPE fix. Waiting on `at_queue` fencing (0 prod-reachable free nodes); predecessor waited ~44 h. |
| `8771637` | **done, FAILED** | t4 converted-seed smoke -- see section 4 |
| `8687863`, `8752939`, `8752824`, `8756071`, `8756072` | H | older held continuations, predate current fixes |
| `8766066` | Q | 80b-dp-bracket, unrelated |
| `8747000` | Q | daos-discovery, unrelated |

Predecessor `8764675` finished cleanly (`Exit_status=-29`, walltime 12:00:31):
20b-512 -> step 10,699 (ckpt 10600), 20b-256 -> step 11,800 (ckpt 11800). Its
three 2B seats never started.

## 3. Restart the dashboard

```
cd torchtitan/experiments/ezpz/utils
nohup python3 prod_dash_web.py --port 8712 > /tmp/pdweb.log 2>&1 < /dev/null &
# -> http://127.0.0.1:8712
```

Loopback only by design. It self-refreshes (browser 30 s / server 60 s /
cluster backbone 1 h), so no manual refresh is needed. NOTE: the server holds
`prod_dash.py` in memory -- after a code change it must be restarted, not
refreshed.

## 4. RESOLVED: t4 smoke `8771637` -- precedence, not a dropped flag

Ran through the umbrella launcher (`MULTI_ONLY=4 MULTI_NNODES_OVERRIDE=4`) and
still died with the same pre-existing error:

    AttributeError: 'dict' object has no attribute 'mul_'

An earlier revision of this section claimed the `--checkpoint.initial-load-path`
flag never reached the trainer. **That was wrong.** It reached the trainer
intact; torchtitan then declined to use it:

    [W] components/checkpoint:686: checkpoint.initial_load_path is provided but
        the checkpoint.folder exists. Checkpointer will use the checkpoints
        from the checkpoint.folder .../constlr-from9500.
    [I] components/checkpoint:706: Loading the checkpoint from
        .../constlr-from9500/step-9500

`initial_load_path` SEEDS A FRESH CHAIN; it never overrides a resumable
checkpoint already sitting in `checkpoint.folder`. t4's real ckpt dir holds the
OLD-format step-9500, so the smoke re-tested the very checkpoint it was meant
to bypass. The hypothesis in the earlier revision was right; the evidence cited
for it was not.

**Why the wrong reading happened, and how to avoid repeating it:** the grep was
against the umbrella's `.o` file, which carries only the wrapper's stdout. The
resolved launch line and that warning live in
`logs/multi-autoretry-<jobid>/trainer-0-*.console.log`. Always grep the
trainer console, not the `.o`.

FIXED in `c2131f3f2`: the smoke now rewrites field 5 (ckpt_dir) to an empty
scratch dir as well as field 8, symlinks the warm blendcorpus index cache
(field 5 also drives `data-cache-path`, and the two resolve differently --
`checkpoint.folder` gets `./outputs/` prepended, `data-cache-path` does not),
bounds the run with a new `MULTI_STEPS_OVERRIDE` (a prod seat derives
`training.steps` from train_tokens: attempt 4 launched with 5,943,018), and
greps the console for the override warning so an invalid run reports itself.

Re-submitted as `8771774`.

## 5. Everything else still open

- **#71** t3 `std::bad_alloc` -- fresh 512N reproduction in `8764675`. Note t3
  died at 522 nodes while t1 ran fine at the SAME 522 nodes in the same job,
  which argues against pure rank count and against the `LAUNCH_STAGGER=180`
  framing. Read the traceback and compare its call site to the 1024N signature
  BEFORE spending nodes.
- **#76** 906 GB reclaim on two mixed ckpt dirs. Recommendation stands: leave
  it. /lus/flare is 67% used with 31 PB free; both destructive options carry
  more risk than the space is worth.
- **#77** constant-LR fork re-check at ~30k. Now wants a CORRECTLY-flavored
  fork -- the step-21000 matched pair (job `8769743`) came back with no
  meaningful separation (mean delta -0.0033, fork ahead 5/7) but boolq
  (-0.0409) is confounded by the RoPE mismatch.
- **#85** the t4 question itself -- unblocked; `8771774` is the re-run.

## 6. Landed today (all pushed)

MDS chain + `lr` in the dashboard; MDS tokens/step corrected in 3 eval plotters
(bit-exact vs W&B `consumed_train_tokens`); eval charts regenerated on complete
ropefix data (72/72, 32 series); per-seat RoPE flavor; stale-clone
`ckpt_key_compat`; `prod_dash` umbrella liveness + `SSH_TIMEOUT` + table
overflow; `MULTI_ONLY`/`MULTI_NNODES_OVERRIDE`.

## 7. Re-arm monitors after restart (optional)

Only `8769730` is worth watching -- it is the next production cycle:

```
# fires on state change only
until [ -z "$(ssh -n aurora '/opt/pbs/bin/qstat -f 8769730 2>/dev/null | grep -o "job_state = R"')" ]; do sleep 300; done
```

The per-trainer watcher used during `8764675` keyed on category (starting /
training / dead) rather than step number -- keying on the step number makes it
fire on every increment of a healthy chain, which is noise.
