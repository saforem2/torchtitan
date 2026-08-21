# Session resume notes -- 2026-08-21

Written before a Claude session restart. Nothing here requires the old session;
every long-running thing lives on the cluster (PBS) or is a one-line restart.

## 1. Nothing is lost by restarting

All monitors were **local watchers**, not workers -- they polled `qstat` and
tailed logs. Killing them stops the notifications, not the work. Every PBS job
below keeps running regardless.

The only local process worth restarting is the dashboard web server.

## 2. Cluster state (as of 2026-08-21 ~12:45 UTC)

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

## 4. The open thread: t4 smoke `8771637` FAILED, and the reason is a lead

Ran through the umbrella launcher (`MULTI_ONLY=4 MULTI_NNODES_OVERRIDE=4`) and
still died with the same pre-existing error:

    AttributeError: 'dict' object has no attribute 'mul_'

**The wrapper reported "injected initial-load-path into the t4 row", but
grepping the trainer's actual launch shows NO `--checkpoint.initial-load-path`
flag.** So the converted seed was never used and this run re-tested the
old-format checkpoint -- the result says nothing about whether step-9500 is
corrupt.

Leading hypothesis, NOT yet verified: t4's own ckpt dir already contains a
resumable `step-9500`, and a resume-from-latest takes precedence over
`initial-load-path` (that flag seeds a FRESH chain). If so the fix is to point
the smoke at an empty ckpt dir so there is nothing to resume from.

Next step is a log read, not an allocation: find where `initial-load-path` is
dropped between the rewritten TRAINERS row and the `ezpz launch` line.

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
- **#85** the t4 question itself, blocked on section 4.

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
