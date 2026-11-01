# Sunspot startup isolation: MDS/TT continuation

Snapshot: `2026-11-01 14:55 UTC`. The comparison remains blocked before
application execution. No new training, loss, or throughput result follows
from these recovery attempts.

## Support follow-up

At `2026-11-01 09:15 CST`, the prepared incident report was sent through `AgentMail`
from the dedicated agent inbox to `support@alcf.anl.gov`, with Sam copied
on CC and all four diagnostic attachments. The returned message ID was
verified against the inbox's `sent` label, recipients, subject, and attachment
count. This confirms sending, not receipt or a site response.

After explicit user approval, `12481933` was cancelled and confirmed
finished. PBS supplied no numeric exit status for this cancellation. The
pre-cancellation `21` failed starts are preserved alongside the post-cancel
scheduler snapshot. `12481932` and `12481935` remain system-held; neither
was cancelled or released.

## Reproduction

| Job | Purpose | Source | Observed result |
|---|---|---|---|
| `12481908` | Native-DDP live-parameter audit and training | `b6dfa77e2` | Finished, exit `265`, `21` failed starts; no `stime` or application log |
| `12481931` | Retry of the live-parameter canary | `a4057f84e` | Cancelled with explicit user approval; finished after `15` starts, no numeric `Exit_status` supplied by PBS |
| `12481932` | Existing 24-rank probe on previously successful hosts | `a4057f84e` | System-held after `21` failed starts; no `stime` or application log |
| `12481933` | One-node shell/storage probe, stdout under home | `dbb46e9bc` | System-held after `21` failed starts; no shell marker or stdout file |
| `12481935` | One-node probe, home filesystem only, explicit `/bin/bash`, different cabinet | `dbb46e9bc` | System-held after `21` failed starts; no shell marker |
| `9025634` (Aurora) | Existing 24-rank device/accumulation probe | `24985bc50` | Finished, exit `0`; all `24` device and accumulation records verified |

`12481931` used `x1922c6s3b0n0` and `x1922c6s4b0n0`.
`12481932` explicitly selected `x1922c1s4b0n0` and `x1922c1s5b0n0`,
the earlier successful pair. `12481933` initially selected the first `c6`
host, then moved to `x1921c0s2b0n0` on a later PBS attempt.
`12481935` explicitly selected `x1921c0s1b0n0`.

## What is established

The active failures report `job_state=R`, `substate=41`, no `stime`, and
no output from the first batch-script marker. The shared PBS source defines
`41` as `JOB_SUBSTATE_PRERUN` and `42` as running. A top-level `R` alone
does not establish application execution.
[PBS state definitions](https://github.com/openpbs/openpbs/blob/master/src/include/job.h).

The shell probe loads no modules, Python, MPI, Torch, or model. Its first
marker is `BATCH_ENTERED`; later markers verify shared-storage access and
the pinned source. Directing stdout to existing writable home storage did
not produce a marker. This narrows the failure to startup before the script
body. The home-only, explicit-shell variant separately isolates tegu mount
and login-shell dependencies.

Changing the two-node pair did not resolve startup. One previously involved
host, `x1922c4s0b0n0`, is offline with
`EXECJOB_END: unable to free VNI (job 12481907)`; the current failures also
occur without that host. This is a related observation, not proof that VNI
allocation causes every failed start.

The exact meaning of exit `265` in this deployment and the responsible
MOM/hook failure remain unverified. Accessible `tracejob` output did not
provide the relevant server/MOM logs. Do not label this a TorchTitan,
DeepSpeed, checkpoint, or distributed-collective failure from these records.

ALCF documents this hold message for node startup problems, output-path
problems, and user-code failures. The probes isolate those boundaries;
they do not decode the daemon error.
[ALCF troubleshooting](https://docs.alcf.anl.gov/running-jobs/known-issues/#comment-job-held-too-many-failed-attempts-to-run).

## Recovery changes

- Added the pinned shell-only probe in `dbb46e9bc`.
- Updated the single campaign monitor in `f35fb4db1` to expose pre-run phase,
  startup count, and `stime`. A finished cancellation without numeric exit
  is terminal, with exit shown as unknown.
- Preserved the full Python child command in the existing `ezpz launch`
  launchers. Installed-parser probes disproved the earlier dropped-command
  diagnosis; separator edits remain reverted.
- Preserved shared runtimes, node state, other jobs, and the unfinished
  Aurora MDS launcher changes.

Every submission passed the repository `alcf-job-preflight` wrapper.
An initial shell-probe preflight correctly refused submission until the
new checkout's tokenizer link was present.

## Aurora fallback

The `debug-scaling` submission passed preflight but was rejected by its
per-user queued-job limit. Existing four-node smoke job `9025608` and
production followup `8986382` were left untouched. The documented separate
`debug` queue accepts two nodes; the unchanged established launcher was
admitted there as `9025634`, with `WANDB_MODE=online`.
[ALCF queue contract](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/).

`9025634` finished on its first startup attempt, PBS exit `0`. All `24`
rank/device records select `FLAT` with `12` devices/node. All `24` rank
JSONs pass unsynchronized first microbatch, final all-rank gradient equality,
GAS-2 versus combined-batch gradient agreement, and manual-wire reference
agreement. Maximum gradient difference is `7.450580596923828e-09`;
manual-wire and global-loss differences are `0`. The probe uses FP32
parameters/gradients and deterministic seed `42`.

This is a small runtime/communication probe on Aurora Torch `2.15`, not the
2B model or a matched Sunspot Torch `2.14` performance comparison. The
[online probe run](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/agpt-aurora-probe-9025634)
has a committed `verification` artifact,
`agpt-replicated-accumulation-9025634:v0`. Complete logs, `24` rank JSONs,
`24` device records, and terminal scheduler data were independently checked. The live model-load audit
and full MDS/TT pair remain open.

## Evidence

[Startup archive in online W&B](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/agpt-startup-evidence-20261101)
contains complete Aurora output, all rank results, submitted scripts, pinned
source changes, preflight logs, sanitized scheduler records, and a size/SHA256
manifest. Artifact `agpt-startup-evidence-20261101:v0` records the first snapshot;
the next version records all three system-held probes.
This archival run contains no training measurements and does not resume any
training run. Held PBS jobs remain held, not finished. No unapproved cancellation,
release, or node-state change was performed. The active monitor was stopped
only after all allocated probes had either finished or reached system holds.

## Site investigation request (sent through AgentMail)

Please inspect the PBS server/MOM and execution-hook logs for Sunspot jobs
`12481908`, `12481931`, `12481932`, `12481933`, and `12481935` on
`2026-11-01`, approximately `05:34-07:00 UTC` and `14:23 UTC` onward.
The jobs repeatedly enter substate `41` with no `stime` and no batch output.
A one-node script with no MPI, Python, modules, or model reproduces the
missing first marker. The stdout parent exists and is writable from login.
Tests cover `c1`, `c6`, and `x1921c0` hosts. Please identify the failed
startup stage, numeric exit mapping, and whether the VNI cleanup failure
recorded on `x1922c4s0b0n0` requires a site repair.

Do not release the system-held probe or submit more training retries until
the shell-start gate is understood. The next application gate remains the
live loaded-parameter audit, followed by the controlled MDS/TT pair.
