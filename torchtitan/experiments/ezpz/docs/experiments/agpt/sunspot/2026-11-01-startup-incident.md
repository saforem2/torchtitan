# Sunspot startup isolation: MDS/TT continuation

Initial startup snapshot: `2026-11-01 14:55 UTC`. The comparison remains blocked before
application execution. No new training, loss, or throughput result follows
from these recovery attempts.

## Direct executable isolation and cross-user evidence

A second isolation round submits `/bin/hostname` with PBS executable mode,
through the existing preflight wrapper from pinned checkout `dbb46e9bc`.
The PBS `executable` attribute records `/bin/hostname`. No batch shell,
Python, MPI, Torch, model, data loader, checkpoint, or compiler executes on
compute nodes in these probes. Login preflight imports remain separate.

| Job | Queue / image | PBS stdout/stderr | Outcome |
|---|---|---|---|
| `12481943` | `workq`, October 13 image | Existing home storage | System-held after `21` failed starts; no `stime` or output |
| `12481944` | `workq`, October 13 image | `/dev/null` | System-held after `21` failed starts; no `stime` |
| `12481945` | `next-eval`, October 28 image | Existing home storage | System-held after `21` failed starts; no `stime` or output |

`next-eval` is enabled on Sunspot, has its own eligible node pool, and
requests `compute_sunspot_test_20261028T222447_47bb71c5`. `workq` requests
`compute_sunspot_test_20261013T165519_97e02396`. An Aurora queue-retirement
claim does not establish Sunspot's live queue configuration. Both paths
reproduce the failure. These are system-binary startup controls, not
performance runs or a runtime migration.

Aggregate public PBS history, captured at `09:36 CST` on November 1, contains
**six matching failures across three other user accounts**: `run_count >= 21`,
no `stime`, and the same failed-start hold comment. The first matching
other-user submission is `05:48:08 UTC`. This account's first matching
submission is `05:34:25 UTC`; its last recorded actual start is `05:28:07 UTC`.
Only counts/times are retained, without other-user identities, job IDs,
commands, paths, or environments. This establishes that the failure is
shared across accounts; it does not identify its initiating service.

### Read-only host and service observations

- `/home` and `/lus/tegu` are logical mounts on the same Lustre filesystem,
  with free space/inodes. `lfs quota` reports no user limit. Moving stdout
  between those paths did not constitute an independent physical-storage test.
- Login profile and sourced dependency mtimes predate the incident. Login
  process pressure is low. These observations do not verify compute mounts
  or compute-side UID/PAM/cgroup setup.
- PALS `1.8` `palstat --ping` fails from the UAN on assigned nodes in both
  pools with `Couldn't send RPC ping ... Resource temporarily unavailable`.
  The valid Sunspot names have the `-hsn0.hsn.cm.sunspot.alcf.anl.gov` suffix,
  verified from successful job `12481905`; bare node names and Aurora-style
  names are not interchangeable.
- TCP connection to the documented PALS port `7919` succeeds on the assigned
  `next-eval` node. An open TCP port does not establish successful PALS RPC
  authentication or request handling. No known-good UAN-originated ping
  during PBS pre-run was captured, so this remains supplemental evidence.
- Read-only `qmgr` hook inspection returns unauthorized for this account.
  Relevant MOM/server logs are not available on the UAN. Normal SSH to the
  currently assigned node closes without a shell; no access workaround,
  daemon restart, hook modification, or node-state change was attempted.

PBS supports direct execution and direct-write output to `/dev/null`.
The latter depends on MoM configuration, which we cannot inspect; its failed
probe alone does not rule out every output-routing issue.
[PBS direct execution](https://docs.alcf.anl.gov/running-jobs/),
[PBS direct-write contract](https://2019.help.altair.com/19.2/PBSProfessional/PBSUserGuide19.2.1.pdf).

### Diagnostic boundary

The shared startup path is now the demonstrated failure domain. A
TorchTitan-specific or account-only explanation cannot explain the complete
observations. The exact MOM response or execution/resource hook failure is
still unverified. Compute-side filesystem/account setup and PALS/VNI hook
state require site inspection.

HPE documents a Slingshot VNID-hook failure producing VNI errors, but its
example is `[409]: Failed to find specified job`; our earlier PALS error is
`[308]` with no response, and our bare one-node jobs also fail. This is an
operator investigation lead, not a confirmed match or an instruction to
reload a hook without inspection.
[HPE VNID-hook troubleshooting](https://support.hpe.com/hpesc/public/docDisplay?docId=dp00007631en_us&docLocale=en_US&page=operations%2Fwlm%2FMulti_node_PALS_launches_fail_with_VNID_request_failed_error.html).

The automated support acknowledgement created **REQ-17196** at `09:19 CST`.
It contains no diagnosis. A prepared follow-up adds the direct-executable,
aggregate cross-user, and qualified PALS results and corrects future CC to
Sam's Argonne address. It remains unsent pending authorization.

## Earlier working TT jobs and later failures

Read-only inspection of the `sunspot-tt-ezpz` pane and exact historical PBS
records confirms that its earlier success was real. Times below are
`America/Chicago`; the daylight-saving transition changes `CDT` to `CST`.

| Job | Local time | Verified outcome |
|---|---|---|
| `12481889`, upstream-sync TT smoke | Oct 31, `22:00-22:05 CDT` | One startup, exit `0`, all three arms `rc=0`, `VERDICT: ok` |
| `12481906`, campaign MDS control | Nov 1, `00:15-00:19 CDT` | One startup, exit `0`; complete 32-update evidence is in the comparison report |
| `12481907`, campaign pair | Nov 1, `00:28 CDT` start | Batch did start; later PALS launch failed; cancelled with approval |
| `12481908`, campaign live-load audit | Nov 1, `00:34 CDT` submission | No `stime`, 21 failed starts, eventual exit `265` |
| `12481940` / `12481941` / `12481942`, pane CPT/SFT/2B smokes | Nov 1, `09:03 CST` submission | All reached system holds after 21 failed starts, no `stime`; subsequently finished with no numeric exit supplied |

The pane had reported the new jobs as running from the top-level PBS `R`
state before they reached actual batch execution. Its old `VERDICT: ok`
remains valid; it does not establish success of the new CPT/SFT/2B jobs.
The current source uses Torch `2.16` while this campaign uses Torch `2.14`,
but the new failures precede either runtime loading.

All listed jobs requested the same recorded compute image,
`compute_sunspot_test_20261013T165519_97e02396`, and the same account/queue.
The later pane jobs also use `tegu:home`, two nodes, and a distinct smoke
launcher. No configuration change or input was sent to that active
agent; no neighboring job was cancelled by this investigation.

The evidence shows earlier successful execution followed by failures across
both submission paths for this account. It does not establish a cluster-wide
outage, a general TorchTitan failure, or the exact failing MOM/hook stage.
The comparison record is `sunspot-context-reconciliation-20261101.json` in
the online evidence archive; private agent transcripts are excluded.

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
