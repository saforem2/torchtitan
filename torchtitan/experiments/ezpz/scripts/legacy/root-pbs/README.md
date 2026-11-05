# Archived root-level PBS launchers

These 53 scripts were moved intact from the repository root on 2026-09-30 to
restore the ezpz Golden Rule: experiment code and launch artifacts belong under
`torchtitan/experiments/ezpz/`.

They are historical experiment provenance, not current launch templates. Many
predate current TorchTitan CLI and configuration contracts. In particular,
several still use removed dotted options documented in
[`dead-cli-flags-in-repo-root-pbs.md`](../../../docs/guides/known-bugs/dead-cli-flags-in-repo-root-pbs.md).

Before reusing one:

1. Compare it with the current launcher for the same workflow.
2. Parse the exact extracted argv against the current configuration loader.
3. Use a job-unique output and checkpoint directory.
4. Run a bounded hardware canary and require real optimizer updates plus the
   promised artifacts.

Do not move these files back to repository root. The launcher-contract test
rejects root-level `.pbs` files.
