# Upstream-sync regression gates

`test_upstream_sync_contracts.py` keeps the local `ezpz` runtime aligned with the
upstream configuration and model contracts.

The static gates scan active launch/runtime modules. They reject imports that are
not exported by `torchtitan.config`, removed `comm.mode` / `fake_backend` /
`local_tensor` options, and old pipeline option names. One-off scripts, tests, and
the vendored Aurora drop-in are excluded because they are not launch/runtime
code. The two active gates are expected failures until the config/runtime agent
lands the corresponding migrations; remove their `xfail` marks when those
failures clear.

The construction gates build every `qwen3` registry flavor on `meta`, validate the
current config schema, and snapshot state-dict keys plus parameter counts for the
dense and MoE debug models. Update those snapshots only after reviewing the full
key and shape diff.

`upstream_sync_baseline.py` records deterministic output, loss, gradient,
optimizer, and post-step model signatures. Tensor signatures include shape,
dtype, sum, L2 norm, and a SHA-256 digest. Keep baseline inputs, seeds, runtime,
and dtype fixed when comparing signatures across revisions.

Run the focused suite with:

```bash
python -m pytest -q tests/unit_tests/cpu/test_upstream_sync_contracts.py
```

The two strict `xfail` cases are negative controls. An `XPASS` means a gate no
longer detects its injected fault and must be repaired before the suite is
trusted.
