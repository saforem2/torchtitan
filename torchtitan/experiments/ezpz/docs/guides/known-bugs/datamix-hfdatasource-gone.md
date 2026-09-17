# Data-mix configs are broken: HFDataSource no longer exists in core

**Status:** live on `origin/ezpz`. NOT caused by the sync-84 merge -- the
pre-merge tip (`a25127f56^1`) fails identically.

## What fails

```
agpt_2b_mds_mix_owm_edu_7525()
ImportError: cannot import name 'HFDataSource' from
             'torchtitan.hf_datasets.text_datasets'
```

`_agpt_2b_mds_mix_blend` (`agpt/config_registry.py:671`) lazily imports
`HFDataSource` and `InterleavedHuggingFaceTextDataLoader`. Neither exists in
core any more: `torchtitan/hf_datasets/text_datasets.py` now defines only
`TextProcessor` and `ChatProcessor`. Three registered flavors route through
that builder (owm/edu 75-25, 50-50, 90-10), plus `_agpt_2b_mds_mix_blend_src`.

## Why nothing caught it

Three filters in a row let it through:

1. It is **off the smoke path** -- the smoke runs `agpt_debugmodel` and
   `moe_debugmodel`, never a mix flavor.
2. The import is **lazy**, inside the function body, so importing the registry
   succeeds. See the same shape in the #4684 `validate_converter_order` break
   (`715d8a16e`), where both registries imported while `agpt_debugmodel` failed
   to BUILD.
3. `git grep` for the symbol finds only OUR reference, which reads like the
   definition unless you check whether core still has one.

Found by walking every lazily-imported core symbol in `experiments/ezpz` with
an AST parse and checking each against the installed core -- 24 symbols across
21 modules, 6 unresolved, of which this is the only real break (`full_dtensor`
and `deepep` are comments and optional guarded paths).

## What it does NOT affect

Nothing on the production or smoke path: `agpt_debugmodel`, `moe_debugmodel`,
the 2B/20B/80B production flavors and the validated sync-84 runs are unaffected.
This is the data-mix experiment arm only (see
`docs/.../project_datamix_experiment` context: owm-control vs edu-100).

## Fixing it

Not obvious enough to guess. The replacement is a `GrainDataLoader` built from
`SingleDatasetConfig(source=HuggingFaceStreamingSource, processor=TextProcessor)`
-- a different object graph, the same porting note core leaves on `c4_test`
(`config_registry.py:426` raises `NotImplementedError` pointing at
`torchtitan/components/data/dataset.py`). Porting the mix arm means rebuilding
the weighted interleave on top of that, and it should be done when the data-mix
experiment is next actually run, with its numbers checked -- not blind.
