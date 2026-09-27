# Verified OLMo-3-vocab GBS=6144 evidence

These are immutable copies of LR-finder CSV artifacts used by the campaign
report. All five files have unique LR samples and finite LR/loss fields.
Those checks do not establish optimizer health by themselves; final-result
status also requires terminal logs showing real finite optimizer updates.

| job | arm | rows | sampled LR range | SHA-256 | interpretation |
|---:|---|---:|---:|---|---|
| `12478508` | 5B AdamW fine | 100 | `1e-6`–`1e-3` | `35cbf0ddc54d9f376f61f9df78c35fb00e9b9f8787b8f911128bed0a3ef7e525` | final, `7.17e-5` |
| `12478509` | 10B AdamW fine | 100 | `1e-6`–`1e-3` | `bd2952987847cc224743500892627bda6b6e8b7d6b45c639c373713be4eeb5a4` | final, `4.16e-5` |
| `12478624` | 5B SophiaG fine | 100 | `2.8e-6`–`2.8e-3` | `ff0a68c049d4328c39118cdbf0959c7f16b94ba7dcea858f6085b4dedee20884` | final, `7.04e-7` |
| `12478569` | 10B SophiaG fine | 100 | `1.66e-6`–`1.66e-4` | `0f59cdea18a2a197dfd0397a7301a0a5bcd71b21d93be7a8e1ea076eefa38d3f` | diagnostic; detector result below sampled range |
| `12478513` | 30B SophiaG coarse | 30 | `1e-8`–`1e-1` | `e029a151243be03bf269b800b6044b69dc6c812b81654c1d6e636bd2481f0be7` | coarse diagnostic only |

The `global_batch_size` CSV field is token batch size: 25,165,824 tokens,
equal to 6,144 sequences × 4,096 tokens.

Partial 30B fine curves are deliberately absent. The superseded 5B SophiaG
coarse curve is also absent; job `12478624` provides the final accepted arm.
