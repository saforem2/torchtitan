"""Do REAL production checkpoints show the embedding freeze the ablation predicts?

v1 (bf16 master) vs v2 (fp32 master), 2B agpt. tok_embeddings inits at
std=1.0 (agpt _EMBEDDING_INIT), the same scale as RMSNorm.weight -- so if the
ablation's finding generalizes, v1's embedding rows should be far coarser
(quantized to the bf16 grid at scale 1.0) than v2's.

The direct test: count DISTINCT values in a slice. A bf16 master at scale ~1.0
can only occupy the bf16 grid (256 values per binade); an fp32 master occupies
far more.
"""
import sys, torch, torch.distributed.checkpoint as dcp

VOCAB, DIM = 256128, 2048
ROWS = 2048  # sample the first ROWS rows to keep the load small

for label, ckpt in [
    ("v1 bf16-master", sys.argv[1]),
    ("v2 fp32-master", sys.argv[2]),
]:
    sd = {"tok_embeddings.weight": torch.zeros(VOCAB, DIM),
          "norm.weight": torch.zeros(DIM)}
    try:
        dcp.load(sd, checkpoint_id=ckpt)
    except Exception as e:
        print(f"{label}: LOAD FAILED {type(e).__name__}: {str(e)[:150]}")
        continue
    e = sd["tok_embeddings.weight"][:ROWS].float()
    n = sd["norm.weight"].float()
    uniq = torch.unique(e).numel()
    # bf16 round-trip identity => the value already lies on the bf16 grid
    on_bf16_grid = (e.bfloat16().float() == e).float().mean().item()
    print(f"{label}:  ckpt={ckpt.split('/')[-2:]}")
    print(f"    tok_embeddings[:{ROWS}]  std={e.std():.5f}  distinct values={uniq:,}"
          f"  frac exactly on bf16 grid={on_bf16_grid:.4f}")
    print(f"    norm.weight              std={n.std():.3e}  var={n.var():.3e}"
          f"  all_ones={torch.allclose(n, torch.ones_like(n))}")
