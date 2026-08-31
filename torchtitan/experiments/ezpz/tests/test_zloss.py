"""Tests for the output z-loss.

The load-bearing claim is that z_loss_coef=0.0 is BIT-IDENTICAL to plain cross
entropy -- that is what makes this safe to adopt as the default loss. The rest
check the penalty does what it says: grows with |log Z|, is shift-sensitive
where CE is not, and contributes gradient.
"""
import torch

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.experiments.ezpz.zloss import CrossEntropyWithZLoss, z_loss_term

torch.manual_seed(0)
T, V = 64, 512
ok = True


def check(label, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}{('  ' + extra) if extra else ''}")


pred = torch.randn(T, V, dtype=torch.float32)
labels = torch.randint(0, V, (T,))

# --- 1. coef=0 is bit-identical to plain CE -------------------------------
ce = CrossEntropyLoss(CrossEntropyLoss.Config())
z0 = CrossEntropyWithZLoss(CrossEntropyWithZLoss.Config(z_loss_coef=0.0))
l_ce, m_ce = ce(pred, labels)
l_z0, m_z0 = z0(pred, labels)
check("coef=0 is BIT-identical to CrossEntropyLoss",
      l_ce.item() == l_z0.item(), f"{l_ce.item():.10f} vs {l_z0.item():.10f}")
check("coef=0 adds no metrics", "z_loss" not in m_z0)

# --- 2. a positive coef increases the loss and reports the term -----------
zc = CrossEntropyWithZLoss(CrossEntropyWithZLoss.Config(z_loss_coef=1e-4))
l_zc, m_zc = zc(pred, labels)
check("coef>0 increases the total loss", l_zc.item() > l_ce.item(),
      f"{l_zc.item():.6f} > {l_ce.item():.6f}")
check("coef>0 reports z_loss separately", "z_loss" in m_zc,
      f"z_loss={m_zc.get('z_loss', float('nan')):.6f}")
# Relative tolerance: these are ~428 in fp32, where 1e-6 absolute is below
# the representable gap (~4e-5). The first version of this test asserted
# 1e-6 absolute and failed on rounding alone.
_diff = abs((l_zc - l_ce).item() - m_zc["z_loss"].item())
check("total - CE equals the reported z_loss",
      _diff / max(1.0, abs(l_ce.item())) < 1e-6, f"abs diff {_diff:.2e}")

# --- 3. the penalty tracks log Z, which CE is blind to --------------------
# Shifting every logit by a constant leaves CE exactly unchanged but moves
# log Z by that constant. This is the invariance z-loss exists to remove.
shifted = pred + 10.0
l_ce_s, _ = ce(shifted, labels)
l_zc_s, m_s = zc(shifted, labels)
check("CE is invariant to a constant logit shift",
      abs(l_ce_s.item() - l_ce.item()) < 1e-3,
      f"{l_ce.item():.6f} -> {l_ce_s.item():.6f}")
# The penalty is coef * sum((log Z)^2), and a constant shift c moves every
# log Z to log Z + c. So the expected ratio is sum((z+c)^2)/sum(z^2),
# computed here from the actual log Z values rather than asserted against an
# invented threshold -- an earlier version of this test demanded >10x and
# failed on a real 6.2x.
with torch.no_grad():
    _lz = torch.logsumexp(pred.float(), dim=-1)
    _expected = ((_lz + 10.0) ** 2).sum().item() / (_lz**2).sum().item()
_actual = m_s["z_loss"].item() / m_zc["z_loss"].item()
check("z-loss IS shift-sensitive, by the predicted factor",
      abs(_actual - _expected) / _expected < 0.01,
      f"{m_zc['z_loss'].item():.4f} -> {m_s['z_loss'].item():.4f} "
      f"= {_actual:.2f}x (predicted {_expected:.2f}x)")

# --- 4. gradient actually flows through the penalty -----------------------
p = pred.clone().requires_grad_(True)
loss, _ = zc(p, labels)
loss.backward()
g_with = p.grad.clone()

p2 = pred.clone().requires_grad_(True)
loss2, _ = ce(p2, labels)
loss2.backward()
g_without = p2.grad.clone()

check("z-loss changes the gradient", not torch.allclose(g_with, g_without))

# --- 5. scaling and validation -------------------------------------------
big = z_loss_term(pred, 1e-3)
small = z_loss_term(pred, 1e-4)
check("penalty scales linearly in coef",
      abs((big / small).item() - 10.0) < 1e-3, f"ratio {(big / small).item():.4f}")

try:
    CrossEntropyWithZLoss(CrossEntropyWithZLoss.Config(z_loss_coef=-1.0))
    check("negative coef is rejected", False)
except ValueError:
    check("negative coef is rejected", True)

# --- 6. fp32 accumulation even when logits are bf16 -----------------------
z_bf16 = z_loss_term(pred.bfloat16(), 1e-4)
check("bf16 logits still give an fp32 penalty",
      z_bf16.dtype == torch.float32, str(z_bf16.dtype))

print("\nZ-LOSS TESTS:", "ALL PASS" if ok else "FAILURES PRESENT")
