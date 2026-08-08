"""Is the new light palette actually more distinguishable than the old one?

Scores pairwise separation in CIE Lab (deltaE-76), which approximates how
different two colors LOOK, unlike raw RGB distance. Only the first N entries
matter -- the board shows ~5-6 chains.
"""
import sys, math
sys.path.insert(0, ".")

OLD_LIGHT = [
    (31, 96, 196), (196, 96, 8), (40, 140, 45), (196, 40, 40),
    (110, 70, 200), (0, 140, 132), (200, 50, 130), (150, 130, 20),
    (110, 72, 40), (50, 150, 90), (80, 100, 120), (190, 110, 40),
]
import ast as _ast, re as _re
_src = open("torchtitan/experiments/ezpz/utils/prod_dash_app.py").read()
NEW_LIGHT = _ast.literal_eval(_re.search(r"_PALETTE_LIGHT = (\[.*?\])", _src, _re.S).group(1))


def _srgb_to_lab(rgb):
    def inv(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (inv(float(v)) for v in rgb)
    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t) + 16 / 116
    fx, fy, fz = f(x), f(y), f(z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def de(a, b):
    la, lb = _srgb_to_lab(a), _srgb_to_lab(b)
    return math.sqrt(sum((u - v) ** 2 for u, v in zip(la, lb)))


def score(pal, n):
    sub = pal[:n]
    pairs = [(i, j, de(sub[i], sub[j]))
             for i in range(n) for j in range(i + 1, n)]
    ds = [d for _, _, d in pairs]
    worst = min(pairs, key=lambda p: p[2])
    return min(ds), sum(ds) / len(ds), worst


print("deltaE-76 pairwise separation (higher = easier to tell apart)")
print("  ~10 = 'just distinguishable', <10 reads as the same color at a glance\n")
ok = True
for n in (5, 6, 8, 12):
    o_min, o_avg, o_w = score(OLD_LIGHT, n)
    n_min, n_avg, n_w = score(NEW_LIGHT, n)
    flag = "ok  " if n_min > o_min else "FAIL"
    if n_min <= o_min:
        ok = False
    print(f"  [{flag}] first {n:2d}: worst-pair dE {o_min:5.1f} -> {n_min:5.1f}   "
          f"avg {o_avg:5.1f} -> {n_avg:5.1f}")
    print(f"          old worst pair: idx {o_w[0]},{o_w[1]}  "
          f"{OLD_LIGHT[o_w[0]]} vs {OLD_LIGHT[o_w[1]]}")
    print(f"          new worst pair: idx {n_w[0]},{n_w[1]}  "
          f"{NEW_LIGHT[n_w[0]]} vs {NEW_LIGHT[n_w[1]]}")

# The 5 chains actually on the board today.
print("\n  chroma (colorfulness) of the first 6, old -> new:")
for i in range(6):
    o, n = OLD_LIGHT[i], NEW_LIGHT[i]
    co = math.hypot(*_srgb_to_lab(o)[1:])
    cn = math.hypot(*_srgb_to_lab(n)[1:])
    print(f"    idx {i}: {co:5.1f} -> {cn:5.1f}   {o} -> {n}")

print()
print("PASS: new light palette separates better at every size" if ok
      else "FAIL: some size regressed")
sys.exit(0 if ok else 1)
