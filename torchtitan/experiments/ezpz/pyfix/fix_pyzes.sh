#!/bin/bash
# Idempotently patch the pyzes hardcoded-libze_loader-path bug in the active venv.
#
# pyzes (imported by torch.xpu) hardcodes the Debian multiarch path
# /usr/lib/x86_64-linux-gnu/libze_loader.so.1 in _LoadZeLibrary(). On
# Sunspot/Aurora (RHEL-family) the loader is in /usr/lib64, so `import pyzes`
# -> OSError -> torch.xpu.is_available() crashes the whole job. Upstream fix:
# github.com/oneapi-src/level-zero PR 496 (fixes #485). Until that ships, this
# rewrites the hardcoded path to the bare soname so the dynamic linker resolves
# it via the normal search path (works on every distro).
#
# Refresh-proof by design: version-controlled in the repo and re-run at job
# start, so a venv rebuild that restores the buggy pyzes is re-patched. No-op if
# already patched or if pyzes is absent. Idempotent.
#
# Usage: source/call after activating the venv, e.g. in a PBS job script:
#   source venvs/2026.1.0/bin/activate
#   bash torchtitan/experiments/ezpz/pyfix/fix_pyzes.sh
set -o pipefail

# Locate pyzes.py by PATH, not by importing it: buggy pyzes CRASHES on import
# (the very OSError we fix), so `import pyzes` would fail here and we'd never
# patch. Scan the venv's site-packages dirs instead.
pz="$(python3 - <<'PYEOF'
import os, site, sys
cands = list(site.getsitepackages()) if hasattr(site, "getsitepackages") else []
su = site.getusersitepackages() if hasattr(site, "getusersitepackages") else None
if su:
    cands.append(su)
cands.append(os.path.join(sys.prefix, "lib", "python%d.%d" % sys.version_info[:2], "site-packages"))
for d in cands:
    p = os.path.join(d, "pyzes.py")
    if os.path.isfile(p):
        print(p)
        break
PYEOF
)"
if [[ -z "${pz}" || ! -f "${pz}" ]]; then
    echo "[fix_pyzes] pyzes.py not found in site-packages; nothing to patch"
    exit 0
fi

if grep -q 'bare soname' "${pz}" 2>/dev/null; then
    echo "[fix_pyzes] already patched: ${pz}"
    exit 0
fi

old='libName = "/usr/lib/x86_64-linux-gnu/lib" + libName + ".so.1"'
if ! grep -qF "${old}" "${pz}" 2>/dev/null; then
    echo "[fix_pyzes] hardcoded path not found (pyzes may already be fixed upstream): ${pz}"
    exit 0
fi

cp "${pz}" "${pz}.bak-pre-soname-fix" 2>/dev/null || true
python3 - "${pz}" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = 'libName = "/usr/lib/x86_64-linux-gnu/lib" + libName + ".so.1"'
new = 'libName = "lib" + libName + ".so.1"  # ezpz: bare soname; resolve via ld search (/usr/lib64), not the hardcoded Debian path -- oneapi-src/level-zero#496'
assert old in s
open(p, "w").write(s.replace(old, new))
print("[fix_pyzes] patched:", p)
PYEOF
