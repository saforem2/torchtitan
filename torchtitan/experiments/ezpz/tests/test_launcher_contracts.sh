#!/usr/bin/env bash
# Static contracts for batch launchers that cannot run off-machine.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
sonic="$repo_root/torchtitan/experiments/ezpz/submit/aurora/submit_agpt_moe_full_sonic_2n_1100.pbs"
production="$repo_root/torchtitan/experiments/ezpz/submit/aurora/submit_agpt_dense_moe_256n_50k.pbs"
resumable="$repo_root/torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_gs138650_tulu_math_uc_mix_8n_gbs6144.sh"
docs="$repo_root/torchtitan/experiments/ezpz/docs/guides/aurora-moe-training.md"

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

assert_contains() {
    local file=$1 pattern=$2 message=$3
    grep -Eq -- "$pattern" "$file" || fail "$message"
}

assert_not_contains() {
    local file=$1 pattern=$2 message=$3
    if grep -Eq -- "$pattern" "$file"; then
        fail "$message"
    fi
}

# Keep the public ezpz invocation; only fix status handling around its pipeline.
assert_contains "$resumable" '^[[:space:]]*ezpz launch ' \
    'resumable SFT must continue to use ezpz launch'
assert_contains "$resumable" 'rc=\$\{PIPESTATUS\[0\]\}' \
    'resumable SFT must capture the ezpz launch status'
assert_contains "$resumable" 'exit "\$\{rc\}"' \
    'resumable SFT must return the ezpz launch status'
assert_not_contains "$resumable" '\|[[:space:]]*tee.*\|\|[[:space:]]*true' \
    'resumable SFT must not swallow launcher/timeout failures'

# A resumable timeout remains a nonzero batch result for schedulers and callers.
assert_contains "$production" 'resumable=1' \
    'production launcher must identify checkpointed timeout as resumable'
assert_contains "$production" 'elif test "\$\{rc\}" -eq 124' \
    'production launcher must distinguish timeout status 124'
python3 - "$production" <<'PY'
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text()
branch = text.split('elif test "${rc}" -eq 124; then', 1)[1].split('else', 1)[0]
assert 'exit "${rc}"' in branch, 'resumable timeout branch must propagate rc=124'
warmup = text.split('warmup_rc=${PIPESTATUS[0]}', 1)[1]
guard = 'if test "${warmup_rc}" -ne 0; then'
assert guard in warmup, 'production warm-up must explicitly guard nonzero status'
guarded = warmup.split(guard, 1)[1].split('fi', 1)[0]
assert 'exit "${warmup_rc}"' in guarded, 'production warm-up must propagate status'
PY

# The two-node Sonic launcher documents these relocatable artifact overrides.
for variable in \
    TT_ROOT TT_VENV_TAR TT_MOE_KERNEL_CACHE TT_DATA_LIST \
    TT_DATA_LIST_SHA256 TT_TOKENIZER TT_TOKENIZER_SHA256; do
    assert_contains "$sonic" "${variable}" \
        "two-node Sonic launcher must honor ${variable}"
done
assert_contains "$sonic" '--dataloader.dataset-path="\$[0-9]"' \
    'two-node Sonic launcher must pass TT_DATA_LIST to training'
assert_contains "$sonic" '--hf-assets-path="\$[0-9]"' \
    'two-node Sonic launcher must pass the TT_TOKENIZER directory to training'
assert_contains "$sonic" 'missing .*TT_' \
    'two-node Sonic launcher must emit clear missing-override diagnostics'
assert_contains "$sonic" 'warmup_rc=\$\{PIPESTATUS\[0\]\}' \
    'two-node Sonic warm-up must capture timeout/launcher status'
assert_contains "$sonic" 'rc=\$\{PIPESTATUS\[0\]\}' \
    'two-node Sonic training must capture timeout/launcher status'
python3 - "$sonic" <<'PY'
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text()
for assignment in ('warmup_rc=${PIPESTATUS[0]}', 'rc=${PIPESTATUS[0]}'):
    tail = text.split(assignment, 1)[1]
    status = assignment.split('=', 1)[0]
    guard = f'if test "${{{status}}}" -ne 0; then'
    assert guard in tail, f'{status} must have an explicit nonzero guard'
    guarded = tail.split(guard, 1)[1].split('fi', 1)[0]
    assert f'exit "${{{status}}}"' in guarded, f'{status} guard must propagate status'
PY

assert_contains "$docs" 'retains all checkpoints' \
    'checkpoint retention documentation must say all checkpoints are retained'
assert_not_contains "$docs" 'retains the newest two checkpoints' \
    'checkpoint retention documentation must not claim only two are retained'

printf 'launcher contract tests: PASS\n'
