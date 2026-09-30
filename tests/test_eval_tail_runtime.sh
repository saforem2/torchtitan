#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
eval_script="$root/torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh"
tail_wrapper="$root/torchtitan/experiments/ezpz/scripts/eval/eval-20b-current-tail.pbs"

assert_contains() {
    local file=$1 pattern=$2 message=$3
    grep -Eq -- "$pattern" "$file" || { echo "FAIL: $message" >&2; exit 1; }
}
assert_not_contains() {
    local file=$1 pattern=$2 message=$3
    ! grep -Eq -- "$pattern" "$file" || { echo "FAIL: $message" >&2; exit 1; }
}

assert_contains "$eval_script" 'CONVERT_VENV=' 'conversion runtime must be explicit'
assert_contains "$eval_script" 'import spmd_types, torchtitan' 'conversion preflight must import spmd_types'
assert_contains "$eval_script" 'Conversion FAILED for step.*rc=' 'conversion failure must be explicit and carry rc'
assert_contains "$eval_script" 'exit "\$\{convert_rc\}"' 'conversion failure must propagate nonzero'
assert_not_contains "$eval_script" 'Conversion FAILED .*continue' 'conversion failure must not continue to PBS zero'
assert_contains "$tail_wrapper" 'STEPS=11100' '512N retry must target durable step 11100'
assert_contains "$tail_wrapper" 'STEPS=17500' '256N retry must target latest durable step 17500'
assert_contains "$tail_wrapper" 'MODEL_FLAVOR=20b_real' 'tail eval must use the verified cos_sin flavor'
assert_contains "$tail_wrapper" 'EVAL_EXPECTED_SHA.*set EVAL_EXPECTED_SHA' 'tail eval must require a tested source SHA'

printf 'eval tail runtime contracts: PASS\n'
