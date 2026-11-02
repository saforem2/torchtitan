#!/usr/bin/env bash
# Load one immutable ezpz utils revision after verifying its content hash.

EZPZ_UTILS_COMMIT=672057d6ebd5d5a05d78cd4b8e33563d3c8d37c6
EZPZ_UTILS_SHA256=5feac78c01f262b13c41d9b1355cbcaeb49fa304db8e6efdb6a8999df50e1119
EZPZ_UTILS_URL="https://raw.githubusercontent.com/saforem2/ezpz/${EZPZ_UTILS_COMMIT}/src/ezpz/bin/utils.sh"

load_pinned_ezpz_utils() {
    local cache_root cache_file building actual
    cache_root="${EZPZ_UTILS_CACHE_ROOT:-${PBS_O_WORKDIR:-$PWD}/.cache/ezpz-utils}"
    cache_file="${cache_root}/utils-${EZPZ_UTILS_COMMIT}.sh"
    mkdir -p "$cache_root"

    if [[ ! -s "$cache_file" ]]; then
        building="${cache_file}.building-${BASHPID}"
        curl -fsSL --max-time 30 "$EZPZ_UTILS_URL" -o "$building" || return 90
        actual=$(sha256sum "$building" | cut -d' ' -f1)
        [[ "$actual" == "$EZPZ_UTILS_SHA256" ]] || {
            printf 'ezpz utils checksum mismatch: expected=%s actual=%s path=%s\n' \
                "$EZPZ_UTILS_SHA256" "$actual" "$building" >&2
            return 91
        }
        mv "$building" "$cache_file"
    fi

    actual=$(sha256sum "$cache_file" | cut -d' ' -f1)
    [[ "$actual" == "$EZPZ_UTILS_SHA256" ]] || {
        printf 'cached ezpz utils checksum mismatch: expected=%s actual=%s path=%s\n' \
            "$EZPZ_UTILS_SHA256" "$actual" "$cache_file" >&2
        return 92
    }
    # shellcheck source=/dev/null
    source "$cache_file"
    declare -F ezpz_setup_job >/dev/null || return 93
    declare -F ezpz_load_modules >/dev/null || return 94
}
