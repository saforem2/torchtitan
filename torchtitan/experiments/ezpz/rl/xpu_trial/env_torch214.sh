#!/bin/bash
# oneAPI 2026.1 toolchain for the torch 2.14.0+xpu RL stack on Aurora; sourced by
# every launcher in this directory (override with ONEAPI_ENV_SCRIPT).
#
# torch 2.14.0+xpu ships the oneAPI 2026.1 SYCL runtime (libsycl.so.9). Never mix
# it with a oneAPI 2025.x environment (libsycl.so.8): Triton builds its launcher
# with the icpx on PATH, and two SYCL runtimes in one process segfault in
# sycl::device::get_backend().
# PBS jobs run as non-login bash (no profile), so lmod/MODULEPATH may be empty.
# Prime them if missing so `module load oneapi` can also pull gcc/14.3.0 (its
# depends-on). Without this, the job silently falls back to system g++ 7.5 and
# Triton JIT fails with `braces around scalar initializer` (job 8937339).
if [ -z "${MODULEPATH:-}" ]; then
    if [ -f /etc/cray-pe.d/cray-pe-configuration.sh ]; then
        # shellcheck disable=SC1091
        . /etc/cray-pe.d/cray-pe-configuration.sh
        module_install_prefix=${module_install_prefix:-/usr/share}
        if [ -e "$module_install_prefix/lmod/lmod/init/bash" ]; then
            # shellcheck disable=SC1091
            . "$module_install_prefix/lmod/lmod/init/bash"
            for _p in $mpaths; do
                [ -d "$_p" ] && MODULEPATH=$MODULEPATH:$_p
            done
            export MODULEPATH
            LMOD_SYSTEM_DEFAULT_MODULES=${LMOD_SYSTEM_DEFAULT_MODULES:-$(echo "${init_module_list:-}" | tr ' ' ':')}
            export LMOD_SYSTEM_DEFAULT_MODULES
            module --initial_load --no_redirect restore 2>/dev/null || true
        fi
        unset _p
    fi
    case ":$MODULEPATH:" in
        *":/opt/aurora/26.181.0/spack/unified/1.1.1/install/modulefiles/Core:"*) ;;
        *) [ -d /opt/aurora/26.181.0/spack/unified/1.1.1/install/modulefiles/Core ] && MODULEPATH=$MODULEPATH:/opt/aurora/26.181.0/spack/unified/1.1.1/install/modulefiles/Core; export MODULEPATH ;;
    esac
fi
if [ -n "${MODULEPATH:-}" ] && ! command -v module >/dev/null 2>&1; then
    for _init in /usr/share/lmod/lmod/init/bash /opt/cray/pe/lmod/lmod/init/bash; do
        if [ -f "$_init" ]; then . "$_init"; break; fi
    done
    unset _init
fi
module reset 2>/dev/null || true
module load oneapi/release/2026.1.0 gcc/14.3.0 2>/dev/null || true
if ! icpx --version 2>/dev/null | head -1 | grep -q "2026\.1"; then
    echo "ERROR: env_torch214.sh expected icpx 2026.1, got: $(icpx --version 2>&1 | head -1)"
fi
# Triton JIT invokes $CXX (or g++ on PATH) to build its XPU driver helper.
# The gcc module sets CC/CXX, but a failed `module load` (silent `|| true`)
# would leave system g++ 7.5 on PATH which rejects `char err[1024] = {{0}}`
# (job 8937339). Pin to Spack GCC 14.3 as hard fallback.
_SPAX_GCC_BIN="/opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64/gcc-14.3.0-siitp7a/bin"
if [ -x "$_SPAX_GCC_BIN/gcc" ]; then
    if ! echo "${CC:-}" | grep -q "14\.3"; then export CC="$_SPAX_GCC_BIN/gcc"; fi
    if ! echo "${CXX:-}" | grep -q "14\.3"; then export CXX="$_SPAX_GCC_BIN/g++"; fi
    case ":$PATH:" in *":$_SPAX_GCC_BIN:"*) ;; *) export PATH="$_SPAX_GCC_BIN:$PATH" ;; esac
fi
unset _SPAX_GCC_BIN
# The 2026.1 module sets ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE (one 128 GB device per
# card, so ZE_AFFINITY_MASK=N picks card N and masks >= 6 see no device), plus
# ZE_ENABLE_API_TRACING=1 and ZE_ENABLE_PCI_ID_DEVICE_ORDER=1. Every launcher's tile
# masks assume FLAT (12 x 64 GB tiles per node). Under COMPOSITE, co-located
# processes OOM each other and sampled outputs are not run-to-run reproducible.
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
unset ZE_ENABLE_API_TRACING ZE_ENABLE_PCI_ID_DEVICE_ORDER

# Per-process thread pools, capped because the PBS job cgroup caps TASKS per
# node: /sys/fs/cgroup/jobs/<jobid>/pids.max is 8192 on Aurora, counting
# processes and threads alike, and clone() past it returns EAGAIN.
#
# Three pools size themselves to the core count and so cost ~440 threads per
# process on a 208-thread node (measured: OMP + OpenBLAS 115 and
# monarch-pytokio 104 on a 104-thread node, 413 observed on hardware). An RL
# node runs 2 procs per tile for the trainer role alone (trainer + its
# TorchStore storage volume) plus the generators' procs, so the node total is
# roughly 440 * 2 * tiles: fine at 4 tiles, and at 8 tiles it measured 8189 of
# 8192. That ceiling is how job 8914411 died -- 8-node PPN=8 hit EAGAIN in
# OpenBLAS init and then SIGSEGV -- and how its 2-node twin (8914526) hung
# outright with no error at all.
#
# Capped, a process costs ~13 threads, which keeps every tile count from 1 to 12
# far below the cap. 8 OMP threads is also still generous on compute: 104
# physical cores over 24 procs at 12 tiles is ~4 cores each. Raise them for a
# low-tile-count run if CPU-side work ever looks starved.
#
# The opt-out names are TITAN_XPU_* rather than the variables themselves because
# PBS exports OMP_NUM_THREADS=208 (the node's hardware thread count) into every
# job: `${OMP_NUM_THREADS:-8}` would silently keep 208 and leave the largest of
# the three pools uncapped. Measured on job 8914626 before this was fixed.
export OMP_NUM_THREADS=${TITAN_XPU_OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${TITAN_XPU_MKL_NUM_THREADS:-$OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${TITAN_XPU_OPENBLAS_NUM_THREADS:-1}
export TOKIO_WORKER_THREADS=${TITAN_XPU_TOKIO_WORKER_THREADS:-4}
