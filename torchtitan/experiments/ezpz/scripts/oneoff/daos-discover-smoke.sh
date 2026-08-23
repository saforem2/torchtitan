#!/bin/bash
# Account: AuroraGPT works here. The /soft/daos examples default to
# Aurora_deployment, which we have no allocation on ('No active allocation
# found'). AuroraGPT on small-daos gives 'Unauthorized Request' -- that is the
# QUEUE's group ACL rejecting us, not the account. aurora_daos_test is a unix
# group, not a PBS project ('not found: Project aurora_daos_test').
#PBS -A AuroraGPT
# alcf_daos_cn, not small-daos: the DAOS queue ACLs are group-gated and we
# only match one of them.
#   small-daos    acl_groups = alcf-ci-cd-tests            (no)
#   daos-scatter  acl_groups = DAOS_HPE_TEST,Intel-...     (no)
#   alcf_daos_cn  acl_groups = ...,aurora_daos_test,...    (YES, we are in it)
#PBS -q alcf_daos_cn
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -N daos-discover-smoke
#PBS -j oe
#
# First DAOS contact for AuroraGPT. Two questions, in order:
#
#   1. WHAT POOL DO WE ACTUALLY HAVE? `daos pool list` cannot run on a login
#      node -- there is no daos_agent.sock there (DER_AGENT_COMM), the agent
#      only runs on compute nodes inside a DAOS-enabled job. So the inventory
#      question itself requires submitting. Our groups include
#      `aurora_daos_test` and `Aurora_deployment` but NOT an AuroraGPT-named
#      DAOS group, so this may come back empty -- which is itself the answer
#      (we need to request a pool from ALCF support).
#
#   2. IF a pool exists, does the basic POSIX path work end to end? Create a
#      container, mount it with dfuse, run ALCF's own posix-write/posix-read.
#
# Motivation: DCP checkpoint saves are metadata-heavy in exactly the way DAOS
# is built for -- a 512N save writes 6,144 shards into ONE directory, and this
# week Lustre wedged in cl_sync_io_wait and took 10+ min on staleness scans.
# But that torch/XPU test is deliberately NOT here: the DAOS examples need a
# DAOS-specific MPICH from /soft/restricted/CNDA/updates/modulefiles, which
# conflicts with our production mpich/50.1. Learn the pool situation first
# with ALCF-supplied binaries only, then decide about the module swap.
#
# Queue note: the DAOS queues are lightly used (0 running at submit time),
# but they are separate from prod/capacity so this does not compete with
# the umbrella.

set -x

echo "############ 1. environment ############"
date
hostname
echo "PBS_JOBID=$PBS_JOBID  nodes=$(wc -l < "$PBS_NODEFILE")"

module load daos/base 2>&1 | tail -2
module list 2>&1 | tail -12

echo "############ 2. is the agent up on the compute node? ############"
ls -la /var/run/daos_agent/ 2>&1 | head -5
pgrep -a daos_agent 2>&1 | head -3

echo "############ 3. THE QUESTION: what pools can we see? ############"
daos pool list 2>&1 | head -40
echo "---- exit=$? ----"

echo "############ 4. per-pool detail ############"
# Parse pool names out of `daos pool list` and query each. If this prints
# nothing, we have no pool and the answer to the whole exercise is
# "request one from ALCF support".
POOLS=$(daos pool list 2>/dev/null | awk 'NR>2 && $1 !~ /^-/ && NF>0 {print $1}')
echo "parsed pools: [${POOLS}]"
for p in $POOLS; do
    echo "---- pool $p"
    daos pool query "$p" 2>&1 | head -20
    echo "---- containers in $p"
    daos container list "$p" 2>&1 | head -10
done

if [ -z "$POOLS" ]; then
    echo "############ NO POOL VISIBLE -- stopping here ############"
    echo "Next step is an ALCF support request for an AuroraGPT DAOS pool."
    echo "Everything else (daos CLI 2.6.5, daos/base module, /soft/daos"
    echo "helpers + examples, the small-daos queue) is already in place."
    exit 0
fi

echo "############ 5. POSIX smoke on the first pool ############"
DAOS_POOL=$(echo "$POOLS" | head -1)
DAOS_CONT="agpt-smoke-$$"
echo "using pool=$DAOS_POOL cont=$DAOS_CONT"

daos container create --type POSIX "$DAOS_POOL" "$DAOS_CONT" 2>&1 | head -12 || {
    echo "container create FAILED -- likely no write permission on this pool"
    exit 1
}

clean-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}" 2>&1 | tail -2
launch-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}" 2>&1 | tail -5

MNT=/tmp/${DAOS_POOL}/${DAOS_CONT}
echo "---- mount check"
ls -la "$MNT" 2>&1 | head -5
df -h "$MNT" 2>&1 | tail -2

echo "############ 6. plain POSIX write/read (no MPI) ############"
# Deliberately shell-only: proves the mount works before involving any MPI
# module swap. 200 MB sequential, then a metadata-ish burst of small files,
# which is closer to what a DCP save actually does.
cd "$MNT" || exit 1
echo "-- sequential 200MB"
time dd if=/dev/zero of=seq.bin bs=1M count=200 oflag=direct 2>&1 | tail -3
time dd if=seq.bin of=/dev/null bs=1M 2>&1 | tail -3
echo "-- 512 small files (DCP-shard-like metadata burst)"
mkdir -p shards
time ( for i in $(seq 1 512); do dd if=/dev/zero of=shards/s$i bs=4k count=1 2>/dev/null; done )
time ls shards | wc -l
echo "-- same burst on Lustre for comparison"
LMNT=/flare/AuroraGPT/foremans/daos-compare-$$
mkdir -p "$LMNT/shards"
time ( for i in $(seq 1 512); do dd if=/dev/zero of=$LMNT/shards/s$i bs=4k count=1 2>/dev/null; done )
time ls "$LMNT/shards" | wc -l
# `backup` (~/.local/bin/backup) rather than rm, per project rule -- even
# though $LMNT is a scratch dir this script created 4 lines above.
PATH="$HOME/.local/bin:$PATH" backup "$LMNT" 2>&1 | tail -1

echo "############ 7. cleanup ############"
cd /
clean-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}" 2>&1 | tail -2
daos container destroy "$DAOS_POOL" "$DAOS_CONT" 2>&1 | tail -3

echo "############ done ############"
date
