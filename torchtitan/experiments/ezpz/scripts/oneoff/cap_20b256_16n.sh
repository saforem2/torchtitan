#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=168:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=16
#PBS -j oe
# 20B-256 on the CAPACITY queue: 16 nodes, 168h walltime. Same low-contention
# trickle that advanced the 20B-512 chain (cap_20b512_16n.sh) -- pays the
# ~11min startup tax ONCE, then trains + checkpoints for up to a week while the
# 256N prod resumes sit queue-starved behind cluster-wide node scarcity.
# GAS=16 on 16 nodes: 16*12*LBS2*GAS16 = 6144, bit-identical to the canonical
# 256N chain (GBS=6144, LBS=2, pure FSDP).  Pinned CKPT_DIR = the real chain.
# COLLISION NOTE: writes the SAME ckpt dir as the queued 256N prod resumes
# (8698753/54). Only one lane may run at a time -- if a 256N prod slot opens,
# qdel this capacity job first (or hold the prod resume) to avoid a shared-DCP
# race.
cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/runs/agpt-20b-n256/torchtitan-ezpz}"
export NHOSTS_TRAIN=16
export GAS=16
export CKPT_DIR=checkpoints/agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144
export CKPT_INTERVAL=25
exec bash torchtitan/experiments/ezpz/scripts/submit_agpt_20b_autoretry.sh
