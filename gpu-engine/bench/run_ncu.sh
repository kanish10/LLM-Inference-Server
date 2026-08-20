#!/usr/bin/env bash
# Collects per-custom-kernel Nsight Compute metrics for the GPU decode step
# (M7 in gpu-engine/CLAUDE.md): gpu__time_duration.sum,
# sm__warps_active.avg.pct_of_peak_sustained_active (achieved occupancy),
# dram__throughput.avg.pct_of_peak_sustained_elapsed,
# lts__t_sector_hit_rate.pct, launch__registers_per_thread.
#
# Requires the `ncu` CLI (Nsight Compute) and an NVIDIA GPU. Not runnable on
# the machine this repo was scaffolded on -- no CUDA device present. See
# gpu-engine/VERIFICATION.md.
#
# Usage:
#   ./run_ncu.sh ./build/bench_decode before-fusion
#   ./run_ncu.sh ./build/bench_decode after-fusion
# Then hand-write results/ncu_summary.md comparing the two tagged runs.
set -euo pipefail

BIN=${1:-./build/bench_decode}
TAG=${2:-untagged}
OUT_DIR="$(dirname "$0")/results"
mkdir -p "$OUT_DIR"

METRICS="gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,launch__registers_per_thread"

ncu --target-processes all \
    --kernel-name-base function \
    --metrics "$METRICS" \
    --csv \
    --log-file "${OUT_DIR}/ncu_${TAG}.csv" \
    "$BIN" --batch 8 --steps 8 --reps 1 --graphs off

echo "Wrote ${OUT_DIR}/ncu_${TAG}.csv"
