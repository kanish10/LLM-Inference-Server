#!/usr/bin/env bash
# Build one PerfLens Tune candidate. Thin wrapper so callers have a stable
# shell entry point; all logic (and its tests) lives in tune_build.py.
#   bench/build_candidate.sh <kernel> '<config-json>' [--force]
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 bench/tune_build.py "$@"
