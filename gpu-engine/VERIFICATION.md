# Verification status

Per the HONESTY hard constraint in `CLAUDE.md`: every number that lands on a
resume must come from a measurement logged in `bench/results/`. This file is
the ledger of what has actually been built and run, vs. what is written to
spec but unverified. It exists because this repo was developed on a machine
with **no NVIDIA GPU** (`nvidia-smi`/`nvcc` absent, integrated AMD GPU only) —
everything under "unverified" needs a CUDA 12.x + NVIDIA GPU environment
before any claim from it can be trusted.

## Verified (built and run on this machine, no GPU required)

| Milestone | What was run | Result |
|---|---|---|
| M0 export | `export/export_gpt2.py` against real HF `gpt2` weights | `weights.bin` (497,759,232 bytes, 148 tensors), `manifest.json`, `config.json` written; `checksum_sha256 = 03cdb9f90df10be3bdbb8e490ebba85a2a729266455333c7f250c6b3816fe9c2` |
| M0 loader round-trip | `WeightsLoader` (host, `src/reference/weights_loader.cpp`) | Recomputed `checksum_sum` matches the manifest's to `rel_err = 4.568e-15` |
| M1 CPU oracle | `gpu_engine_tests` → `tests/test_end_to_end.cpp`, greedy 32-token decode vs. `export/verify_parity.py`'s HF fp32 reference, 10 fixed prompts | **10/10 exact top-1 token match** |
| M3 (host-side allocator only) | `gpu_engine_tests` → `tests/test_kv_cache.cpp`: alloc/free cycles, exhaustion path, double-free rejection, out-of-range rejection, `block_bytes` constant | All pass (`8 test cases, 1536 assertions, 0 failed`) |

Reproduce: from `gpu-engine/`,
```
export/venv/bin/python export/export_gpt2.py --out models
export/venv/bin/python export/verify_parity.py --out models/parity_reference.json
mkdir -p build && cd build && cmake .. && make -j
GPU_ENGINE_MODELS_DIR=../models ./gpu_engine_tests
```

## Unverified (written to spec, never compiled or run — needs CUDA + NVIDIA GPU)

Everything CMake gates behind `CMAKE_CUDA_COMPILER` being found:
`gpu_kernels` (all of `src/kernels/*.cu`, `src/engine.cu`, `src/kv_cache.cu`,
`src/loader.cpp`), `gpu_kernel_tests` (`tests/test_kernels.cpp`), and
`bench_decode` (`bench/bench_decode.cpp`).

This includes:
- M2 naive GPU path (cuBLAS GEMMs, per-kernel FP16 parity vs. CPU-FP32
  reference, end-to-end greedy parity on ≥8/10 prompts)
- M3's device side (the actual paged KV-cache arena and CUDA kernels reading
  it — only the host-side `BlockPool`/`BlockTable` bookkeeping is verified
  above)
- M4 fusion timing (`bench/results/fusion.csv` — does not exist yet)
- M5 CUDA Graphs on/off comparison (`bench/results/graphs.csv` — does not
  exist yet)
- M6 scheduler integration and the re-measured continuous-vs-static
  benchmark on real GPU compute
- M7 Nsight Compute profiling (`bench/results/ncu_summary.md` — does not
  exist yet)

Known, deliberate deviations from the literal `CLAUDE.md` kernel-design
suggestions (documented in-file, called out again here because they affect
what "matches spec" means once verification starts):
- `src/kernels/attention.cu`: paged attention uses `blockDim.x == head_dim`
  (one thread per lane, streaming online softmax) instead of the suggested
  128-threads/8-per-key cooperative layout. Functionally equivalent (still
  single-pass online softmax, FP32 accumulators); revisit for occupancy
  once M7 profiling is possible.
- `src/kernels/logits.cu`: argmax is a hand-written block reduction instead
  of `cub::DeviceReduce`/`cub::DeviceSegmentedReduce`, to avoid depending on
  a CUB API surface that could not be checked against an installed CUDA
  toolkit's actual headers here.

## What to do once GPU access is available

1. `cmake .. -DCMAKE_BUILD_TYPE=Release` should now report "CUDA toolchain
   found" and configure `gpu_kernels`, `gpu_kernel_tests`, `bench_decode`.
2. `make -j && ctest --output-on-failure` — this is where M2/M3-device bugs
   will surface for the first time. Expect to need real debugging here;
   nothing below `src/` and `tests/test_kernels.cpp` has been iterated
   against a compiler or a GPU yet.
3. Once `gpu_kernel_tests` is green, run `bench/bench_decode.cpp` per the
   commands in the "Benchmark protocol" table in `CLAUDE.md`, redirecting
   output into `bench/results/`.
4. Run `bench/run_ncu.sh` before/after the fusion kernels for the M7
   occupancy table.
5. Only after 1-4 are green and logged should any tok/s, speedup, or
   occupancy number from this module be written down as a resume claim.
