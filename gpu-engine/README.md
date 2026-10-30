# gpu-engine

GPU-native GPT-2 124M decode path (cuBLAS + custom CUDA kernels + paged
KV-cache + CUDA Graphs), meant to eventually plug into the scheduler in
`../cpp-engine` behind `include/engine/engine_api.h`. Full spec:
`CLAUDE.md`. Current verification status (what's actually been run vs. what
is spec-only): `VERIFICATION.md` — **read that before trusting any number
or claim from this module.**

## Layout

```
export/          Python: HF gpt2 -> weights.bin/manifest.json/config.json, HF parity reference, tokenizer helper
include/engine/   Public headers: engine_api.h (scheduler boundary), model.h, kv_cache.h, kernels.h, gpu_weights.h
src/              Engine implementation: engine.cu, loader.cpp, kv_cache.cu, kernels/*.cu, kv_cache_pool.cpp (host allocator), reference/ (CPU oracle)
tests/            doctest: test_end_to_end.cpp (M1, CPU-only), test_kv_cache.cpp (CPU-only), test_kernels.cpp (M2, needs CUDA)
bench/            bench_decode.cpp (needs CUDA), run_ncu.sh (needs Nsight Compute + GPU), results/ (CSV/MD outputs)
```

## Building the CPU-only pieces (no GPU required)

This is the M0/M1 path plus the host-side KV allocator (M3's bookkeeping
half) -- genuinely verifiable on any machine.

```bash
# One-time: export real GPT-2 124M weights + HF fp32 parity reference
python3 -m venv export/venv
export/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
export/venv/bin/pip install transformers numpy
export/venv/bin/python export/export_gpt2.py --out models
export/venv/bin/python export/verify_parity.py --out models/parity_reference.json

# Build + run
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
GPU_ENGINE_MODELS_DIR=../models ./gpu_engine_tests
```

Expect `CPU oracle exact top-1 match: 10/10` and all `BlockPool`/`BlockTable`
tests passing.

## Building the CUDA pieces (needs an NVIDIA GPU + CUDA 12.x toolkit)

CMake auto-detects a CUDA compiler. On a machine that has one, the same
`cmake .. && make -j` above additionally configures and builds
`gpu_kernels`, `gpu_kernel_tests`, and `bench_decode`. Then:

```bash
./gpu_kernel_tests                                    # M2 kernel + end-to-end parity
./bench_decode --weights ../models --batch 8 --steps 512 --reps 5 --graphs both --out ../bench/results/decode.csv
../bench/run_ncu.sh ./bench_decode before-fusion
```

None of this has been run yet -- see `VERIFICATION.md` for exactly what's
been checked and what a first real run should expect to shake out.

## Kernel tuning harness

`bench/kernel_bench.cu` and `bench/build_candidate.sh` build, check, and
time one kernel under a given set of compile flags and launch parameters.
[PerfLens](https://github.com/kanish10/PerfLens)'s `perflens tune` uses them
to search those settings. See [`bench/TUNING.md`](bench/TUNING.md).
