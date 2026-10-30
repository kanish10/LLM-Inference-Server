# Kernel tuning harness

The pieces [PerfLens](https://github.com/kanish10/PerfLens) drives when it
searches compile flags and launch parameters for gpu-engine kernels
(`perflens tune`). This directory only builds, checks, and times
candidates; the search strategies, the agent, and the results database
live in PerfLens.

## What can be tuned

All knobs are compile-time macros in `include/engine/tune_params.h`. Their
defaults equal the values the engine shipped with, so a normal CMake build
is unchanged.

| Kernel | Macros | Notes |
|---|---|---|
| `bias_gelu` | `GELU_THREADS`, `GELU_BLOCKS_PER_ROW`, `GELU_VEC` (1 or 2), `GELU_MIN_BLOCKS` | `BLOCKS_PER_ROW > 1` splits each row across several blocks; `VEC=2` uses half2 loads/stores |
| `add_layernorm` | `LN_THREADS`, `LN_MIN_BLOCKS` | one block per row (the reduction needs the whole row) |
| `argmax` | `ARGMAX_THREADS`, `ARGMAX_MIN_BLOCKS` | one block per row |

Every kernel can also take `-maxrregcount=N` and `--use_fast_math`.
`*_THREADS` must be a power of two in [32, 1024], and
`THREADS * MIN_BLOCKS` must fit sm_86's 1536 threads per SM. The header
enforces both with `static_assert`.

## Building one candidate

```bash
# from gpu-engine/
bench/build_candidate.sh bias_gelu '{"threads": 512, "blocks_per_row": 4, "vec": 2, "fast_math": false}'
```

This compiles `bench/kernel_bench.cu` together with only that kernel's
translation unit and the candidate's flags, using `-Xptxas -v`. It prints
a JSON record with `ok`, `compile_ms` (nvcc wall time), the full flag
list, the binary path, and `ptxas_log` (the raw compiler output, which
PerfLens parses for registers, spills, and shared memory).

Builds are cached under `build/tune/<kernel>-<hash>/`. A cached record
keeps its original `compile_ms`; pass `--force` to rebuild. Set
`NVCC=/path/to/nvcc` or `TUNE_ARCH=sm_XX` to override the defaults
(`nvcc`, `sm_86`).

## Running one candidate (GPU required)

```bash
build/tune/bias_gelu-<hash>/kernel_bench --batch 8
```

This prints one JSON line, for example:

```
{"kernel":"bias_gelu","batch":8,"correct":true,"max_abs_err":0.0039,
 "median_us":...,"mad_us":...,"samples":200,"inner":10,...}
```

- **Inputs:** GPT-2 124M shapes (d_model 768, d_ff 3072, vocab 50257) and
  deterministic seeded data.
- **Correctness:** the same rule as `tests/test_kernels.cpp`.
  `max_abs_err < 5e-2` against an FP32 CPU reference computed on the same
  half-rounded inputs. For layernorm, both `out` and `residual_out` are
  checked. For argmax, the result must match exactly, with one planted
  winner per row.
- **Timing:** 20 warmup launches, then 200 cudaEvent-timed groups of 10
  back-to-back launches. It reports the per-launch median and MAD
  (median absolute deviation) across the groups.

## Tests

```bash
python3 -m unittest discover -s bench -p "test_*.py"   # no CUDA needed
```

These cover config validation, config-to-flag mapping, cache keys, and the
build/caching/failure paths, using a fake `nvcc`.

## Notes from the compiler (nvcc 12.8, sm_86)

- Every configuration in PerfLens's default search spaces compiles.
- **Register usage is low everywhere.** bias_gelu uses 13–22 registers
  (VEC=2 adds about 8), add_layernorm 18–21, and argmax 16.
- **`-maxrregcount` and `__launch_bounds__` can't bind for these
  kernels.** ptxas raises any `-maxrregcount` below 24 to 24 on sm_86
  ("adjusting per thread register count of 16 to lower bound of 24"), and
  no kernel here uses more than 22 registers. That is why PerfLens's
  default spaces leave both knobs out. The harness still supports them for
  kernels that do use more registers.
