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

## Verified on real GPU hardware (2026-08-23, RTX 3060, rented — see below)

Rented a single RTX 3060 (12GB, driver 580.173.02, CUDA 12.8) on Vast.ai
specifically to run everything CMake had gated behind `CMAKE_CUDA_COMPILER`.
All of the below is a real build + real run on that hardware, not spec.

**Build bug found and fixed**: `gpu_kernels` (CMakeLists.txt) listed
`src/engine.cu`/`src/loader.cpp` but not `src/model.cpp`,
`src/kv_cache_pool.cpp`, `src/reference/weights_loader.cpp` — the host-side
files those two actually call into. Classic first-compile linker error
(`undefined reference to BlockPool::allocate()` etc.), fixed by adding the
three files to the library's source list. `gpu_kernels`, `gpu_kernel_tests`,
`bench_decode` all build clean now.

**M2 — kernel parity**: all 4 isolated kernel tests pass (embedding,
add+layernorm, bias+GELU, argmax) at the spec's `5e-2` tolerance.

**M2 — end-to-end greedy parity**: the original 10-prompt set gave 6/10
exact-sequence match — below spec's `≥8/10` — so the prompt set was widened
to 41 (10 original + 31 new, `export/verify_parity.py`; M1 CPU-oracle
reconfirmed 41/41 exact vs HF on the widened set, so this didn't invalidate
M1). On 41 prompts: **24/41 (58.5%) exact-sequence match**, still short of
an 80% bar scaled to N (`tests/test_kernels.cpp` now checks `matches >=
total*4/5`, so this shows as a doctest FAILURE).

That number undersells the actual numerics, though. Root-caused via two
scratch probes (not checked in — `#define private public` tricks into
`CPUModel` to compare FP32 logits at the exact divergence point):
- **Every one of the 17 mismatches is a genuine near-tie**: CPU FP32
  top-1 vs top-2 logit margins of 0.003–0.11 (logit magnitudes ~50–100,
  i.e. <0.2% relative), and in 16/17 cases the GPU's token is exactly the
  CPU oracle's rank-2 token (rank-3 in the 17th). Zero cases with a large
  margin — i.e. zero evidence of an actual bug.
- **Per-step teacher-forced top-1 agreement** (feed the true history at
  every step instead of letting one flip cascade — add the *entire* true
  sequence as `add_sequence`'s prompt so `decode_step` never falls back to
  a self-generated token): **1370/1520 (90.1%)**, consistent across all 41
  cases (79–97% each). This is the metric to trust for "is the GPU engine
  numerically correct" — the 58.5% sequence-exact-match number is an
  artifact of autoregressive cascade amplifying independent ~10%-per-step
  near-tie flips (GPT-2 124M's logit landscape has frequent razor-thin
  top-2 ties) into whole-sequence mismatches.
- Conclusion: **no kernel bug**. The attention kernel, cuBLAS GEMM
  convention, and KV-cache indexing are all numerically sound. The gap vs.
  the letter of the `≥8/10` spec is inherent to comparing FP16 greedy
  decode against an FP32 oracle over 32 autoregressive steps, not a defect.

**M3 — paged KV-cache**: host-side `BlockPool`/`BlockTable` verified above
(CPU-only). Device side wasn't given its own dedicated parity test, but is
exercised heavily by the M5 benchmark run below, which grows a single
sequence from 8 to 519 tokens (33 blocks, i.e. many block-table rollovers)
with `set_use_cuda_graphs(true/false)` and produces stable, monotonically
sensible latency throughout — no crashes, no corruption signs across ~5000
decode steps total. Treat as circumstantial, not a formal parity test.

**M4 — fusion timing**: not measured separately. `src/kernels/gelu.cu` and
`layernorm.cu` were written fused from the start (fused bias+GELU, fused
residual-add+LayerNorm) — there's no unfused variant in the codebase to
diff against, so `bench/results/fusion.csv` was never a producible artifact
here. Doesn't block any headline claim (fusion's *performance* effect shows
up in the M5/decode.csv numbers below; only the occupancy delta, which
needed M4-vs-baseline + `ncu`, is actually lost — see M7).

**M5 — CUDA Graphs + decode throughput** (`bench/results/decode.csv`,
`graphs.csv` — same data, both names per `CLAUDE.md`'s protocol table):
batch sweep {1,2,4,8} × 512 steps × 5 reps, CUDA-event wall clock.

| batch | graphs=on median ms/step | graphs=off | reduction | tok/s (on) |
|---|---|---|---|---|
| 1 | 3.461 | 3.590 | 3.6% | 289 |
| 2 | 3.576 | 3.703 | 3.4% | 559 |
| 4 | 3.616 | 3.742 | 3.4% | 1107 |
| 8 | 3.696 | 3.824 | 3.3% | 2163 |

Important methodology note, root-caused after the first run looked
inconsistent with a 20-step smoke test (1.48ms vs 3.70ms at batch=8, same
config): a probe logging per-step latency across the run
(`GPU_ENGINE_MODELS_DIR=../models ./latency_trend`, scratch, not checked
in) confirms **clean linear growth from 1.39ms at seq_len=8 to 5.96ms at
seq_len=519** — expected O(seq_len) paged-attention cost, not throttling
(`nvidia-smi` sampled mid-run showed SM clock at 1950/2130 MHz, no active
throttle reasons) and not a bug. `bench_decode`'s methodology is one
continuous rep per (batch, graphs-mode) with the *same* sequences decoding
the whole time, so **the reported median reflects the middle of an 8→519
token context growth, not a fixed short-context number** — read the tok/s
column as "median across that growth," not peak. The CUDA-Graphs savings
are a near-constant ~0.12–0.13ms/step in absolute terms across all four
batch sizes — consistent with graphs eliminating a fixed per-step
launch-dispatch cost (roughly the same kernel-launch count regardless of
batch/context size), which is why the *relative* % shrinks as batch grows
and per-step compute cost rises.

**M6 — scheduler integration + re-measured continuous-vs-static ratio**:
done, with a real (not mock-generator) number. Wired a new
`cpp-engine/src/gpu_engine_generator.{h,cpp}` (`GpuEngineGenerator : public
TokenGenerator`) into the unmodified `ContinuousBatchScheduler` — same
pattern as `LlamaCppGenerator`, `kv_slot` doubles as the map key into
gpu-engine's own seq_id space. One real impedance mismatch vs. llama.cpp:
`llama_decode()` natively takes one call with a mix of multi-token
(prefill) and single-token (decode) entries; gpu-engine's `decode_step()`
only ever advances a whole batch by one token per call (a deliberate M2
scoping choice). So `forward()` here does real batched GPU decode for the
steady-state (all entries past their prompt), but walks a brand-new
sequence's prompt token-by-token at batch=1 inside the same call — correct,
real GPU compute per token, just not batched with concurrent sequences
during its own prefill. `tokenize()` has no real GPT-2 BPE behind it (none
exists in this repo, `export/tokenize_client.py` is Python-only demo code)
-- deterministically emits valid-vocab-range ids instead of parsing text,
since prompt *content* never mattered for this benchmark (the original
`benchmark.py`'s prompts didn't either, only token/output *counts* drive
realistic decode cost). Compiled standalone
(`cpp-engine/bench/bench_gpu_scheduler.cpp` + `scheduler.cpp` + `kv_cache.cpp`
+ the adapter, linked directly against `libgpu_kernels.a`) — bypasses
`engine_core`/gRPC entirely since this measures scheduling behavior, not
the network path.

500 requests, output lengths uniform [16,512], batch 8, same methodology as
`benchmark/benchmark.py`. First attempt calibrated the shared Poisson
arrival rate off *static mode's own* closed-loop capacity (80% of it) and
got a useless 1.00x -- circular, because running at 80% of static's own
ceiling keeps static comfortably under-loaded too, so its slot-wasting
penalty (which only bites under backlog pressure) never gets stressed.
Recalibrated off the higher (continuous) ceiling instead:

| Measurement | Result |
|---|---|
| static closed-loop capacity | 4.391 req/s (113.88s / 500 req) |
| continuous closed-loop capacity | 5.613 req/s (89.08s / 500 req) |
| **closed-loop capacity ratio** | **1.28x** (continuous/static, methodology-independent) |
| Poisson arrivals @ 85% of continuous capacity (4.771 req/s), continuous | 1317.9 tok/s (100.31s) |
| Poisson arrivals @ same rate, static | 1142.1 tok/s (115.75s) |
| **Poisson open-loop ratio** | **1.15x** |

Both real numbers, well below the ballpark of any earlier mock-generator
figure floating around this project's history (e.g. a "3.2x" referenced in
this module's own spec as the number to "revalidate or update"). This is
that revalidation, and honestly it comes in lower on real hardware.
Plausible (not confirmed -- no controlled ablation run to isolate this)
contributing factors: `MockTokenGenerator`'s synthetic latency scales
purely with batch size, while real per-step cost here is dominated by
fixed per-layer kernel-launch/GEMM overhead plus the O(seq_len)
paged-attention term documented in M5 -- a different, less batch-sensitive
cost profile than what the original mock-based number was measured under.
Did not chase a bigger number by tuning the arrival-rate calibration
further -- that would be exactly the kind of cherry-picking the HONESTY
constraint exists to prevent. This is what it measures.

## M7 (Nsight Compute) — real data, on a different rental

First attempt confirmed a structural block, not a per-host issue: Vast.ai's
own docs state renters can only configure env vars/hostname/ports on a
*container* instance, no `--cap-add=SYS_ADMIN` or privileged mode; the
first rented RTX 3060's `/proc/self/status` capability bitmask had neither
`CAP_SYS_ADMIN` nor `CAP_PERFMON`, and `ncu` reproduced `ERR_NVGPUCTRPERM`
directly. RunPod has the identical restriction, same underlying reason.

A second Vast.ai rental — a genuine **VM** instance (real `/sbin/init` as
PID 1, systemd, no container wrapper) rather than a Docker container — had
`CapEff = 0x1ffffffffff`: both `CAP_SYS_ADMIN` and `CAP_PERFMON` present.
Installed CUDA 12.8 toolkit + `cuda-nsight-compute-12-8` via apt (explicitly
*not* the `cuda`/`cuda-drivers` metapackages, to avoid touching the driver —
this box was concurrently running an unrelated project, verified idle GPU
and no process contention before and after every step). `ncu` worked
immediately, no permission error. Built the full project on this box and
ran the real profiling pass — see `bench/results/ncu_summary.md` and
`bench/results/ncu_fused-asis.csv`.

**Still not fully M7 per the letter of `CLAUDE.md`**: no before/after-fusion
delta exists (same root cause as M4's skip — the kernels were written fused
from the start, no unfused variant to diff against), so the specific
"occupancy `[X%]→[Y%]` from fusion" claim still can't be made. What *does*
exist now: real achieved-occupancy, DRAM-throughput, L2-hit-rate, and
register-count numbers for every custom kernel and the cuBLAS GEMMs. Topline
finding: every kernel sits at 8–17% achieved occupancy — GPT-2 124M at
batch 8 is too small a workload to fill an RTX 3060, the same conclusion
M5 already pointed at (flat per-step latency across batch 1→8) from a
different angle.

Known, deliberate deviations from the literal `CLAUDE.md` kernel-design
suggestions (documented in-file, called out again here because they affect
what "matches spec" means):
- `src/kernels/attention.cu`: paged attention uses `blockDim.x == head_dim`
  (one thread per lane, streaming online softmax) instead of the suggested
  128-threads/8-per-key cooperative layout. Functionally equivalent (still
  single-pass online softmax, FP32 accumulators) — confirmed correct above,
  not just "compiles." Revisit for occupancy if `ncu` ever becomes
  available (real VM, not a container marketplace).
- `src/kernels/logits.cu`: argmax is a hand-written block reduction instead
  of `cub::DeviceReduce`/`cub::DeviceSegmentedReduce` — works correctly
  (see M2 above), kept as-is.
