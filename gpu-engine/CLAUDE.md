# CLAUDE.md — P1: GPU-Native Inference Path (LLM Inference Server upgrade)
# Rename this file to CLAUDE.md in the repo root before running Claude Code.

## Mission

Upgrade the existing LLM inference server (C++ continuous-batching scheduler + Go gRPC
gateway) from a CPU/stub decode step to a real GPU decode path for GPT-2 124M on an
RTX 3060 (sm_86): cuBLAS GEMMs, custom fused CUDA kernels, a paged KV-cache with
per-sequence block tables, and CUDA Graphs for the decode step. Profile everything
with Nsight Compute/Systems.

The existing scheduler semantics are preserved: single-threaded scheduler is the sole
writer to batch state; mutex only on the waiting-queue splice; decode step runs
lock-free. The GPU engine slots in behind a narrow interface.

## Hard constraints

1. HONESTY: every number that lands on a resume must come from a measurement logged in
   bench/results/. No estimated or interpolated claims. Placeholders stay placeholders
   until measured.
2. Hardware: single RTX 3060 (assume 12GB; keep configurable so 6GB laptop variant
   works — GPT-2 124M weights are ~250MB in FP16, this is not tight).
3. Model: GPT-2 124M exactly (12 layers, 12 heads, d_model=768, head_dim=64,
   vocab=50257, ctx=1024, pre-LayerNorm, learned positional embeddings, weight-tied
   lm_head, GELU tanh approximation — must match HuggingFace `gpt2` numerics).
4. Precision: weights and activations FP16; accumulation FP32 (cuBLAS
   CUBLAS_COMPUTE_32F); LayerNorm and softmax accumulate in FP32.
5. No cuDNN, no TensorRT, no external inference libs. cuBLAS + cub/thrust + custom
   kernels only. (Resume claims cuBLAS; nothing else may be implied.)
6. Every kernel gets a CPU-reference parity test before optimization.

## Repo layout

```
gpu-engine/
  CLAUDE.md                  (this file)
  CMakeLists.txt             (CUDA 12.x, sm_86, doctest via FetchContent)
  export/
    export_gpt2.py           (HF gpt2 -> weights.bin + manifest.json + config.json)
    verify_parity.py         (HF fp32 logits vs engine logits over test prompts)
    tokenize_client.py       (BPE encode/decode helper for demos; engine speaks token IDs)
  include/engine/
    engine_api.h             (public boundary used by the existing scheduler)
    model.h                  (GPT2Config, GPT2Weights, layer pointer structs)
    kv_cache.h               (BlockPool, BlockTable, constants)
    kernels.h                (all kernel launchers)
  src/
    engine.cu                (EngineAPI impl, decode step orchestration, CUDA Graphs)
    loader.cpp               (weights.bin + manifest loader -> device memory)
    kv_cache.cu              (block pool arena, alloc/free, table upload)
    kernels/
      embedding.cu           (wte + wpe gather-add)
      layernorm.cu           (fused residual-add + LayerNorm, FP32 accum)
      gelu.cu                (fused bias + GELU tanh)
      attention.cu           (decode-only paged attention, online softmax)
      logits.cu              (argmax over vocab via cub::DeviceReduce; top-k stretch)
    reference/
      cpu_model.cpp          (straightforward FP32 CPU forward pass — the oracle)
  tests/
    test_kernels.cpp         (per-kernel parity vs CPU reference, doctest)
    test_kv_cache.cpp        (allocator invariants)
    test_end_to_end.cpp      (greedy continuation parity)
  bench/
    bench_decode.cpp         (tok/s, per-step latency, graphs on/off, batch sweep)
    run_ncu.sh               (Nsight Compute metric collection per kernel)
    results/                 (CSV/JSON outputs — the claims ledger sources)
```

## Public interface (what the existing scheduler calls)

```cpp
// engine_api.h — adapt names to the existing scheduler's conventions during M6.
struct EngineConfig { int max_batch; int max_seq_len; int kv_block_budget; };

class Engine {
 public:
  bool init(const std::string& weights_dir, const EngineConfig& cfg);
  // Returns a sequence handle; allocates first KV block. Fails if pool exhausted.
  int  add_sequence(const int* prompt_tokens, int prompt_len);
  // One decode step for the current batch. seq_ids/out_tokens are host arrays.
  // Internally: upload per-step buffers -> (graph) launch -> download next tokens.
  void decode_step(const int* seq_ids, int batch, int* out_next_tokens);
  void free_sequence(int seq_id);   // returns all blocks to the pool
  size_t free_blocks() const;       // scheduler admission control input
};
```

Prefill strategy (deliberate scoping): prompts are consumed token-by-token through the
same decode path (correct, zero extra kernels). Report decode-phase tok/s as the
headline (standard practice). Batched prefill with a masked attention GEMM path is a
stretch goal, not required.

## Data structures

### Paged KV-cache
- BLOCK_TOKENS = 16.
- One block covers ALL layers for 16 token positions:
  layout `[layer][2(K,V)][BLOCK_TOKENS][n_heads][head_dim]` half.
  Block bytes = 12 * 2 * 16 * 12 * 64 * 2 = 589,824 (~0.56 MB... compute: 12*2*16*768*2 =
  589,824 bytes). One cudaMalloc arena of kv_block_budget blocks.
- BlockPool (host): `std::vector<int> free_stack`; O(1) alloc/free; assert on
  double-free (debug map).
- BlockTable per sequence (host): `std::vector<int> block_ids`; grows when
  seq_len % BLOCK_TOKENS == 0 rolls over.
- Device-side per-step buffers (fixed max_batch capacity, persistent allocations —
  REQUIRED for CUDA Graphs stability):
  `int d_token_ids[max_batch]`, `int d_seq_lens[max_batch]`,
  `int d_block_tables[max_batch][max_blocks_per_seq]` (row-padded).
  Host writes these each step via one cudaMemcpyAsync each, then launches the graph.

### Weights
- `GPT2Weights`: contiguous device buffer; per-layer pointer struct
  {ln1_g, ln1_b, qkv_w, qkv_b, proj_w, proj_b, ln2_g, ln2_b, fc_w, fc_b, fc_proj_w,
  fc_proj_b} + {wte, wpe, lnf_g, lnf_b}. lm_head aliases wte (weight tying).
- Export gotcha: HF GPT-2 uses Conv1D — weight shape (in, out). export_gpt2.py must
  TRANSPOSE to (out, in) row-major for standard GEMM, and record shapes in
  manifest.json {name, dtype, shape, byte_offset}.

## Decode step (per token, per batch) — kernel sequence

For each of 12 layers:
1. fused residual-add + LayerNorm (ln1)            — custom kernel, FP32 accum
2. QKV projection: GEMM [batch,768]x[768,2304]     — cuBLAS
3. append K,V for this position into paged blocks  — small scatter kernel (block
   table indexed write)
4. paged attention (decode-only)                   — custom kernel (below)
5. output projection GEMM + fused bias             — cuBLAS + epilogue in gelu.cu style
6. fused residual-add + LayerNorm (ln2)
7. MLP up GEMM [batch,768]x[768,3072]              — cuBLAS
8. fused bias + GELU(tanh)                         — custom kernel
9. MLP down GEMM + bias                            — cuBLAS
Then: final LayerNorm -> logits GEMM [batch,768]x[768,50257] (cuBLAS, lm_head=wte^T)
-> argmax kernel -> d_next_tokens.

### Paged attention kernel (the centerpiece)
- Grid: (batch * n_heads) thread blocks; 128 threads each.
- Each block: load its query vector (64 halfs) into registers/smem; iterate the
  sequence's block table; for each KV block, compute up to 16 q·k dots (threads
  cooperate across head_dim), maintain ONLINE SOFTMAX running state (m = running max,
  l = running denom, acc[64] = running weighted V sum) so only one pass over KV;
  FP32 accumulators; write out acc/l as the head output.
- Fallback milestone (if online softmax stalls you): two-pass (pass 1 max, pass 2
  exp-sum + weighted V). Ship two-pass first, upgrade after parity is green.
- Causality is implicit in decode (all cached positions precede the query).

### CUDA Graphs
- Capture the ENTIRE decode step (all 12 layers + logits + argmax) once per batch-size
  bucket {1, 2, 4, 8}; pad the live batch up to the bucket.
- Stability requirement: all kernel parameters point at the persistent per-step device
  buffers (indirection), so no per-step graph rebuilds; cudaGraphExecUpdate only if a
  buffer address ever changes (it shouldn't).
- Measure per-step wall time graphs-ON vs graphs-OFF at each bucket — this is the
  launch-overhead reduction number for the resume.

## Milestones & acceptance criteria

- M0 export: export_gpt2.py produces weights.bin/manifest/config; loader round-trips
  shapes; a printed weight checksum matches the Python side.
- M1 CPU oracle: cpu_model.cpp greedy-decodes 32 tokens matching HF fp32 top-1 exactly
  on 10 fixed prompts (this validates architecture/numerics before any CUDA).
- M2 naive GPU: cuBLAS GEMMs + unfused kernels + contiguous (non-paged) KV; per-kernel
  parity tests pass (FP16 tolerance: max_abs_err vs CPU-FP32 reference under 5e-2 per
  activation tensor; end-to-end: greedy matches CPU oracle for 32 tokens on >=8/10
  prompts, mismatches only at near-tie logits, logged).
- M3 paged KV: allocator tests (alloc/free cycles, exhaustion path, no double-free);
  paged attention parity vs M2 contiguous attention bit-for-bit on same seeds is NOT
  required — require max_abs_err < 1e-3 on attention output and identical greedy
  continuations.
- M4 fusion: fused bias+GELU and fused residual+LN parity tests; record per-kernel
  duration before/after in bench/results/fusion.csv.
- M5 CUDA Graphs: identical outputs graphs-ON vs OFF; record per-step latency table
  (bucket x on/off) -> bench/results/graphs.csv. THIS produces the [X]% claim.
- M6 integration: wire EngineAPI into the existing scheduler; the original benchmark
  methodology (500 requests, Poisson arrivals at 80% of static capacity, output lengths
  16–512, batch 8) reruns end-to-end on real GPU compute; continuous-vs-static
  comparison re-measured and logged (the 3.2× claim gets revalidated or updated —
  whatever it measures is what the resume says).
- M7 profiling: bench/run_ncu.sh collects, per custom kernel: gpu__time_duration.sum,
  sm__warps_active.avg.pct_of_peak_sustained_active (achieved occupancy),
  dram__throughput.avg.pct_of_peak_sustained_elapsed, lts__t_sector_hit_rate.pct,
  launch__registers_per_thread. Write bench/results/ncu_summary.md with a
  before/after-fusion table. THIS produces the occupancy [X%→Y%] claim.

## Benchmark protocol (claims ledger)

| Resume claim | Source file | Command |
|---|---|---|
| [N] tok/s decode, batch 8, GPT-2 124M, RTX 3060 | bench/results/decode.csv | bench_decode --batch 8 --steps 512 --reps 5 (report median) |
| CUDA Graphs cut per-step launch overhead [X]% | bench/results/graphs.csv | bench_decode --graphs on/off sweep |
| occupancy [X]%→[Y]% from fusion | bench/results/ncu_summary.md | run_ncu.sh pre/post tags |
| KV fragmentation eliminated across concurrent seqs | tests/test_kv_cache.cpp | ctest |

Rules: 5 reps minimum, report medians, fixed GPU clocks if possible
(`nvidia-smi -lgc` needs admin; if unavailable, note thermal variance in results).

## Pitfalls Claude Code must avoid
- HF Conv1D transpose (most common GPT-2 port bug — M1 catches it).
- GELU must be tanh approximation, not erf (HF gpt2 uses gelu_new).
- cuBLAS is column-major: either use CUBLAS_OP flags to compute row-major as
  transposed column-major, or standardize on (B^T A^T)=(AB)^T convention — document
  the chosen convention once in model.h and never deviate.
- FP16 attention logits overflow: scale q by 1/sqrt(64) BEFORE the dot, accumulate
  in FP32.
- Graph capture fails on any cudaMalloc/host-sync inside the step — all allocations
  at init only.
- ncu durations are replay-based: use them for kernel-level claims; use CUDA-event
  wall clock in bench_decode for step/throughput claims.

## Stretch (only after M7)
- Batched prefill path (masked attention over prompt in one pass)
- Top-k sampling kernel
- Online-softmax upgrade if two-pass was shipped
- INT8 weight-only quantization of MLP GEMMs (ties to the AgroScan quantization story)

## Timebox
Weekend 1: M0–M2. Weekend 2: M3–M5. Weeknights after: M6–M7.
If remote GPU (Tailscale to the 3060 box): develop over SSH; keep builds incremental
(ccache); run ncu with `--target-processes all` under the same user.