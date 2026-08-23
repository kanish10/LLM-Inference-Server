# M7 — Nsight Compute per-kernel profile

Real `ncu` run on an RTX 3060 (VM instance with genuine root — `CAP_SYS_ADMIN`
and `CAP_PERFMON` present, unlike the earlier Docker-container rental this
project was otherwise built on; see `VERIFICATION.md` for why that
distinction matters). `bash bench/run_ncu.sh`-equivalent invocation: batch 8,
8 decode steps, graphs off, metric set `gpu__time_duration.sum,
sm__warps_active.avg.pct_of_peak_sustained_active,
dram__throughput.avg.pct_of_peak_sustained_elapsed,
lts__t_sector_hit_rate.pct, launch__registers_per_thread`. Raw data:
`ncu_fused-asis.csv` (8,198 rows).

**No before/after-fusion comparison** — same reason as M4: the kernels were
written fused from the start, so there's no unfused variant in the codebase
to diff against. This is the as-is profile of the fused kernels, not a
before/after delta. `gpu__time_duration.sum` below is `ncu` replay-based and
**not** a real-world timing claim (that's `bench/results/decode.csv`, CUDA
events, per the project's own documented pitfall) — read the other four
columns as the real per-kernel characterization.

## Custom kernels

| Kernel | Achieved occupancy | DRAM throughput | L2 hit rate | Registers/thread |
|---|---|---|---|---|
| `embedding_kernel` | 16.4% | 0.7% | 89.1% | 22 |
| `add_layernorm_kernel` (fused residual+LN) | 16.5% | 1.7% | 68.4% | 27 |
| `bias_gelu_kernel` (fused bias+GELU) | 16.4% | 2.8% | 66.4% | 16 |
| `kv_append_kernel` | 13.7% | 3.3% | 64.3% | 22 |
| `paged_attention_kernel` | 14.0% | 4.8% | 26.9% | 40 |
| `argmax_kernel` | 16.7% | 3.3% | 11.8% | 16 |
| `add_bias_kernel` | 16.4% | 2.2% | 69.5% | 16 |
| `add_one_kernel` | 2.1% | 0.4% | 88.0% | 16 |

## cuBLAS GEMMs (for context, not ours)

| Kernel | Achieved occupancy | DRAM throughput | L2 hit rate | Registers/thread |
|---|---|---|---|---|
| `ampere_fp16_s16816gemm...` | 8.3% | 58.7% | 11.9% | 102 |
| `cutlass::Kernel2<...80_wmma...>` (large) | 9.1% | 65.9% | 13.8% | 124 |
| `cutlass::Kernel2<...80_wmma...>` (small) | 8.3% | 42.7% | 22.2% | 80 |
| `cutlass::Kernel2<...75_wmma...>` (lm_head, once/step) | 16.5% | 65.7% | 10.6% | 254 |

## What this actually shows

Every kernel — ours and cuBLAS's — sits at **8–17% achieved occupancy**.
That's not a fusion-tuning problem; it's GPT-2 124M at batch 8 being far too
small a workload to fill an RTX 3060. Consistent with M5's finding that
per-step latency barely moves across batch 1→8 (launch/overhead-bound, not
compute-bound) — this is the same story from a different instrument.
`add_one_kernel`'s 2.1% is the outlier worth flagging honestly rather than
hiding: it's a tiny utility kernel with essentially nothing to parallelize,
not evidence of a problem to fix.

`paged_attention_kernel`'s 26.9% L2 hit rate is the lowest among our custom
kernels — expected, since it strides across the paged KV-cache arena rather
than reading a small resident working set the way `embedding_kernel`
(89.1%) or `add_bias_kernel` (69.5%) do.
