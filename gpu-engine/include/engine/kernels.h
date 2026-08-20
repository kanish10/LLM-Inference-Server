#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Kernel launchers for the GPT-2 decode step. Every kernel accumulates in
// FP32 internally even though inputs/outputs are half (FP16), per the
// precision constraint in gpu-engine/CLAUDE.md.
//
// cuBLAS convention (documented once, per the CLAUDE.md pitfall list): all
// weight matrices are stored row-major as [out_features, in_features]
// (export_gpt2.py already transposed HF's Conv1D layout to this). To
// compute row-major Y[batch,O] = X[batch,I] @ W^T with W stored row-major
// [O,I], call:
//   cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, O, batch, I, &alpha,
//               W, I, X, I, &beta, Y, O);
// (A row-major (R,C) buffer is bit-identical to a column-major (C,R)
// buffer of the same data; opA=T reads W's column-major (I,O) view as its
// transpose (O,I) = W itself, without any explicit data transpose.) This
// convention is used everywhere a GEMM is issued in src/engine.cu; do not
// deviate from it there.
//
// seq_lens convention: launch_kv_append expects seq_lens[b] = number of
// tokens already cached for sequence b (i.e. the position this step's new
// token is written to). launch_paged_attention expects seq_lens[b] to
// already include this step's token (num_keys = cached count AFTER the
// append). Callers must append first, then increment seq_lens by one
// on-device, then run attention -- see src/engine.cu.

namespace kernels {

// out[b,:] = wte[token_ids[b],:] + wpe[positions[b],:]
void launch_embedding(half* out, const half* wte, const half* wpe, const int* token_ids,
                       const int* positions, int batch, int d_model, cudaStream_t stream);

// out[b,:] = x[b,:] + bias[:]  (broadcast bias over batch)
void launch_add_bias(half* out, const half* x, const half* bias, int batch, int dim,
                      cudaStream_t stream);

// Fused residual-add + LayerNorm, FP32 accumulation.
// residual_out = residual_in + x   (kept for the next residual branch)
// out = LayerNorm(residual_out) * gamma + beta
void launch_add_layernorm(half* out, half* residual_out, const half* x, const half* residual_in,
                           const half* gamma, const half* beta, int batch, int d_model, float eps,
                           cudaStream_t stream);

// Fused bias-add + GELU(tanh approximation, HF gpt2 "gelu_new").
// out[b,d] = gelu_tanh(x[b,d] + bias[d])
void launch_bias_gelu(half* out, const half* x, const half* bias, int batch, int d_ff,
                       cudaStream_t stream);

// Scatter this step's K,V (one token per sequence) into their paged block
// slot. k/v may point into a strided buffer (e.g. the K/V slice of a
// combined QKV GEMM output where each row is [Q|K|V]) -- kv_row_stride is
// the element distance between consecutive batch rows in k and v (pass
// head_dim*n_head if k/v are their own tightly packed [batch,D] buffers).
// See seq_lens convention above.
void launch_kv_append(half* kv_pool, const half* k, const half* v, int kv_row_stride,
                       const int* block_tables, const int* seq_lens, int batch, int n_head,
                       int head_dim, int max_blocks_per_seq, int layer_idx, int n_layer,
                       cudaStream_t stream);

// out[b] = in[b] + 1. Writes to a separate buffer rather than in place so
// callers can keep the pre-append seq_lens around (needed by kv_append)
// alongside the post-append count (needed by attention) within the same
// step.
void launch_add_one(int* out, const int* in, int batch, cudaStream_t stream);

struct PagedAttentionArgs {
    const half* q;            // [batch, n_head, head_dim], row stride q_row_stride
    int q_row_stride;         // element distance between consecutive batch rows in q
    const half* kv_pool;      // device arena, block-indexed (see kv_cache.h layout)
    const int* block_tables;  // [batch, max_blocks_per_seq], row-padded
    const int* seq_lens;      // [batch], INCLUDES this step's token (post kv_append + increment)
    half* out;                 // [batch, n_head, head_dim], tightly packed
    int batch;
    int n_head;
    int head_dim;
    int max_blocks_per_seq;
    int layer_idx;
    int n_layer;
};

// Decode-only paged attention. One thread block per (batch, head); one
// thread per head_dim lane; single streaming pass over cached keys with
// online-softmax running state (m, l, acc), FP32 accumulators. Causality is
// implicit: a decode query attends to every position already in the cache.
void launch_paged_attention(const PagedAttentionArgs& args, cudaStream_t stream);

// Greedy argmax over the vocab dimension, one block per batch row.
void launch_argmax(int* out_tokens, const half* logits, int batch, int vocab_size,
                    cudaStream_t stream);

}  // namespace kernels
