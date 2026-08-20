// EngineAPI implementation: decode-step orchestration and CUDA Graphs.
//
// UNVERIFIED: this file has never been compiled or run. Development
// happened on a machine with no NVIDIA GPU (see gpu-engine/VERIFICATION.md
// for the full status ledger); it needs a real build + the M2 parity tests
// in tests/test_kernels.cpp before any claim here can be trusted.
//
// Design note on prompt handling: per gpu-engine/CLAUDE.md's prefill
// strategy, prompts are consumed token-by-token through this same
// decode_step path (T=1 always, batched across sequences). For a sequence
// still inside its prompt, decode_step still computes and returns a
// prediction every step (the same forward pass runs regardless -- "zero
// extra kernels"), but that prediction is meaningless until the sequence's
// LAST prompt token has been fed as input; only from that point on does
// the returned token reflect real generation and get fed back as the next
// input. Whoever wires this into the scheduler (M6) must track prompt_len
// per sequence and discard outputs before that point -- Engine itself does
// not special-case it.
#include "engine/engine_api.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/gpu_weights.h"
#include "engine/kernels.h"
#include "engine/kv_cache.h"
#include "engine/kv_cache_gpu.h"
#include "engine/model.h"

namespace {

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

void cublas_check(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": cuBLAS error " + std::to_string(st));
    }
}

constexpr int kBuckets[] = {1, 2, 4, 8};

int pick_bucket(int batch, int max_batch) {
    for (int b : kBuckets) {
        if (b >= batch && b <= max_batch) return b;
    }
    return max_batch;
}

struct SequenceState {
    BlockTable block_table;
    int seq_len = 0;
    std::vector<int32_t> prompt_tokens;
    int prompt_pos = 0;
    int32_t last_token = 0;
};

}  // namespace

struct Engine::Impl {
    GPT2Config config;
    EngineConfig engine_config;
    GPT2WeightsGPU weights;
    std::unique_ptr<KVCacheArenaGPU> kv_arena;
    std::unique_ptr<BlockPool> block_pool;
    int max_blocks_per_seq = 0;

    cublasHandle_t cublas = nullptr;
    cudaStream_t stream = nullptr;

    std::unordered_map<int, SequenceState> sequences;
    int next_seq_id = 0;
    bool use_graphs = true;

    // Persistent per-step device buffers (fixed addresses/capacity == max_batch,
    // REQUIRED for CUDA Graph stability -- see gpu-engine/CLAUDE.md).
    int* d_token_ids = nullptr;
    int* d_positions = nullptr;
    int* d_seq_lens = nullptr;       // pre-append cached length, used by kv_append
    int* d_seq_lens_post = nullptr;  // +1, used by paged_attention
    int* d_block_tables = nullptr;   // [max_batch, max_blocks_per_seq]
    int* d_next_tokens = nullptr;

    half* d_residual = nullptr;      // running residual stream [max_batch, D]
    half* d_x = nullptr;             // LN output [max_batch, D]
    half* d_zero_delta = nullptr;    // persistent zero buffer [max_batch, D]
    half* d_qkv = nullptr;           // [max_batch, 3D] interleaved Q|K|V per row
    half* d_attn_concat = nullptr;   // [max_batch, D]
    half* d_attn_out = nullptr;      // [max_batch, D]
    half* d_ff = nullptr;            // [max_batch, 4D]
    half* d_ff_out = nullptr;        // [max_batch, D]
    half* d_logits = nullptr;        // [max_batch, vocab]

    std::unordered_map<int, cudaGraphExec_t> graph_execs;

    ~Impl() {
        for (auto& kv : graph_execs) cudaGraphExecDestroy(kv.second);
        if (cublas) cublasDestroy(cublas);
        if (stream) cudaStreamDestroy(stream);
        free_gpu_buffers();
        free_gpt2_weights_gpu(weights);
    }

    void free_gpu_buffers() {
        void* ptrs[] = {d_token_ids, d_positions,   d_seq_lens,     d_seq_lens_post,
                         d_block_tables, d_next_tokens, d_residual,     d_x,
                         d_zero_delta,   d_qkv,          d_attn_concat, d_attn_out,
                         d_ff,           d_ff_out,       d_logits};
        for (void* p : ptrs)
            if (p) cudaFree(p);
    }

    void gemm_rowmajor(int batch, int I, int O, const half* X, const half* W, half* Y) {
        // Row-major Y[batch,O] = X[batch,I] @ W^T, W stored row-major [O,I].
        // See the convention block-comment in include/engine/kernels.h.
        const float alpha = 1.0f, beta = 0.0f;
        cublas_check(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, O, batch, I, &alpha, W,
                                   CUDA_R_16F, I, X, CUDA_R_16F, I, &beta, Y, CUDA_R_16F, O,
                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                     "gemm_rowmajor");
    }

    // Issues the full decode-step kernel/GEMM sequence for `bucket` lanes of
    // the persistent buffers. Captured verbatim into a CUDA graph per
    // bucket, and also callable directly when use_graphs is false.
    void run_decode_kernels(int bucket) {
        const int D = config.d_model;
        const int D3 = 3 * D;
        const int D4 = 4 * D;
        const int H = config.n_head;
        const int Dh = config.head_dim;

        kernels::launch_embedding(d_residual, weights.wte, weights.wpe, d_token_ids, d_positions,
                                   bucket, D, stream);
        kernels::launch_add_one(d_seq_lens_post, d_seq_lens, bucket, stream);

        const half* prev_delta = d_zero_delta;

        for (int layer = 0; layer < config.n_layer; ++layer) {
            const GPT2LayerWeightsGPU& lw = weights.layers[layer];

            // ln1, fused with adding the previous sub-layer's output (zero on layer 0).
            kernels::launch_add_layernorm(d_x, d_residual, prev_delta, d_residual, lw.ln1_g,
                                           lw.ln1_b, bucket, D, config.layer_norm_epsilon, stream);

            gemm_rowmajor(bucket, D, D3, d_x, lw.qkv_w, d_qkv);
            kernels::launch_add_bias(d_qkv, d_qkv, lw.qkv_b, bucket, D3, stream);

            half* q = d_qkv;          // row stride D3
            half* k = d_qkv + D;      // row stride D3
            half* v = d_qkv + 2 * D;  // row stride D3

            kernels::launch_kv_append(weights_kv_pool(), k, v, D3, d_block_tables, d_seq_lens,
                                       bucket, H, Dh, max_blocks_per_seq, layer, config.n_layer,
                                       stream);

            kernels::PagedAttentionArgs attn_args{};
            attn_args.q = q;
            attn_args.q_row_stride = D3;
            attn_args.kv_pool = weights_kv_pool();
            attn_args.block_tables = d_block_tables;
            attn_args.seq_lens = d_seq_lens_post;
            attn_args.out = d_attn_concat;
            attn_args.batch = bucket;
            attn_args.n_head = H;
            attn_args.head_dim = Dh;
            attn_args.max_blocks_per_seq = max_blocks_per_seq;
            attn_args.layer_idx = layer;
            attn_args.n_layer = config.n_layer;
            kernels::launch_paged_attention(attn_args, stream);

            gemm_rowmajor(bucket, D, D, d_attn_concat, lw.proj_w, d_attn_out);
            kernels::launch_add_bias(d_attn_out, d_attn_out, lw.proj_b, bucket, D, stream);

            // ln2, fused with adding attn_out.
            kernels::launch_add_layernorm(d_x, d_residual, d_attn_out, d_residual, lw.ln2_g,
                                           lw.ln2_b, bucket, D, config.layer_norm_epsilon, stream);

            gemm_rowmajor(bucket, D, D4, d_x, lw.fc_w, d_ff);
            kernels::launch_bias_gelu(d_ff, d_ff, lw.fc_b, bucket, D4, stream);

            gemm_rowmajor(bucket, D4, D, d_ff, lw.fc_proj_w, d_ff_out);
            kernels::launch_add_bias(d_ff_out, d_ff_out, lw.fc_proj_b, bucket, D, stream);

            prev_delta = d_ff_out;  // fed into next layer's ln1 (or final LN below)
        }

        // Final LayerNorm, fused with adding the last layer's ff_out.
        kernels::launch_add_layernorm(d_x, d_residual, prev_delta, d_residual, weights.lnf_g,
                                       weights.lnf_b, bucket, D, config.layer_norm_epsilon, stream);

        gemm_rowmajor(bucket, D, config.vocab_size, d_x, weights.wte, d_logits);
        kernels::launch_argmax(d_next_tokens, d_logits, bucket, config.vocab_size, stream);
    }

    half* weights_kv_pool() const { return kv_arena->pool(); }

    cudaGraphExec_t get_or_capture_graph(int bucket) {
        auto it = graph_execs.find(bucket);
        if (it != graph_execs.end()) return it->second;

        cudaGraph_t graph;
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                   "cudaStreamBeginCapture");
        run_decode_kernels(bucket);
        cuda_check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");

        cudaGraphExec_t exec;
        cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "cudaGraphInstantiate");
        cudaGraphDestroy(graph);

        graph_execs[bucket] = exec;
        return exec;
    }

    void ensure_capacity(SequenceState& seq) {
        while (static_cast<int>(seq.block_table.num_blocks()) * BLOCK_TOKENS <= seq.seq_len) {
            int block_id = block_pool->allocate();
            if (block_id < 0) throw std::runtime_error("KV cache block pool exhausted mid-decode");
            seq.block_table.append_block(block_id);
        }
    }
};

Engine::Engine() : impl_(new Impl()) {}

Engine::~Engine() { delete impl_; }

bool Engine::init(const std::string& weights_dir, const EngineConfig& cfg) {
    try {
        impl_->engine_config = cfg;
        impl_->config = GPT2Config::load_from_json(weights_dir + "/config.json");
        impl_->weights = load_gpt2_weights_gpu(weights_dir, impl_->config);

        impl_->max_blocks_per_seq =
            (cfg.max_seq_len + BLOCK_TOKENS - 1) / BLOCK_TOKENS;

        impl_->kv_arena = std::make_unique<KVCacheArenaGPU>(
            cfg.kv_block_budget, impl_->config.n_layer, impl_->config.n_head,
            impl_->config.head_dim);
        impl_->block_pool = std::make_unique<BlockPool>(cfg.kv_block_budget);

        cuda_check(cudaStreamCreate(&impl_->stream), "cudaStreamCreate");
        cublas_check(cublasCreate(&impl_->cublas), "cublasCreate");
        cublas_check(cublasSetStream(impl_->cublas, impl_->stream), "cublasSetStream");

        const int B = cfg.max_batch;
        const int D = impl_->config.d_model;
        const int V = impl_->config.vocab_size;

        cuda_check(cudaMalloc(&impl_->d_token_ids, B * sizeof(int)), "malloc token_ids");
        cuda_check(cudaMalloc(&impl_->d_positions, B * sizeof(int)), "malloc positions");
        cuda_check(cudaMalloc(&impl_->d_seq_lens, B * sizeof(int)), "malloc seq_lens");
        cuda_check(cudaMalloc(&impl_->d_seq_lens_post, B * sizeof(int)), "malloc seq_lens_post");
        cuda_check(cudaMalloc(&impl_->d_block_tables, B * impl_->max_blocks_per_seq * sizeof(int)),
                   "malloc block_tables");
        cuda_check(cudaMalloc(&impl_->d_next_tokens, B * sizeof(int)), "malloc next_tokens");

        cuda_check(cudaMalloc(&impl_->d_residual, static_cast<size_t>(B) * D * sizeof(half)),
                   "malloc residual");
        cuda_check(cudaMalloc(&impl_->d_x, static_cast<size_t>(B) * D * sizeof(half)), "malloc x");
        cuda_check(cudaMalloc(&impl_->d_zero_delta, static_cast<size_t>(B) * D * sizeof(half)),
                   "malloc zero_delta");
        cuda_check(cudaMemset(impl_->d_zero_delta, 0, static_cast<size_t>(B) * D * sizeof(half)),
                   "memset zero_delta");
        cuda_check(cudaMalloc(&impl_->d_qkv, static_cast<size_t>(B) * 3 * D * sizeof(half)),
                   "malloc qkv");
        cuda_check(cudaMalloc(&impl_->d_attn_concat, static_cast<size_t>(B) * D * sizeof(half)),
                   "malloc attn_concat");
        cuda_check(cudaMalloc(&impl_->d_attn_out, static_cast<size_t>(B) * D * sizeof(half)),
                   "malloc attn_out");
        cuda_check(cudaMalloc(&impl_->d_ff, static_cast<size_t>(B) * 4 * D * sizeof(half)),
                   "malloc ff");
        cuda_check(cudaMalloc(&impl_->d_ff_out, static_cast<size_t>(B) * D * sizeof(half)),
                   "malloc ff_out");
        cuda_check(cudaMalloc(&impl_->d_logits, static_cast<size_t>(B) * V * sizeof(half)),
                   "malloc logits");

        return true;
    } catch (const std::exception&) {
        return false;
    }
}

int Engine::add_sequence(const int* prompt_tokens, int prompt_len) {
    int block_id = impl_->block_pool->allocate();
    if (block_id < 0) return -1;

    SequenceState seq;
    seq.block_table.append_block(block_id);
    seq.prompt_tokens.assign(prompt_tokens, prompt_tokens + prompt_len);

    const int seq_id = impl_->next_seq_id++;
    impl_->sequences.emplace(seq_id, std::move(seq));
    return seq_id;
}

void Engine::decode_step(const int* seq_ids, int batch, int* out_next_tokens) {
    Impl& impl = *impl_;
    const int bucket = pick_bucket(batch, impl.engine_config.max_batch);

    std::vector<int> h_token_ids(bucket), h_positions(bucket), h_seq_lens(bucket);
    std::vector<int> h_block_tables(static_cast<size_t>(bucket) * impl.max_blocks_per_seq, 0);

    for (int i = 0; i < batch; ++i) {
        SequenceState& seq = impl.sequences.at(seq_ids[i]);
        impl.ensure_capacity(seq);

        const int32_t token = (seq.prompt_pos < static_cast<int>(seq.prompt_tokens.size()))
                                   ? seq.prompt_tokens[seq.prompt_pos]
                                   : seq.last_token;
        h_token_ids[i] = token;
        h_positions[i] = seq.seq_len;
        h_seq_lens[i] = seq.seq_len;
        const auto& blocks = seq.block_table.block_ids();
        for (size_t j = 0; j < blocks.size(); ++j) {
            h_block_tables[static_cast<size_t>(i) * impl.max_blocks_per_seq + j] = blocks[j];
        }
    }
    // Pad lanes replay the last real lane so padded block tables stay
    // valid; their outputs are simply never read back.
    for (int i = batch; i < bucket; ++i) {
        h_token_ids[i] = h_token_ids[batch - 1];
        h_positions[i] = h_positions[batch - 1];
        h_seq_lens[i] = h_seq_lens[batch - 1];
        for (int j = 0; j < impl.max_blocks_per_seq; ++j) {
            h_block_tables[static_cast<size_t>(i) * impl.max_blocks_per_seq + j] =
                h_block_tables[static_cast<size_t>(batch - 1) * impl.max_blocks_per_seq + j];
        }
    }

    cuda_check(cudaMemcpyAsync(impl.d_token_ids, h_token_ids.data(), bucket * sizeof(int),
                                cudaMemcpyHostToDevice, impl.stream),
               "upload token_ids");
    cuda_check(cudaMemcpyAsync(impl.d_positions, h_positions.data(), bucket * sizeof(int),
                                cudaMemcpyHostToDevice, impl.stream),
               "upload positions");
    cuda_check(cudaMemcpyAsync(impl.d_seq_lens, h_seq_lens.data(), bucket * sizeof(int),
                                cudaMemcpyHostToDevice, impl.stream),
               "upload seq_lens");
    cuda_check(cudaMemcpyAsync(impl.d_block_tables, h_block_tables.data(),
                                h_block_tables.size() * sizeof(int), cudaMemcpyHostToDevice,
                                impl.stream),
               "upload block_tables");

    if (impl.use_graphs) {
        cudaGraphExec_t exec = impl.get_or_capture_graph(bucket);
        cuda_check(cudaGraphLaunch(exec, impl.stream), "cudaGraphLaunch");
    } else {
        impl.run_decode_kernels(bucket);
    }

    std::vector<int> h_next_tokens(bucket);
    cuda_check(cudaMemcpyAsync(h_next_tokens.data(), impl.d_next_tokens, bucket * sizeof(int),
                                cudaMemcpyDeviceToHost, impl.stream),
               "download next_tokens");
    cuda_check(cudaStreamSynchronize(impl.stream), "sync decode_step");

    for (int i = 0; i < batch; ++i) {
        SequenceState& seq = impl.sequences.at(seq_ids[i]);
        if (seq.prompt_pos < static_cast<int>(seq.prompt_tokens.size())) seq.prompt_pos++;
        seq.seq_len++;
        seq.last_token = h_next_tokens[i];
        out_next_tokens[i] = h_next_tokens[i];
    }
}

void Engine::free_sequence(int seq_id) {
    auto it = impl_->sequences.find(seq_id);
    if (it == impl_->sequences.end()) return;
    for (int block_id : it->second.block_table.block_ids()) impl_->block_pool->free(block_id);
    impl_->sequences.erase(it);
}

size_t Engine::free_blocks() const { return static_cast<size_t>(impl_->block_pool->num_free()); }

void Engine::set_use_cuda_graphs(bool enabled) { impl_->use_graphs = enabled; }
