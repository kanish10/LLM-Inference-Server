#pragma once
#include <string>
#include <vector>

#include "engine/model.h"
#include "weights_loader.h"

// Straightforward FP32 CPU forward pass for GPT-2 124M. This is the
// numerics oracle (M1): it must match HF's fp32 top-1 greedy continuation
// exactly. No batching, no paging, no fusion -- one sequence, one head at a
// time, KV cache as a plain contiguous per-layer buffer. Correctness over
// speed; the GPU engine is where speed lives.
class CPUModel {
public:
    CPUModel(const GPT2Config& config, const WeightsLoader& weights);

    // Greedy-decodes num_new_tokens tokens starting from prompt_tokens.
    // Returns just the newly generated token IDs (length num_new_tokens),
    // stopping early only if eos_token_id is produced.
    std::vector<int32_t> greedy_generate(const std::vector<int32_t>& prompt_tokens,
                                          int num_new_tokens);

private:
    struct LayerWeights {
        const float* ln1_g;
        const float* ln1_b;
        const float* qkv_w;  // [3*d_model, d_model]
        const float* qkv_b;  // [3*d_model]
        const float* proj_w; // [d_model, d_model]
        const float* proj_b; // [d_model]
        const float* ln2_g;
        const float* ln2_b;
        const float* fc_w;      // [4*d_model, d_model]
        const float* fc_b;      // [4*d_model]
        const float* fc_proj_w; // [d_model, 4*d_model]
        const float* fc_proj_b; // [d_model]
    };

    // Runs tokens (already embedded at absolute positions [pos_start,
    // pos_start+tokens.size())) through all layers, appending to the KV
    // cache. Returns hidden state for the LAST token only (all that's
    // needed for greedy next-token prediction).
    std::vector<float> forward_step(const std::vector<int32_t>& tokens, int pos_start);

    int32_t argmax_logits(const std::vector<float>& last_hidden) const;

    GPT2Config config_;
    const WeightsLoader& weights_;
    const float* wte_;  // [vocab, d_model]
    const float* wpe_;  // [n_ctx, d_model]
    const float* lnf_g_;
    const float* lnf_b_;
    std::vector<LayerWeights> layers_;

    // KV cache: per layer, [n_ctx, d_model] for K and for V (d_model =
    // n_head * head_dim, heads are contiguous slices).
    std::vector<std::vector<float>> k_cache_;
    std::vector<std::vector<float>> v_cache_;
    int cache_len_ = 0;  // number of positions currently filled
};
