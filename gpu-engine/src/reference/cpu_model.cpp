#include "cpu_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

// out[t*O + o] = sum_i in[t*I + i] * W[o*I + i] + b[o]     (W is [O, I] row-major)
void linear(std::vector<float>& out, const std::vector<float>& in, int T, int I, int O,
            const float* W, const float* b) {
    out.assign(static_cast<size_t>(T) * O, 0.0f);
    for (int t = 0; t < T; ++t) {
        const float* x = &in[static_cast<size_t>(t) * I];
        float* y = &out[static_cast<size_t>(t) * O];
        for (int o = 0; o < O; ++o) {
            const float* w = &W[static_cast<size_t>(o) * I];
            float acc = b ? b[o] : 0.0f;
            for (int i = 0; i < I; ++i) acc += x[i] * w[i];
            y[o] = acc;
        }
    }
}

void layernorm(std::vector<float>& out, const std::vector<float>& in, int T, int D,
               const float* gamma, const float* beta, float eps) {
    out.assign(static_cast<size_t>(T) * D, 0.0f);
    for (int t = 0; t < T; ++t) {
        const float* x = &in[static_cast<size_t>(t) * D];
        float* y = &out[static_cast<size_t>(t) * D];
        double mean = 0.0;
        for (int d = 0; d < D; ++d) mean += x[d];
        mean /= D;
        double var = 0.0;
        for (int d = 0; d < D; ++d) {
            double diff = x[d] - mean;
            var += diff * diff;
        }
        var /= D;
        const float inv_std = static_cast<float>(1.0 / std::sqrt(var + eps));
        for (int d = 0; d < D; ++d) {
            y[d] = static_cast<float>((x[d] - mean) * inv_std) * gamma[d] + beta[d];
        }
    }
}

// HF gpt2 "gelu_new": tanh approximation, NOT erf-based gelu.
float gelu_tanh(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    const float inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

}  // namespace

CPUModel::CPUModel(const GPT2Config& config, const WeightsLoader& weights)
    : config_(config), weights_(weights) {
    wte_ = weights_.get("wte").data;
    wpe_ = weights_.get("wpe").data;
    lnf_g_ = weights_.get("lnf_g").data;
    lnf_b_ = weights_.get("lnf_b").data;

    layers_.resize(config_.n_layer);
    for (int i = 0; i < config_.n_layer; ++i) {
        const std::string p = "layer" + std::to_string(i) + ".";
        LayerWeights& lw = layers_[i];
        lw.ln1_g = weights_.get(p + "ln1_g").data;
        lw.ln1_b = weights_.get(p + "ln1_b").data;
        lw.qkv_w = weights_.get(p + "qkv_w").data;
        lw.qkv_b = weights_.get(p + "qkv_b").data;
        lw.proj_w = weights_.get(p + "proj_w").data;
        lw.proj_b = weights_.get(p + "proj_b").data;
        lw.ln2_g = weights_.get(p + "ln2_g").data;
        lw.ln2_b = weights_.get(p + "ln2_b").data;
        lw.fc_w = weights_.get(p + "fc_w").data;
        lw.fc_b = weights_.get(p + "fc_b").data;
        lw.fc_proj_w = weights_.get(p + "fc_proj_w").data;
        lw.fc_proj_b = weights_.get(p + "fc_proj_b").data;
    }

    k_cache_.resize(config_.n_layer);
    v_cache_.resize(config_.n_layer);
    for (int i = 0; i < config_.n_layer; ++i) {
        k_cache_[i].assign(static_cast<size_t>(config_.n_ctx) * config_.d_model, 0.0f);
        v_cache_[i].assign(static_cast<size_t>(config_.n_ctx) * config_.d_model, 0.0f);
    }
    cache_len_ = 0;
}

std::vector<float> CPUModel::forward_step(const std::vector<int32_t>& tokens, int pos_start) {
    const int T = static_cast<int>(tokens.size());
    const int D = config_.d_model;
    const int H = config_.n_head;
    const int Dh = config_.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));

    if (pos_start + T > config_.n_ctx) {
        throw std::runtime_error("CPUModel: sequence exceeds n_ctx");
    }

    std::vector<float> hidden(static_cast<size_t>(T) * D);
    for (int t = 0; t < T; ++t) {
        const float* tok_emb = &wte_[static_cast<size_t>(tokens[t]) * D];
        const float* pos_emb = &wpe_[static_cast<size_t>(pos_start + t) * D];
        for (int d = 0; d < D; ++d) hidden[static_cast<size_t>(t) * D + d] = tok_emb[d] + pos_emb[d];
    }

    std::vector<float> x, qkv, attn_concat, attn_out, ff, ff_out;

    for (int layer = 0; layer < config_.n_layer; ++layer) {
        const LayerWeights& lw = layers_[layer];

        layernorm(x, hidden, T, D, lw.ln1_g, lw.ln1_b, config_.layer_norm_epsilon);
        linear(qkv, x, T, D, 3 * D, lw.qkv_w, lw.qkv_b);

        // Append this step's K, V into the cache before computing attention,
        // so causal attention (key positions 0..query position inclusive)
        // reads uniformly from the cache.
        std::vector<float>& kc = k_cache_[layer];
        std::vector<float>& vc = v_cache_[layer];
        for (int t = 0; t < T; ++t) {
            const int pos = pos_start + t;
            const float* k_src = &qkv[static_cast<size_t>(t) * 3 * D + D];
            const float* v_src = &qkv[static_cast<size_t>(t) * 3 * D + 2 * D];
            std::copy(k_src, k_src + D, &kc[static_cast<size_t>(pos) * D]);
            std::copy(v_src, v_src + D, &vc[static_cast<size_t>(pos) * D]);
        }

        attn_concat.assign(static_cast<size_t>(T) * D, 0.0f);
        std::vector<float> scores;
        for (int t = 0; t < T; ++t) {
            const int query_pos = pos_start + t;
            const int num_keys = query_pos + 1;  // causal: keys [0, query_pos]
            const float* q_row = &qkv[static_cast<size_t>(t) * 3 * D];

            for (int h = 0; h < H; ++h) {
                const float* q_head = &q_row[h * Dh];
                scores.assign(num_keys, 0.0f);
                float max_score = -std::numeric_limits<float>::infinity();
                for (int j = 0; j < num_keys; ++j) {
                    const float* k_head = &kc[static_cast<size_t>(j) * D + h * Dh];
                    float dot = 0.0f;
                    for (int d = 0; d < Dh; ++d) dot += q_head[d] * k_head[d];
                    dot *= scale;
                    scores[j] = dot;
                    max_score = std::max(max_score, dot);
                }
                double denom = 0.0;
                for (int j = 0; j < num_keys; ++j) {
                    scores[j] = std::exp(scores[j] - max_score);
                    denom += scores[j];
                }
                float* out_head = &attn_concat[static_cast<size_t>(t) * D + h * Dh];
                for (int j = 0; j < num_keys; ++j) {
                    const float w = static_cast<float>(scores[j] / denom);
                    const float* v_head = &vc[static_cast<size_t>(j) * D + h * Dh];
                    for (int d = 0; d < Dh; ++d) out_head[d] += w * v_head[d];
                }
            }
        }

        linear(attn_out, attn_concat, T, D, D, lw.proj_w, lw.proj_b);
        for (size_t i = 0; i < hidden.size(); ++i) hidden[i] += attn_out[i];

        layernorm(x, hidden, T, D, lw.ln2_g, lw.ln2_b, config_.layer_norm_epsilon);
        linear(ff, x, T, D, 4 * D, lw.fc_w, lw.fc_b);
        for (float& v : ff) v = gelu_tanh(v);
        linear(ff_out, ff, T, 4 * D, D, lw.fc_proj_w, lw.fc_proj_b);
        for (size_t i = 0; i < hidden.size(); ++i) hidden[i] += ff_out[i];
    }

    cache_len_ = pos_start + T;

    std::vector<float> hidden_last(hidden.end() - D, hidden.end());
    std::vector<float> normed(D);
    layernorm(normed, hidden_last, 1, D, lnf_g_, lnf_b_, config_.layer_norm_epsilon);
    return normed;
}

int32_t CPUModel::argmax_logits(const std::vector<float>& last_hidden) const {
    const int D = config_.d_model;
    int32_t best_tok = 0;
    float best_logit = -std::numeric_limits<float>::infinity();
    for (int v = 0; v < config_.vocab_size; ++v) {
        const float* row = &wte_[static_cast<size_t>(v) * D];
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) acc += row[d] * last_hidden[d];
        if (acc > best_logit) {
            best_logit = acc;
            best_tok = v;
        }
    }
    return best_tok;
}

std::vector<int32_t> CPUModel::greedy_generate(const std::vector<int32_t>& prompt_tokens,
                                                int num_new_tokens) {
    cache_len_ = 0;
    std::vector<int32_t> generated;
    generated.reserve(num_new_tokens);

    std::vector<float> last_hidden = forward_step(prompt_tokens, 0);
    int32_t next_tok = argmax_logits(last_hidden);
    generated.push_back(next_tok);
    if (next_tok == config_.eos_token_id) return generated;

    for (int step = 1; step < num_new_tokens; ++step) {
        last_hidden = forward_step({next_tok}, cache_len_);
        next_tok = argmax_logits(last_hidden);
        generated.push_back(next_tok);
        if (next_tok == config_.eos_token_id) break;
    }
    return generated;
}
