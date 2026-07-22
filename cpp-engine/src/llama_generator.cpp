// llama_generator.cpp — real-model backend, only built with -DWITH_LLAMA=ON.
#include "llama_generator.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <random>

LlamaCppGenerator::LlamaCppGenerator(Config config) : config_(config) {
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;
    model_ = llama_load_model_from_file(config.model_path.c_str(), model_params);
    assert(model_ && "Failed to load model");

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config.n_ctx;
    ctx_params.n_batch = config.n_batch;
    ctx_params.n_threads = config.n_threads;
    ctx_ = llama_new_context_with_model(model_, ctx_params);
    assert(ctx_ && "Failed to create context");

    n_vocab_ = llama_n_vocab(model_);
}

LlamaCppGenerator::~LlamaCppGenerator() {
    if (ctx_) llama_free(ctx_);
    if (model_) llama_free_model(model_);
    llama_backend_free();
}

TokenGenerator::BatchOutput
LlamaCppGenerator::forward(const BatchInput& input) {
    // Build llama_batch from our BatchInput
    llama_batch batch = llama_batch_init(
        config_.n_batch, 0, 1);

    // Track which batch positions need logits (last token of each entry)
    std::vector<int> logit_positions;
    int pos_in_batch = 0;

    for (const auto& entry : input.entries) {
        for (int i = 0; i < static_cast<int>(entry.tokens.size()); ++i) {
            bool need_logits = (i == static_cast<int>(entry.tokens.size()) - 1);
            llama_batch_add(
                batch,
                entry.tokens[i],
                entry.seq_position + i,   // position
                {entry.kv_slot},          // seq_id = kv_slot
                need_logits
            );
            if (need_logits) {
                logit_positions.push_back(pos_in_batch);
            }
            pos_in_batch++;
        }
    }

    // Run forward pass
    int rc = llama_decode(ctx_, batch);
    assert(rc == 0 && "llama_decode failed");

    // Sample one token per entry
    BatchOutput output;
    output.next_tokens.reserve(input.entries.size());
    for (int i = 0; i < static_cast<int>(input.entries.size()); ++i) {
        float* logits = llama_get_logits_ith(ctx_, logit_positions[i]);
        int32_t token = sample_token(logits, n_vocab_,
                                     input.entries[i].temperature);
        output.next_tokens.push_back(token);
    }

    llama_batch_free(batch);
    return output;
}

void LlamaCppGenerator::clear_kv_slot(int seq_id) {
    llama_kv_cache_seq_rm(ctx_, seq_id, -1, -1);
}

int32_t LlamaCppGenerator::sample_token(
    float* logits, int n_vocab, float temperature) {
    if (temperature <= 0.0f) {
        // Greedy
        return static_cast<int32_t>(
            std::max_element(logits, logits + n_vocab) - logits);
    }

    // Temperature scaling + softmax + multinomial sampling
    std::vector<float> probs(n_vocab);
    float max_logit = *std::max_element(logits, logits + n_vocab);
    float sum = 0;
    for (int i = 0; i < n_vocab; ++i) {
        probs[i] = std::exp((logits[i] - max_logit) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < n_vocab; ++i) probs[i] /= sum;

    // Multinomial sample
    static thread_local std::mt19937 rng(std::random_device{}());
    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
    return dist(rng);
}

std::vector<int32_t> LlamaCppGenerator::tokenize(const std::string& text) {
    std::vector<int32_t> tokens(text.size() + 16);
    int n = llama_tokenize(model_, text.c_str(), text.size(),
                           tokens.data(), tokens.size(), true, false);
    assert(n >= 0);
    tokens.resize(n);
    return tokens;
}

std::string LlamaCppGenerator::detokenize(int32_t token_id) {
    char buf[64];
    int n = llama_token_to_piece(model_, token_id, buf, sizeof(buf), 0, false);
    if (n < 0) return "";
    return std::string(buf, n);
}

int32_t LlamaCppGenerator::eos_token_id() const {
    return llama_token_eos(model_);
}
