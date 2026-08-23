#include "gpu_engine_generator.h"

#include <random>
#include <stdexcept>

GpuEngineGenerator::GpuEngineGenerator(Config config) : config_(config) {
    EngineConfig cfg;
    cfg.max_batch = config_.max_batch_size;
    cfg.max_seq_len = config_.max_seq_len;
    cfg.kv_block_budget = config_.kv_block_budget;
    if (!engine_.init(config_.weights_dir, cfg)) {
        throw std::runtime_error("GpuEngineGenerator: engine.init failed for " + config_.weights_dir);
    }
}

TokenGenerator::BatchOutput GpuEngineGenerator::forward(const BatchInput& input) {
    BatchOutput output;
    output.next_tokens.assign(input.entries.size(), 0);

    std::vector<int> decode_gpu_ids;
    std::vector<int> decode_out_idx;
    decode_gpu_ids.reserve(input.entries.size());
    decode_out_idx.reserve(input.entries.size());

    for (size_t i = 0; i < input.entries.size(); ++i) {
        const Entry& entry = input.entries[i];
        auto it = kv_slot_to_gpu_seq_.find(entry.kv_slot);
        if (it == kv_slot_to_gpu_seq_.end()) {
            // First time we've seen this kv_slot: register with gpu-engine
            // and walk its full prompt token-by-token (batch=1) right here,
            // real GPU compute per token, matching gpu-engine's own
            // documented prefill scoping.
            int gpu_seq = engine_.add_sequence(entry.tokens.data(), static_cast<int>(entry.tokens.size()));
            if (gpu_seq < 0) {
                throw std::runtime_error("GpuEngineGenerator: gpu-engine KV block pool exhausted");
            }
            kv_slot_to_gpu_seq_[entry.kv_slot] = gpu_seq;
            int next_tok = 0;
            for (size_t t = 0; t < entry.tokens.size(); ++t) {
                engine_.decode_step(&gpu_seq, 1, &next_tok);
            }
            output.next_tokens[i] = next_tok;
        } else {
            decode_gpu_ids.push_back(it->second);
            decode_out_idx.push_back(static_cast<int>(i));
        }
    }

    if (!decode_gpu_ids.empty()) {
        std::vector<int> next_toks(decode_gpu_ids.size());
        engine_.decode_step(decode_gpu_ids.data(), static_cast<int>(decode_gpu_ids.size()), next_toks.data());
        for (size_t j = 0; j < decode_gpu_ids.size(); ++j) {
            output.next_tokens[decode_out_idx[j]] = next_toks[j];
        }
    }

    return output;
}

void GpuEngineGenerator::reset_sequence(int kv_slot) {
    auto it = kv_slot_to_gpu_seq_.find(kv_slot);
    if (it == kv_slot_to_gpu_seq_.end()) return;
    engine_.free_sequence(it->second);
    kv_slot_to_gpu_seq_.erase(it);
}

// No C++ GPT-2 BPE tokenizer exists in gpu-engine (export/tokenize_client.py
// is Python-only, used for demos). For the M6 scheduling/throughput
// benchmark, prompt TEXT content is irrelevant -- benchmark.py's own
// prompts ("Write a detailed essay about topic number N") were never
// meaningful either, only prompt/output token COUNTS drive realistic decode
// cost. So tokenize() deterministically produces a small sequence of valid
// GPT-2 vocab ids (any of gpt2's 50257 tokens is a valid embedding lookup)
// instead of parsing text. detokenize() is a placeholder for the same
// reason -- this generator is for throughput measurement, not readable
// output. eos_token_id() returns an id outside gpt2's vocab so it can never
// spontaneously match a real generated token, matching MockTokenGenerator's
// eos_prob=0 default: output length is controlled entirely by the
// benchmark driver's max_new_tokens, not by chance EOS, so the 16-512
// uniform-length methodology is reproducible.
std::vector<int32_t> GpuEngineGenerator::tokenize(const std::string& text) {
    std::mt19937 rng(static_cast<uint32_t>(text.size()) * 2654435761u + tokenize_counter_++);
    std::uniform_int_distribution<int32_t> dist(0, 50256);
    int n = 4 + static_cast<int>(text.size() % 8);  // 4-11 tokens, deterministic per call
    std::vector<int32_t> tokens(n);
    for (auto& t : tokens) t = dist(rng);
    return tokens;
}

std::string GpuEngineGenerator::detokenize(int32_t token_id) {
    return "<" + std::to_string(token_id) + ">";
}

int32_t GpuEngineGenerator::eos_token_id() const { return -1; }
