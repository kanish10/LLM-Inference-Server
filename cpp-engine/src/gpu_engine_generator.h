#pragma once
#include "token_generator.h"
#include "engine/engine_api.h"

#include <string>
#include <unordered_map>

// Wraps ../gpu-engine's real CUDA GPT-2 124M decode path behind the same
// TokenGenerator interface LlamaCppGenerator uses -- see gpu-engine/CLAUDE.md
// M6 and gpu-engine/VERIFICATION.md for why this exists and what it measures.
//
// IMPEDANCE MISMATCH vs. LlamaCppGenerator: llama_decode() natively accepts
// one call with a heterogeneous mix of multi-token (prefill) and
// single-token (decode) entries. gpu-engine's Engine::decode_step() only
// ever advances every sequence in its batch by exactly one token per call
// (a deliberate M2 scoping choice -- see gpu-engine/CLAUDE.md's "Prefill
// strategy" note: prompts are consumed token-by-token, batched prefill is a
// stretch goal). So forward() here does real batched GPU decode for the
// steady-state case (all entries already past their prompt), but a brand
// new sequence's prompt is walked token-by-token, batch=1, inside this same
// forward() call -- correct, real GPU compute for every token, just not
// batched with concurrent sequences during its own prefill.
//
// kv_slot doubles as the map key into gpu-engine's own seq_id space, same
// pattern as LlamaCppGenerator using kv_slot as llama_seq_id.
class GpuEngineGenerator : public TokenGenerator {
public:
    struct Config {
        std::string weights_dir;
        int max_batch_size = 8;
        int max_seq_len = 1024;
        int kv_block_budget = 512;
    };

    explicit GpuEngineGenerator(Config config);
    ~GpuEngineGenerator() override = default;

    BatchOutput forward(const BatchInput& input) override;
    std::vector<int32_t> tokenize(const std::string& text) override;
    std::string detokenize(int32_t token_id) override;
    int32_t eos_token_id() const override;
    void reset_sequence(int kv_slot) override;

private:
    Config config_;
    Engine engine_;
    std::unordered_map<int, int> kv_slot_to_gpu_seq_;
    uint32_t tokenize_counter_ = 0;  // deterministic pseudo-tokenizer seed, see .cpp
};
