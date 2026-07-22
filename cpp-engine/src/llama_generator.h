#pragma once
#include "token_generator.h"
#include "llama.h"  // llama.cpp header (fetched via CMake FetchContent, see CMakeLists.txt)
#include <string>

// Wraps llama.cpp's C API for real inference with TinyLlama.
//
// Model: TinyLlama-1.1B-Chat-v1.0 in GGUF format (Q4_K_M quantization).
// Download: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
// File: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (~669 MB)
//
// Only compiled when the engine is built with -DWITH_LLAMA=ON. The
// scheduler, KV cache bookkeeping, and gRPC service are fully exercised by
// the mock generator; this class swaps in a real model behind the same
// TokenGenerator interface without touching scheduling logic.
//
// IMPORTANT: llama.cpp manages its own KV cache internally (llama_memory_t,
// obtained via llama_get_memory()). Our KVCacheAllocator tracks slot
// *assignments* for the scheduler's logic — which sequence owns which
// integer slot — but the actual cache memory and its per-sequence state are
// owned by llama.cpp. Each sequence's kv_slot doubles as its llama_seq_id;
// max_batch_size must not exceed the context's n_seq_max (see Config).

class LlamaCppGenerator : public TokenGenerator {
public:
    struct Config {
        std::string model_path;
        int n_ctx = 2048;           // context length
        int n_batch = 512;          // max tokens per llama_decode call
        int n_threads = 4;          // CPU threads for inference
        int n_gpu_layers = 0;       // layers to offload to GPU (0 = CPU only)
        int max_batch_size = 8;     // must match ContinuousBatchScheduler::Config::max_batch_size —
                                    // sets llama's n_seq_max so every kv_slot is a valid seq_id
    };

    explicit LlamaCppGenerator(Config config);
    ~LlamaCppGenerator() override;

    LlamaCppGenerator(const LlamaCppGenerator&) = delete;
    LlamaCppGenerator& operator=(const LlamaCppGenerator&) = delete;

    BatchOutput forward(const BatchInput& input) override;
    std::vector<int32_t> tokenize(const std::string& text) override;
    std::string detokenize(int32_t token_id) override;
    int32_t eos_token_id() const override;
    void reset_sequence(int kv_slot) override;

private:
    int32_t sample_token(int32_t logits_index, float temperature);

    Config config_;
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
    int32_t n_vocab_ = 0;
    llama_batch batch_{};
};
