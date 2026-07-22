#pragma once
#include "token_generator.h"
#include "llama.h"  // llama.cpp header
#include <string>

// Wraps llama.cpp's C API for real inference with TinyLlama.
//
// Model: TinyLlama-1.1B-Chat-v1.0 in GGUF format (Q4_K_M quantization).
// Download: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
// File: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (~669 MB)
//
// Only compiled when the engine is built with -DWITH_LLAMA=ON and
// third_party/llama.cpp is present (see cpp-engine/CMakeLists.txt). The
// scheduler, KV cache, and gRPC service are fully exercised by the mock
// generator; this class swaps in a real model behind the same
// TokenGenerator interface without touching scheduling logic.
//
// IMPORTANT: llama.cpp manages its own KV cache internally.
// Our KVCacheAllocator tracks slot *assignments* for the scheduler's
// logic, but llama.cpp's llama_kv_cache_seq_* functions manage the
// actual cache. We must keep them in sync:
//   - On admit: call llama_kv_cache_seq_rm() to clear the slot
//   - On evict: call llama_kv_cache_seq_rm() to free the slot
//   - Each sequence is assigned a unique seq_id = kv_slot index

class LlamaCppGenerator : public TokenGenerator {
public:
    struct Config {
        std::string model_path;
        int n_ctx = 2048;           // context length
        int n_batch = 512;          // max tokens per llama_decode call
        int n_threads = 4;          // CPU threads for inference
        int n_gpu_layers = 0;       // layers to offload to GPU (0 = CPU only)
    };

    explicit LlamaCppGenerator(Config config);
    ~LlamaCppGenerator();

    BatchOutput forward(const BatchInput& input) override;
    std::vector<int32_t> tokenize(const std::string& text) override;
    std::string detokenize(int32_t token_id) override;
    int32_t eos_token_id() const override;

private:
    int32_t sample_token(float* logits, int n_vocab, float temperature);
    void clear_kv_slot(int seq_id);

    Config config_;
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    int n_vocab_ = 0;
};
