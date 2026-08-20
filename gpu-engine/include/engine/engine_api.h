#pragma once
#include <cstddef>
#include <string>

// Narrow boundary the existing C++ scheduler (../cpp-engine/src/scheduler.*)
// calls into. Deliberately CUDA-free so scheduler.h/scheduler.cpp never
// need to see a CUDA header -- Impl (defined in src/engine.cu) hides
// everything behind a pointer.
struct EngineConfig {
    int max_batch = 8;
    int max_seq_len = 1024;
    int kv_block_budget = 4096;
};

class Engine {
public:
    Engine();
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    bool init(const std::string& weights_dir, const EngineConfig& cfg);

    // Registers a new sequence and allocates its first KV block. Returns a
    // sequence handle, or -1 if the block pool is exhausted.
    int add_sequence(const int* prompt_tokens, int prompt_len);

    // One decode step for the given batch of sequence handles (prompts are
    // consumed token-by-token through this same path -- see
    // gpu-engine/CLAUDE.md's prefill-strategy note). Writes one next-token
    // id per entry into out_next_tokens, same order as seq_ids.
    void decode_step(const int* seq_ids, int batch, int* out_next_tokens);

    void free_sequence(int seq_id);

    // Free KV blocks remaining in the pool -- scheduler admission control input.
    size_t free_blocks() const;

    // Bench/test-only knob: capture+replay a CUDA graph per batch-size
    // bucket (default) vs. issuing the same kernel/GEMM sequence directly
    // every step. Not used by the production scheduler integration; exists
    // so bench_decode can measure the CUDA-Graphs launch-overhead claim.
    void set_use_cuda_graphs(bool enabled);

private:
    struct Impl;
    Impl* impl_;
};
