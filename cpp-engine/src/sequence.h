#pragma once
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

enum class SequenceState {
    WAITING,    // in queue, not yet admitted to batch
    PREFILL,    // admitted, full prompt being processed this step
    DECODE,     // generating tokens one-by-one
    FINISHED    // hit EOS or max_new_tokens — pending eviction
};

struct Sequence {
    std::string request_id;

    // Input
    std::vector<int32_t> prompt_tokens;
    int32_t max_new_tokens = 0;
    float temperature = 0.0f;

    // Output (accumulated during decode)
    std::vector<int32_t> generated_tokens;

    // Scheduler state
    SequenceState state = SequenceState::WAITING;
    int kv_slot = -1;  // index into KVCacheAllocator's pool; -1 = unassigned

    // Timing (all set by scheduler)
    std::chrono::steady_clock::time_point arrival_time;
    std::chrono::steady_clock::time_point first_token_time;
    std::chrono::steady_clock::time_point completion_time;

    // Derived
    int total_length() const {
        return static_cast<int>(prompt_tokens.size() + generated_tokens.size());
    }
    int prompt_length() const {
        return static_cast<int>(prompt_tokens.size());
    }
    int generated_length() const {
        return static_cast<int>(generated_tokens.size());
    }
    bool hit_max() const {
        return generated_length() >= max_new_tokens;
    }
};
