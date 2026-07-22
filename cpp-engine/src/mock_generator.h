#pragma once
#include "token_generator.h"
#include <chrono>
#include <random>

// Deterministic mock for testing the scheduler in isolation.
//
// Behavior:
// - tokenize(): maps each character to its ASCII value (simple, reversible)
// - forward(): sleeps latency_per_token * batch_size to simulate compute,
//   then returns random tokens. EOS emitted with probability eos_prob
//   per sequence per step (default 0 — sequences run to max_new_tokens).
// - Configurable parameters let the benchmark control output length
//   distribution by adjusting max_new_tokens per request rather than
//   relying on random EOS.
//
// WHY THIS EXISTS: the scheduler's throughput advantage over static batching
// comes from its *scheduling logic*, not from the model. The mock isolates
// the scheduler so the benchmark measures what matters. If mock latency
// per token = T, then:
//   Static throughput = batch_size / (T * max_steps_in_batch)
//   Continuous throughput ~= batch_size / (T * mean_steps) under full load
// The ratio is driven by output length variance, exactly as it should be.

class MockTokenGenerator : public TokenGenerator {
public:
    struct Config {
        int vocab_size = 32000;
        int32_t eos_id = 2;
        double eos_prob = 0.0;       // probability of emitting EOS per step
        std::chrono::microseconds latency_per_token{500};  // per sequence
        uint64_t seed = 42;
    };

    MockTokenGenerator();
    explicit MockTokenGenerator(Config config);

    BatchOutput forward(const BatchInput& input) override;
    std::vector<int32_t> tokenize(const std::string& text) override;
    std::string detokenize(int32_t token_id) override;
    int32_t eos_token_id() const override { return config_.eos_id; }

private:
    Config config_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> dist_{0.0, 1.0};
};
