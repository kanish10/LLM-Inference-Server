#pragma once
#include "token_generator.h"
#include <chrono>
#include <random>

// Deterministic mock for testing the scheduler in isolation.
//
// Behavior:
// - tokenize(): maps each character to its ASCII value (simple, reversible)
// - forward(): sleeps fixed_overhead + latency_per_token * batch_size to
//   simulate compute, then returns random tokens. EOS emitted with
//   probability eos_prob per sequence per step (default 0 — sequences run
//   to max_new_tokens).
// - Configurable parameters let the benchmark control output length
//   distribution by adjusting max_new_tokens per request rather than
//   relying on random EOS.
//
// WHY fixed_overhead EXISTS (default 0, so existing scheduler tests are
// unaffected): a real batched forward pass has a large batch-size-independent
// cost — reading model weights from memory once per step dominates at small
// batch sizes — plus a small marginal cost per additional sequence, since
// the GPU parallelizes across the batch dimension. Without a fixed cost,
// this mock's per-step latency is exactly proportional to how many
// sequences are actively decoding, so a batch sitting at effective size 2
// (six slots occupied by finished-but-unevicted sequences, in static mode)
// costs exactly 2/8 of a full step — no different, in aggregate tokens/sec,
// than continuous batching keeping all 8 slots full. That hides the entire
// point of the project. With a fixed_overhead, an underfull batch pays the
// same fixed cost for fewer tokens, so static batching's idle slots show up
// as a real throughput loss, matching real inference servers.
//
// WHY THIS EXISTS: the scheduler's throughput advantage over static batching
// comes from its *scheduling logic*, not from the model. The mock isolates
// the scheduler so the benchmark measures what matters.

class MockTokenGenerator : public TokenGenerator {
public:
    struct Config {
        int vocab_size = 32000;
        int32_t eos_id = 2;
        double eos_prob = 0.0;       // probability of emitting EOS per step
        std::chrono::microseconds latency_per_token{500};  // marginal, per sequence
        std::chrono::microseconds fixed_overhead{0};       // per step, independent of batch size
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
