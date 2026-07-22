#include "mock_generator.h"
#include <thread>

// Delegates instead of giving Config a default argument in the header:
// a nested class's default member initializers can't be used in a default
// argument of the enclosing class's constructor ([class.mem]) — both are
// deferred to "end of enclosing class" and the deferral is circular.
MockTokenGenerator::MockTokenGenerator() : MockTokenGenerator(Config{}) {}

MockTokenGenerator::MockTokenGenerator(Config config)
    : config_(config), rng_(config.seed) {}

TokenGenerator::BatchOutput
MockTokenGenerator::forward(const BatchInput& input) {
    // Simulate compute: latency scales with batch size
    auto total_latency = config_.latency_per_token * input.entries.size();
    std::this_thread::sleep_for(total_latency);

    BatchOutput output;
    output.next_tokens.reserve(input.entries.size());
    for (size_t i = 0; i < input.entries.size(); ++i) {
        if (config_.eos_prob > 0 && dist_(rng_) < config_.eos_prob) {
            output.next_tokens.push_back(config_.eos_id);
        } else {
            // Random non-EOS token
            int32_t tok = static_cast<int32_t>(
                dist_(rng_) * (config_.vocab_size - 1)) + 1;
            if (tok == config_.eos_id) tok++;  // avoid accidental EOS
            output.next_tokens.push_back(tok);
        }
    }
    return output;
}

std::vector<int32_t> MockTokenGenerator::tokenize(const std::string& text) {
    // ASCII encoding: one token per character
    std::vector<int32_t> tokens;
    tokens.reserve(text.size());
    for (char c : text) {
        tokens.push_back(static_cast<int32_t>(static_cast<unsigned char>(c)));
    }
    return tokens;
}

std::string MockTokenGenerator::detokenize(int32_t token_id) {
    if (token_id < 0 || token_id > 127) return "?";
    return std::string(1, static_cast<char>(token_id));
}
