#include <gtest/gtest.h>
#include "mock_generator.h"
#include <chrono>

TEST(MockTokenGeneratorTest, ForwardLatencyScalesWithBatchSizeNoFixedOverhead) {
    MockTokenGenerator::Config config;
    config.latency_per_token = std::chrono::microseconds(2000);
    config.fixed_overhead = std::chrono::microseconds(0);
    MockTokenGenerator gen(config);

    TokenGenerator::BatchInput input;
    input.entries.resize(3, TokenGenerator::Entry{0, {1}, 0, 0.0f});

    auto start = std::chrono::steady_clock::now();
    gen.forward(input);
    auto elapsed = std::chrono::steady_clock::now() - start;

    // ~3 entries * 2000us = 6000us, no fixed cost.
    EXPECT_GE(elapsed, std::chrono::microseconds(5500));
    EXPECT_LT(elapsed, std::chrono::microseconds(20000));
}

TEST(MockTokenGeneratorTest, ForwardLatencyIncludesFixedOverheadRegardlessOfBatchSize) {
    // A batch of 1 must still pay the fixed cost — this is what makes an
    // underfull batch (static batching's finished-but-unevicted slots)
    // measurably less efficient than a full one.
    MockTokenGenerator::Config config;
    config.latency_per_token = std::chrono::microseconds(100);
    config.fixed_overhead = std::chrono::microseconds(5000);
    MockTokenGenerator gen(config);

    TokenGenerator::BatchInput input;
    input.entries.push_back(TokenGenerator::Entry{0, {1}, 0, 0.0f});

    auto start = std::chrono::steady_clock::now();
    gen.forward(input);
    auto elapsed = std::chrono::steady_clock::now() - start;

    // ~5000us fixed + 100us marginal = 5100us, dominated by the fixed cost.
    EXPECT_GE(elapsed, std::chrono::microseconds(4800));
    EXPECT_LT(elapsed, std::chrono::microseconds(20000));
}

TEST(MockTokenGeneratorTest, TokenizeIsReversibleAscii) {
    MockTokenGenerator gen;
    auto tokens = gen.tokenize("Hi!");
    ASSERT_EQ(tokens.size(), 3u);
    EXPECT_EQ(gen.detokenize(tokens[0]), "H");
    EXPECT_EQ(gen.detokenize(tokens[1]), "i");
    EXPECT_EQ(gen.detokenize(tokens[2]), "!");
}
