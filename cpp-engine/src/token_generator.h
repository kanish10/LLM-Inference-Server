#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Abstract interface for the model backend.
// The scheduler calls forward() once per decode step with the full batch.
// Implementations: MockTokenGenerator (testing), LlamaCppGenerator (real).
//
// DESIGN DECISION: forward() takes the entire batch, not individual sequences.
// This matters because real models batch all sequences into a single
// matrix multiply. The mock simulates this by sleeping proportional
// to batch size.

class TokenGenerator {
public:
    virtual ~TokenGenerator() = default;

    struct Entry {
        int kv_slot;                     // which KV cache slot
        std::vector<int32_t> tokens;     // prefill: all prompt tokens
                                         // decode: just [last_generated_token]
        int seq_position;                // position offset for rotary embeddings
        float temperature;               // per-sequence temperature
    };

    struct BatchInput {
        std::vector<Entry> entries;
    };

    struct BatchOutput {
        std::vector<int32_t> next_tokens; // one per entry, same order
    };

    // Run forward pass on the batch. Returns one next-token per entry.
    virtual BatchOutput forward(const BatchInput& input) = 0;

    // Tokenize raw text into token IDs.
    virtual std::vector<int32_t> tokenize(const std::string& text) = 0;

    // Detokenize a single token ID to text.
    virtual std::string detokenize(int32_t token_id) = 0;

    // The end-of-sequence token ID for this model.
    virtual int32_t eos_token_id() const = 0;
};
