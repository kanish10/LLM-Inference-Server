// llama_generator.cpp — real-model backend, only built with -DWITH_LLAMA=ON.
#include "llama_generator.h"
#include <random>
#include <stdexcept>

LlamaCppGenerator::LlamaCppGenerator(Config config) : config_(config) {
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;
    model_ = llama_model_load_from_file(config.model_path.c_str(), model_params);
    if (!model_) {
        throw std::runtime_error("Failed to load model: " + config.model_path);
    }
    vocab_ = llama_model_get_vocab(model_);
    n_vocab_ = llama_vocab_n_tokens(vocab_);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config.n_ctx;
    ctx_params.n_batch = config.n_batch;
    ctx_params.n_threads = config.n_threads;
    ctx_params.n_threads_batch = config.n_threads;
    // Every Sequence::kv_slot the scheduler hands us doubles as a
    // llama_seq_id, so the context must be able to track at least
    // max_batch_size distinct sequences concurrently.
    ctx_params.n_seq_max = config.max_batch_size;

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        llama_model_free(model_);
        throw std::runtime_error("Failed to create llama context for: " + config.model_path);
    }

    // One seq_id per token is all this scheduler ever needs (no beam search,
    // no shared prefixes across sequences).
    batch_ = llama_batch_init(config_.n_batch, /*embd=*/0, /*n_seq_max=*/1);
}

LlamaCppGenerator::~LlamaCppGenerator() {
    llama_batch_free(batch_);
    if (ctx_) llama_free(ctx_);
    if (model_) llama_model_free(model_);
    llama_backend_free();
}

TokenGenerator::BatchOutput
LlamaCppGenerator::forward(const BatchInput& input) {
    batch_.n_tokens = 0;

    // Track which batch position holds the last token of each entry — that's
    // the position whose logits we sample from to get the next token.
    std::vector<int32_t> logit_positions;
    logit_positions.reserve(input.entries.size());

    for (const auto& entry : input.entries) {
        for (int i = 0; i < static_cast<int>(entry.tokens.size()); ++i) {
            bool need_logits = (i == static_cast<int>(entry.tokens.size()) - 1);
            int idx = batch_.n_tokens;

            batch_.token[idx]    = entry.tokens[i];
            batch_.pos[idx]      = entry.seq_position + i;
            batch_.n_seq_id[idx] = 1;
            batch_.seq_id[idx][0] = entry.kv_slot;
            batch_.logits[idx]   = need_logits;

            if (need_logits) logit_positions.push_back(idx);
            batch_.n_tokens++;
        }
    }

    int rc = llama_decode(ctx_, batch_);
    if (rc != 0) {
        throw std::runtime_error("llama_decode failed with code " + std::to_string(rc));
    }

    BatchOutput output;
    output.next_tokens.reserve(input.entries.size());
    for (size_t i = 0; i < input.entries.size(); ++i) {
        output.next_tokens.push_back(
            sample_token(logit_positions[i], input.entries[i].temperature));
    }
    return output;
}

int32_t LlamaCppGenerator::sample_token(int32_t logits_index, float temperature) {
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(sparams);

    if (temperature <= 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(std::random_device{}()));
    }

    llama_token token = llama_sampler_sample(chain, ctx_, logits_index);
    llama_sampler_free(chain);
    return token;
}

void LlamaCppGenerator::reset_sequence(int kv_slot) {
    llama_memory_seq_rm(llama_get_memory(ctx_), kv_slot, -1, -1);
}

std::vector<int32_t> LlamaCppGenerator::tokenize(const std::string& text) {
    std::vector<int32_t> tokens(text.size() + 16);
    int n = llama_tokenize(vocab_, text.c_str(), static_cast<int32_t>(text.size()),
                           tokens.data(), static_cast<int32_t>(tokens.size()),
                           /*add_special=*/true, /*parse_special=*/false);
    if (n < 0) {
        // Buffer was too small; n is the negated required size.
        tokens.resize(-n);
        n = llama_tokenize(vocab_, text.c_str(), static_cast<int32_t>(text.size()),
                           tokens.data(), static_cast<int32_t>(tokens.size()),
                           /*add_special=*/true, /*parse_special=*/false);
    }
    tokens.resize(n);
    return tokens;
}

std::string LlamaCppGenerator::detokenize(int32_t token_id) {
    char buf[64];
    int n = llama_token_to_piece(vocab_, token_id, buf, sizeof(buf), /*lstrip=*/0, /*special=*/false);
    if (n < 0) return "";
    return std::string(buf, n);
}

int32_t LlamaCppGenerator::eos_token_id() const {
    return llama_vocab_eos(vocab_);
}
