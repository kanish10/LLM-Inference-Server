#pragma once
#include <cstdint>
#include <string>

// GPT-2 124M hyperparameters. Shared by the CPU oracle (M1) and the GPU
// engine (M2+) so both read the same config.json produced by export_gpt2.py.
struct GPT2Config {
    int n_layer = 12;
    int n_head = 12;
    int d_model = 768;
    int head_dim = 64;
    int vocab_size = 50257;
    int n_ctx = 1024;
    float layer_norm_epsilon = 1e-5f;
    int eos_token_id = 50256;

    static GPT2Config load_from_json(const std::string& path);
};
