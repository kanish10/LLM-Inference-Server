// M2 acceptance tests: per-kernel parity vs the CPU-FP32 reference
// (max_abs_err < 5e-2 per activation tensor), plus an end-to-end greedy
// parity check against the CPU oracle (>=8/10 prompts, mismatches only at
// near-tie logits). Only built when a CUDA toolchain is present (see
// CMakeLists.txt) -- this file has never been compiled or run; see
// gpu-engine/VERIFICATION.md.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <vector>

#include "engine/engine_api.h"
#include "engine/kernels.h"
#include "engine/model.h"
#include "reference/cpu_model.h"
#include "reference/weights_loader.h"

namespace {

std::string models_dir() {
    if (const char* env = std::getenv("GPU_ENGINE_MODELS_DIR")) return env;
    return "../models";
}

float max_abs_err(const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
    return worst;
}

std::vector<float> half_to_float(const std::vector<half>& in) {
    std::vector<float> out(in.size());
    for (size_t i = 0; i < in.size(); ++i) out[i] = __half2float(in[i]);
    return out;
}

std::vector<half> float_to_half(const std::vector<float>& in) {
    std::vector<half> out(in.size());
    for (size_t i = 0; i < in.size(); ++i) out[i] = __float2half(in[i]);
    return out;
}

}  // namespace

TEST_CASE("kernel: embedding matches wte[tok] + wpe[pos]") {
    const int batch = 4, d_model = 16;
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> wte_f(64 * d_model), wpe_f(32 * d_model);
    for (auto& v : wte_f) v = dist(rng);
    for (auto& v : wpe_f) v = dist(rng);
    auto wte_h = float_to_half(wte_f);
    auto wpe_h = float_to_half(wpe_f);

    std::vector<int> tok_ids = {3, 10, 40, 63};
    std::vector<int> positions = {0, 1, 2, 3};

    half *d_wte, *d_wpe, *d_out;
    int *d_tok, *d_pos;
    cudaMalloc(&d_wte, wte_h.size() * sizeof(half));
    cudaMalloc(&d_wpe, wpe_h.size() * sizeof(half));
    cudaMalloc(&d_out, static_cast<size_t>(batch) * d_model * sizeof(half));
    cudaMalloc(&d_tok, batch * sizeof(int));
    cudaMalloc(&d_pos, batch * sizeof(int));
    cudaMemcpy(d_wte, wte_h.data(), wte_h.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wpe, wpe_h.data(), wpe_h.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tok, tok_ids.data(), batch * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, positions.data(), batch * sizeof(int), cudaMemcpyHostToDevice);

    kernels::launch_embedding(d_out, d_wte, d_wpe, d_tok, d_pos, batch, d_model, nullptr);
    cudaDeviceSynchronize();

    std::vector<half> out_h(static_cast<size_t>(batch) * d_model);
    cudaMemcpy(out_h.data(), d_out, out_h.size() * sizeof(half), cudaMemcpyDeviceToHost);
    auto out_f = half_to_float(out_h);

    std::vector<float> expected(static_cast<size_t>(batch) * d_model);
    for (int b = 0; b < batch; ++b)
        for (int d = 0; d < d_model; ++d)
            expected[b * d_model + d] =
                wte_f[tok_ids[b] * d_model + d] + wpe_f[positions[b] * d_model + d];

    CHECK(max_abs_err(out_f, expected) < 5e-2f);

    cudaFree(d_wte);
    cudaFree(d_wpe);
    cudaFree(d_out);
    cudaFree(d_tok);
    cudaFree(d_pos);
}

TEST_CASE("kernel: fused add+layernorm matches CPU reference") {
    const int batch = 3, d_model = 32;
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> x_f(batch * d_model), r_f(batch * d_model), gamma_f(d_model), beta_f(d_model);
    for (auto& v : x_f) v = dist(rng);
    for (auto& v : r_f) v = dist(rng);
    for (auto& v : gamma_f) v = 1.0f + 0.1f * dist(rng);
    for (auto& v : beta_f) v = 0.1f * dist(rng);

    // CPU reference: residual_out = x + r; out = LN(residual_out) * gamma + beta.
    std::vector<float> expected_out(batch * d_model);
    const float eps = 1e-5f;
    for (int b = 0; b < batch; ++b) {
        double mean = 0.0;
        std::vector<float> row(d_model);
        for (int d = 0; d < d_model; ++d) {
            row[d] = x_f[b * d_model + d] + r_f[b * d_model + d];
            mean += row[d];
        }
        mean /= d_model;
        double var = 0.0;
        for (int d = 0; d < d_model; ++d) var += (row[d] - mean) * (row[d] - mean);
        var /= d_model;
        const float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + eps);
        for (int d = 0; d < d_model; ++d) {
            expected_out[b * d_model + d] =
                static_cast<float>((row[d] - mean) * inv_std) * gamma_f[d] + beta_f[d];
        }
    }

    auto x_h = float_to_half(x_f), r_h = float_to_half(r_f);
    auto gamma_h = float_to_half(gamma_f), beta_h = float_to_half(beta_f);

    half *d_x, *d_r, *d_gamma, *d_beta, *d_out, *d_rout;
    const size_t n = static_cast<size_t>(batch) * d_model;
    cudaMalloc(&d_x, n * sizeof(half));
    cudaMalloc(&d_r, n * sizeof(half));
    cudaMalloc(&d_gamma, d_model * sizeof(half));
    cudaMalloc(&d_beta, d_model * sizeof(half));
    cudaMalloc(&d_out, n * sizeof(half));
    cudaMalloc(&d_rout, n * sizeof(half));
    cudaMemcpy(d_x, x_h.data(), n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, r_h.data(), n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma_h.data(), d_model * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta_h.data(), d_model * sizeof(half), cudaMemcpyHostToDevice);

    kernels::launch_add_layernorm(d_out, d_rout, d_x, d_r, d_gamma, d_beta, batch, d_model, eps,
                                   nullptr);
    cudaDeviceSynchronize();

    std::vector<half> out_h(n);
    cudaMemcpy(out_h.data(), d_out, n * sizeof(half), cudaMemcpyDeviceToHost);
    CHECK(max_abs_err(half_to_float(out_h), expected_out) < 5e-2f);

    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_out);
    cudaFree(d_rout);
}

TEST_CASE("kernel: bias+GELU(tanh) matches CPU reference") {
    const int batch = 2, d_ff = 24;
    std::mt19937 rng(2);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    std::vector<float> x_f(batch * d_ff), bias_f(d_ff);
    for (auto& v : x_f) v = dist(rng);
    for (auto& v : bias_f) v = dist(rng);

    auto gelu = [](float v) {
        const float k0 = 0.7978845608028654f;
        return 0.5f * v * (1.0f + std::tanh(k0 * (v + 0.044715f * v * v * v)));
    };
    std::vector<float> expected(batch * d_ff);
    for (int b = 0; b < batch; ++b)
        for (int d = 0; d < d_ff; ++d)
            expected[b * d_ff + d] = gelu(x_f[b * d_ff + d] + bias_f[d]);

    auto x_h = float_to_half(x_f), bias_h = float_to_half(bias_f);
    half *d_x, *d_bias, *d_out;
    const size_t n = static_cast<size_t>(batch) * d_ff;
    cudaMalloc(&d_x, n * sizeof(half));
    cudaMalloc(&d_bias, d_ff * sizeof(half));
    cudaMalloc(&d_out, n * sizeof(half));
    cudaMemcpy(d_x, x_h.data(), n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias_h.data(), d_ff * sizeof(half), cudaMemcpyHostToDevice);

    kernels::launch_bias_gelu(d_out, d_x, d_bias, batch, d_ff, nullptr);
    cudaDeviceSynchronize();

    std::vector<half> out_h(n);
    cudaMemcpy(out_h.data(), d_out, n * sizeof(half), cudaMemcpyDeviceToHost);
    CHECK(max_abs_err(half_to_float(out_h), expected) < 5e-2f);

    cudaFree(d_x);
    cudaFree(d_bias);
    cudaFree(d_out);
}

TEST_CASE("kernel: argmax matches manual max") {
    const int batch = 2, vocab = 100;
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::vector<float> logits_f(batch * vocab);
    for (auto& v : logits_f) v = dist(rng);
    // Force an unambiguous winner per row.
    logits_f[0 * vocab + 42] = 100.0f;
    logits_f[1 * vocab + 7] = 100.0f;

    auto logits_h = float_to_half(logits_f);
    half* d_logits;
    int* d_out;
    cudaMalloc(&d_logits, logits_h.size() * sizeof(half));
    cudaMalloc(&d_out, batch * sizeof(int));
    cudaMemcpy(d_logits, logits_h.data(), logits_h.size() * sizeof(half), cudaMemcpyHostToDevice);

    kernels::launch_argmax(d_out, d_logits, batch, vocab, nullptr);
    cudaDeviceSynchronize();

    std::vector<int> out(batch);
    cudaMemcpy(out.data(), d_out, batch * sizeof(int), cudaMemcpyDeviceToHost);
    CHECK(out[0] == 42);
    CHECK(out[1] == 7);

    cudaFree(d_logits);
    cudaFree(d_out);
}

TEST_CASE("end-to-end: GPU engine greedy decode matches CPU oracle on >=8/10 prompts") {
    const std::string dir = models_dir();
    const std::string ref_path = dir + "/parity_reference.json";
    std::ifstream rf(ref_path);
    REQUIRE_MESSAGE(rf.good(), "missing " << ref_path << " -- run export/verify_parity.py first");
    nlohmann::json ref;
    rf >> ref;
    const int num_new_tokens = ref.at("num_new_tokens").get<int>();

    EngineConfig cfg;
    cfg.max_batch = 1;
    cfg.max_seq_len = 256;
    cfg.kv_block_budget = 64;
    Engine engine;
    REQUIRE(engine.init(dir, cfg));

    int matches = 0, total = 0;
    for (const auto& c : ref.at("cases")) {
        std::vector<int32_t> prompt_tokens;
        for (auto& t : c.at("prompt_tokens")) prompt_tokens.push_back(t.get<int32_t>());
        std::vector<int32_t> expected;
        for (auto& t : c.at("generated_tokens")) expected.push_back(t.get<int32_t>());

        int seq_id = engine.add_sequence(prompt_tokens.data(), static_cast<int>(prompt_tokens.size()));
        REQUIRE(seq_id >= 0);

        std::vector<int32_t> actual;
        const int total_steps = static_cast<int>(prompt_tokens.size()) - 1 + num_new_tokens;
        for (int step = 0; step < total_steps; ++step) {
            int next_tok = 0;
            engine.decode_step(&seq_id, 1, &next_tok);
            if (step >= static_cast<int>(prompt_tokens.size()) - 1) actual.push_back(next_tok);
        }
        engine.free_sequence(seq_id);

        ++total;
        if (actual == expected) ++matches;
    }

    MESSAGE("GPU engine exact top-1 match: ", matches, "/", total);
    CHECK(matches >= 8);
}
