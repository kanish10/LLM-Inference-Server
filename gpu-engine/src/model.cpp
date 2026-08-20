#include "engine/model.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

GPT2Config GPT2Config::load_from_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open config file: " + path);
    nlohmann::json j;
    f >> j;

    GPT2Config cfg;
    cfg.n_layer = j.at("n_layer").get<int>();
    cfg.n_head = j.at("n_head").get<int>();
    cfg.d_model = j.at("d_model").get<int>();
    cfg.head_dim = j.at("head_dim").get<int>();
    cfg.vocab_size = j.at("vocab_size").get<int>();
    cfg.n_ctx = j.at("n_ctx").get<int>();
    cfg.layer_norm_epsilon = j.at("layer_norm_epsilon").get<float>();
    cfg.eos_token_id = j.at("eos_token_id").get<int>();
    return cfg;
}
