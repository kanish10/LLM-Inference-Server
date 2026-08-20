// M1 acceptance test: the CPU oracle must greedy-decode 32 tokens matching
// HF gpt2's fp32 top-1 continuation on the 10 fixed prompts in
// export/verify_parity.py. Run export_gpt2.py and verify_parity.py first to
// produce models/{weights.bin,manifest.json,config.json,parity_reference.json}.
#include <doctest/doctest.h>

#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "engine/model.h"
#include "reference/cpu_model.h"
#include "reference/weights_loader.h"

namespace {

std::string models_dir() {
    if (const char* env = std::getenv("GPU_ENGINE_MODELS_DIR")) return env;
    return "../models";
}

}  // namespace

TEST_CASE("CPU oracle matches HF fp32 greedy decode on fixed prompts") {
    const std::string dir = models_dir();
    const std::string ref_path = dir + "/parity_reference.json";

    std::ifstream rf(ref_path);
    REQUIRE_MESSAGE(rf.good(), "missing " << ref_path << " -- run export/verify_parity.py first");
    nlohmann::json ref;
    rf >> ref;

    GPT2Config config = GPT2Config::load_from_json(dir + "/config.json");
    WeightsLoader weights(dir);
    CPUModel model(config, weights);

    const int num_new_tokens = ref.at("num_new_tokens").get<int>();
    int num_cases = 0;
    int num_exact_match = 0;

    for (const auto& c : ref.at("cases")) {
        std::vector<int32_t> prompt_tokens;
        for (auto& t : c.at("prompt_tokens")) prompt_tokens.push_back(t.get<int32_t>());
        std::vector<int32_t> expected;
        for (auto& t : c.at("generated_tokens")) expected.push_back(t.get<int32_t>());

        std::vector<int32_t> actual = model.greedy_generate(prompt_tokens, num_new_tokens);

        ++num_cases;
        const bool exact = (actual == expected);
        if (exact) ++num_exact_match;

        INFO("prompt: ", c.at("prompt").get<std::string>());
        CHECK_MESSAGE(exact, "token mismatch for prompt " << c.at("prompt").get<std::string>());
    }

    MESSAGE("CPU oracle exact top-1 match: ", num_exact_match, "/", num_cases);
    CHECK(num_exact_match == num_cases);
}
