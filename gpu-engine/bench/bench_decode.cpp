// Measures decode-step tok/s and per-step latency, CUDA-Graphs on vs off,
// across a batch sweep -- the source of the resume numbers in
// gpu-engine/CLAUDE.md's benchmark protocol table. Timed with CUDA events
// (per the CLAUDE.md pitfall note: ncu durations are replay-based, use
// them only for kernel-level claims; wall-clock step/throughput claims
// come from here).
//
// UNVERIFIED: never compiled or run (no CUDA device on the dev machine).
//
// Usage:
//   bench_decode --weights ../models --batch 8 --steps 512 --reps 5 --graphs both --out results/decode.csv
#include "engine/engine_api.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string weights_dir = "../models";
    int batch = 8;
    int steps = 512;
    int reps = 5;
    std::string graphs = "both";  // "on", "off", "both"
    std::string out = "results/decode.csv";
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (arg == "--weights") a.weights_dir = next();
        else if (arg == "--batch") a.batch = std::stoi(next());
        else if (arg == "--steps") a.steps = std::stoi(next());
        else if (arg == "--reps") a.reps = std::stoi(next());
        else if (arg == "--graphs") a.graphs = next();
        else if (arg == "--out") a.out = next();
    }
    return a;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

double run_one_rep(Engine& engine, int batch, int steps) {
    std::vector<int> seq_ids(batch);
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    for (int i = 0; i < batch; ++i) {
        seq_ids[i] = engine.add_sequence(prompt.data(), static_cast<int>(prompt.size()));
    }
    std::vector<int> next_tokens(batch);

    // Warmup: also where the CUDA graph for this bucket gets captured, so
    // capture cost never lands inside the timed loop below.
    for (int i = 0; i < 3; ++i) engine.decode_step(seq_ids.data(), batch, next_tokens.data());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<double> step_ms;
    step_ms.reserve(steps);
    for (int s = 0; s < steps; ++s) {
        cudaEventRecord(start);
        engine.decode_step(seq_ids.data(), batch, next_tokens.data());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        step_ms.push_back(ms);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for (int seq_id : seq_ids) engine.free_sequence(seq_id);
    return median(step_ms);
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    EngineConfig cfg;
    cfg.max_batch = std::max(args.batch, 8);
    cfg.max_seq_len = 512;
    cfg.kv_block_budget = 512;

    std::ofstream out(args.out);
    if (!out) {
        std::fprintf(stderr, "cannot open output file: %s\n", args.out.c_str());
        return 1;
    }
    out << "batch,graphs,rep,median_step_ms,tokens_per_sec\n";

    std::vector<bool> graph_modes;
    if (args.graphs == "on") graph_modes = {true};
    else if (args.graphs == "off") graph_modes = {false};
    else graph_modes = {true, false};

    for (bool use_graphs : graph_modes) {
        Engine engine;
        if (!engine.init(args.weights_dir, cfg)) {
            std::fprintf(stderr, "engine.init failed for weights_dir=%s\n", args.weights_dir.c_str());
            return 1;
        }
        engine.set_use_cuda_graphs(use_graphs);

        for (int rep = 0; rep < args.reps; ++rep) {
            const double median_ms = run_one_rep(engine, args.batch, args.steps);
            const double tok_per_s = (args.batch * 1000.0) / median_ms;
            out << args.batch << "," << (use_graphs ? "on" : "off") << "," << rep << ","
                << median_ms << "," << tok_per_s << "\n";
            std::printf("batch=%d graphs=%s rep=%d median_step_ms=%.4f tok/s=%.1f\n", args.batch,
                        use_graphs ? "on" : "off", rep, median_ms, tok_per_s);
        }
    }

    std::printf("Wrote %s\n", args.out.c_str());
    return 0;
}
