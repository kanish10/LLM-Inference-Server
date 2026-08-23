// M6: continuous-vs-static batching, re-measured on real GPU compute
// (gpu-engine's CUDA GPT-2 124M decode path via GpuEngineGenerator) instead
// of MockTokenGenerator's synthetic sleep. Same methodology as the original
// benchmark/benchmark.py: Poisson arrivals, output lengths uniform
// [16,512], batch 8 -- see gpu-engine/VERIFICATION.md for the result.
//
// Standalone (bypasses engine_core/gRPC -- this measures scheduling
// behavior, not the network path): links scheduler.cpp + kv_cache.cpp +
// gpu_engine_generator.cpp directly against gpu-engine's libgpu_kernels.a.
#include "scheduler.h"
#include "gpu_engine_generator.h"

#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

using Clock = std::chrono::steady_clock;

struct RunResult {
    double wall_seconds = 0.0;
    long total_tokens = 0;
    int completed = 0;
};

// arrival_rate_per_sec <= 0 means closed-loop: submit as fast as admission
// allows (used only for the static-capacity calibration pass).
RunResult run_once(bool static_mode, int num_requests, double arrival_rate_per_sec,
                    const std::vector<int32_t>& output_lengths,
                    const std::vector<double>& arrival_offsets, const std::string& weights_dir,
                    int max_batch, int max_queue) {
    ContinuousBatchScheduler::Config cfg;
    cfg.max_batch_size = max_batch;
    cfg.max_queue_depth = max_queue;
    cfg.static_batching = static_mode;

    GpuEngineGenerator::Config gcfg;
    gcfg.weights_dir = weights_dir;
    gcfg.max_batch_size = max_batch;
    gcfg.max_seq_len = 1024;
    gcfg.kv_block_budget = 512;
    auto gen = std::make_unique<GpuEngineGenerator>(gcfg);

    ContinuousBatchScheduler sched(cfg, std::move(gen));

    std::mutex mu;
    int completed = 0;
    long total_tokens = 0;
    std::unordered_map<std::string, int> id_to_idx;
    std::vector<int> tokens_per_req(num_requests, 0);

    sched.set_token_callback([&](const std::string& rid, const std::string&, int32_t, bool done, int) {
        std::lock_guard<std::mutex> lock(mu);
        int idx = id_to_idx.at(rid);
        tokens_per_req[idx]++;
        if (done) {
            ++completed;
            total_tokens += tokens_per_req[idx];
        }
    });

    std::thread sched_thread([&] { sched.run(); });

    Clock::time_point bench_start = Clock::now();
    for (int i = 0; i < num_requests; ++i) {
        std::string rid = "req-" + std::to_string(i);
        {
            std::lock_guard<std::mutex> lock(mu);
            id_to_idx[rid] = i;
        }
        if (arrival_rate_per_sec > 0) {
            auto target = bench_start + std::chrono::duration_cast<Clock::duration>(
                                             std::chrono::duration<double>(arrival_offsets[i]));
            std::this_thread::sleep_until(target);
        }
        while (!sched.submit(rid, "bench prompt", output_lengths[i], 0.0f)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    while (true) {
        {
            std::lock_guard<std::mutex> lock(mu);
            if (completed >= num_requests) break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    Clock::time_point bench_end = Clock::now();

    sched.stop();
    sched_thread.join();

    RunResult r;
    r.wall_seconds = std::chrono::duration<double>(bench_end - bench_start).count();
    r.total_tokens = total_tokens;
    r.completed = completed;
    return r;
}

int main(int argc, char** argv) {
    std::string weights_dir = argc > 1 ? argv[1] : "../../gpu-engine/models";
    const int num_requests = 500;
    const int max_batch = 8;
    const int max_queue = 128;

    std::mt19937 len_rng(42);
    std::uniform_int_distribution<int32_t> len_dist(16, 512);
    std::vector<int32_t> output_lengths(num_requests);
    for (auto& l : output_lengths) l = len_dist(len_rng);

    std::printf("=== Phase 1a: static-mode closed-loop capacity calibration ===\n");
    std::vector<double> no_delay;
    RunResult cal_static =
        run_once(true, num_requests, -1.0, output_lengths, no_delay, weights_dir, max_batch, max_queue);
    double static_req_per_sec = cal_static.completed / cal_static.wall_seconds;
    std::printf("static closed-loop: %d/%d done in %.2fs (%.3f req/s, %ld tokens)\n", cal_static.completed,
                num_requests, cal_static.wall_seconds, static_req_per_sec, cal_static.total_tokens);

    std::printf("=== Phase 1b: continuous-mode closed-loop capacity calibration ===\n");
    RunResult cal_cont =
        run_once(false, num_requests, -1.0, output_lengths, no_delay, weights_dir, max_batch, max_queue);
    double cont_req_per_sec = cal_cont.completed / cal_cont.wall_seconds;
    std::printf("continuous closed-loop: %d/%d done in %.2fs (%.3f req/s, %ld tokens)\n", cal_cont.completed,
                num_requests, cal_cont.wall_seconds, cont_req_per_sec, cal_cont.total_tokens);
    std::printf("closed-loop capacity ratio (continuous/static): %.3fx\n", cont_req_per_sec / static_req_per_sec);

    // Calibrating off static's OWN ceiling is circular: 80% of static's own
    // max keeps static comfortably under-loaded too, so its slot-wasting
    // penalty (which only bites when there's a backlog eager to fill freed
    // slots) never gets stressed. Calibrate off the HIGHER (continuous)
    // ceiling instead, so static gets pushed into backlog while continuous
    // doesn't -- that's the regime the whole algorithm is supposed to win in.
    double arrival_rate = 0.85 * cont_req_per_sec;
    std::printf("=== Target arrival rate (85%% of continuous capacity): %.3f req/s ===\n", arrival_rate);

    std::mt19937 arng(123);
    std::exponential_distribution<double> exp_dist(arrival_rate);
    std::vector<double> arrival_offsets(num_requests);
    double t = 0.0;
    for (auto& off : arrival_offsets) {
        t += exp_dist(arng);
        off = t;
    }

    std::printf("=== Phase 2: continuous batching, Poisson arrivals ===\n");
    RunResult cont = run_once(false, num_requests, arrival_rate, output_lengths, arrival_offsets,
                               weights_dir, max_batch, max_queue);
    double cont_tps = cont.total_tokens / cont.wall_seconds;
    std::printf("continuous: %d/%d done in %.2fs, %ld tokens, %.1f tok/s\n", cont.completed, num_requests,
                cont.wall_seconds, cont.total_tokens, cont_tps);

    std::printf("=== Phase 3: static batching, Poisson arrivals (same rate/lengths) ===\n");
    RunResult stat = run_once(true, num_requests, arrival_rate, output_lengths, arrival_offsets,
                               weights_dir, max_batch, max_queue);
    double stat_tps = stat.total_tokens / stat.wall_seconds;
    std::printf("static: %d/%d done in %.2fs, %ld tokens, %.1f tok/s\n", stat.completed, num_requests,
                stat.wall_seconds, stat.total_tokens, stat_tps);

    std::printf("\n=== RESULT ===\ncontinuous tok/s: %.1f\nstatic tok/s: %.1f\nratio: %.2fx\n", cont_tps,
                stat_tps, cont_tps / stat_tps);
    return 0;
}
