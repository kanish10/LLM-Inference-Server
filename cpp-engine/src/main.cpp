#include "grpc_service.h"
#include "mock_generator.h"
#include "scheduler.h"
#ifdef WITH_LLAMA
#include "llama_generator.h"
#endif
#include <grpcpp/grpcpp.h>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

int main(int argc, char* argv[]) {
    // Parse CLI flags (keep it simple — no library needed)
    std::string mode = "mock";        // "mock" or "llama"
    std::string model_path;
    std::string listen_addr = "unix:///tmp/inference-engine.sock";
    int max_batch_size = 8;
    int max_queue_depth = 64;
    bool static_batching = false;
    int mock_latency_us = 500;
    int mock_fixed_overhead_us = 0;
    int n_ctx = 2048;
    int n_gpu_layers = 0;
    int n_threads = 4;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--listen" && i + 1 < argc) listen_addr = argv[++i];
        else if (arg == "--batch-size" && i + 1 < argc) max_batch_size = std::stoi(argv[++i]);
        else if (arg == "--max-queue" && i + 1 < argc) max_queue_depth = std::stoi(argv[++i]);
        else if (arg == "--static") static_batching = true;
        else if (arg == "--mock-latency-us" && i + 1 < argc) mock_latency_us = std::stoi(argv[++i]);
        else if (arg == "--mock-fixed-overhead-us" && i + 1 < argc) mock_fixed_overhead_us = std::stoi(argv[++i]);
        else if (arg == "--n-ctx" && i + 1 < argc) n_ctx = std::stoi(argv[++i]);
        else if (arg == "--n-gpu-layers" && i + 1 < argc) n_gpu_layers = std::stoi(argv[++i]);
        else if (arg == "--n-threads" && i + 1 < argc) n_threads = std::stoi(argv[++i]);
        else if (arg == "--help") {
            std::cout << "Usage: engine [--mode mock|llama] [--model PATH] "
                         "[--listen ADDR] [--batch-size N] [--max-queue N] [--static] "
                         "[--mock-latency-us N] [--mock-fixed-overhead-us N] "
                         "[--n-ctx N] [--n-gpu-layers N] [--n-threads N]\n";
            return 0;
        }
    }

    // Create token generator
    std::unique_ptr<TokenGenerator> generator;
    if (mode == "llama") {
#ifdef WITH_LLAMA
        if (model_path.empty()) {
            std::cerr << "--mode llama requires --model PATH\n";
            return 1;
        }
        LlamaCppGenerator::Config cfg;
        cfg.model_path = model_path;
        cfg.n_ctx = n_ctx;
        cfg.n_gpu_layers = n_gpu_layers;
        cfg.n_threads = n_threads;
        cfg.max_batch_size = max_batch_size;
        generator = std::make_unique<LlamaCppGenerator>(cfg);
        std::cout << "Loaded model: " << model_path << "\n";
#else
        std::cerr << "Engine was built without WITH_LLAMA support. "
                     "Rebuild with -DWITH_LLAMA=ON.\n";
        return 1;
#endif
    } else {
        MockTokenGenerator::Config mock_config;
        mock_config.latency_per_token = std::chrono::microseconds(mock_latency_us);
        mock_config.fixed_overhead = std::chrono::microseconds(mock_fixed_overhead_us);
        generator = std::make_unique<MockTokenGenerator>(mock_config);
        std::cout << "Using mock generator (latency_per_token=" << mock_latency_us
                  << "us, fixed_overhead=" << mock_fixed_overhead_us << "us)\n";
    }

    // Create scheduler
    ContinuousBatchScheduler::Config sched_config;
    sched_config.max_batch_size = max_batch_size;
    sched_config.max_queue_depth = max_queue_depth;
    sched_config.static_batching = static_batching;
    ContinuousBatchScheduler scheduler(sched_config, std::move(generator));

    std::cout << "Scheduling mode: "
              << (static_batching ? "static" : "continuous") << "\n";

    // Create gRPC service
    EngineServiceImpl service(&scheduler);

    // Start gRPC server
    grpc::ServerBuilder builder;
    builder.AddListeningPort(listen_addr,
                             grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    if (!server) {
        std::cerr << "Failed to start gRPC server on " << listen_addr << "\n";
        return 1;
    }
    std::cout << "Engine listening on " << listen_addr << "\n";

    // Start scheduler on dedicated thread
    std::thread scheduler_thread([&scheduler] { scheduler.run(); });

    server->Wait();

    scheduler.stop();
    scheduler_thread.join();
    return 0;
}
