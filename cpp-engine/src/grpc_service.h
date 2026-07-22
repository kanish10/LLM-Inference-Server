#pragma once
#include "scheduler.h"
#include "engine.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <unordered_map>

// Per-request token queue: the scheduler pushes tokens here,
// the gRPC handler pops them and streams to the client.
struct RequestStream {
    std::mutex mu;
    std::condition_variable cv;
    std::queue<engine::EngineToken> tokens;
    bool done = false;
};

class EngineServiceImpl final : public engine::InferenceEngine::Service {
public:
    explicit EngineServiceImpl(ContinuousBatchScheduler* scheduler);

    grpc::Status SubmitRequest(
        grpc::ServerContext* context,
        const engine::EngineRequest* request,
        grpc::ServerWriter<engine::EngineToken>* writer) override;

    grpc::Status GetMetrics(
        grpc::ServerContext* context,
        const engine::MetricsRequest* request,
        engine::EngineMetrics* response) override;

private:
    ContinuousBatchScheduler* scheduler_;  // non-owning

    // Registry of active request streams
    std::mutex streams_mutex_;
    std::unordered_map<std::string, std::shared_ptr<RequestStream>> streams_;

    // Called by scheduler via token_callback_
    void on_token(const std::string& request_id,
                  const std::string& text,
                  int32_t token_id,
                  bool is_finished,
                  int finish_reason);
};
