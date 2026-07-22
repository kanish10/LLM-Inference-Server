// End-to-end test: MockTokenGenerator + ContinuousBatchScheduler +
// EngineServiceImpl, driven over a real gRPC channel (loopback TCP, OS-picked
// port) exactly the way the Go gateway drives the real engine process. Unlike
// scheduler_test.cpp (which calls Scheduler::submit() directly) and the
// manual smoke tests run during development, this is the only automated
// check that the gRPC service layer itself — streaming, per-request
// callbacks, GetMetrics — is wired up correctly end-to-end.
#include <gtest/gtest.h>
#include "grpc_service.h"
#include "scheduler.h"
#include "mock_generator.h"
#include <grpcpp/grpcpp.h>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>

class EngineIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        ContinuousBatchScheduler::Config config;
        config.max_batch_size = 4;
        config.max_queue_depth = 16;
        auto gen = std::make_unique<MockTokenGenerator>();
        scheduler_ = std::make_unique<ContinuousBatchScheduler>(
            config, std::move(gen));
        service_ = std::make_unique<EngineServiceImpl>(scheduler_.get());

        int selected_port = 0;
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(),
                                  &selected_port);
        builder.RegisterService(service_.get());
        server_ = builder.BuildAndStart();
        ASSERT_NE(server_, nullptr);
        ASSERT_NE(selected_port, 0);

        scheduler_thread_ = std::thread([this] { scheduler_->run(); });

        auto channel = grpc::CreateChannel(
            "127.0.0.1:" + std::to_string(selected_port),
            grpc::InsecureChannelCredentials());
        stub_ = engine::InferenceEngine::NewStub(channel);
    }

    void TearDown() override {
        scheduler_->stop();
        scheduler_thread_.join();
        server_->Shutdown();
    }

    // Submits a request and blocks until its stream completes, returning all
    // received tokens in order.
    std::vector<engine::EngineToken> submitAndDrain(
        const std::string& request_id, const std::string& prompt,
        int32_t max_new_tokens) {
        engine::EngineRequest req;
        req.set_request_id(request_id);
        req.set_prompt(prompt);
        req.set_max_new_tokens(max_new_tokens);
        req.set_temperature(0.0f);

        grpc::ClientContext ctx;
        auto reader = stub_->SubmitRequest(&ctx, req);

        std::vector<engine::EngineToken> tokens;
        engine::EngineToken token;
        while (reader->Read(&token)) {
            tokens.push_back(token);
        }
        grpc::Status status = reader->Finish();
        EXPECT_TRUE(status.ok()) << status.error_message();
        return tokens;
    }

    std::unique_ptr<ContinuousBatchScheduler> scheduler_;
    std::unique_ptr<EngineServiceImpl> service_;
    std::unique_ptr<grpc::Server> server_;
    std::thread scheduler_thread_;
    std::unique_ptr<engine::InferenceEngine::Stub> stub_;
};

TEST_F(EngineIntegrationTest, SubmitRequestStreamsTokensEndToEnd) {
    auto tokens = submitAndDrain("it-basic", "hello", 5);

    ASSERT_EQ(tokens.size(), 5u);
    for (size_t i = 0; i < tokens.size(); ++i) {
        EXPECT_EQ(tokens[i].request_id(), "it-basic");
        bool expect_finished = (i == tokens.size() - 1);
        EXPECT_EQ(tokens[i].is_finished(), expect_finished);
    }
    EXPECT_EQ(tokens.back().finish_reason(), engine::FINISH_REASON_LENGTH);
}

TEST_F(EngineIntegrationTest, GetMetricsReflectsCompletedWork) {
    submitAndDrain("it-metrics", "hi", 3);

    grpc::ClientContext ctx;
    engine::MetricsRequest req;
    engine::EngineMetrics resp;
    auto status = stub_->GetMetrics(&ctx, req, &resp);

    ASSERT_TRUE(status.ok()) << status.error_message();
    EXPECT_GE(resp.total_tokens_generated(), 3);
    EXPECT_EQ(resp.max_batch_size(), 4);
    EXPECT_EQ(resp.kv_cache_slots_total(), 4);
}

TEST_F(EngineIntegrationTest, MultipleConcurrentRequestsAllComplete) {
    // Exercises real batching over the wire: more requests than
    // max_batch_size (4), so some must be admitted mid-flight as others
    // finish — the same property SchedulerTest.ContinuousBatchingAdmitsMidFlight
    // checks at the scheduler level, here checked through the full gRPC path.
    constexpr int kNumRequests = 8;
    std::vector<std::thread> threads;
    std::vector<std::vector<engine::EngineToken>> results(kNumRequests);

    for (int i = 0; i < kNumRequests; ++i) {
        threads.emplace_back([&, i] {
            results[i] = submitAndDrain(
                "it-concurrent-" + std::to_string(i), "x", 4 + i);
        });
    }
    for (auto& t : threads) t.join();

    for (int i = 0; i < kNumRequests; ++i) {
        ASSERT_EQ(results[i].size(), static_cast<size_t>(4 + i))
            << "request " << i;
        EXPECT_TRUE(results[i].back().is_finished());
    }
}

TEST_F(EngineIntegrationTest, QueueFullReturnsResourceExhaustedOverGrpc) {
    // max_batch_size=4 and max_queue_depth=16: the first 4 long-running
    // holders get admitted straight into the active batch (and stay there,
    // since they never finish within this test), so it takes
    // 4 + max_queue_depth = 20 holders to actually fill waiting_queue_ to
    // its limit. Verify the 21st submission is then rejected with
    // RESOURCE_EXHAUSTED — the same contract SchedulerTest.QueueFull checks
    // directly against Scheduler::submit(), here checked through the gRPC
    // service's error path.
    constexpr int kHolderCount = 4 /* max_batch_size */ + 16 /* max_queue_depth */;
    std::atomic<bool> stop_reading{false};
    std::vector<std::thread> holders;
    std::vector<std::unique_ptr<grpc::ClientContext>> contexts;

    for (int i = 0; i < kHolderCount; ++i) {
        contexts.push_back(std::make_unique<grpc::ClientContext>());
        engine::EngineRequest req;
        req.set_request_id("it-holder-" + std::to_string(i));
        req.set_prompt("x");
        req.set_max_new_tokens(100000);  // long enough to hold its slot/queue spot
        req.set_temperature(0.0f);

        auto* ctx = contexts.back().get();
        holders.emplace_back([this, ctx, req, &stop_reading] {
            auto reader = stub_->SubmitRequest(ctx, req);
            engine::EngineToken token;
            while (!stop_reading.load() && reader->Read(&token)) {
                // Drain so the server's per-request buffer doesn't grow
                // unbounded; the request itself is deliberately never
                // allowed to finish within this test.
            }
        });
    }

    // Give the holders time to actually land in the queue/batch before
    // testing the boundary.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    engine::EngineRequest overflow_req;
    overflow_req.set_request_id("it-overflow");
    overflow_req.set_prompt("x");
    overflow_req.set_max_new_tokens(1);
    overflow_req.set_temperature(0.0f);

    grpc::ClientContext overflow_ctx;
    auto reader = stub_->SubmitRequest(&overflow_ctx, overflow_req);
    engine::EngineToken token;
    while (reader->Read(&token)) {
    }
    grpc::Status status = reader->Finish();

    EXPECT_EQ(status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);

    stop_reading.store(true);
    for (auto& ctx : contexts) ctx->TryCancel();
    for (auto& t : holders) t.join();
}
