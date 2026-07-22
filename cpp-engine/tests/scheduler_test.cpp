#include <gtest/gtest.h>
#include "scheduler.h"
#include "mock_generator.h"
#include <thread>
#include <unordered_map>
#include <vector>

class SchedulerTest : public ::testing::Test {
protected:
    void SetUp() override {
        ContinuousBatchScheduler::Config config;
        config.max_batch_size = 4;
        config.max_queue_depth = 16;
        auto gen = std::make_unique<MockTokenGenerator>();
        scheduler = std::make_unique<ContinuousBatchScheduler>(
            config, std::move(gen));
    }

    std::unique_ptr<ContinuousBatchScheduler> scheduler;
};

TEST_F(SchedulerTest, SubmitAndComplete) {
    std::vector<std::string> received_tokens;
    std::mutex mu;
    bool finished = false;

    scheduler->set_token_callback(
        [&](const std::string& rid, const std::string& text,
            int32_t tid, bool done, int reason) {
            std::lock_guard<std::mutex> lock(mu);
            received_tokens.push_back(text);
            if (done) finished = true;
        });

    // Start scheduler
    std::thread sched_thread([this] { scheduler->run(); });

    // Submit request
    EXPECT_TRUE(scheduler->submit("req-1", "hello", 5, 0.0f));

    // Wait for completion
    for (int i = 0; i < 1000 && !finished; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::lock_guard<std::mutex> lock(mu);
        if (finished) break;
    }

    scheduler->stop();
    sched_thread.join();

    EXPECT_TRUE(finished);
    EXPECT_EQ(received_tokens.size(), 5u);
}

TEST_F(SchedulerTest, QueueFull) {
    // Fill queue beyond max_queue_depth (16)
    for (int i = 0; i < 16; ++i) {
        EXPECT_TRUE(scheduler->submit(
            "req-" + std::to_string(i), "hello", 100, 0.0f));
    }
    // 17th should be rejected
    EXPECT_FALSE(scheduler->submit("req-16", "hello", 100, 0.0f));
}

TEST_F(SchedulerTest, BatchSizeRespected) {
    scheduler->set_token_callback(
        [](const std::string&, const std::string&,
           int32_t, bool, int) {});

    std::thread sched_thread([this] { scheduler->run(); });

    // Submit more requests than batch size
    for (int i = 0; i < 8; ++i) {
        scheduler->submit("req-" + std::to_string(i), "hello", 100, 0.0f);
    }

    // Let scheduler run a few steps
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Batch size should never exceed 4
    EXPECT_LE(scheduler->metrics().current_batch_size.load(), 4);

    scheduler->stop();
    sched_thread.join();
}

TEST_F(SchedulerTest, ContinuousBatchingAdmitsMidFlight) {
    // This test verifies THE core behavior:
    // sequences that finish should be replaced by waiting ones
    // WITHOUT waiting for the entire batch to complete.
    //
    // Strategy: submit 6 requests (batch=4), with requests 0-1
    // set to max_tokens=2 and requests 2-5 set to max_tokens=100.
    // Requests 0-1 finish quickly, and requests 4-5 should be
    // admitted while 2-3 are still running.

    std::unordered_map<std::string, bool> completed;
    std::mutex mu;

    scheduler->set_token_callback(
        [&](const std::string& rid, const std::string&,
            int32_t, bool done, int) {
            if (done) {
                std::lock_guard<std::mutex> lock(mu);
                completed[rid] = true;
            }
        });

    std::thread sched_thread([this] { scheduler->run(); });

    // Short requests
    scheduler->submit("short-0", "a", 2, 0.0f);
    scheduler->submit("short-1", "b", 2, 0.0f);
    // Long requests
    scheduler->submit("long-2", "c", 100, 0.0f);
    scheduler->submit("long-3", "d", 100, 0.0f);
    // These should be admitted AFTER short-0/1 finish
    scheduler->submit("late-4", "e", 5, 0.0f);
    scheduler->submit("late-5", "f", 5, 0.0f);

    // Wait enough for short + late to finish, but not long
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    {
        std::lock_guard<std::mutex> lock(mu);
        // late-4 and late-5 should have completed (they were short)
        // even though long-2/3 are still running
        EXPECT_TRUE(completed.count("short-0"));
        EXPECT_TRUE(completed.count("short-1"));
        EXPECT_TRUE(completed.count("late-4"));
        EXPECT_TRUE(completed.count("late-5"));
        // long-2/3 might or might not be done depending on timing
    }

    scheduler->stop();
    sched_thread.join();
}

TEST_F(SchedulerTest, FinishesOnFirstTokenWhenMaxTokensIsOne) {
    // Regression test: a sequence with max_new_tokens=1 must finish
    // after the token produced during PREFILL, not generate a second
    // token during DECODE (off-by-one in the finish check).
    std::atomic<int> tokens_for_req{0};
    std::atomic<bool> finished{false};

    scheduler->set_token_callback(
        [&](const std::string& rid, const std::string&,
            int32_t, bool done, int) {
            if (rid == "one-shot") {
                tokens_for_req.fetch_add(1);
                if (done) finished.store(true);
            }
        });

    std::thread sched_thread([this] { scheduler->run(); });
    scheduler->submit("one-shot", "hi", 1, 0.0f);

    for (int i = 0; i < 200 && !finished.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    scheduler->stop();
    sched_thread.join();

    EXPECT_TRUE(finished.load());
    EXPECT_EQ(tokens_for_req.load(), 1);
}

class StaticBatchingSchedulerTest : public ::testing::Test {
protected:
    void SetUp() override {
        ContinuousBatchScheduler::Config config;
        config.max_batch_size = 2;
        config.max_queue_depth = 16;
        config.static_batching = true;
        // Slow, deterministic per-token latency so the test has a wide
        // timing margin: "long" alone takes ~490ms to decode, comfortably
        // longer than the sleeps below.
        MockTokenGenerator::Config gen_config;
        gen_config.latency_per_token = std::chrono::milliseconds(10);
        auto gen = std::make_unique<MockTokenGenerator>(gen_config);
        scheduler = std::make_unique<ContinuousBatchScheduler>(
            config, std::move(gen));
    }

    std::unique_ptr<ContinuousBatchScheduler> scheduler;
};

TEST_F(StaticBatchingSchedulerTest, DoesNotAdmitUntilWholeBatchFinishes) {
    // THE key behavioral difference from continuous batching: a finished
    // sequence's slot is not reclaimed until every sequence in the batch
    // is finished, so a waiting request cannot be admitted mid-flight even
    // though there's a "logically" free slot.
    scheduler->set_token_callback(
        [](const std::string&, const std::string&, int32_t, bool, int) {});

    std::thread sched_thread([this] { scheduler->run(); });

    scheduler->submit("short", "a", 1, 0.0f);   // finishes after prefill (~20ms)
    scheduler->submit("long", "b", 50, 0.0f);   // ~490ms to decode alone

    // Give "short" plenty of time to finish and get stuck in FINISHED state,
    // while "long" is still far from done.
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // Now submit a third request. In static mode it must NOT be admitted
    // while "long" is still running, even though "short" already finished.
    scheduler->submit("queued", "c", 1, 0.0f);
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    EXPECT_EQ(scheduler->metrics().current_batch_size.load(), 2);
    EXPECT_EQ(scheduler->queue_depth(), 1);  // "queued" still waiting

    scheduler->stop();
    sched_thread.join();
}
