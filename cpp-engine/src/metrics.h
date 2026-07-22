#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>

struct SchedulerMetrics {
    // Counters (monotonically increasing)
    std::atomic<int64_t> total_tokens_generated{0};
    std::atomic<int64_t> total_requests_completed{0};
    std::atomic<int64_t> total_requests_submitted{0};

    // Gauges (current value)
    std::atomic<int> current_batch_size{0};
    std::atomic<int> queue_depth{0};
    std::atomic<int> kv_slots_used{0};

    // Rolling averages (updated by scheduler thread)
    std::atomic<float> batch_occupancy_pct{0.0f};   // mean(batch_size / max_batch_size)
    std::atomic<float> tokens_per_second{0.0f};     // last 5s window

    // For tokens_per_second calculation
    struct Window {
        static constexpr int SIZE = 50;  // 50 steps ~= 5 seconds at 100ms/step
        int64_t token_counts[SIZE] = {};
        std::chrono::steady_clock::time_point timestamps[SIZE] = {};
        int head = 0;
        int count = 0;

        void record(int64_t tokens, std::chrono::steady_clock::time_point now) {
            token_counts[head] = tokens;
            timestamps[head] = now;
            head = (head + 1) % SIZE;
            if (count < SIZE) ++count;
        }

        float compute_rate() const {
            if (count < 2) return 0;
            int oldest = (count < SIZE) ? 0 : head;
            int newest = (head - 1 + SIZE) % SIZE;
            auto dt = std::chrono::duration<float>(
                timestamps[newest] - timestamps[oldest]).count();
            if (dt <= 0) return 0;
            int64_t total = 0;
            for (int i = 0; i < count; i++) total += token_counts[i];
            return static_cast<float>(total) / dt;
        }
    };
};
