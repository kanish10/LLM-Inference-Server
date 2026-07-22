#pragma once
#include "kv_cache.h"
#include "metrics.h"
#include "sequence.h"
#include "token_generator.h"
#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

// ContinuousBatchScheduler — the heart of the project.
//
// ONE THREAD runs the main loop. That thread is the sole writer to
// active_batch_. gRPC threads only touch waiting_queue_ (mutex-guarded)
// and read atomic metrics. This eliminates data races on the hot path.
//
// THE MAIN LOOP, every iteration:
//   1. ADMIT:   fill empty batch slots from waiting_queue_
//   2. PREFILL: for newly admitted sequences, process full prompt
//   3. DECODE:  for all DECODE-state sequences, generate one token
//   4. CHECK:   mark sequences that hit EOS or max_new_tokens as FINISHED
//   5. EVICT:   remove FINISHED sequences, free KV slots, fire callbacks
//   6. METRICS: update rolling counters
//
// This order matters: ADMIT before PREFILL means a newly arrived request
// can start generating in the same loop iteration it's admitted —
// minimizing TTFT. EVICT after CHECK (not combined) keeps the code
// paths clean and testable.
//
// STATIC BATCHING MODE (config_.static_batching): evict_finished() is
// replaced by evict_finished_static(), which does nothing until every
// sequence in the batch is FINISHED, then frees them all at once. That
// single branch is the entire difference between the two scheduling
// policies — see CLAUDE.md "Static Batching Baseline".

class ContinuousBatchScheduler {
public:
    struct Config {
        int max_batch_size = 8;
        int max_sequence_length = 2048;
        int max_queue_depth = 64;

        // Model params (used for KV cache sizing)
        int num_layers = 22;          // TinyLlama-1.1B
        int num_kv_heads = 4;         // TinyLlama uses GQA with 4 KV heads
        int head_dim = 64;            // TinyLlama

        // If true, use static batching (evict only when the whole batch
        // is finished) instead of continuous batching. Used to produce
        // the baseline comparison for the benchmark.
        bool static_batching = false;
    };

    // Token callback: called from scheduler thread when a token is generated.
    // Implementations must be thread-safe if they post to other threads.
    // Args: request_id, token_text, is_finished, finish_reason
    using TokenCallback = std::function<void(
        const std::string& request_id,
        const std::string& token_text,
        int32_t token_id,
        bool is_finished,
        int finish_reason  // 0=none, 1=eos, 2=length
    )>;

    ContinuousBatchScheduler(Config config,
                             std::unique_ptr<TokenGenerator> generator);

    // Submit a new request. Thread-safe (called from gRPC threads).
    // Returns false if queue is full (caller should return RESOURCE_EXHAUSTED).
    bool submit(const std::string& request_id,
                const std::string& prompt,
                int32_t max_new_tokens,
                float temperature);

    // Set callback for token output. Must be called before run().
    void set_token_callback(TokenCallback cb);

    // Run the scheduler loop. Blocks until stop() is called.
    // Call from a dedicated thread.
    void run();

    // Signal the scheduler to stop after the current iteration.
    void stop();

    // Read-only metrics. Thread-safe (atomics).
    const SchedulerMetrics& metrics() const { return metrics_; }

    // Current queue depth (for admission control in Go gateway).
    int queue_depth() const { return metrics_.queue_depth.load(); }

    int max_batch_size() const { return config_.max_batch_size; }

private:
    // Main loop phases (called only from scheduler thread)
    void admit_new();
    void prefill_step();
    void decode_step();
    void check_finished();
    void evict_finished();
    void evict_finished_static();
    void update_metrics();

    Config config_;
    std::unique_ptr<TokenGenerator> generator_;
    KVCacheAllocator kv_cache_;
    TokenCallback token_callback_;

    // Active batch — ONLY touched by scheduler thread. No lock.
    std::vector<Sequence> active_batch_;

    // Waiting queue — shared between gRPC threads (writers) and
    // scheduler thread (reader). Protected by queue_mutex_.
    std::mutex queue_mutex_;
    std::deque<Sequence> waiting_queue_;

    std::atomic<bool> running_{false};
    SchedulerMetrics metrics_;
    SchedulerMetrics::Window rate_window_;
};
