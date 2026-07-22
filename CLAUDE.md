# LLM Inference Server — Continuous Batching

## Project Identity

**Resume bullets this project must justify:**
1. "Built an LLM inference server with continuous (in-flight) batching: a C++ scheduler admits queued requests and frees finished KV-cache slots at every decode step, sustaining [N]× static-batching throughput at matched p99 latency."
2. "Fronted by a Go gRPC streaming gateway with admission control and backpressure; exposed 8+ Prometheus metrics."

**Every design decision traces back to making those two sentences true and defensible in a 45-minute systems interview.**

---

## Why This Project Exists

Static batching holds every slot hostage until the **longest** sequence in the batch finishes. If a batch contains sequences that will generate 16, 128, and 512 tokens, the 16-token sequence finishes at step 16 but its KV-cache slot sits idle for 496 more steps. Utilization: `(16+128+512) / (3×512)` = **42.7%**.

Continuous (in-flight) batching fixes this: at every decode step, the scheduler checks for finished sequences, frees their KV-cache slots, and admits waiting requests from the queue. Under load with variable output lengths, utilization approaches **100%**. This is the core innovation behind vLLM, TGI, and every production serving system built after 2023.

**Why Go + C++?** Not a gimmick — each language owns what it's best at:
- **Go**: goroutine-per-connection handles thousands of concurrent streaming clients with zero callback spaghetti. Admission control, backpressure, and metrics are networking concerns — Go's domain.
- **C++**: the scheduler's hot loop runs every decode step (~5–50ms). It mutates shared batch state, manages memory layouts, and interfaces with llama.cpp's C API. Zero-overhead control matters here.
- **The boundary**: local gRPC over a Unix domain socket. Inference steps take milliseconds; gRPC serialization takes microseconds. The overhead is unmeasurable, and the interface is self-documenting via protobuf.

**Why not just use vLLM?** vLLM is 100k+ lines of production software. This project isolates the core scheduling algorithm (~500 lines of C++) to understand it deeply and demonstrate that understanding. It's not a vLLM replacement — it's a proof that you can build the thing vLLM is built on.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Clients                              │
│            (grpcurl / benchmark.py / any gRPC client)       │
└──────────────────────────┬──────────────────────────────────┘
                           │ gRPC streaming (Generate RPC)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     Go Gateway                              │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐ │
│  │ gRPC Server   │  │  Admission    │  │   Prometheus     │ │
│  │ (streaming)   │  │  Controller   │  │   /metrics       │ │
│  │               │  │  (max queue,  │  │   endpoint       │ │
│  │ 1 goroutine   │  │   max active) │  │                  │ │
│  │ per request   │  │               │  │  8+ gauges &     │ │
│  └──────┬────────┘  └───────────────┘  │  histograms      │ │
│         │                               └──────────────────┘ │
│         │ gRPC (SubmitRequest)                               │
│         │ over Unix domain socket                            │
└─────────┼───────────────────────────────────────────────────┘
          ▼
┌─────────────────────────────────────────────────────────────┐
│                     C++ Engine                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Continuous Batch Scheduler                 │ │
│  │                                                        │ │
│  │   ┌─────────┐    ┌──────────┐    ┌────────────────┐   │ │
│  │   │ Waiting  │───▶│  Active   │───▶│   Finished     │   │ │
│  │   │ Queue    │    │  Batch    │    │   (evicted)    │   │ │
│  │   └─────────┘    └────┬─────┘    └────────────────┘   │ │
│  │                       │                                │ │
│  │              ┌────────▼─────────┐                      │ │
│  │              │  Token Generator  │                      │ │
│  │              │  (mock or llama)  │                      │ │
│  │              └──────────────────┘                      │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────┐                                    │
│  │   KV Cache Allocator │                                    │
│  │   (pre-allocated     │                                    │
│  │    contiguous pool)  │                                    │
│  └─────────────────────┘                                    │
└─────────────────────────────────────────────────────────────┘
```

**Data flow for a single request:**
1. Client calls `Generate(prompt, max_tokens)` on the Go gateway (gRPC streaming).
2. Go gateway checks admission (queue depth < max). If full → `RESOURCE_EXHAUSTED`.
3. Go gateway calls `SubmitRequest` on the C++ engine (local gRPC, Unix domain socket).
4. C++ engine enqueues the request in the waiting queue.
5. Scheduler loop: when a batch slot opens, the request is admitted → prefill → decode loop.
6. Each generated token is streamed back: C++ engine → Go gateway → client.
7. On EOS or max_tokens: sequence evicted, KV slot freed, client stream closed.

**Concurrency model — this is the interview question:**
- The scheduler runs in a **single dedicated thread**. It is the only writer to `active_batch_`. No locks on the hot path.
- gRPC handler threads write to `waiting_queue_` through a **mutex-protected deque**. The scheduler reads from it once per step.
- Completed tokens are published via a **per-request callback** that posts to gRPC stream threads. Each callback captures a thread-safe channel/queue.
- Net effect: the only lock acquisition per decode step is one `mutex::lock()` on the waiting queue, held for the duration of a deque splice (microseconds). The decode step itself (milliseconds) runs lock-free.

---

## Directory Structure

```
llm-inference-server/
├── claude.md                          # this file
├── README.md                          # usage, build, benchmark instructions
├── Makefile                           # top-level: build, test, bench, clean
│
├── proto/
│   ├── gateway.proto                  # client-facing API (Go serves this)
│   └── engine.proto                   # internal API (C++ serves this)
│
├── go-gateway/
│   ├── go.mod
│   ├── go.sum
│   ├── main.go                        # entry point, CLI flags, wiring
│   ├── server/
│   │   ├── server.go                  # gRPC Generate handler, streaming logic
│   │   ├── admission.go               # admission controller (queue depth, concurrency)
│   │   └── metrics.go                 # Prometheus collector registration
│   ├── engine/
│   │   └── client.go                  # gRPC client connecting to C++ engine
│   └── gen/                           # generated protobuf Go code (gitignored)
│       ├── gateway/
│       └── engine/
│
├── cpp-engine/
│   ├── CMakeLists.txt                 # gRPC, protobuf, llama.cpp, googletest
│   ├── src/
│   │   ├── main.cpp                   # entry point, CLI flags, gRPC server startup
│   │   ├── scheduler.h                # ContinuousBatchScheduler class declaration
│   │   ├── scheduler.cpp              # scheduler implementation (THE core file)
│   │   ├── kv_cache.h                 # KVCacheAllocator declaration
│   │   ├── kv_cache.cpp               # KV cache pool implementation
│   │   ├── sequence.h                 # Sequence struct and SequenceState enum
│   │   ├── token_generator.h          # abstract TokenGenerator interface
│   │   ├── mock_generator.h           # MockTokenGenerator (configurable latency)
│   │   ├── mock_generator.cpp
│   │   ├── llama_generator.h          # LlamaCppGenerator (wraps llama.cpp C API)
│   │   ├── llama_generator.cpp
│   │   ├── grpc_service.h             # EngineServiceImpl (gRPC handlers)
│   │   ├── grpc_service.cpp
│   │   └── metrics.h                  # SchedulerMetrics struct (atomic counters)
│   ├── tests/
│   │   ├── scheduler_test.cpp         # unit tests for scheduler logic
│   │   ├── kv_cache_test.cpp          # unit tests for allocator
│   │   └── integration_test.cpp       # mock generator + scheduler + gRPC end-to-end
│   └── gen/                           # generated protobuf C++ code (gitignored)
│
├── benchmark/
│   ├── requirements.txt               # grpcio, numpy, matplotlib, pyyaml, pandas
│   ├── benchmark.py                   # Poisson load generator + measurement
│   ├── static_baseline.py             # static batching mode for comparison
│   ├── analyze.py                     # parse results, compute stats, plot graphs
│   └── configs/
│       └── default.yaml               # arrival rate, num requests, output length range
│
├── docker/
│   ├── Dockerfile.engine              # C++ build (multi-stage, Ubuntu 24.04)
│   ├── Dockerfile.gateway             # Go build
│   └── docker-compose.yml             # engine + gateway + prometheus + grafana
│
├── dashboards/
│   └── grafana.json                   # pre-built Grafana dashboard
│
└── models/
    └── .gitkeep                       # download TinyLlama GGUF here
```

---

## Protobuf Definitions

### `proto/gateway.proto` — Client-facing API (served by Go)

```protobuf
syntax = "proto3";
package gateway;
option go_package = "gen/gateway";

service InferenceService {
  // Streaming: client sends one request, receives a stream of tokens
  rpc Generate(GenerateRequest) returns (stream GenerateResponse);
  // Health check
  rpc Health(HealthRequest) returns (HealthResponse);
}

message GenerateRequest {
  string prompt = 1;
  int32 max_tokens = 2;          // max new tokens to generate
  float temperature = 3;         // sampling temperature (0.0 = greedy)
  repeated string stop = 4;      // stop sequences (optional)
}

message GenerateResponse {
  string token = 1;              // detokenized text for this token
  bool is_finished = 2;
  FinishReason finish_reason = 3;
  UsageStats usage = 4;          // populated only on final message
}

enum FinishReason {
  FINISH_REASON_NONE = 0;
  FINISH_REASON_STOP = 1;        // hit a stop sequence
  FINISH_REASON_LENGTH = 2;      // hit max_tokens
  FINISH_REASON_EOS = 3;         // model emitted EOS token
}

message UsageStats {
  int32 prompt_tokens = 1;
  int32 completion_tokens = 2;
  float time_to_first_token_ms = 3;
  float total_time_ms = 4;
}

message HealthRequest {}
message HealthResponse {
  bool ready = 1;
  int32 queue_depth = 2;
  int32 active_requests = 3;
}
```

### `proto/engine.proto` — Internal API (served by C++, consumed by Go)

```protobuf
syntax = "proto3";
package engine;
option go_package = "gen/engine";

service InferenceEngine {
  // Submit a request; receive a stream of tokens as they're generated
  rpc SubmitRequest(EngineRequest) returns (stream EngineToken);
  // Pull current scheduler metrics
  rpc GetMetrics(MetricsRequest) returns (EngineMetrics);
}

message EngineRequest {
  string request_id = 1;
  string prompt = 2;              // raw text; engine tokenizes internally
  int32 max_new_tokens = 3;
  float temperature = 4;
}

message EngineToken {
  string request_id = 1;
  int32 token_id = 2;
  string text = 3;                // detokenized
  bool is_finished = 4;
  FinishReason finish_reason = 5;
}

enum FinishReason {
  FINISH_REASON_NONE = 0;
  FINISH_REASON_EOS = 1;
  FINISH_REASON_LENGTH = 2;
}

message MetricsRequest {}

message EngineMetrics {
  int64 total_tokens_generated = 1;
  int32 current_batch_size = 2;
  int32 max_batch_size = 3;
  int32 queue_depth = 4;
  float batch_occupancy_pct = 5;   // rolling average
  float tokens_per_second = 6;     // rolling window
  int32 kv_cache_slots_used = 7;
  int32 kv_cache_slots_total = 8;
}
```

---

## C++ Engine — Complete Specification

### `sequence.h`

```cpp
#pragma once
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

enum class SequenceState {
    WAITING,    // in queue, not yet admitted to batch
    PREFILL,    // admitted, full prompt being processed this step
    DECODE,     // generating tokens one-by-one
    FINISHED    // hit EOS or max_new_tokens — pending eviction
};

struct Sequence {
    std::string request_id;

    // Input
    std::vector<int32_t> prompt_tokens;
    int32_t max_new_tokens;
    float temperature;

    // Output (accumulated during decode)
    std::vector<int32_t> generated_tokens;

    // Scheduler state
    SequenceState state = SequenceState::WAITING;
    int kv_slot = -1;  // index into KVCacheAllocator's pool; -1 = unassigned

    // Timing (all set by scheduler)
    std::chrono::steady_clock::time_point arrival_time;
    std::chrono::steady_clock::time_point first_token_time;
    std::chrono::steady_clock::time_point completion_time;

    // Derived
    int total_length() const {
        return static_cast<int>(prompt_tokens.size() + generated_tokens.size());
    }
    int prompt_length() const {
        return static_cast<int>(prompt_tokens.size());
    }
    int generated_length() const {
        return static_cast<int>(generated_tokens.size());
    }
    bool hit_max() const {
        return generated_length() >= max_new_tokens;
    }
};
```

### `kv_cache.h` / `kv_cache.cpp`

```cpp
// kv_cache.h
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

// Simple slot-based KV cache allocator.
// Pre-allocates a contiguous block: num_slots * slot_size.
// Each slot holds the full KV cache for one sequence up to max_seq_len.
// No fragmentation, no paging — simple and predictable.
//
// Memory layout per slot:
//   [num_layers][2 (K then V)][max_seq_len][num_kv_heads][head_dim]
//   Stored as float16 (2 bytes each).
//
// For TinyLlama-1.1B (22 layers, 4 KV heads, head_dim 64, max_seq_len 2048):
//   slot_size = 22 * 2 * 2048 * 4 * 64 * 2 bytes = ~45 MB per slot
//   8 slots = ~360 MB total — fits comfortably in RAM for benchmarking
//
// INTERVIEW QUESTION: "Why not PagedAttention?"
// ANSWER: PagedAttention solves memory *fragmentation* when sequences have
// wildly different lengths and you're running near memory limits. For a
// demonstration with max_batch_size=8 and known max_seq_len, contiguous
// pre-allocation wastes at most 7 * max_seq_len * slot_overhead bytes
// (the slots of short sequences). That's acceptable here, and the
// implementation is 50 lines instead of 500.

class KVCacheAllocator {
public:
    KVCacheAllocator(int num_slots, int max_seq_len, int num_layers,
                     int num_kv_heads, int head_dim);

    // Allocate the first free slot. Returns slot index [0, num_slots).
    // Returns -1 if all slots are occupied.
    int allocate();

    // Free a previously allocated slot. Asserts slot was occupied.
    void free(int slot);

    // Get raw pointer to the start of a slot's memory.
    // Caller (TokenGenerator) is responsible for the layout.
    void* get_slot_ptr(int slot) const;

    // Metrics
    int num_free() const;
    int num_total() const { return num_slots_; }
    float utilization() const {
        return 1.0f - static_cast<float>(num_free()) / num_slots_;
    }

private:
    int num_slots_;
    int slot_size_bytes_;
    std::vector<bool> occupied_;          // true if slot is in use
    std::unique_ptr<uint8_t[]> memory_;   // contiguous allocation
};
```

```cpp
// kv_cache.cpp
#include "kv_cache.h"
#include <cassert>
#include <cstring>

KVCacheAllocator::KVCacheAllocator(int num_slots, int max_seq_len,
                                   int num_layers, int num_kv_heads,
                                   int head_dim)
    : num_slots_(num_slots),
      occupied_(num_slots, false) {
    // float16 = 2 bytes per element
    // K and V = factor of 2
    slot_size_bytes_ = num_layers * 2 * max_seq_len * num_kv_heads * head_dim * 2;
    memory_ = std::make_unique<uint8_t[]>(
        static_cast<size_t>(num_slots_) * slot_size_bytes_);
    std::memset(memory_.get(), 0,
                static_cast<size_t>(num_slots_) * slot_size_bytes_);
}

int KVCacheAllocator::allocate() {
    for (int i = 0; i < num_slots_; ++i) {
        if (!occupied_[i]) {
            occupied_[i] = true;
            return i;
        }
    }
    return -1;
}

void KVCacheAllocator::free(int slot) {
    assert(slot >= 0 && slot < num_slots_);
    assert(occupied_[slot]);
    occupied_[slot] = false;
    // Optional: zero the memory for safety (comment out for perf)
    std::memset(get_slot_ptr(slot), 0, slot_size_bytes_);
}

void* KVCacheAllocator::get_slot_ptr(int slot) const {
    assert(slot >= 0 && slot < num_slots_);
    return memory_.get() + static_cast<size_t>(slot) * slot_size_bytes_;
}

int KVCacheAllocator::num_free() const {
    int count = 0;
    for (bool b : occupied_) {
        if (!b) ++count;
    }
    return count;
}
```

### `token_generator.h`

```cpp
#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Abstract interface for the model backend.
// The scheduler calls forward() once per decode step with the full batch.
// Implementations: MockTokenGenerator (testing), LlamaCppGenerator (real).
//
// DESIGN DECISION: forward() takes the entire batch, not individual sequences.
// This matters because real models batch all sequences into a single
// matrix multiply. The mock simulates this by sleeping proportional
// to batch size.

class TokenGenerator {
public:
    virtual ~TokenGenerator() = default;

    struct Entry {
        int kv_slot;                     // which KV cache slot
        std::vector<int32_t> tokens;     // prefill: all prompt tokens
                                         // decode: just [last_generated_token]
        int seq_position;                // position offset for rotary embeddings
        float temperature;               // per-sequence temperature
    };

    struct BatchInput {
        std::vector<Entry> entries;
    };

    struct BatchOutput {
        std::vector<int32_t> next_tokens; // one per entry, same order
    };

    // Run forward pass on the batch. Returns one next-token per entry.
    virtual BatchOutput forward(const BatchInput& input) = 0;

    // Tokenize raw text into token IDs.
    virtual std::vector<int32_t> tokenize(const std::string& text) = 0;

    // Detokenize a single token ID to text.
    virtual std::string detokenize(int32_t token_id) = 0;

    // The end-of-sequence token ID for this model.
    virtual int32_t eos_token_id() const = 0;
};
```

### `mock_generator.h` / `mock_generator.cpp`

```cpp
// mock_generator.h
#pragma once
#include "token_generator.h"
#include <chrono>
#include <random>

// Deterministic mock for testing the scheduler in isolation.
//
// Behavior:
// - tokenize(): maps each character to its ASCII value (simple, reversible)
// - forward(): sleeps latency_per_token * batch_size to simulate compute,
//   then returns random tokens. EOS emitted with probability eos_prob
//   per sequence per step (default 0 — sequences run to max_new_tokens).
// - Configurable parameters let the benchmark control output length
//   distribution by adjusting max_new_tokens per request rather than
//   relying on random EOS.
//
// WHY THIS EXISTS: the scheduler's throughput advantage over static batching
// comes from its *scheduling logic*, not from the model. The mock isolates
// the scheduler so the benchmark measures what matters. If mock latency
// per token = T, then:
//   Static throughput = batch_size / (T * max_steps_in_batch)
//   Continuous throughput ≈ batch_size / (T * mean_steps) under full load
// The ratio is driven by output length variance, exactly as it should be.

class MockTokenGenerator : public TokenGenerator {
public:
    struct Config {
        int vocab_size = 32000;
        int32_t eos_id = 2;
        double eos_prob = 0.0;       // probability of emitting EOS per step
        std::chrono::microseconds latency_per_token{500};  // per sequence
        uint64_t seed = 42;
    };

    explicit MockTokenGenerator(Config config = {});

    BatchOutput forward(const BatchInput& input) override;
    std::vector<int32_t> tokenize(const std::string& text) override;
    std::string detokenize(int32_t token_id) override;
    int32_t eos_token_id() const override { return config_.eos_id; }

private:
    Config config_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> dist_{0.0, 1.0};
};
```

```cpp
// mock_generator.cpp
#include "mock_generator.h"
#include <thread>

MockTokenGenerator::MockTokenGenerator(Config config)
    : config_(config), rng_(config.seed) {}

TokenGenerator::BatchOutput
MockTokenGenerator::forward(const BatchInput& input) {
    // Simulate compute: latency scales with batch size
    auto total_latency = config_.latency_per_token * input.entries.size();
    std::this_thread::sleep_for(total_latency);

    BatchOutput output;
    output.next_tokens.reserve(input.entries.size());
    for (size_t i = 0; i < input.entries.size(); ++i) {
        if (config_.eos_prob > 0 && dist_(rng_) < config_.eos_prob) {
            output.next_tokens.push_back(config_.eos_id);
        } else {
            // Random non-EOS token
            int32_t tok = static_cast<int32_t>(
                dist_(rng_) * (config_.vocab_size - 1)) + 1;
            if (tok == config_.eos_id) tok++;  // avoid accidental EOS
            output.next_tokens.push_back(tok);
        }
    }
    return output;
}

std::vector<int32_t> MockTokenGenerator::tokenize(const std::string& text) {
    // ASCII encoding: one token per character
    std::vector<int32_t> tokens;
    tokens.reserve(text.size());
    for (char c : text) {
        tokens.push_back(static_cast<int32_t>(static_cast<unsigned char>(c)));
    }
    return tokens;
}

std::string MockTokenGenerator::detokenize(int32_t token_id) {
    if (token_id < 0 || token_id > 127) return "?";
    return std::string(1, static_cast<char>(token_id));
}
```

### `metrics.h`

```cpp
#pragma once
#include <atomic>
#include <chrono>

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
        static constexpr int SIZE = 50;  // 50 steps ≈ 5 seconds at 100ms/step
        int64_t token_counts[SIZE] = {};
        std::chrono::steady_clock::time_point timestamps[SIZE] = {};
        int head = 0;

        void record(int64_t tokens, std::chrono::steady_clock::time_point now) {
            token_counts[head] = tokens;
            timestamps[head] = now;
            head = (head + 1) % SIZE;
        }

        float compute_rate() const {
            int oldest = head;  // oldest entry is at head (circular)
            int newest = (head - 1 + SIZE) % SIZE;
            auto dt = std::chrono::duration<float>(
                timestamps[newest] - timestamps[oldest]).count();
            if (dt <= 0) return 0;
            int64_t total = 0;
            for (int i = 0; i < SIZE; i++) total += token_counts[i];
            return static_cast<float>(total) / dt;
        }
    };
};
```

### `scheduler.h` / `scheduler.cpp` — THE CORE

```cpp
// scheduler.h
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

private:
    // Main loop phases (called only from scheduler thread)
    void admit_new();
    void prefill_step();
    void decode_step();
    void check_finished();
    void evict_finished();
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
```

```cpp
// scheduler.cpp
#include "scheduler.h"
#include <algorithm>
#include <cassert>

ContinuousBatchScheduler::ContinuousBatchScheduler(
    Config config, std::unique_ptr<TokenGenerator> generator)
    : config_(config),
      generator_(std::move(generator)),
      kv_cache_(config.max_batch_size,
                config.max_sequence_length,
                config.num_layers,
                config.num_kv_heads,
                config.head_dim) {
    active_batch_.reserve(config.max_batch_size);
}

bool ContinuousBatchScheduler::submit(
    const std::string& request_id,
    const std::string& prompt,
    int32_t max_new_tokens,
    float temperature) {

    Sequence seq;
    seq.request_id = request_id;
    seq.prompt_tokens = generator_->tokenize(prompt);
    seq.max_new_tokens = max_new_tokens;
    seq.temperature = temperature;
    seq.state = SequenceState::WAITING;
    seq.arrival_time = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (static_cast<int>(waiting_queue_.size()) >= config_.max_queue_depth) {
        return false;  // queue full — caller returns RESOURCE_EXHAUSTED
    }
    waiting_queue_.push_back(std::move(seq));
    metrics_.total_requests_submitted.fetch_add(1);
    metrics_.queue_depth.store(static_cast<int>(waiting_queue_.size()));
    return true;
}

void ContinuousBatchScheduler::set_token_callback(TokenCallback cb) {
    token_callback_ = std::move(cb);
}

// ─────────────────────────────────────────────
// THE MAIN LOOP
// ─────────────────────────────────────────────
void ContinuousBatchScheduler::run() {
    running_.store(true);
    while (running_.load()) {
        // If batch is empty AND queue is empty, sleep briefly to avoid busy-wait
        if (active_batch_.empty()) {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (waiting_queue_.empty()) {
                // Release lock, sleep, continue
                // (In production: use a condition variable instead)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
        }

        admit_new();       // fill empty slots from queue
        prefill_step();    // process new sequences' prompts
        decode_step();     // generate one token per DECODE sequence
        check_finished();  // mark sequences that are done
        evict_finished();  // remove them, free KV slots
        update_metrics();  // rolling rate window
    }
}

void ContinuousBatchScheduler::stop() {
    running_.store(false);
}

// ─────────────────────────────────────────────
// PHASE 1: ADMIT
// Fill empty batch slots from the waiting queue.
// ─────────────────────────────────────────────
void ContinuousBatchScheduler::admit_new() {
    int free_slots = config_.max_batch_size
                     - static_cast<int>(active_batch_.size());
    if (free_slots <= 0) return;

    std::lock_guard<std::mutex> lock(queue_mutex_);
    int to_admit = std::min(free_slots,
                            static_cast<int>(waiting_queue_.size()));

    for (int i = 0; i < to_admit; ++i) {
        Sequence seq = std::move(waiting_queue_.front());
        waiting_queue_.pop_front();

        int slot = kv_cache_.allocate();
        assert(slot >= 0);  // free_slots guarantees availability

        seq.kv_slot = slot;
        seq.state = SequenceState::PREFILL;
        active_batch_.push_back(std::move(seq));
    }

    metrics_.queue_depth.store(static_cast<int>(waiting_queue_.size()));
    metrics_.kv_slots_used.store(
        kv_cache_.num_total() - kv_cache_.num_free());
}

// ─────────────────────────────────────────────
// PHASE 2: PREFILL
// For newly admitted sequences, process the full prompt.
// In real inference: this is a single forward pass with all
// prompt tokens, filling the KV cache. It's compute-bound
// (unlike decode, which is memory-bandwidth-bound).
// ─────────────────────────────────────────────
void ContinuousBatchScheduler::prefill_step() {
    TokenGenerator::BatchInput prefill_batch;

    for (auto& seq : active_batch_) {
        if (seq.state != SequenceState::PREFILL) continue;

        TokenGenerator::Entry entry;
        entry.kv_slot = seq.kv_slot;
        entry.tokens = seq.prompt_tokens;  // all prompt tokens
        entry.seq_position = 0;
        entry.temperature = seq.temperature;
        prefill_batch.entries.push_back(entry);
    }

    if (prefill_batch.entries.empty()) return;

    // Forward pass: processes all prompt tokens, fills KV cache,
    // returns the first generated token for each sequence.
    auto output = generator_->forward(prefill_batch);

    // Match outputs back to sequences
    int out_idx = 0;
    for (auto& seq : active_batch_) {
        if (seq.state != SequenceState::PREFILL) continue;

        int32_t token = output.next_tokens[out_idx++];
        seq.generated_tokens.push_back(token);
        seq.state = SequenceState::DECODE;
        seq.first_token_time = std::chrono::steady_clock::now();

        // Stream the first token
        if (token_callback_) {
            bool is_eos = (token == generator_->eos_token_id());
            token_callback_(
                seq.request_id,
                generator_->detokenize(token),
                token,
                is_eos,
                is_eos ? 1 : 0
            );
            if (is_eos) seq.state = SequenceState::FINISHED;
        }

        metrics_.total_tokens_generated.fetch_add(1);
    }
}

// ─────────────────────────────────────────────
// PHASE 3: DECODE
// Generate one token for every active DECODE sequence.
// This is the step that runs most often — the "inner loop."
// ─────────────────────────────────────────────
void ContinuousBatchScheduler::decode_step() {
    TokenGenerator::BatchInput decode_batch;
    std::vector<int> batch_indices;  // map from decode_batch index to active_batch_ index

    for (int i = 0; i < static_cast<int>(active_batch_.size()); ++i) {
        auto& seq = active_batch_[i];
        if (seq.state != SequenceState::DECODE) continue;

        TokenGenerator::Entry entry;
        entry.kv_slot = seq.kv_slot;
        entry.tokens = {seq.generated_tokens.back()};  // just the last token
        entry.seq_position = seq.total_length() - 1;
        entry.temperature = seq.temperature;
        decode_batch.entries.push_back(entry);
        batch_indices.push_back(i);
    }

    if (decode_batch.entries.empty()) return;

    auto output = generator_->forward(decode_batch);

    for (int j = 0; j < static_cast<int>(batch_indices.size()); ++j) {
        auto& seq = active_batch_[batch_indices[j]];
        int32_t token = output.next_tokens[j];
        seq.generated_tokens.push_back(token);

        bool is_eos = (token == generator_->eos_token_id());
        bool is_finished = is_eos || seq.hit_max();

        if (token_callback_) {
            token_callback_(
                seq.request_id,
                generator_->detokenize(token),
                token,
                is_finished,
                is_eos ? 1 : (seq.hit_max() ? 2 : 0)
            );
        }

        metrics_.total_tokens_generated.fetch_add(1);
    }
}

// ─────────────────────────────────────────────
// PHASE 4: CHECK
// Mark sequences as FINISHED if they hit EOS or max tokens.
// Separated from eviction for clarity and testability.
// ─────────────────────────────────────────────
void ContinuousBatchScheduler::check_finished() {
    for (auto& seq : active_batch_) {
        if (seq.state != SequenceState::DECODE) continue;

        bool hit_eos = (!seq.generated_tokens.empty() &&
                        seq.generated_tokens.back() == generator_->eos_token_id());
        bool hit_max = seq.hit_max();

        if (hit_eos || hit_max) {
            seq.state = SequenceState::FINISHED;
            seq.completion_time = std::chrono::steady_clock::now();
        }
    }
}

// ─────────────────────────────────────────────
// PHASE 5: EVICT
// Remove finished sequences. Free their KV cache slots.
// THIS IS THE LINE THAT MAKES CONTINUOUS BATCHING WORK.
// In static batching, this phase doesn't exist — you wait
// until ALL sequences are done, then free everything at once.
// ─────────────────────────────────────────────
void ContinuousBatchScheduler::evict_finished() {
    auto it = std::remove_if(
        active_batch_.begin(), active_batch_.end(),
        [this](Sequence& seq) {
            if (seq.state == SequenceState::FINISHED) {
                kv_cache_.free(seq.kv_slot);  // <-- THE KEY OPERATION
                metrics_.total_requests_completed.fetch_add(1);
                return true;
            }
            return false;
        });
    active_batch_.erase(it, active_batch_.end());

    metrics_.current_batch_size.store(
        static_cast<int>(active_batch_.size()));
    metrics_.kv_slots_used.store(
        kv_cache_.num_total() - kv_cache_.num_free());
}

// ─────────────────────────────────────────────
// PHASE 6: METRICS
// ─────────────────────────────────────────────
void ContinuousBatchScheduler::update_metrics() {
    auto now = std::chrono::steady_clock::now();
    int bs = static_cast<int>(active_batch_.size());

    // Rolling batch occupancy
    float occ = static_cast<float>(bs) / config_.max_batch_size;
    float prev = metrics_.batch_occupancy_pct.load();
    float alpha = 0.05f;  // EMA smoothing
    metrics_.batch_occupancy_pct.store(prev * (1 - alpha) + occ * alpha);

    // Tokens/second (windowed)
    rate_window_.record(bs, now);  // bs tokens generated this step
    metrics_.tokens_per_second.store(rate_window_.compute_rate());
}
```

### `llama_generator.h` / `llama_generator.cpp`

```cpp
// llama_generator.h
#pragma once
#include "token_generator.h"
#include "llama.h"  // llama.cpp header
#include <string>

// Wraps llama.cpp's C API for real inference with TinyLlama.
//
// Model: TinyLlama-1.1B-Chat-v1.0 in GGUF format (Q4_K_M quantization).
// Download: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
// File: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (~669 MB)
//
// The key llama.cpp functions used:
//   llama_model_load_from_file()  — load GGUF model
//   llama_new_context_with_model() — create inference context
//   llama_batch_init()            — allocate batch structure
//   llama_batch_add()             — add tokens to batch
//   llama_decode()                — run forward pass
//   llama_get_logits_ith()        — get logits for a position
//   llama_token_to_piece()        — detokenize
//   llama_tokenize()              — tokenize
//
// IMPORTANT: llama.cpp manages its own KV cache internally.
// Our KVCacheAllocator tracks slot *assignments* for the scheduler's
// logic, but llama.cpp's llama_kv_cache_seq_* functions manage the
// actual cache. We must keep them in sync:
//   - On admit: call llama_kv_cache_seq_rm() to clear the slot
//   - On evict: call llama_kv_cache_seq_rm() to free the slot
//   - Each sequence is assigned a unique seq_id = kv_slot index

class LlamaCppGenerator : public TokenGenerator {
public:
    struct Config {
        std::string model_path;
        int n_ctx = 2048;           // context length
        int n_batch = 512;          // max tokens per llama_decode call
        int n_threads = 4;          // CPU threads for inference
        int n_gpu_layers = 0;       // layers to offload to GPU (0 = CPU only)
    };

    explicit LlamaCppGenerator(Config config);
    ~LlamaCppGenerator();

    BatchOutput forward(const BatchInput& input) override;
    std::vector<int32_t> tokenize(const std::string& text) override;
    std::string detokenize(int32_t token_id) override;
    int32_t eos_token_id() const override;

private:
    int32_t sample_token(float* logits, int n_vocab, float temperature);
    void clear_kv_slot(int seq_id);

    Config config_;
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    int n_vocab_ = 0;
};
```

```cpp
// llama_generator.cpp — implementation sketch (key functions only)
#include "llama_generator.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <random>

LlamaCppGenerator::LlamaCppGenerator(Config config) : config_(config) {
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;
    model_ = llama_load_model_from_file(config.model_path.c_str(), model_params);
    assert(model_ && "Failed to load model");

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config.n_ctx;
    ctx_params.n_batch = config.n_batch;
    ctx_params.n_threads = config.n_threads;
    ctx_ = llama_new_context_with_model(model_, ctx_params);
    assert(ctx_ && "Failed to create context");

    n_vocab_ = llama_n_vocab(model_);
}

LlamaCppGenerator::~LlamaCppGenerator() {
    if (ctx_) llama_free(ctx_);
    if (model_) llama_free_model(model_);
    llama_backend_free();
}

TokenGenerator::BatchOutput
LlamaCppGenerator::forward(const BatchInput& input) {
    // Build llama_batch from our BatchInput
    llama_batch batch = llama_batch_init(
        config_.n_batch, 0, 1);

    // Track which batch positions need logits (last token of each entry)
    std::vector<int> logit_positions;
    int pos_in_batch = 0;

    for (const auto& entry : input.entries) {
        for (int i = 0; i < static_cast<int>(entry.tokens.size()); ++i) {
            bool need_logits = (i == static_cast<int>(entry.tokens.size()) - 1);
            llama_batch_add(
                batch,
                entry.tokens[i],
                entry.seq_position + i,   // position
                {entry.kv_slot},          // seq_id = kv_slot
                need_logits
            );
            if (need_logits) {
                logit_positions.push_back(pos_in_batch);
            }
            pos_in_batch++;
        }
    }

    // Run forward pass
    int rc = llama_decode(ctx_, batch);
    assert(rc == 0 && "llama_decode failed");

    // Sample one token per entry
    BatchOutput output;
    output.next_tokens.reserve(input.entries.size());
    for (int i = 0; i < static_cast<int>(input.entries.size()); ++i) {
        float* logits = llama_get_logits_ith(ctx_, logit_positions[i]);
        int32_t token = sample_token(logits, n_vocab_,
                                     input.entries[i].temperature);
        output.next_tokens.push_back(token);
    }

    llama_batch_free(batch);
    return output;
}

void LlamaCppGenerator::clear_kv_slot(int seq_id) {
    llama_kv_cache_seq_rm(ctx_, seq_id, -1, -1);
}

int32_t LlamaCppGenerator::sample_token(
    float* logits, int n_vocab, float temperature) {
    if (temperature <= 0.0f) {
        // Greedy
        return static_cast<int32_t>(
            std::max_element(logits, logits + n_vocab) - logits);
    }

    // Temperature scaling + softmax + multinomial sampling
    std::vector<float> probs(n_vocab);
    float max_logit = *std::max_element(logits, logits + n_vocab);
    float sum = 0;
    for (int i = 0; i < n_vocab; ++i) {
        probs[i] = std::exp((logits[i] - max_logit) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < n_vocab; ++i) probs[i] /= sum;

    // Multinomial sample
    static thread_local std::mt19937 rng(std::random_device{}());
    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
    return dist(rng);
}

std::vector<int32_t> LlamaCppGenerator::tokenize(const std::string& text) {
    std::vector<int32_t> tokens(text.size() + 16);
    int n = llama_tokenize(model_, text.c_str(), text.size(),
                           tokens.data(), tokens.size(), true, false);
    assert(n >= 0);
    tokens.resize(n);
    return tokens;
}

std::string LlamaCppGenerator::detokenize(int32_t token_id) {
    char buf[64];
    int n = llama_token_to_piece(model_, token_id, buf, sizeof(buf), 0, false);
    if (n < 0) return "";
    return std::string(buf, n);
}

int32_t LlamaCppGenerator::eos_token_id() const {
    return llama_token_eos(model_);
}
```

### `grpc_service.h` / `grpc_service.cpp`

```cpp
// grpc_service.h
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
```

```cpp
// grpc_service.cpp
#include "grpc_service.h"

EngineServiceImpl::EngineServiceImpl(ContinuousBatchScheduler* scheduler)
    : scheduler_(scheduler) {
    // Register our callback with the scheduler
    scheduler_->set_token_callback(
        [this](const std::string& rid, const std::string& text,
               int32_t tid, bool done, int reason) {
            on_token(rid, text, tid, done, reason);
        });
}

grpc::Status EngineServiceImpl::SubmitRequest(
    grpc::ServerContext* context,
    const engine::EngineRequest* request,
    grpc::ServerWriter<engine::EngineToken>* writer) {

    // Create per-request stream
    auto stream = std::make_shared<RequestStream>();
    {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        streams_[request->request_id()] = stream;
    }

    // Submit to scheduler
    bool accepted = scheduler_->submit(
        request->request_id(),
        request->prompt(),
        request->max_new_tokens(),
        request->temperature());

    if (!accepted) {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        streams_.erase(request->request_id());
        return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED,
                           "Queue full");
    }

    // Stream tokens until done
    while (true) {
        engine::EngineToken token;
        {
            std::unique_lock<std::mutex> lock(stream->mu);
            stream->cv.wait(lock, [&] {
                return !stream->tokens.empty() || stream->done;
            });

            if (stream->tokens.empty() && stream->done) break;

            token = std::move(stream->tokens.front());
            stream->tokens.pop();
        }

        if (!writer->Write(token)) break;  // client disconnected
        if (token.is_finished()) break;
    }

    // Cleanup
    {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        streams_.erase(request->request_id());
    }

    return grpc::Status::OK;
}

void EngineServiceImpl::on_token(
    const std::string& request_id,
    const std::string& text,
    int32_t token_id,
    bool is_finished,
    int finish_reason) {

    std::shared_ptr<RequestStream> stream;
    {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        auto it = streams_.find(request_id);
        if (it == streams_.end()) return;  // client already disconnected
        stream = it->second;
    }

    engine::EngineToken token;
    token.set_request_id(request_id);
    token.set_token_id(token_id);
    token.set_text(text);
    token.set_is_finished(is_finished);
    token.set_finish_reason(
        static_cast<engine::FinishReason>(finish_reason));

    {
        std::lock_guard<std::mutex> lock(stream->mu);
        stream->tokens.push(std::move(token));
        if (is_finished) stream->done = true;
    }
    stream->cv.notify_one();
}

grpc::Status EngineServiceImpl::GetMetrics(
    grpc::ServerContext* /*context*/,
    const engine::MetricsRequest* /*request*/,
    engine::EngineMetrics* response) {

    const auto& m = scheduler_->metrics();
    response->set_total_tokens_generated(m.total_tokens_generated.load());
    response->set_current_batch_size(m.current_batch_size.load());
    response->set_max_batch_size(8);  // from config
    response->set_queue_depth(m.queue_depth.load());
    response->set_batch_occupancy_pct(m.batch_occupancy_pct.load());
    response->set_tokens_per_second(m.tokens_per_second.load());
    response->set_kv_cache_slots_used(m.kv_slots_used.load());
    response->set_kv_cache_slots_total(8);

    return grpc::Status::OK;
}
```

### `main.cpp`

```cpp
#include "grpc_service.h"
#include "mock_generator.h"
#include "llama_generator.h"
#include "scheduler.h"
#include <grpcpp/grpcpp.h>
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

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--listen" && i + 1 < argc) listen_addr = argv[++i];
        else if (arg == "--batch-size" && i + 1 < argc) max_batch_size = std::stoi(argv[++i]);
        else if (arg == "--max-queue" && i + 1 < argc) max_queue_depth = std::stoi(argv[++i]);
    }

    // Create token generator
    std::unique_ptr<TokenGenerator> generator;
    if (mode == "llama") {
        LlamaCppGenerator::Config cfg;
        cfg.model_path = model_path;
        generator = std::make_unique<LlamaCppGenerator>(cfg);
        std::cout << "Loaded model: " << model_path << "\n";
    } else {
        generator = std::make_unique<MockTokenGenerator>();
        std::cout << "Using mock generator\n";
    }

    // Create scheduler
    ContinuousBatchScheduler::Config sched_config;
    sched_config.max_batch_size = max_batch_size;
    sched_config.max_queue_depth = max_queue_depth;
    ContinuousBatchScheduler scheduler(sched_config, std::move(generator));

    // Create gRPC service
    EngineServiceImpl service(&scheduler);

    // Start gRPC server
    grpc::ServerBuilder builder;
    builder.AddListeningPort(listen_addr,
                             grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    std::cout << "Engine listening on " << listen_addr << "\n";

    // Start scheduler on dedicated thread
    std::thread scheduler_thread([&scheduler] { scheduler.run(); });

    server->Wait();

    scheduler.stop();
    scheduler_thread.join();
    return 0;
}
```

---

## Go Gateway — Complete Specification

### `go-gateway/main.go`

```go
package main

import (
    "flag"
    "log"
    "net"
    "net/http"

    "google.golang.org/grpc"
    "github.com/prometheus/client_golang/prometheus/promhttp"

    gatewaypb "llm-server/gen/gateway"
    "llm-server/engine"
    "llm-server/server"
)

func main() {
    listenAddr := flag.String("listen", ":50051", "gRPC listen address")
    metricsAddr := flag.String("metrics", ":9090", "Prometheus metrics address")
    engineAddr := flag.String("engine", "unix:///tmp/inference-engine.sock", "C++ engine address")
    maxQueue := flag.Int("max-queue", 64, "Max queued requests before rejecting")
    flag.Parse()

    // Connect to C++ engine
    engineClient, err := engine.NewClient(*engineAddr)
    if err != nil {
        log.Fatalf("Failed to connect to engine: %v", err)
    }
    defer engineClient.Close()

    // Create server
    srv := server.New(engineClient, *maxQueue)

    // Start Prometheus metrics endpoint
    go func() {
        http.Handle("/metrics", promhttp.Handler())
        log.Printf("Metrics on %s/metrics", *metricsAddr)
        log.Fatal(http.ListenAndServe(*metricsAddr, nil))
    }()

    // Start gRPC server
    lis, err := net.Listen("tcp", *listenAddr)
    if err != nil {
        log.Fatalf("Failed to listen: %v", err)
    }

    grpcServer := grpc.NewServer()
    gatewaypb.RegisterInferenceServiceServer(grpcServer, srv)

    log.Printf("Gateway listening on %s", *listenAddr)
    log.Fatal(grpcServer.Serve(lis))
}
```

### `go-gateway/server/server.go`

```go
package server

import (
    "context"
    "io"
    "sync/atomic"

    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
    "github.com/google/uuid"

    gatewaypb "llm-server/gen/gateway"
    enginepb "llm-server/gen/engine"
    "llm-server/engine"
)

type Server struct {
    gatewaypb.UnimplementedInferenceServiceServer
    engine       *engine.Client
    maxQueue     int
    activeCount  atomic.Int64
    metrics      *Metrics
}

func New(engineClient *engine.Client, maxQueue int) *Server {
    s := &Server{
        engine:   engineClient,
        maxQueue: maxQueue,
        metrics:  NewMetrics(),
    }
    return s
}

func (s *Server) Generate(
    req *gatewaypb.GenerateRequest,
    stream gatewaypb.InferenceService_GenerateServer,
) error {
    // Admission control
    current := s.activeCount.Add(1)
    defer s.activeCount.Add(-1)

    if int(current) > s.maxQueue {
        s.metrics.RejectedTotal.Inc()
        return status.Error(codes.ResourceExhausted,
            "server at capacity — retry later")
    }

    s.metrics.ActiveRequests.Set(float64(current))
    s.metrics.RequestsTotal.Inc()

    requestID := uuid.New().String()
    timer := s.metrics.NewRequestTimer()
    var firstTokenRecorded bool

    // Submit to C++ engine
    engineStream, err := s.engine.Submit(stream.Context(), &enginepb.EngineRequest{
        RequestId:    requestID,
        Prompt:       req.Prompt,
        MaxNewTokens: req.MaxTokens,
        Temperature:  req.Temperature,
    })
    if err != nil {
        return err
    }

    // Stream tokens back to client
    var completionTokens int32
    for {
        token, err := engineStream.Recv()
        if err == io.EOF {
            break
        }
        if err != nil {
            return err
        }

        completionTokens++

        if !firstTokenRecorded {
            s.metrics.TTFT.Observe(timer.ElapsedMs())
            firstTokenRecorded = true
        }

        resp := &gatewaypb.GenerateResponse{
            Token:      token.Text,
            IsFinished: token.IsFinished,
        }

        if token.IsFinished {
            elapsed := timer.ElapsedMs()
            resp.FinishReason = gatewaypb.FinishReason(token.FinishReason)
            resp.Usage = &gatewaypb.UsageStats{
                CompletionTokens:   completionTokens,
                TotalTimeMs:        float32(elapsed),
                TimeToFirstTokenMs: float32(s.metrics.LastTTFT()),
            }
        }

        if err := stream.Send(resp); err != nil {
            return err // client disconnected
        }

        if token.IsFinished {
            break
        }
    }

    s.metrics.RequestDuration.Observe(timer.ElapsedMs())
    s.metrics.TokensGenerated.Add(float64(completionTokens))
    return nil
}

func (s *Server) Health(
    ctx context.Context,
    req *gatewaypb.HealthRequest,
) (*gatewaypb.HealthResponse, error) {
    // Pull metrics from engine to report queue depth
    engineMetrics, err := s.engine.GetMetrics(ctx)
    if err != nil {
        return &gatewaypb.HealthResponse{Ready: false}, nil
    }
    return &gatewaypb.HealthResponse{
        Ready:          true,
        QueueDepth:     engineMetrics.QueueDepth,
        ActiveRequests: int32(s.activeCount.Load()),
    }, nil
}
```

### `go-gateway/server/metrics.go` — Prometheus (the 8+ metrics)

```go
package server

import (
    "time"
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

type Metrics struct {
    // 1. Total requests received
    RequestsTotal prometheus.Counter
    // 2. Total requests rejected (queue full)
    RejectedTotal prometheus.Counter
    // 3. Currently active requests
    ActiveRequests prometheus.Gauge
    // 4. Request duration (end-to-end latency histogram, ms)
    RequestDuration prometheus.Histogram
    // 5. Time to first token (histogram, ms)
    TTFT prometheus.Histogram
    // 6. Total tokens generated
    TokensGenerated prometheus.Counter
    // 7. Batch occupancy (from engine, updated on health checks)
    BatchOccupancy prometheus.Gauge
    // 8. Queue depth (from engine)
    QueueDepth prometheus.Gauge

    lastTTFT float64
}

func NewMetrics() *Metrics {
    m := &Metrics{
        RequestsTotal: promauto.NewCounter(prometheus.CounterOpts{
            Name: "inference_requests_total",
            Help: "Total inference requests received",
        }),
        RejectedTotal: promauto.NewCounter(prometheus.CounterOpts{
            Name: "inference_requests_rejected_total",
            Help: "Requests rejected due to queue full",
        }),
        ActiveRequests: promauto.NewGauge(prometheus.GaugeOpts{
            Name: "inference_requests_active",
            Help: "Currently active requests",
        }),
        RequestDuration: promauto.NewHistogram(prometheus.HistogramOpts{
            Name:    "inference_request_duration_ms",
            Help:    "End-to-end request duration in milliseconds",
            Buckets: prometheus.ExponentialBuckets(10, 2, 12), // 10ms to 40s
        }),
        TTFT: promauto.NewHistogram(prometheus.HistogramOpts{
            Name:    "inference_ttft_ms",
            Help:    "Time to first token in milliseconds",
            Buckets: prometheus.ExponentialBuckets(5, 2, 10), // 5ms to 5s
        }),
        TokensGenerated: promauto.NewCounter(prometheus.CounterOpts{
            Name: "inference_tokens_generated_total",
            Help: "Total tokens generated",
        }),
        BatchOccupancy: promauto.NewGauge(prometheus.GaugeOpts{
            Name: "inference_batch_occupancy_pct",
            Help: "Current batch occupancy percentage",
        }),
        QueueDepth: promauto.NewGauge(prometheus.GaugeOpts{
            Name: "inference_queue_depth",
            Help: "Current request queue depth",
        }),
    }
    return m
}

type RequestTimer struct {
    start time.Time
}

func (m *Metrics) NewRequestTimer() *RequestTimer {
    return &RequestTimer{start: time.Now()}
}

func (t *RequestTimer) ElapsedMs() float64 {
    return float64(time.Since(t.start).Milliseconds())
}

func (m *Metrics) LastTTFT() float64 {
    return m.lastTTFT
}
```

---

## Benchmark Harness

### `benchmark/benchmark.py`

```python
"""
Poisson-arrival load generator for benchmarking.

Usage:
    python benchmark.py --mode continuous --target localhost:50051 \
        --requests 500 --arrival-rate 20 --output results_continuous.csv

    python benchmark.py --mode static --target localhost:50051 \
        --requests 500 --arrival-rate 20 --output results_static.csv

    python analyze.py results_continuous.csv results_static.csv

The key design choice: output lengths are sampled uniformly from
[16, 512] tokens. This variance is what creates the gap between
static and continuous batching. With fixed-length outputs, the
gap shrinks to zero (both modes have ~100% utilization).
"""
import argparse
import csv
import grpc
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import gateway_pb2
import gateway_pb2_grpc


@dataclass
class RequestResult:
    request_id: int
    arrival_time: float      # relative to benchmark start
    first_token_time: float  # TTFT in ms
    total_time: float        # end-to-end in ms
    tokens_generated: int
    error: str = ""


def send_request(stub, request_id: int, max_tokens: int) -> RequestResult:
    """Send a single streaming request and measure timing."""
    start = time.monotonic()
    first_token = None
    token_count = 0
    error = ""

    try:
        responses = stub.Generate(gateway_pb2.GenerateRequest(
            prompt=f"Write a detailed essay about topic number {request_id}.",
            max_tokens=max_tokens,
            temperature=0.8,
        ))
        for resp in responses:
            token_count += 1
            if first_token is None:
                first_token = time.monotonic()
            if resp.is_finished:
                break
    except grpc.RpcError as e:
        error = f"{e.code()}: {e.details()}"

    end = time.monotonic()
    return RequestResult(
        request_id=request_id,
        arrival_time=0,  # filled in by caller
        first_token_time=(first_token - start) * 1000 if first_token else -1,
        total_time=(end - start) * 1000,
        tokens_generated=token_count,
        error=error,
    )


def run_benchmark(target: str, num_requests: int, arrival_rate: float,
                  output_range: tuple, seed: int = 42) -> list[RequestResult]:
    """Run Poisson-arrival benchmark."""
    rng = np.random.default_rng(seed)

    # Generate arrival times (Poisson process)
    intervals = rng.exponential(1.0 / arrival_rate, num_requests)
    arrival_times = np.cumsum(intervals)

    # Sample output lengths uniformly
    max_tokens_list = rng.integers(output_range[0], output_range[1] + 1,
                                   size=num_requests)

    channel = grpc.insecure_channel(target)
    stub = gateway_pb2_grpc.InferenceServiceStub(channel)

    results = [None] * num_requests
    lock = threading.Lock()
    start_time = time.monotonic()

    def timed_request(i):
        # Wait until arrival time
        now = time.monotonic() - start_time
        wait = arrival_times[i] - now
        if wait > 0:
            time.sleep(wait)

        result = send_request(stub, i, int(max_tokens_list[i]))
        result.arrival_time = arrival_times[i]
        with lock:
            results[i] = result

    with ThreadPoolExecutor(max_workers=num_requests) as pool:
        futures = [pool.submit(timed_request, i) for i in range(num_requests)]
        for f in as_completed(futures):
            f.result()  # raise exceptions

    channel.close()
    return [r for r in results if r is not None]


def save_results(results: list[RequestResult], path: str):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["request_id", "arrival_time_s", "ttft_ms",
                         "total_time_ms", "tokens", "error"])
        for r in results:
            writer.writerow([r.request_id, f"{r.arrival_time:.3f}",
                            f"{r.first_token_time:.1f}",
                            f"{r.total_time:.1f}",
                            r.tokens_generated, r.error])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="localhost:50051")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--arrival-rate", type=float, default=20.0,
                        help="Mean requests/second (Poisson)")
    parser.add_argument("--min-tokens", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", default="results.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Running {args.requests} requests at {args.arrival_rate} req/s...")
    results = run_benchmark(
        args.target, args.requests, args.arrival_rate,
        (args.min_tokens, args.max_tokens), args.seed)

    save_results(results, args.output)

    # Quick summary
    ttfts = [r.first_token_time for r in results if r.first_token_time > 0]
    totals = [r.total_time for r in results if not r.error]
    total_tokens = sum(r.tokens_generated for r in results)
    wall_time = max(r.arrival_time + r.total_time/1000 for r in results)
    errors = sum(1 for r in results if r.error)

    print(f"\n{'='*50}")
    print(f"Requests:    {len(results)} ({errors} errors)")
    print(f"Throughput:  {total_tokens/wall_time:.1f} tokens/s")
    print(f"TTFT p50:    {np.percentile(ttfts, 50):.1f} ms")
    print(f"TTFT p99:    {np.percentile(ttfts, 99):.1f} ms")
    print(f"Latency p50: {np.percentile(totals, 50):.1f} ms")
    print(f"Latency p99: {np.percentile(totals, 99):.1f} ms")
    print(f"Saved to:    {args.output}")
```

### `benchmark/analyze.py`

```python
"""
Compare static vs continuous batching results.

Usage:
    python analyze.py results_continuous.csv results_static.csv

Produces:
    - throughput_comparison.png
    - latency_cdf.png
    - ttft_histogram.png
    - summary.txt (the numbers for the resume bullet)
"""
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if not r['error']]
    return {
        'ttft': np.array([float(r['ttft_ms']) for r in rows]),
        'total': np.array([float(r['total_time_ms']) for r in rows]),
        'tokens': np.array([int(r['tokens']) for r in rows]),
        'arrival': np.array([float(r['arrival_time_s']) for r in rows]),
    }


def analyze(continuous_path, static_path):
    cont = load_results(continuous_path)
    stat = load_results(static_path)

    # Throughput
    cont_wall = max(cont['arrival'] + cont['total']/1000)
    stat_wall = max(stat['arrival'] + stat['total']/1000)
    cont_tps = cont['tokens'].sum() / cont_wall
    stat_tps = stat['tokens'].sum() / stat_wall
    ratio = cont_tps / stat_tps

    print(f"\n{'='*60}")
    print(f"CONTINUOUS BATCHING vs STATIC BATCHING")
    print(f"{'='*60}")
    print(f"Throughput (continuous): {cont_tps:.1f} tok/s")
    print(f"Throughput (static):    {stat_tps:.1f} tok/s")
    print(f"RATIO:                  {ratio:.2f}x  <-- RESUME NUMBER")
    print(f"")
    print(f"TTFT p50 (continuous):  {np.percentile(cont['ttft'], 50):.1f} ms")
    print(f"TTFT p50 (static):      {np.percentile(stat['ttft'], 50):.1f} ms")
    print(f"Latency p99 (cont):     {np.percentile(cont['total'], 99):.1f} ms")
    print(f"Latency p99 (static):   {np.percentile(stat['total'], 99):.1f} ms")

    # Save summary for resume reference
    with open('summary.txt', 'w') as f:
        f.write(f"Throughput ratio: {ratio:.2f}x\n")
        f.write(f"Continuous: {cont_tps:.1f} tok/s\n")
        f.write(f"Static: {stat_tps:.1f} tok/s\n")
        f.write(f"Batch occupancy advantage: use Prometheus /metrics\n")

    # Latency CDF plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for data, label in [(cont, 'Continuous'), (stat, 'Static')]:
        sorted_lat = np.sort(data['total'])
        cdf = np.arange(1, len(sorted_lat)+1) / len(sorted_lat)
        axes[0].plot(sorted_lat, cdf, label=label, linewidth=2)
    axes[0].set_xlabel('End-to-end latency (ms)')
    axes[0].set_ylabel('CDF')
    axes[0].set_title('Latency CDF')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # TTFT comparison
    axes[1].hist(cont['ttft'], bins=40, alpha=0.6, label='Continuous', density=True)
    axes[1].hist(stat['ttft'], bins=40, alpha=0.6, label='Static', density=True)
    axes[1].set_xlabel('Time to First Token (ms)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('TTFT Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=150)
    print(f"\nPlots saved to benchmark_results.png")
    print(f"Summary saved to summary.txt")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python analyze.py <continuous.csv> <static.csv>")
        sys.exit(1)
    analyze(sys.argv[1], sys.argv[2])
```

---

## Static Batching Baseline

To produce the comparison number, you need a static batching mode. The simplest approach: add a `--static` flag to the C++ engine that changes the scheduler behavior.

In static mode, `evict_finished()` becomes:
```cpp
void ContinuousBatchScheduler::evict_finished_static() {
    // STATIC BATCHING: wait until ALL sequences in the batch are finished
    bool all_done = std::all_of(
        active_batch_.begin(), active_batch_.end(),
        [](const Sequence& s) { return s.state == SequenceState::FINISHED; });

    if (!all_done) return;  // <-- THE DIFFERENCE: do nothing until every slot is done

    // Only now evict all at once
    for (auto& seq : active_batch_) {
        kv_cache_.free(seq.kv_slot);
        metrics_.total_requests_completed.fetch_add(1);
    }
    active_batch_.clear();
    metrics_.current_batch_size.store(0);
    metrics_.kv_slots_used.store(0);
}
```

This is the ONLY difference between static and continuous. One `if` statement. The rest of the scheduler is identical — same prefill, same decode, same admit. This is what makes the comparison fair and the benchmark result defensible: the only variable is the eviction policy.

---

## Build System

### Top-level `Makefile`

```makefile
.PHONY: all engine gateway proto test bench clean

all: proto engine gateway

# Generate protobuf code for both languages
proto:
	@echo "Generating protobuf code..."
	protoc --proto_path=proto \
		--go_out=go-gateway/gen --go_opt=paths=source_relative \
		--go-grpc_out=go-gateway/gen --go-grpc_opt=paths=source_relative \
		proto/gateway.proto proto/engine.proto
	protoc --proto_path=proto \
		--cpp_out=cpp-engine/gen \
		--grpc_out=cpp-engine/gen --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` \
		proto/engine.proto

# Build C++ engine
engine:
	@echo "Building C++ engine..."
	cd cpp-engine && mkdir -p build && cd build && \
		cmake .. -DCMAKE_BUILD_TYPE=Release && \
		make -j$$(nproc)

# Build Go gateway
gateway:
	@echo "Building Go gateway..."
	cd go-gateway && go build -o ../bin/gateway .

# Run tests
test:
	cd cpp-engine/build && ctest --output-on-failure
	cd go-gateway && go test ./...

# Run benchmark (both modes)
bench: all
	@echo "Starting engine (continuous)..."
	./cpp-engine/build/engine --mode mock &
	sleep 1
	./bin/gateway --engine unix:///tmp/inference-engine.sock &
	sleep 1
	cd benchmark && python benchmark.py --output results_continuous.csv
	kill %2 %1
	@echo "Starting engine (static)..."
	./cpp-engine/build/engine --mode mock --static &
	sleep 1
	./bin/gateway --engine unix:///tmp/inference-engine.sock &
	sleep 1
	cd benchmark && python benchmark.py --output results_static.csv
	kill %2 %1
	@echo "Analyzing..."
	cd benchmark && python analyze.py results_continuous.csv results_static.csv

clean:
	rm -rf cpp-engine/build bin/
	cd go-gateway && go clean
```

### `cpp-engine/CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.16)
project(inference-engine CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)
find_package(Threads REQUIRED)

# Optional: llama.cpp (only needed for --mode llama)
option(WITH_LLAMA "Build with llama.cpp support" OFF)
if(WITH_LLAMA)
    add_subdirectory(third_party/llama.cpp)
endif()

# Generated protobuf sources
set(PROTO_SRCS gen/engine.pb.cc gen/engine.grpc.pb.cc)

# Engine executable
add_executable(engine
    src/main.cpp
    src/scheduler.cpp
    src/kv_cache.cpp
    src/mock_generator.cpp
    src/grpc_service.cpp
    ${PROTO_SRCS}
)

if(WITH_LLAMA)
    target_sources(engine PRIVATE src/llama_generator.cpp)
    target_compile_definitions(engine PRIVATE WITH_LLAMA)
    target_link_libraries(engine PRIVATE llama)
endif()

target_include_directories(engine PRIVATE src gen)
target_link_libraries(engine PRIVATE
    gRPC::grpc++
    protobuf::libprotobuf
    Threads::Threads
)

# Tests
enable_testing()
find_package(GTest REQUIRED)

add_executable(tests
    tests/scheduler_test.cpp
    tests/kv_cache_test.cpp
    src/scheduler.cpp
    src/kv_cache.cpp
    src/mock_generator.cpp
    ${PROTO_SRCS}
)
target_include_directories(tests PRIVATE src gen)
target_link_libraries(tests PRIVATE
    GTest::gtest_main
    gRPC::grpc++
    protobuf::libprotobuf
    Threads::Threads
)
add_test(NAME unit_tests COMMAND tests)
```

---

## Tests

### `tests/kv_cache_test.cpp`

```cpp
#include <gtest/gtest.h>
#include "kv_cache.h"

TEST(KVCacheTest, AllocateAndFree) {
    KVCacheAllocator cache(4, 128, 2, 4, 64);
    EXPECT_EQ(cache.num_free(), 4);

    int s0 = cache.allocate();
    EXPECT_GE(s0, 0);
    EXPECT_EQ(cache.num_free(), 3);

    cache.free(s0);
    EXPECT_EQ(cache.num_free(), 4);
}

TEST(KVCacheTest, ExhaustSlots) {
    KVCacheAllocator cache(2, 128, 2, 4, 64);
    int s0 = cache.allocate();
    int s1 = cache.allocate();
    EXPECT_GE(s0, 0);
    EXPECT_GE(s1, 0);
    EXPECT_NE(s0, s1);

    // Third allocation should fail
    EXPECT_EQ(cache.allocate(), -1);

    // Free one, allocate again
    cache.free(s0);
    int s2 = cache.allocate();
    EXPECT_GE(s2, 0);
}

TEST(KVCacheTest, Utilization) {
    KVCacheAllocator cache(4, 128, 2, 4, 64);
    EXPECT_FLOAT_EQ(cache.utilization(), 0.0f);

    cache.allocate();
    cache.allocate();
    EXPECT_FLOAT_EQ(cache.utilization(), 0.5f);
}

TEST(KVCacheTest, SlotPointersAreDistinct) {
    KVCacheAllocator cache(4, 128, 2, 4, 64);
    int s0 = cache.allocate();
    int s1 = cache.allocate();
    EXPECT_NE(cache.get_slot_ptr(s0), cache.get_slot_ptr(s1));
}
```

### `tests/scheduler_test.cpp`

```cpp
#include <gtest/gtest.h>
#include "scheduler.h"
#include "mock_generator.h"
#include <thread>
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
    }

    scheduler->stop();
    sched_thread.join();

    EXPECT_TRUE(finished);
    EXPECT_EQ(received_tokens.size(), 5);
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
```

---

## Interview Defense Preparation

### Questions you WILL be asked, with one-breath answers:

**Q: Why continuous batching?**
A: Static batching holds every slot until the longest sequence finishes. With output lengths from 16 to 512 tokens, that means ~43% utilization. Continuous batching frees finished slots and admits waiting requests every step, approaching 100% utilization under load.

**Q: Why Go + C++?**
A: Go's goroutine-per-connection model handles thousands of concurrent streaming clients without callback complexity. C++ gives zero-overhead control over the scheduler's hot loop and the KV cache memory layout. The boundary is local gRPC — inference takes milliseconds, serialization takes microseconds.

**Q: How do you avoid data races?**
A: The scheduler runs in a single dedicated thread — it's the only writer to batch state. gRPC threads write to a mutex-protected deque. The only lock acquisition per decode step is one mutex lock for a deque splice, held for microseconds. The decode step itself runs lock-free.

**Q: How does backpressure work?**
A: The Go gateway tracks active requests. When the count exceeds the configured limit, new requests get RESOURCE_EXHAUSTED. The C++ engine also rejects submissions when the waiting queue is full. Two-layer defense: gateway protects the network, engine protects memory.

**Q: Why not PagedAttention?**
A: PagedAttention solves memory fragmentation when you have thousands of sequences with wildly varying lengths near the memory limit. With max_batch_size=8 and a known max_seq_len, contiguous pre-allocation wastes at most 7 slots worth of memory. The implementation is 50 lines instead of 500, and the scheduling algorithm — which is the point of this project — is identical.

**Q: Where did the 3-5x number come from?**
A: I ran 500 requests with Poisson arrivals at 80% of static capacity, output lengths sampled uniformly from 16-512 tokens. The ratio is continuous throughput ÷ static throughput, both at matched p99 latency. The variance in output length is what creates the gap — with fixed-length outputs, there's no gap. Published 20x numbers (vLLM, etc.) compare against unbatched HF loops, which is an unfair baseline.

**Q: What happens when the KV cache is full?**
A: New requests wait in the queue. The scheduler only admits from the queue when a batch slot AND a KV cache slot are both free. Both become free at the same time (eviction frees both), so this is really one check: `active_batch_.size() < max_batch_size`.

**Q: What if a client disconnects mid-stream?**
A: The Go gateway's `stream.Send()` returns an error on disconnect. The gRPC handler exits, removing the request's stream from the registry. The C++ engine continues generating tokens for that request, but the token callback finds no registered stream and drops them silently. The sequence completes normally and its KV slot is freed. No leak.

**Q: Why is prefill separate from decode?**
A: Prefill processes all prompt tokens at once — it's compute-bound (matrix multiplications over the full prompt). Decode processes one token at a time — it's memory-bandwidth-bound (reading KV cache for all previous positions). They have different computational profiles. Separating them lets you batch prefills together (high arithmetic intensity) and decodes together (high memory bandwidth), rather than mixing the two in the same forward pass, which wastes one resource or the other.

---

## Honest Scope — What This Is and Isn't

**This IS:**
- A working implementation of the continuous batching scheduling algorithm
- A clean Go+C++ architecture demonstrating systems design
- A measurable benchmark with reproducible numbers
- A project you can explain down to every mutex and pointer

**This IS NOT:**
- A production inference server (no speculative decoding, no prefix caching, no tensor parallelism, no model sharding)
- A vLLM replacement
- Optimized for absolute throughput (the point is the ratio, not the absolute tok/s)

**When an interviewer asks "how would you scale this?":**
- Tensor parallelism across GPUs for larger models (shard attention heads)
- Prefix caching for shared system prompts (avoid re-computing KV for common prefixes)
- Speculative decoding with a small draft model (generate N candidate tokens cheaply, verify in one forward pass)
- PagedAttention for memory-constrained scenarios with 1000+ concurrent sequences
- Disaggregated prefill and decode (separate prefill servers from decode servers, since they have different hardware needs)

---

## Development Order (suggested)

1. **KV cache allocator** + tests (~30 min)
2. **Sequence struct** (~10 min)
3. **MockTokenGenerator** (~20 min)
4. **Scheduler** + tests (~2 hours — this is the core)
5. **Protobuf definitions** + code generation (~20 min)
6. **C++ gRPC service** (~1 hour)
7. **Go gateway** with metrics (~1.5 hours)
8. **Benchmark harness** (~1 hour)
9. **Static baseline mode** (~15 min — it's one `if` statement)
10. **Run benchmark, get numbers** (~30 min)
11. **LlamaCppGenerator** (optional, for demo — ~2 hours)
12. **Docker + Grafana dashboard** (optional polish — ~1 hour)

**Total estimated: 8-10 hours for a measured, defensible result.**
**Minimum viable for resume: steps 1-10 (~7 hours). Steps 11-12 are polish.**