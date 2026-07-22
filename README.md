# LLM Inference Server — Continuous Batching

A from-scratch implementation of continuous (in-flight) batching, the scheduling
algorithm behind vLLM, TGI, and every production LLM serving system built after 2023.

A single-threaded C++ scheduler admits queued requests and frees finished KV-cache
slots at every decode step. A Go gRPC gateway fronts it with admission control,
backpressure, and Prometheus metrics.

See `CLAUDE.md` for the full design rationale and specification.

## Architecture

```
Client --gRPC(stream)--> Go Gateway --gRPC(unix socket)--> C++ Engine
                              |                                  |
                         Prometheus                   ContinuousBatchScheduler
                          /metrics                      (single thread, lock-free
                                                          hot path)
```

## Prerequisites

- CMake >= 3.16, a C++17 compiler
- `protobuf-compiler`, `libprotobuf-dev`, `libgrpc++-dev`, `protobuf-compiler-grpc`
- Go >= 1.21
- `libgtest-dev` (or CMake will fetch GoogleTest automatically)
- Python 3 with `grpcio`, `grpcio-tools`, `numpy`, `matplotlib`, `pandas`

On Ubuntu/Debian:

```bash
sudo apt-get install -y protobuf-compiler libprotobuf-dev protobuf-compiler-grpc \
    libgrpc++-dev libgrpc-dev golang-go libgtest-dev build-essential cmake
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
export PATH="$PATH:$(go env GOPATH)/bin"
pip install -r benchmark/requirements.txt
```

## Build

```bash
make proto      # generate Go + C++ protobuf/gRPC stubs
make engine      # build the C++ engine (cpp-engine/build/engine)
make gateway      # build the Go gateway (bin/gateway)
```

## Test

```bash
make test        # C++ unit tests (ctest) + Go tests
```

## Run

```bash
# Terminal 1: C++ engine (mock generator — no model download needed)
./cpp-engine/build/engine --mode mock --listen unix:///tmp/inference-engine.sock

# Terminal 2: Go gateway
./bin/gateway --engine unix:///tmp/inference-engine.sock --listen :50051 --metrics :9090

# Terminal 3: send a request
grpcurl -plaintext -d '{"prompt": "hello", "max_tokens": 16}' \
    localhost:50051 gateway.InferenceService/Generate
```

Prometheus metrics are served at `http://localhost:9090/metrics`.

## Benchmark: continuous vs. static batching

```bash
make bench
```

This runs the same scheduler in continuous mode (`evict_finished`) and static mode
(`--static`, which waits for the whole batch to finish before evicting anyone —
see "Static Batching Baseline" in CLAUDE.md), against identical Poisson-arrival
traffic with output lengths sampled uniformly from [16, 512] tokens. The result
(`benchmark_results.png`, `summary.txt`) is the throughput ratio that backs the
resume claim.

**Mock latency model matters here.** `MockTokenGenerator::forward()` sleeps
`fixed_overhead + latency_per_token * batch_size`. With `fixed_overhead = 0`
(the constructor default, kept for scheduler-test determinism), per-step cost
is exactly proportional to how many sequences are actively decoding — so an
underfull static batch (six slots occupied by finished-but-unevicted
sequences) costs exactly as much per active sequence as a full one, and the
aggregate throughput gap between static and continuous batching disappears.
Real batched inference isn't like that: reading model weights from memory is
a large per-step cost independent of batch size, amortized over however many
sequences are actually in the batch. `make bench` passes
`--mock-fixed-overhead-us 2000 --mock-latency-us 100` to reproduce that, and
`ARRIVAL_RATE=2.4` (~80% of the static engine's measured capacity at
`--batch-size 8` with those settings — static's capacity collapses sharply
above ~3 req/s in this configuration, while continuous stays comfortable past
10 req/s). Override any of `MOCK_LATENCY_US`, `MOCK_FIXED_OVERHEAD_US`,
`BATCH_SIZE`, `ARRIVAL_RATE`, `BENCH_REQUESTS` as `make` variables to
re-tune for a different machine.

## Real model inference (optional)

Download a GGUF model into `models/` and build with `-DWITH_LLAMA=ON` to run the
engine against real TinyLlama weights instead of the mock generator (`--mode llama
--model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`). See CLAUDE.md for details —
this is polish on top of the core scheduler, not required to validate the
scheduling algorithm itself.

## What this is / isn't

This isolates the ~500-line core scheduling algorithm to understand and demonstrate
it deeply. It is not a production inference server (no speculative decoding, no
prefix caching, no tensor parallelism) and not a vLLM replacement.
