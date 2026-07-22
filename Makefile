.PHONY: all engine gateway proto test test-cpp test-go bench bench-capacity clean

# Mock-generator tuning shared by both bench targets: a real batched forward
# pass has a large per-step fixed cost (reading model weights) amortized
# across the batch, plus a small marginal cost per sequence. Without the
# fixed term, static batching's idle (finished-but-unevicted) slots don't
# cost anything extra, and the whole point of the benchmark disappears — see
# cpp-engine/src/mock_generator.h.
MOCK_LATENCY_US ?= 100
MOCK_FIXED_OVERHEAD_US ?= 2000
BATCH_SIZE ?= 8
BENCH_REQUESTS ?= 500

# `bench`: identical offered load for both modes, at ~80% of static's
# measured capacity. Below both systems' capacity, so throughput ties
# (~1.0x) — this experiment isolates the latency effect of head-of-line
# blocking (static's p99/TTFT is far worse at the same load).
ARRIVAL_RATE ?= 2.4

# `bench-capacity`: the "N× throughput" resume number. Static is driven at
# its own measured capacity ceiling (collapses above ~3 req/s at these mock
# settings; 2.8 req/s is the last stable point found by probing). Continuous
# is driven faster (6 req/s) until ITS p99 latency matches static's p99 at
# that ceiling (~1.7s) — i.e., "same tail latency budget, how much more
# throughput". Measured on this machine: continuous 1581 tok/s vs static
# 743 tok/s = 2.13x, with p99 latency 1688ms vs 1674ms. Re-probe with `make
# bench-capacity STATIC_ARRIVAL_RATE=... CONTINUOUS_ARRIVAL_RATE=...` if you
# change MOCK_*/BATCH_SIZE — the sustainable rates are specific to those.
STATIC_ARRIVAL_RATE ?= 2.8
CONTINUOUS_ARRIVAL_RATE ?= 6

all: proto engine gateway

# Generate protobuf code for both languages.
# (The C++ side also regenerates automatically at `cmake --build` time via
# the generate_proto custom command in cpp-engine/CMakeLists.txt — this
# target is mainly needed for the Go gateway, which has no such hook.)
proto:
	@echo "Generating protobuf code..."
	mkdir -p go-gateway/gen cpp-engine/gen
	protoc --proto_path=proto \
		--go_out=go-gateway --go_opt=module=llm-server \
		--go-grpc_out=go-gateway --go-grpc_opt=module=llm-server \
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
		$(MAKE) -j$$(nproc)

# Build Go gateway
gateway:
	@echo "Building Go gateway..."
	cd go-gateway && go build -o ../bin/gateway .

# Run all tests
test: test-cpp test-go

test-cpp:
	cd cpp-engine/build && ctest --output-on-failure

test-go:
	cd go-gateway && go test ./...

# Run benchmark (both modes) against the mock generator
bench: all
	@echo "Starting engine (continuous)..."
	./cpp-engine/build/engine --mode mock --batch-size $(BATCH_SIZE) --max-queue 256 \
		--mock-latency-us $(MOCK_LATENCY_US) --mock-fixed-overhead-us $(MOCK_FIXED_OVERHEAD_US) \
		--listen unix:///tmp/inference-engine.sock &
	sleep 1
	./bin/gateway --engine unix:///tmp/inference-engine.sock --max-queue 256 &
	sleep 1
	cd benchmark && python3 benchmark.py --requests $(BENCH_REQUESTS) --arrival-rate $(ARRIVAL_RATE) \
		--output results_continuous.csv
	-pkill -f 'cpp-engine/build/engine' ; pkill -f 'bin/gateway'
	@echo "Starting engine (static)..."
	./cpp-engine/build/engine --mode mock --static --batch-size $(BATCH_SIZE) --max-queue 256 \
		--mock-latency-us $(MOCK_LATENCY_US) --mock-fixed-overhead-us $(MOCK_FIXED_OVERHEAD_US) \
		--listen unix:///tmp/inference-engine.sock &
	sleep 1
	./bin/gateway --engine unix:///tmp/inference-engine.sock --max-queue 256 &
	sleep 1
	cd benchmark && python3 benchmark.py --requests $(BENCH_REQUESTS) --arrival-rate $(ARRIVAL_RATE) \
		--output results_static.csv
	-pkill -f 'cpp-engine/build/engine' ; pkill -f 'bin/gateway'
	@echo "Analyzing..."
	cd benchmark && python3 analyze.py results_continuous.csv results_static.csv

# Throughput comparison at matched p99 latency — see STATIC_ARRIVAL_RATE /
# CONTINUOUS_ARRIVAL_RATE above for what this actually measures.
bench-capacity: all
	@echo "Starting engine (continuous @ $(CONTINUOUS_ARRIVAL_RATE) req/s)..."
	./cpp-engine/build/engine --mode mock --batch-size $(BATCH_SIZE) --max-queue 256 \
		--mock-latency-us $(MOCK_LATENCY_US) --mock-fixed-overhead-us $(MOCK_FIXED_OVERHEAD_US) \
		--listen unix:///tmp/inference-engine.sock &
	sleep 1
	./bin/gateway --engine unix:///tmp/inference-engine.sock --max-queue 256 &
	sleep 1
	cd benchmark && python3 benchmark.py --requests $(BENCH_REQUESTS) --arrival-rate $(CONTINUOUS_ARRIVAL_RATE) \
		--output results_continuous.csv
	-pkill -f 'cpp-engine/build/engine' ; pkill -f 'bin/gateway'
	@echo "Starting engine (static @ $(STATIC_ARRIVAL_RATE) req/s)..."
	./cpp-engine/build/engine --mode mock --static --batch-size $(BATCH_SIZE) --max-queue 256 \
		--mock-latency-us $(MOCK_LATENCY_US) --mock-fixed-overhead-us $(MOCK_FIXED_OVERHEAD_US) \
		--listen unix:///tmp/inference-engine.sock &
	sleep 1
	./bin/gateway --engine unix:///tmp/inference-engine.sock --max-queue 256 &
	sleep 1
	cd benchmark && python3 benchmark.py --requests $(BENCH_REQUESTS) --arrival-rate $(STATIC_ARRIVAL_RATE) \
		--output results_static.csv
	-pkill -f 'cpp-engine/build/engine' ; pkill -f 'bin/gateway'
	@echo "Analyzing..."
	cd benchmark && python3 analyze.py results_continuous.csv results_static.csv

clean:
	rm -rf cpp-engine/build bin/
	cd go-gateway && go clean
