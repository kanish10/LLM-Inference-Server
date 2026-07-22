.PHONY: all engine gateway proto test test-cpp test-go bench clean

# Mock-generator tuning for `bench`: a real batched forward pass has a large
# per-step fixed cost (reading model weights) that's amortized across the
# batch, plus a small marginal cost per sequence. Without the fixed term,
# static batching's idle (finished-but-unevicted) slots don't cost anything
# extra, and the whole point of the benchmark disappears — see
# cpp-engine/src/mock_generator.h. ARRIVAL_RATE is calibrated to ~80% of the
# static baseline's measured capacity at these settings (see README).
MOCK_LATENCY_US ?= 100
MOCK_FIXED_OVERHEAD_US ?= 2000
BATCH_SIZE ?= 8
ARRIVAL_RATE ?= 2.4
BENCH_REQUESTS ?= 500

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

clean:
	rm -rf cpp-engine/build bin/
	cd go-gateway && go clean
