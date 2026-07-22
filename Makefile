.PHONY: all engine gateway proto test test-cpp test-go bench bench-mock clean

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
	./cpp-engine/build/engine --mode mock --listen unix:///tmp/inference-engine.sock &
	sleep 1
	./bin/gateway --engine unix:///tmp/inference-engine.sock &
	sleep 1
	cd benchmark && python3 benchmark.py --output results_continuous.csv
	-pkill -f 'cpp-engine/build/engine' ; pkill -f 'bin/gateway'
	@echo "Starting engine (static)..."
	./cpp-engine/build/engine --mode mock --static --listen unix:///tmp/inference-engine.sock &
	sleep 1
	./bin/gateway --engine unix:///tmp/inference-engine.sock &
	sleep 1
	cd benchmark && python3 benchmark.py --output results_static.csv
	-pkill -f 'cpp-engine/build/engine' ; pkill -f 'bin/gateway'
	@echo "Analyzing..."
	cd benchmark && python3 analyze.py results_continuous.csv results_static.csv

clean:
	rm -rf cpp-engine/build bin/
	cd go-gateway && go clean
