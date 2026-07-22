"""
Poisson-arrival load generator for benchmarking.

Usage:
    python benchmark.py --target localhost:50051 \
        --requests 500 --arrival-rate 20 --output results_continuous.csv

    python benchmark.py --target localhost:50051 \
        --requests 500 --arrival-rate 20 --output results_static.csv

    python analyze.py results_continuous.csv results_static.csv

"continuous" vs. "static" is not a benchmark.py flag — it's which mode the
C++ engine process behind --target was started with (`engine --mode mock`
vs. `engine --mode mock --static`, see CLAUDE.md "Static Batching
Baseline"). This script just generates identical traffic against whichever
engine is listening.

The key design choice: output lengths are sampled uniformly from
[16, 512] tokens. This variance is what creates the gap between
static and continuous batching. With fixed-length outputs, the
gap shrinks to zero (both modes have ~100% utilization).
"""
import argparse
import csv
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import grpc
import numpy as np
import yaml

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
                   output_range: tuple, seed: int = 42) -> list:
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

    with ThreadPoolExecutor(max_workers=min(num_requests, 256)) as pool:
        futures = [pool.submit(timed_request, i) for i in range(num_requests)]
        for f in as_completed(futures):
            f.result()  # raise exceptions

    channel.close()
    return [r for r in results if r is not None]


def save_results(results: list, path: str):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["request_id", "arrival_time_s", "ttft_ms",
                          "total_time_ms", "tokens", "error"])
        for r in results:
            writer.writerow([r.request_id, f"{r.arrival_time:.3f}",
                              f"{r.first_token_time:.1f}",
                              f"{r.total_time:.1f}",
                              r.tokens_generated, r.error])


def load_yaml_defaults(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args():
    defaults = {
        "target": "localhost:50051",
        "requests": 500,
        "arrival_rate": 20.0,
        "min_tokens": 16,
        "max_tokens": 512,
        "seed": 42,
    }

    # First pass: only look for --config, so its values can seed argparse
    # defaults (explicit CLI flags below still take precedence over both).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    if pre_args.config:
        defaults.update(load_yaml_defaults(pre_args.config))

    parser = argparse.ArgumentParser(parents=[pre], description=__doc__)
    parser.add_argument("--target", default=defaults["target"])
    parser.add_argument("--requests", type=int, default=defaults["requests"])
    parser.add_argument("--arrival-rate", type=float, default=defaults["arrival_rate"],
                         help="Mean requests/second (Poisson)")
    parser.add_argument("--min-tokens", type=int, default=defaults["min_tokens"])
    parser.add_argument("--max-tokens", type=int, default=defaults["max_tokens"])
    parser.add_argument("--output", default="results.csv")
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Running {args.requests} requests at {args.arrival_rate} req/s "
          f"against {args.target}...")
    results = run_benchmark(
        args.target, args.requests, args.arrival_rate,
        (args.min_tokens, args.max_tokens), args.seed)

    save_results(results, args.output)

    # Quick summary
    ttfts = [r.first_token_time for r in results if r.first_token_time > 0]
    totals = [r.total_time for r in results if not r.error]
    total_tokens = sum(r.tokens_generated for r in results)
    wall_time = max(r.arrival_time + r.total_time / 1000 for r in results)
    errors = sum(1 for r in results if r.error)

    print(f"\n{'=' * 50}")
    print(f"Requests:    {len(results)} ({errors} errors)")
    print(f"Throughput:  {total_tokens / wall_time:.1f} tokens/s")
    if ttfts:
        print(f"TTFT p50:    {np.percentile(ttfts, 50):.1f} ms")
        print(f"TTFT p99:    {np.percentile(ttfts, 99):.1f} ms")
    if totals:
        print(f"Latency p50: {np.percentile(totals, 50):.1f} ms")
        print(f"Latency p99: {np.percentile(totals, 99):.1f} ms")
    print(f"Saved to:    {args.output}")


if __name__ == "__main__":
    main()
