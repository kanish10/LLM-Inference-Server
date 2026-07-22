"""
Static-batching baseline runner.

Static vs. continuous batching is a server-side switch, not a client-side
one: the C++ engine's scheduler is byte-for-byte identical except for one
branch in the evict phase (see CLAUDE.md "Static Batching Baseline"). This
script exists only so the directory layout has an explicit, named entry
point for "run the baseline" — it applies benchmark.py's identical Poisson
traffic generator and just defaults the output filename to results_static.csv
and reminds you to point --target at a `engine --static` instance.

Usage:
    ./cpp-engine/build/engine --mode mock --static --listen unix:///tmp/inference-engine.sock &
    ./bin/gateway --engine unix:///tmp/inference-engine.sock &
    python static_baseline.py --target localhost:50051 --requests 500 --arrival-rate 20
"""
import sys

from benchmark import parse_args, run_benchmark, save_results


def main():
    print("NOTE: static_baseline.py assumes --target is an engine started "
          "with --static. It does not start or configure the engine itself.",
          file=sys.stderr)

    args = parse_args()
    if args.output == "results.csv":
        args.output = "results_static.csv"

    print(f"Running {args.requests} requests at {args.arrival_rate} req/s "
          f"against {args.target} (expected: static batching mode)...")
    results = run_benchmark(
        args.target, args.requests, args.arrival_rate,
        (args.min_tokens, args.max_tokens), args.seed)
    save_results(results, args.output)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
