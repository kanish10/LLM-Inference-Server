"""
Compare static vs continuous batching results.

Usage:
    python analyze.py results_continuous.csv results_static.csv

Produces:
    - benchmark_results.png (latency CDF + TTFT distribution)
    - summary.txt (the numbers for the resume bullet)
"""
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if not r['error']]
    if not rows:
        raise ValueError(f"{path} has no successful (non-error) rows to analyze")
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
    cont_wall = max(cont['arrival'] + cont['total'] / 1000)
    stat_wall = max(stat['arrival'] + stat['total'] / 1000)
    cont_tps = cont['tokens'].sum() / cont_wall
    stat_tps = stat['tokens'].sum() / stat_wall
    ratio = cont_tps / stat_tps if stat_tps > 0 else float('inf')

    print(f"\n{'=' * 60}")
    print(f"CONTINUOUS BATCHING vs STATIC BATCHING")
    print(f"{'=' * 60}")
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
        f.write(f"TTFT p50 continuous/static: "
                f"{np.percentile(cont['ttft'], 50):.1f}ms / "
                f"{np.percentile(stat['ttft'], 50):.1f}ms\n")
        f.write(f"Latency p99 continuous/static: "
                f"{np.percentile(cont['total'], 99):.1f}ms / "
                f"{np.percentile(stat['total'], 99):.1f}ms\n")

    # Latency CDF plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for data, label in [(cont, 'Continuous'), (stat, 'Static')]:
        sorted_lat = np.sort(data['total'])
        cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
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
