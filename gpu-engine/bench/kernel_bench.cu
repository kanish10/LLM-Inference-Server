// Per-kernel microbenchmark + correctness check for PerfLens Tune
// (see bench/TUNING.md).
//
// Built once per tuning candidate by bench/tune_build.py, together with only
// the one kernel translation unit under test and that candidate's -D /
// -maxrregcount / --use_fast_math flags. Exactly one of BENCH_GELU,
// BENCH_LAYERNORM, BENCH_ARGMAX must be defined. This file is host-only on
// purpose: ptxas output for the build then describes the tuned kernel alone.
//
// Shapes are GPT-2 124M's (d_model 768, d_ff 3072, vocab 50257). Inputs are
// deterministic (seeded). Correctness uses the same rule as
// tests/test_kernels.cpp: max_abs_err < 5e-2 against an FP32 CPU reference
// (computed on the same half-rounded inputs the GPU sees); argmax must match
// exactly, with one planted, unambiguous winner per row.
//
// Timing: warmup launches, then `samples` cudaEvent-timed groups of `inner`
// back-to-back launches each; per-launch time = group time / inner. Prints
// one JSON line on stdout with the median and MAD across samples.
//
// Usage: kernel_bench --batch 8 [--samples 200] [--inner 10] [--warmup 20] [--seed 1234]

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "engine/kernels.h"
#include "engine/tune_params.h"

#if (defined(BENCH_GELU) + defined(BENCH_LAYERNORM) + defined(BENCH_ARGMAX)) != 1
#error "define exactly one of BENCH_GELU, BENCH_LAYERNORM, BENCH_ARGMAX"
#endif

namespace {

// Not every constant/helper is used by every BENCH_* variant.
[[maybe_unused]] constexpr int kDModel = 768;
[[maybe_unused]] constexpr int kDFF = 3072;
[[maybe_unused]] constexpr int kVocab = 50257;
[[maybe_unused]] constexpr float kTol = 5e-2f;

struct Args {
    int batch = 8;
    int samples = 200;
    int inner = 10;
    int warmup = 20;
    unsigned seed = 1234;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", k.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (k == "--batch") a.batch = std::atoi(next());
        else if (k == "--samples") a.samples = std::atoi(next());
        else if (k == "--inner") a.inner = std::atoi(next());
        else if (k == "--warmup") a.warmup = std::atoi(next());
        else if (k == "--seed") a.seed = static_cast<unsigned>(std::strtoul(next(), nullptr, 10));
        else {
            std::fprintf(stderr, "unknown argument: %s\n", k.c_str());
            std::exit(2);
        }
    }
    if (a.batch < 1 || a.samples < 1 || a.inner < 1 || a.warmup < 0) {
        std::fprintf(stderr, "batch/samples/inner must be >= 1, warmup >= 0\n");
        std::exit(2);
    }
    return a;
}

std::vector<half> to_half(const std::vector<float>& in) {
    std::vector<half> out(in.size());
    for (size_t i = 0; i < in.size(); ++i) out[i] = __float2half(in[i]);
    return out;
}

// Round-trip through half so the CPU reference sees exactly what the GPU sees.
[[maybe_unused]] std::vector<float> round_to_half(const std::vector<float>& in) {
    std::vector<float> out(in.size());
    for (size_t i = 0; i < in.size(); ++i) out[i] = __half2float(__float2half(in[i]));
    return out;
}

[[maybe_unused]] float max_abs_err(const std::vector<half>& got, const std::vector<float>& want) {
    float worst = 0.0f;
    for (size_t i = 0; i < want.size(); ++i) {
        const float g = __half2float(got[i]);
        if (!std::isfinite(g)) return INFINITY;
        worst = std::max(worst, std::fabs(g - want[i]));
    }
    return worst;
}

template <typename T>
T* to_device(const std::vector<T>& host) {
    T* d = nullptr;
    cudaMalloc(&d, host.size() * sizeof(T));
    cudaMemcpy(d, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

template <typename T>
std::vector<T> from_device(const T* d, size_t n) {
    std::vector<T> host(n);
    cudaMemcpy(host.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return host;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

void die_on_cuda_error(const char* where) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("{\"error\":\"%s: %s\",\"correct\":false}\n", where, cudaGetErrorString(err));
        std::exit(1);
    }
}

// --- per-kernel setup: each block builds inputs, a CPU reference, and a launcher ---

struct Bench {
    const char* kernel = "";
    void (*launch)(void*) = nullptr;
    void* ctx = nullptr;
};

#if defined(BENCH_GELU)
struct GeluCtx {
    half *out, *x, *bias;
    int batch;
};
void gelu_launch(void* p) {
    auto* c = static_cast<GeluCtx*>(p);
    kernels::launch_bias_gelu(c->out, c->x, c->bias, c->batch, kDFF, nullptr);
}
#endif

#if defined(BENCH_LAYERNORM)
struct LnCtx {
    half *out, *rout, *x, *r, *gamma, *beta;
    int batch;
};
void ln_launch(void* p) {
    auto* c = static_cast<LnCtx*>(p);
    kernels::launch_add_layernorm(c->out, c->rout, c->x, c->r, c->gamma, c->beta, c->batch, kDModel,
                                  1e-5f, nullptr);
}
#endif

#if defined(BENCH_ARGMAX)
struct ArgmaxCtx {
    int* out;
    half* logits;
    int batch;
};
void argmax_launch(void* p) {
    auto* c = static_cast<ArgmaxCtx*>(p);
    kernels::launch_argmax(c->out, c->logits, c->batch, kVocab, nullptr);
}
#endif

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    std::mt19937 rng(args.seed);

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::printf("{\"error\":\"no CUDA device\",\"correct\":false}\n");
        return 1;
    }

    Bench bench;
    bool correct = false;
    float err = 0.0f;
    char macros[256];

#if defined(BENCH_GELU)
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    const size_t n = static_cast<size_t>(args.batch) * kDFF;
    std::vector<float> x_f(n), bias_f(kDFF);
    for (auto& v : x_f) v = dist(rng);
    for (auto& v : bias_f) v = dist(rng);
    x_f = round_to_half(x_f);
    bias_f = round_to_half(bias_f);
    std::vector<float> expected(n);
    for (int b = 0; b < args.batch; ++b) {
        for (int d = 0; d < kDFF; ++d) {
            const float v = x_f[static_cast<size_t>(b) * kDFF + d] + bias_f[d];
            const float k0 = 0.7978845608028654f;
            expected[static_cast<size_t>(b) * kDFF + d] =
                0.5f * v * (1.0f + std::tanh(k0 * (v + 0.044715f * v * v * v)));
        }
    }
    static GeluCtx ctx;
    ctx.x = to_device(to_half(x_f));
    ctx.bias = to_device(to_half(bias_f));
    cudaMalloc(&ctx.out, n * sizeof(half));
    ctx.batch = args.batch;
    bench = {"bias_gelu", gelu_launch, &ctx};
    std::snprintf(macros, sizeof(macros),
                  "{\"GELU_THREADS\":%d,\"GELU_BLOCKS_PER_ROW\":%d,\"GELU_VEC\":%d}",
                  GELU_THREADS, GELU_BLOCKS_PER_ROW, GELU_VEC);
    bench.launch(bench.ctx);
    cudaDeviceSynchronize();
    die_on_cuda_error("bias_gelu launch");
    err = max_abs_err(from_device(ctx.out, n), expected);
    correct = err < kTol;
#endif

#if defined(BENCH_LAYERNORM)
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    const size_t n = static_cast<size_t>(args.batch) * kDModel;
    std::vector<float> x_f(n), r_f(n), gamma_f(kDModel), beta_f(kDModel);
    for (auto& v : x_f) v = dist(rng);
    for (auto& v : r_f) v = dist(rng);
    for (auto& v : gamma_f) v = 1.0f + 0.1f * dist(rng);
    for (auto& v : beta_f) v = 0.1f * dist(rng);
    x_f = round_to_half(x_f);
    r_f = round_to_half(r_f);
    gamma_f = round_to_half(gamma_f);
    beta_f = round_to_half(beta_f);
    std::vector<float> expected_out(n), expected_rout(n);
    for (int b = 0; b < args.batch; ++b) {
        double mean = 0.0;
        std::vector<float> row(kDModel);
        for (int d = 0; d < kDModel; ++d) {
            row[d] = x_f[static_cast<size_t>(b) * kDModel + d] + r_f[static_cast<size_t>(b) * kDModel + d];
            expected_rout[static_cast<size_t>(b) * kDModel + d] = row[d];
            mean += row[d];
        }
        mean /= kDModel;
        double var = 0.0;
        for (int d = 0; d < kDModel; ++d) var += (row[d] - mean) * (row[d] - mean);
        var /= kDModel;
        const float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + 1e-5f);
        for (int d = 0; d < kDModel; ++d) {
            expected_out[static_cast<size_t>(b) * kDModel + d] =
                static_cast<float>((row[d] - mean) * inv_std) * gamma_f[d] + beta_f[d];
        }
    }
    static LnCtx ctx;
    ctx.x = to_device(to_half(x_f));
    ctx.r = to_device(to_half(r_f));
    ctx.gamma = to_device(to_half(gamma_f));
    ctx.beta = to_device(to_half(beta_f));
    cudaMalloc(&ctx.out, n * sizeof(half));
    cudaMalloc(&ctx.rout, n * sizeof(half));
    ctx.batch = args.batch;
    bench = {"add_layernorm", ln_launch, &ctx};
    std::snprintf(macros, sizeof(macros), "{\"LN_THREADS\":%d}", LN_THREADS);
    bench.launch(bench.ctx);
    cudaDeviceSynchronize();
    die_on_cuda_error("add_layernorm launch");
    err = std::max(max_abs_err(from_device(ctx.out, n), expected_out),
                   max_abs_err(from_device(ctx.rout, n), expected_rout));
    correct = err < kTol;
#endif

#if defined(BENCH_ARGMAX)
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::uniform_int_distribution<int> pick(0, kVocab - 1);
    const size_t n = static_cast<size_t>(args.batch) * kVocab;
    std::vector<float> logits_f(n);
    for (auto& v : logits_f) v = dist(rng);
    std::vector<int> expected(args.batch);
    for (int b = 0; b < args.batch; ++b) {
        expected[b] = pick(rng);
        logits_f[static_cast<size_t>(b) * kVocab + expected[b]] = 100.0f;  // unambiguous winner
    }
    static ArgmaxCtx ctx;
    ctx.logits = to_device(to_half(logits_f));
    cudaMalloc(&ctx.out, args.batch * sizeof(int));
    ctx.batch = args.batch;
    bench = {"argmax", argmax_launch, &ctx};
    std::snprintf(macros, sizeof(macros), "{\"ARGMAX_THREADS\":%d}", ARGMAX_THREADS);
    bench.launch(bench.ctx);
    cudaDeviceSynchronize();
    die_on_cuda_error("argmax launch");
    const std::vector<int> got = from_device(ctx.out, static_cast<size_t>(args.batch));
    int mismatches = 0;
    for (int b = 0; b < args.batch; ++b) mismatches += got[b] != expected[b];
    err = static_cast<float>(mismatches);
    correct = mismatches == 0;
#endif

    for (int i = 0; i < args.warmup; ++i) bench.launch(bench.ctx);
    cudaDeviceSynchronize();
    die_on_cuda_error("warmup");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::vector<double> per_launch_us(args.samples);
    for (int s = 0; s < args.samples; ++s) {
        cudaEventRecord(start);
        for (int i = 0; i < args.inner; ++i) bench.launch(bench.ctx);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        per_launch_us[s] = 1000.0 * ms / args.inner;
    }
    die_on_cuda_error("timed launches");

    const double med = median(per_launch_us);
    std::vector<double> dev(per_launch_us.size());
    for (size_t i = 0; i < dev.size(); ++i) dev[i] = std::fabs(per_launch_us[i] - med);
    const double mad = median(dev);

    std::printf(
        "{\"kernel\":\"%s\",\"batch\":%d,\"correct\":%s,\"max_abs_err\":%.6g,"
        "\"median_us\":%.4f,\"mad_us\":%.4f,\"samples\":%d,\"inner\":%d,\"warmup\":%d,"
        "\"seed\":%u,\"device\":\"%s\",\"macros\":%s}\n",
        bench.kernel, args.batch, correct ? "true" : "false", err, med, mad, args.samples,
        args.inner, args.warmup, args.seed, prop.name, macros);
    return 0;
}
