#pragma once

// Compile-time launch parameters for the kernels PerfLens Tune searches over
// (see bench/TUNING.md). Every default below equals the value the
// engine shipped with, so a normal CMake build is unchanged; the tuner
// overrides them per candidate with -D flags (bench/tune_build.py).
//
// Space limits enforced here at compile time so an invalid candidate fails
// to build instead of silently launching something different:
//   *_THREADS      power of two in [32, 1024] (block tree reductions need it)
//   *_MIN_BLOCKS   optional; THREADS * MIN_BLOCKS must fit one SM's
//                  1536-thread limit on sm_86, or ptxas ignores the bound
//   GELU_BLOCKS_PER_ROW >= 1
//   GELU_VEC       1 or 2

#ifndef GELU_THREADS
#define GELU_THREADS 256
#endif
#ifndef GELU_BLOCKS_PER_ROW
#define GELU_BLOCKS_PER_ROW 1
#endif
#ifndef GELU_VEC
#define GELU_VEC 1  // elements per load/store: 1 (half) or 2 (half2)
#endif
#ifndef LN_THREADS
#define LN_THREADS 256
#endif
#ifndef ARGMAX_THREADS
#define ARGMAX_THREADS 256
#endif

#define TUNE_IS_POW2_IN_RANGE(n) ((n) >= 32 && (n) <= 1024 && (((n) & ((n) - 1)) == 0))
static_assert(TUNE_IS_POW2_IN_RANGE(GELU_THREADS), "GELU_THREADS must be a power of two in [32,1024]");
static_assert(TUNE_IS_POW2_IN_RANGE(LN_THREADS), "LN_THREADS must be a power of two in [32,1024]");
static_assert(TUNE_IS_POW2_IN_RANGE(ARGMAX_THREADS), "ARGMAX_THREADS must be a power of two in [32,1024]");
static_assert(GELU_BLOCKS_PER_ROW >= 1, "GELU_BLOCKS_PER_ROW must be >= 1");
static_assert(GELU_VEC == 1 || GELU_VEC == 2, "GELU_VEC must be 1 or 2");

#define TUNE_MAX_THREADS_PER_SM 1536  // sm_86

// Optional __launch_bounds__(THREADS, MIN_BLOCKS). Leaving *_MIN_BLOCKS
// undefined (the default) emits no launch bounds at all.
#ifdef GELU_MIN_BLOCKS
static_assert(GELU_THREADS * GELU_MIN_BLOCKS <= TUNE_MAX_THREADS_PER_SM, "GELU launch bounds exceed one SM");
#define GELU_LAUNCH_BOUNDS __launch_bounds__(GELU_THREADS, GELU_MIN_BLOCKS)
#else
#define GELU_LAUNCH_BOUNDS
#endif

#ifdef LN_MIN_BLOCKS
static_assert(LN_THREADS * LN_MIN_BLOCKS <= TUNE_MAX_THREADS_PER_SM, "LN launch bounds exceed one SM");
#define LN_LAUNCH_BOUNDS __launch_bounds__(LN_THREADS, LN_MIN_BLOCKS)
#else
#define LN_LAUNCH_BOUNDS
#endif

#ifdef ARGMAX_MIN_BLOCKS
static_assert(ARGMAX_THREADS * ARGMAX_MIN_BLOCKS <= TUNE_MAX_THREADS_PER_SM,
              "ARGMAX launch bounds exceed one SM");
#define ARGMAX_LAUNCH_BOUNDS __launch_bounds__(ARGMAX_THREADS, ARGMAX_MIN_BLOCKS)
#else
#define ARGMAX_LAUNCH_BOUNDS
#endif
