#include "engine/kernels.h"
#include "engine/tune_params.h"

namespace kernels {
namespace {

// HF gpt2 uses "gelu_new" (tanh approximation), NOT erf-based gelu.
__device__ __forceinline__ float gelu_tanh_f(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    const float inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Grid is (batch, GELU_BLOCKS_PER_ROW): with more than one block per row,
// blocks in the same row stride over d_ff together, so a small batch can
// still spread across more SMs. With the defaults (1 block per row,
// GELU_VEC 1) this is the original one-block-per-row scalar loop.
__global__ void GELU_LAUNCH_BOUNDS bias_gelu_kernel(half* out, const half* x, const half* bias,
                                                    int d_ff) {
    const int b = blockIdx.x;
    const half* x_row = x + static_cast<size_t>(b) * d_ff;
    half* out_row = out + static_cast<size_t>(b) * d_ff;
    const int stride = static_cast<int>(gridDim.y * blockDim.x);
    const int start = static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x);
#if GELU_VEC == 2
    // half2 path: two elements per load/store. Requires 4-byte-aligned row
    // starts, so it is only taken for even d_ff (every GPT-2 size) on
    // cudaMalloc'd buffers; odd d_ff falls through to the scalar loop.
    // __floats2half2_rn rounds exactly like __float2half, so output matches
    // the scalar path bit for bit.
    if ((d_ff & 1) == 0) {
        const int n2 = d_ff / 2;
        const half2* x2 = reinterpret_cast<const half2*>(x_row);
        const half2* b2 = reinterpret_cast<const half2*>(bias);
        half2* o2 = reinterpret_cast<half2*>(out_row);
        for (int i = start; i < n2; i += stride) {
            const float2 xv = __half22float2(x2[i]);
            const float2 bv = __half22float2(b2[i]);
            o2[i] = __floats2half2_rn(gelu_tanh_f(xv.x + bv.x), gelu_tanh_f(xv.y + bv.y));
        }
        return;
    }
#endif
    for (int d = start; d < d_ff; d += stride) {
        const float v = __half2float(x_row[d]) + __half2float(bias[d]);
        out_row[d] = __float2half(gelu_tanh_f(v));
    }
}

}  // namespace

void launch_bias_gelu(half* out, const half* x, const half* bias, int batch, int d_ff,
                       cudaStream_t stream) {
    const dim3 grid(batch, GELU_BLOCKS_PER_ROW);
    bias_gelu_kernel<<<grid, GELU_THREADS, 0, stream>>>(out, x, bias, d_ff);
}

}  // namespace kernels
