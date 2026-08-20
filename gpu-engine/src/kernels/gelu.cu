#include "engine/kernels.h"

namespace kernels {
namespace {

// HF gpt2 uses "gelu_new" (tanh approximation), NOT erf-based gelu.
__device__ __forceinline__ float gelu_tanh_f(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    const float inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void bias_gelu_kernel(half* out, const half* x, const half* bias, int d_ff) {
    const int b = blockIdx.x;
    const half* x_row = x + static_cast<size_t>(b) * d_ff;
    half* out_row = out + static_cast<size_t>(b) * d_ff;
    for (int d = threadIdx.x; d < d_ff; d += blockDim.x) {
        const float v = __half2float(x_row[d]) + __half2float(bias[d]);
        out_row[d] = __float2half(gelu_tanh_f(v));
    }
}

}  // namespace

void launch_bias_gelu(half* out, const half* x, const half* bias, int batch, int d_ff,
                       cudaStream_t stream) {
    bias_gelu_kernel<<<batch, 256, 0, stream>>>(out, x, bias, d_ff);
}

}  // namespace kernels
