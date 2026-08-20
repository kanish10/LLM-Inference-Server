#include "engine/kernels.h"

namespace kernels {
namespace {

// One block per batch row, blockDim.x threads (power of two) cooperate on
// two tree reductions (mean, then variance) via shared memory. FP32
// accumulation throughout even though in/out tensors are half.
__global__ void add_layernorm_kernel(half* out, half* residual_out, const half* x,
                                      const half* residual_in, const half* gamma,
                                      const half* beta, int d_model, float eps) {
    extern __shared__ float smem[];
    const int b = blockIdx.x;
    const half* x_row = x + static_cast<size_t>(b) * d_model;
    const half* r_row = residual_in + static_cast<size_t>(b) * d_model;
    half* rout_row = residual_out + static_cast<size_t>(b) * d_model;
    half* out_row = out + static_cast<size_t>(b) * d_model;

    float local_sum = 0.0f;
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        const float v = __half2float(x_row[d]) + __half2float(r_row[d]);
        rout_row[d] = __float2half(v);
        local_sum += v;
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    const float mean = smem[0] / d_model;
    __syncthreads();

    float local_var = 0.0f;
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        const float v = __half2float(rout_row[d]) - mean;
        local_var += v * v;
    }
    smem[threadIdx.x] = local_var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    const float inv_std = rsqrtf(smem[0] / d_model + eps);
    __syncthreads();

    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        const float v = (__half2float(rout_row[d]) - mean) * inv_std;
        out_row[d] = __float2half(v * __half2float(gamma[d]) + __half2float(beta[d]));
    }
}

}  // namespace

void launch_add_layernorm(half* out, half* residual_out, const half* x, const half* residual_in,
                           const half* gamma, const half* beta, int batch, int d_model, float eps,
                           cudaStream_t stream) {
    const int threads = 256;
    const size_t smem_bytes = threads * sizeof(float);
    add_layernorm_kernel<<<batch, threads, smem_bytes, stream>>>(out, residual_out, x, residual_in,
                                                                   gamma, beta, d_model, eps);
}

}  // namespace kernels
