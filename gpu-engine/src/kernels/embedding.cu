#include "engine/kernels.h"

namespace kernels {
namespace {

__global__ void embedding_kernel(half* out, const half* wte, const half* wpe,
                                  const int* token_ids, const int* positions, int d_model) {
    const int b = blockIdx.x;
    const half* tok_row = wte + static_cast<size_t>(token_ids[b]) * d_model;
    const half* pos_row = wpe + static_cast<size_t>(positions[b]) * d_model;
    half* out_row = out + static_cast<size_t>(b) * d_model;
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        out_row[d] = __hadd(tok_row[d], pos_row[d]);
    }
}

__global__ void add_bias_kernel(half* out, const half* x, const half* bias, int dim) {
    const int b = blockIdx.x;
    const half* x_row = x + static_cast<size_t>(b) * dim;
    half* out_row = out + static_cast<size_t>(b) * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        out_row[d] = __hadd(x_row[d], bias[d]);
    }
}

__global__ void add_one_kernel(int* out, const int* in) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = in[i] + 1;
}

}  // namespace

void launch_embedding(half* out, const half* wte, const half* wpe, const int* token_ids,
                       const int* positions, int batch, int d_model, cudaStream_t stream) {
    embedding_kernel<<<batch, 256, 0, stream>>>(out, wte, wpe, token_ids, positions, d_model);
}

void launch_add_bias(half* out, const half* x, const half* bias, int batch, int dim,
                      cudaStream_t stream) {
    add_bias_kernel<<<batch, 256, 0, stream>>>(out, x, bias, dim);
}

void launch_add_one(int* out, const int* in, int batch, cudaStream_t stream) {
    add_one_kernel<<<1, batch, 0, stream>>>(out, in);
}

}  // namespace kernels
