#include "engine/kernels.h"

#include <cfloat>

// Deviation from the repo-layout suggestion of cub::DeviceReduce: this uses
// a small self-contained block-reduction kernel instead of cub's
// segmented-argmax API. That API's exact call shape (temp-storage sizing
// query, iterator types) varies across CUB versions bundled with different
// CUDA toolkits, and getting it wrong is not something that can be caught
// here (no CUDA device on this machine to compile/run against). A
// hand-written reduction is fully self-contained and easier to verify by
// inspection. cub/thrust remain available if a future profiling pass shows
// this is worth revisiting.

namespace kernels {
namespace {

__global__ void argmax_kernel(int* out_tokens, const half* logits, int vocab_size) {
    extern __shared__ unsigned char smem_raw[];
    float* smem_val = reinterpret_cast<float*>(smem_raw);
    int* smem_idx = reinterpret_cast<int*>(smem_val + blockDim.x);

    const int b = blockIdx.x;
    const half* row = logits + static_cast<size_t>(b) * vocab_size;

    float best_val = -FLT_MAX;
    int best_idx = 0;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        const float val = __half2float(row[v]);
        if (val > best_val) {
            best_val = val;
            best_idx = v;
        }
    }
    smem_val[threadIdx.x] = best_val;
    smem_idx[threadIdx.x] = best_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && smem_val[threadIdx.x + s] > smem_val[threadIdx.x]) {
            smem_val[threadIdx.x] = smem_val[threadIdx.x + s];
            smem_idx[threadIdx.x] = smem_idx[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) out_tokens[b] = smem_idx[0];
}

}  // namespace

void launch_argmax(int* out_tokens, const half* logits, int batch, int vocab_size,
                    cudaStream_t stream) {
    const int threads = 256;
    const size_t smem_bytes = static_cast<size_t>(threads) * (sizeof(float) + sizeof(int));
    argmax_kernel<<<batch, threads, smem_bytes, stream>>>(out_tokens, logits, vocab_size);
}

}  // namespace kernels
