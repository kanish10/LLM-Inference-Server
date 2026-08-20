#include "engine/kv_cache_gpu.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "engine/kv_cache.h"

namespace {
void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}
}  // namespace

KVCacheArenaGPU::KVCacheArenaGPU(int num_blocks, int n_layer, int n_heads, int head_dim)
    : num_blocks_(num_blocks) {
    const size_t bytes = static_cast<size_t>(num_blocks) * block_bytes(n_layer, n_heads, head_dim);
    cuda_check(cudaMalloc(&d_pool_, bytes), "cudaMalloc KV cache arena");
    cuda_check(cudaMemset(d_pool_, 0, bytes), "cudaMemset KV cache arena");
}

KVCacheArenaGPU::~KVCacheArenaGPU() {
    if (d_pool_) cudaFree(d_pool_);
}
