#pragma once
#include <cuda_fp16.h>

// Device-side KV cache arena: one cudaMalloc'd block of
// num_blocks * block_bytes(n_layer, n_heads, head_dim). Which blocks belong
// to which sequence is tracked entirely on the host by BlockPool +
// BlockTable (include/engine/kv_cache.h, CUDA-free, already tested in
// tests/test_kv_cache.cpp); this class owns only the raw device memory.
class KVCacheArenaGPU {
public:
    KVCacheArenaGPU(int num_blocks, int n_layer, int n_heads, int head_dim);
    ~KVCacheArenaGPU();
    KVCacheArenaGPU(const KVCacheArenaGPU&) = delete;
    KVCacheArenaGPU& operator=(const KVCacheArenaGPU&) = delete;

    half* pool() const { return d_pool_; }
    int num_blocks() const { return num_blocks_; }

private:
    half* d_pool_ = nullptr;
    int num_blocks_;
};
