#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

// One block covers ALL layers for BLOCK_TOKENS token positions. Layout
// (device side, M2+): [layer][2(K,V)][BLOCK_TOKENS][n_heads][head_dim] half.
constexpr int BLOCK_TOKENS = 16;

// Bytes per block for GPT-2 124M (12 layers, 12 heads, head_dim 64, fp16):
// 12 * 2 * 16 * 12 * 64 * 2 = 589,824 bytes (~0.56 MB).
inline size_t block_bytes(int n_layer, int n_heads, int head_dim) {
    return static_cast<size_t>(n_layer) * 2 * BLOCK_TOKENS * n_heads * head_dim * sizeof(uint16_t);
}

// Host-side free-list allocator for KV cache block indices. Pure bookkeeping
// -- no device memory here, so this is testable without a GPU. The device
// arena (kv_block_budget * block_bytes) is allocated separately in
// src/kv_cache.cu (CUDA-only, needs GPU hardware); this class just tracks
// which block indices are free/occupied. O(1) alloc/free via a stack of
// free indices.
class BlockPool {
public:
    explicit BlockPool(int num_blocks);

    // Returns a free block index, or -1 if the pool is exhausted.
    int allocate();

    // Returns block_id to the free pool. Asserts (throws) on double-free or
    // an out-of-range id.
    void free(int block_id);

    int num_free() const { return static_cast<int>(free_stack_.size()); }
    int num_total() const { return num_blocks_; }

private:
    int num_blocks_;
    std::vector<int> free_stack_;
    std::vector<bool> occupied_;
};

// Per-sequence table of block ids, in allocation order. block_ids()[i]
// covers token positions [i*BLOCK_TOKENS, (i+1)*BLOCK_TOKENS).
class BlockTable {
public:
    void append_block(int block_id) { block_ids_.push_back(block_id); }
    const std::vector<int>& block_ids() const { return block_ids_; }
    int num_blocks() const { return static_cast<int>(block_ids_.size()); }
    void clear() { block_ids_.clear(); }

    // True when writing the token at (0-indexed) seq_len requires a fresh
    // block to have been allocated first.
    static bool needs_new_block(int seq_len) { return seq_len % BLOCK_TOKENS == 0; }

private:
    std::vector<int> block_ids_;
};
