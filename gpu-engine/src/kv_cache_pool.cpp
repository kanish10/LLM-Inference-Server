#include "engine/kv_cache.h"

#include <stdexcept>
#include <string>

BlockPool::BlockPool(int num_blocks) : num_blocks_(num_blocks), occupied_(num_blocks, false) {
    free_stack_.reserve(num_blocks);
    for (int i = num_blocks - 1; i >= 0; --i) free_stack_.push_back(i);
}

int BlockPool::allocate() {
    if (free_stack_.empty()) return -1;
    int id = free_stack_.back();
    free_stack_.pop_back();
    occupied_[id] = true;
    return id;
}

void BlockPool::free(int block_id) {
    if (block_id < 0 || block_id >= num_blocks_) {
        throw std::out_of_range("BlockPool::free: block_id out of range");
    }
    if (!occupied_[block_id]) {
        throw std::logic_error("BlockPool::free: double-free of block_id " + std::to_string(block_id));
    }
    occupied_[block_id] = false;
    free_stack_.push_back(block_id);
}
