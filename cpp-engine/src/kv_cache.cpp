#include "kv_cache.h"
#include <cassert>
#include <cstring>
#include <stdexcept>

KVCacheAllocator::KVCacheAllocator(int num_slots, int max_seq_len,
                                   int num_layers, int num_kv_heads,
                                   int head_dim)
    : num_slots_(num_slots),
      occupied_(num_slots, false) {
    if (num_slots <= 0) {
        throw std::invalid_argument("num_slots must be positive");
    }
    // float16 = 2 bytes per element
    // K and V = factor of 2
    slot_size_bytes_ = num_layers * 2 * max_seq_len * num_kv_heads * head_dim * 2;
    memory_ = std::make_unique<uint8_t[]>(
        static_cast<size_t>(num_slots_) * slot_size_bytes_);
    std::memset(memory_.get(), 0,
                static_cast<size_t>(num_slots_) * slot_size_bytes_);
}

int KVCacheAllocator::allocate() {
    for (int i = 0; i < num_slots_; ++i) {
        if (!occupied_[i]) {
            occupied_[i] = true;
            return i;
        }
    }
    return -1;
}

void KVCacheAllocator::free(int slot) {
    assert(slot >= 0 && slot < num_slots_);
    assert(occupied_[slot]);
    occupied_[slot] = false;
    // Zeroed for safety/determinism; comment out if profiling shows this
    // memset is hot (it isn't — it runs once per eviction, not per step).
    std::memset(get_slot_ptr(slot), 0, slot_size_bytes_);
}

void* KVCacheAllocator::get_slot_ptr(int slot) const {
    assert(slot >= 0 && slot < num_slots_);
    return memory_.get() + static_cast<size_t>(slot) * slot_size_bytes_;
}

int KVCacheAllocator::num_free() const {
    int count = 0;
    for (bool b : occupied_) {
        if (!b) ++count;
    }
    return count;
}
