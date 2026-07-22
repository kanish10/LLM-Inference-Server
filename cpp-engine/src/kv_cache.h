#pragma once
#include <cstdint>
#include <memory>
#include <vector>

// Simple slot-based KV cache allocator.
// Pre-allocates a contiguous block: num_slots * slot_size.
// Each slot holds the full KV cache for one sequence up to max_seq_len.
// No fragmentation, no paging — simple and predictable.
//
// Memory layout per slot:
//   [num_layers][2 (K then V)][max_seq_len][num_kv_heads][head_dim]
//   Stored as float16 (2 bytes each).
//
// For TinyLlama-1.1B (22 layers, 4 KV heads, head_dim 64, max_seq_len 2048):
//   slot_size = 22 * 2 * 2048 * 4 * 64 * 2 bytes = ~45 MB per slot
//   8 slots = ~360 MB total — fits comfortably in RAM for benchmarking
//
// INTERVIEW QUESTION: "Why not PagedAttention?"
// ANSWER: PagedAttention solves memory *fragmentation* when sequences have
// wildly different lengths and you're running near memory limits. For a
// demonstration with max_batch_size=8 and known max_seq_len, contiguous
// pre-allocation wastes at most 7 * max_seq_len * slot_overhead bytes
// (the slots of short sequences). That's acceptable here, and the
// implementation is 50 lines instead of 500.

class KVCacheAllocator {
public:
    KVCacheAllocator(int num_slots, int max_seq_len, int num_layers,
                     int num_kv_heads, int head_dim);

    // Allocate the first free slot. Returns slot index [0, num_slots).
    // Returns -1 if all slots are occupied.
    int allocate();

    // Free a previously allocated slot. Asserts slot was occupied.
    void free(int slot);

    // Get raw pointer to the start of a slot's memory.
    // Caller (TokenGenerator) is responsible for the layout.
    void* get_slot_ptr(int slot) const;

    // Metrics
    int num_free() const;
    int num_total() const { return num_slots_; }
    float utilization() const {
        return 1.0f - static_cast<float>(num_free()) / num_slots_;
    }

private:
    int num_slots_;
    int slot_size_bytes_;
    std::vector<bool> occupied_;          // true if slot is in use
    std::unique_ptr<uint8_t[]> memory_;   // contiguous allocation
};
