#include <gtest/gtest.h>
#include "kv_cache.h"

TEST(KVCacheTest, AllocateAndFree) {
    KVCacheAllocator cache(4, 128, 2, 4, 64);
    EXPECT_EQ(cache.num_free(), 4);

    int s0 = cache.allocate();
    EXPECT_GE(s0, 0);
    EXPECT_EQ(cache.num_free(), 3);

    cache.free(s0);
    EXPECT_EQ(cache.num_free(), 4);
}

TEST(KVCacheTest, ExhaustSlots) {
    KVCacheAllocator cache(2, 128, 2, 4, 64);
    int s0 = cache.allocate();
    int s1 = cache.allocate();
    EXPECT_GE(s0, 0);
    EXPECT_GE(s1, 0);
    EXPECT_NE(s0, s1);

    // Third allocation should fail
    EXPECT_EQ(cache.allocate(), -1);

    // Free one, allocate again
    cache.free(s0);
    int s2 = cache.allocate();
    EXPECT_GE(s2, 0);
}

TEST(KVCacheTest, Utilization) {
    KVCacheAllocator cache(4, 128, 2, 4, 64);
    EXPECT_FLOAT_EQ(cache.utilization(), 0.0f);

    cache.allocate();
    cache.allocate();
    EXPECT_FLOAT_EQ(cache.utilization(), 0.5f);
}

TEST(KVCacheTest, SlotPointersAreDistinct) {
    KVCacheAllocator cache(4, 128, 2, 4, 64);
    int s0 = cache.allocate();
    int s1 = cache.allocate();
    EXPECT_NE(cache.get_slot_ptr(s0), cache.get_slot_ptr(s1));
}

TEST(KVCacheTest, FreedSlotIsZeroed) {
    KVCacheAllocator cache(2, 128, 2, 4, 64);
    int s0 = cache.allocate();
    uint8_t* ptr = static_cast<uint8_t*>(cache.get_slot_ptr(s0));
    ptr[0] = 0xAB;
    cache.free(s0);
    EXPECT_EQ(ptr[0], 0);
}

TEST(KVCacheTest, NumTotalMatchesConstruction) {
    KVCacheAllocator cache(7, 128, 2, 4, 64);
    EXPECT_EQ(cache.num_total(), 7);
    EXPECT_EQ(cache.num_free(), 7);
}
