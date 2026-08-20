#include <doctest/doctest.h>

#include <stdexcept>
#include <vector>

#include "engine/kv_cache.h"

TEST_CASE("BlockPool allocate and free") {
    BlockPool pool(4);
    CHECK(pool.num_free() == 4);

    int b0 = pool.allocate();
    CHECK(b0 >= 0);
    CHECK(pool.num_free() == 3);

    pool.free(b0);
    CHECK(pool.num_free() == 4);
}

TEST_CASE("BlockPool exhaustion path") {
    BlockPool pool(2);
    int b0 = pool.allocate();
    int b1 = pool.allocate();
    CHECK(b0 >= 0);
    CHECK(b1 >= 0);
    CHECK(b0 != b1);

    CHECK(pool.allocate() == -1);  // exhausted

    pool.free(b0);
    int b2 = pool.allocate();
    CHECK(b2 == b0);  // stack reuse
}

TEST_CASE("BlockPool double-free is rejected") {
    BlockPool pool(2);
    int b0 = pool.allocate();
    pool.free(b0);
    CHECK_THROWS_AS(pool.free(b0), std::logic_error);
}

TEST_CASE("BlockPool out-of-range free is rejected") {
    BlockPool pool(2);
    CHECK_THROWS_AS(pool.free(5), std::out_of_range);
    CHECK_THROWS_AS(pool.free(-1), std::out_of_range);
}

TEST_CASE("BlockPool many alloc/free cycles preserve invariants") {
    const int N = 8;
    BlockPool pool(N);
    std::vector<int> held;
    for (int iter = 0; iter < 1000; ++iter) {
        if (held.size() < static_cast<size_t>(N) && (iter % 3 != 0)) {
            int id = pool.allocate();
            REQUIRE(id >= 0);
            held.push_back(id);
        } else if (!held.empty()) {
            pool.free(held.back());
            held.pop_back();
        }
        CHECK(pool.num_free() == N - static_cast<int>(held.size()));
    }
}

TEST_CASE("BlockTable needs_new_block boundaries") {
    CHECK(BlockTable::needs_new_block(0));
    CHECK_FALSE(BlockTable::needs_new_block(1));
    CHECK_FALSE(BlockTable::needs_new_block(BLOCK_TOKENS - 1));
    CHECK(BlockTable::needs_new_block(BLOCK_TOKENS));

    BlockTable table;
    CHECK(table.num_blocks() == 0);
    table.append_block(3);
    table.append_block(7);
    CHECK(table.num_blocks() == 2);
    CHECK(table.block_ids()[0] == 3);
    CHECK(table.block_ids()[1] == 7);
}

TEST_CASE("block_bytes matches GPT-2 124M spec constant") {
    // 12 layers * 2 (K,V) * 16 tokens * 12 heads * 64 head_dim * 2 bytes (fp16)
    CHECK(block_bytes(12, 12, 64) == 589824);
}
