#include "sliding_window.hpp"
#include <gtest/gtest.h>

static CanFrame mk(uint32_t id) { CanFrame f; f.id = id; f.dlc = 8; return f; }

TEST(SlidingWindow, NotReadyUntilFull) {
    SlidingWindow w(4, 2);
    EXPECT_FALSE(w.push(mk(1)));
    EXPECT_FALSE(w.push(mk(2)));
    EXPECT_FALSE(w.push(mk(3)));
    EXPECT_TRUE(w.push(mk(4)));   // full and stride reached
}

TEST(SlidingWindow, FiresEveryStride) {
    SlidingWindow w(4, 2);
    for (uint32_t i = 1; i <= 4; ++i) w.push(mk(i));
    EXPECT_FALSE(w.push(mk(5)));
    EXPECT_TRUE(w.push(mk(6)));
}

TEST(SlidingWindow, SnapshotIsOldestToNewestAfterWrap) {
    SlidingWindow w(3, 1);
    for (uint32_t i = 1; i <= 5; ++i) w.push(mk(i));   // buffer wrapped
    auto s = w.snapshot();
    ASSERT_EQ(s.size(), 3u);
    EXPECT_EQ(s[0].id, 3u);
    EXPECT_EQ(s[1].id, 4u);
    EXPECT_EQ(s[2].id, 5u);
}
