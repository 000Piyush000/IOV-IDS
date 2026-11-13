#include "features.hpp"
#include <gtest/gtest.h>
#include <cmath>

static CanFrame mk(uint32_t id, uint8_t fill) {
    CanFrame f; f.id = id; f.dlc = 8;
    for (auto& b : f.data) b = fill;
    return f;
}

TEST(Features, DosWindowHasZeroEntropy) {
    std::vector<CanFrame> w(100, mk(0x000, 0x00));   // pure DoS flood of ID 0
    auto f = extract_features(w);
    EXPECT_DOUBLE_EQ(f[0], 1.0);   // one unique ID
    EXPECT_DOUBLE_EQ(f[1], 0.0);   // id_entropy
    EXPECT_DOUBLE_EQ(f[2], 1.0);   // top_id_ratio
    EXPECT_DOUBLE_EQ(f[3], 1.0);   // frac_id_zero
    EXPECT_DOUBLE_EQ(f[38], 1.0);  // max_repeat_run: whole window
}

TEST(Features, TwoEqualIdsGiveOneBitEntropy) {
    std::vector<CanFrame> w;
    for (int i = 0; i < 50; ++i) { w.push_back(mk(0x100, 1)); w.push_back(mk(0x200, 1)); }
    auto f = extract_features(w);
    EXPECT_NEAR(f[1], 1.0, 1e-12);   // H = 1 bit
    EXPECT_DOUBLE_EQ(f[8], 1.0);     // ID changes every frame
}

TEST(Features, PayloadChangeDetected) {
    std::vector<CanFrame> w = {mk(0x100, 0x00), mk(0x100, 0xFF)};
    auto f = extract_features(w);
    for (int b = 0; b < 8; ++b) EXPECT_DOUBLE_EQ(f[26 + b], 1.0);  // every byte changed
    EXPECT_DOUBLE_EQ(f[37], 1.0);                                  // all 64 bits flipped
}

TEST(Features, EmptyWindowIsAllZeros) {
    auto f = extract_features({});
    for (double v : f) EXPECT_EQ(v, 0.0);
}

TEST(Features, NamesCountMatches) {
    EXPECT_EQ(feature_names().size(), static_cast<size_t>(kNumFeatures));
}
