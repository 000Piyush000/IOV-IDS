#include "zero_day.hpp"
#include "logreg.hpp"
#include <gtest/gtest.h>
#include <random>

static FeatureVec normal(std::mt19937& rng) {
    std::normal_distribution<double> ent(5.0, 0.1), top(0.05, 0.005), ham(0.2, 0.01);
    FeatureVec f{};
    f[1] = ent(rng); f[2] = top(rng); f[37] = ham(rng);
    return f;
}

TEST(ZeroDay, NoAlertsDuringCalibration) {
    ZeroDayScorer z(50, 5.0);
    std::mt19937 rng(1);
    for (int i = 0; i < 50; ++i) EXPECT_EQ(z.score(normal(rng)), 0.0);
    EXPECT_TRUE(z.calibrated());
}

TEST(ZeroDay, NormalTrafficStaysBelowThreshold) {
    ZeroDayScorer z(200, 5.0);
    std::mt19937 rng(2);
    for (int i = 0; i < 200; ++i) z.score(normal(rng));
    int flagged = 0;
    for (int i = 0; i < 1000; ++i) flagged += z.is_anomaly(z.score(normal(rng)));
    EXPECT_LE(flagged, 2);    // 5-sigma: essentially no false positives on normal data
}

TEST(ZeroDay, FloodIsFlagged) {
    ZeroDayScorer z(200, 5.0);
    std::mt19937 rng(3);
    for (int i = 0; i < 200; ++i) z.score(normal(rng));
    FeatureVec flood{};
    flood[1] = 1.0; flood[2] = 0.5; flood[37] = 0.2;   // one ID dominates: entropy collapses
    EXPECT_TRUE(z.is_anomaly(z.score(flood)));
}

TEST(LogReg, LearnsSeparableData) {
    std::vector<Sample> d;
    for (int i = 0; i < 200; ++i) {
        Sample s; s.x = {i % 2 ? 2.0f : -2.0f}; s.y = i % 2; s.cls = s.y ? "A" : "BENIGN";
        d.push_back(s);
    }
    LogReg m(1);
    std::mt19937 rng(4);
    for (int e = 0; e < 5; ++e) m.train_epoch(d, 0.1f, 1.0f, rng);
    EXPECT_GT(evaluate(m, d).acc, 0.99);
}
