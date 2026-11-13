#pragma once
// Zero-day (unknown attack) scoring, independent of the trained classifier.
//
// Idea: learn what NORMAL traffic looks like for a few statistics, then flag
// windows that deviate by more than k standard deviations.
//   - id_entropy     (feature 1):  floods/injection make one ID dominate -> entropy drops
//   - top_id_ratio   (feature 2):  same effect, seen from the other side
//   - mean_hamming   (feature 37): spoofed payloads jump more than real signals
// Needs NO attack labels, so it can catch attack types the classifier never saw.
// Baseline is learned from the first `calib` windows (assumed attack-free),
// using Welford's online algorithm (numerically stable running mean/variance).
#include "features.hpp"

#include <array>
#include <cmath>
#include <cstddef>

class ZeroDayScorer {
public:
    static constexpr std::array<int, 3> kFeatures = {1, 2, 37};

    explicit ZeroDayScorer(std::size_t calib_windows = 200, double k = 5.0)
        : calib_(calib_windows), k_(k) {}

    bool calibrated() const { return n_ >= calib_; }
    double threshold() const { return k_; }

    // Returns the anomaly score (max |z| over the tracked features).
    // While calibrating, updates the baseline and returns 0.
    double score(const FeatureVec& f) {
        if (!calibrated()) {
            ++n_;
            for (std::size_t j = 0; j < kFeatures.size(); ++j) {
                double x = f[kFeatures[j]];
                double d = x - mean_[j];
                mean_[j] += d / static_cast<double>(n_);
                m2_[j] += d * (x - mean_[j]);
            }
            return 0.0;
        }
        double worst = 0.0;
        for (std::size_t j = 0; j < kFeatures.size(); ++j) {
            double var = m2_[j] / static_cast<double>(n_ - 1);
            double sd = std::sqrt(var) + 1e-6;            // avoid divide-by-zero
            double z = std::fabs(f[kFeatures[j]] - mean_[j]) / sd;
            if (z > worst) worst = z;
        }
        return worst;
    }

    bool is_anomaly(double score) const { return calibrated() && score > k_; }

private:
    std::size_t calib_;
    double k_;
    std::size_t n_ = 0;
    std::array<double, 3> mean_{}, m2_{};
};
