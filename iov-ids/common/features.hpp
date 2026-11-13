#pragma once
// 39 window-level features. MUST stay identical to tools/features.py
// (checked by the golden test) -- otherwise the model sees different
// inputs in production than in training ("train/serve skew").
#include "frame_source.hpp"
#include <array>
#include <string>
#include <vector>

constexpr int kNumFeatures = 39;
using FeatureVec = std::array<double, kNumFeatures>;

FeatureVec extract_features(const std::vector<CanFrame>& window);
const std::array<std::string, kNumFeatures>& feature_names();
