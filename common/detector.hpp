#pragma once
// Wraps ONNX Runtime: 39 raw features in -> class probabilities out.
// Normalization and softmax live inside the ONNX graph.
#include "features.hpp"

#include <memory>
#include <string>
#include <vector>

struct Detection {
    int cls = 0;              // index into classes()
    float prob = 0.f;         // probability of that class
    std::vector<float> probs; // all class probabilities
};

class Detector {
public:
    Detector(const std::string& model_path, const std::string& classes_path);
    ~Detector();
    Detection predict(const FeatureVec& f);
    const std::vector<std::string>& classes() const { return classes_; }
    int benign_index() const { return benign_idx_; }
private:
    struct Impl;                       // PIMPL: keeps ONNX Runtime headers
    std::unique_ptr<Impl> impl_;       // out of every file that includes this one
    std::vector<std::string> classes_;
    int benign_idx_ = -1;
};
