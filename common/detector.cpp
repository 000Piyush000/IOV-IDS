#include "detector.hpp"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <fstream>
#include <stdexcept>

struct Detector::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "iov-ids"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::string in_name, out_name;
    std::array<float, kNumFeatures> input{};         // reused buffer: no allocation per call
    std::array<int64_t, 2> shape{1, kNumFeatures};
};

Detector::Detector(const std::string& model_path, const std::string& classes_path)
    : impl_(std::make_unique<Impl>()) {
    // 1 thread: for a tiny model, thread hand-off costs more than it saves,
    // and a single thread gives predictable (low-jitter) latency.
    impl_->opts.SetIntraOpNumThreads(1);
    impl_->opts.SetInterOpNumThreads(1);
    impl_->opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), impl_->opts);

    Ort::AllocatorWithDefaultOptions alloc;
    impl_->in_name = impl_->session->GetInputNameAllocated(0, alloc).get();
    impl_->out_name = impl_->session->GetOutputNameAllocated(0, alloc).get();

    std::ifstream in(classes_path);
    if (!in) throw std::runtime_error("cannot open " + classes_path);
    for (std::string line; std::getline(in, line);)
        if (!line.empty()) classes_.push_back(line);
    for (std::size_t i = 0; i < classes_.size(); ++i)
        if (classes_[i] == "BENIGN") benign_idx_ = static_cast<int>(i);
}

Detector::~Detector() = default;

Detection Detector::predict(const FeatureVec& f) {
    for (int i = 0; i < kNumFeatures; ++i) impl_->input[i] = static_cast<float>(f[i]);
    Ort::Value in = Ort::Value::CreateTensor<float>(impl_->mem, impl_->input.data(),
                                                    impl_->input.size(), impl_->shape.data(), 2);
    const char* in_names[] = {impl_->in_name.c_str()};
    const char* out_names[] = {impl_->out_name.c_str()};
    auto out = impl_->session->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);

    const float* p = out[0].GetTensorData<float>();
    std::size_t n = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
    Detection d;
    d.probs.assign(p, p + n);
    for (std::size_t i = 1; i < n; ++i)
        if (p[i] > p[d.cls]) d.cls = static_cast<int>(i);
    d.prob = p[d.cls];
    return d;
}
