#pragma once
// Tiny logistic-regression model used for FEDERATED learning:
//   p(attack | x) = sigmoid(w . x + b)
// Small on purpose: each vehicle trains it locally with SGD and only the
// 40 parameters (39 weights + bias) travel over the network -- never raw CAN data.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Sample {
    std::vector<float> x;   // standardized features
    int y = 0;              // 1 = attack, 0 = benign
    std::string cls;        // original class name (for per-class reporting)
};

// CSV written by tools/fl_partition.py: class,y,f0,...,f38 (header line first)
inline std::vector<Sample> load_samples(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<Sample> out;
    std::string line;
    std::getline(in, line);  // header
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        Sample s;
        std::getline(ss, s.cls, ',');
        std::getline(ss, tok, ',');
        s.y = std::stoi(tok);
        while (std::getline(ss, tok, ',')) s.x.push_back(std::stof(tok));
        out.push_back(std::move(s));
    }
    return out;
}

class LogReg {
public:
    explicit LogReg(std::size_t dim = 0) : p_(dim + 1, 0.f) {}    // params = [w..., b]
    std::size_t dim() const { return p_.size() - 1; }
    std::vector<float>& params() { return p_; }
    const std::vector<float>& params() const { return p_; }

    float prob(const std::vector<float>& x) const {
        double z = p_.back();
        for (std::size_t i = 0; i < x.size(); ++i) z += p_[i] * x[i];
        return static_cast<float>(1.0 / (1.0 + std::exp(-z)));
    }

    // One epoch of SGD on binary cross-entropy. pos_weight > 1 up-weights the
    // rare attack class (class imbalance). L2 keeps weights small.
    void train_epoch(const std::vector<Sample>& data, float lr, float pos_weight,
                     std::mt19937& rng, float l2 = 1e-4f) {
        std::vector<std::size_t> idx(data.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);
        for (std::size_t k : idx) {
            const auto& s = data[k];
            float g = prob(s.x) - static_cast<float>(s.y);     // dLoss/dz
            if (s.y == 1) g *= pos_weight;
            for (std::size_t i = 0; i < s.x.size(); ++i) p_[i] -= lr * (g * s.x[i] + l2 * p_[i]);
            p_.back() -= lr * g;
        }
    }

private:
    std::vector<float> p_;
};

struct BinaryMetrics { double acc = 0, precision = 0, recall = 0, f1 = 0; };

inline BinaryMetrics evaluate(const LogReg& m, const std::vector<Sample>& data, float thr = 0.5f) {
    std::size_t tp = 0, fp = 0, tn = 0, fn = 0;
    for (const auto& s : data) {
        bool pred = m.prob(s.x) >= thr;
        if (pred && s.y) ++tp; else if (pred) ++fp; else if (s.y) ++fn; else ++tn;
    }
    BinaryMetrics r;
    std::size_t n = tp + fp + tn + fn;
    r.acc = n ? double(tp + tn) / n : 0;
    r.precision = (tp + fp) ? double(tp) / (tp + fp) : 0;
    r.recall = (tp + fn) ? double(tp) / (tp + fn) : 0;
    r.f1 = (r.precision + r.recall) > 0 ? 2 * r.precision * r.recall / (r.precision + r.recall) : 0;
    return r;
}

// Detection rate per attack class (how many windows of that class were flagged).
inline std::vector<std::pair<std::string, double>> per_class_recall(const LogReg& m,
                                                                    const std::vector<Sample>& data) {
    std::vector<std::string> names;
    std::vector<std::size_t> hit, tot;
    for (const auto& s : data) {
        if (!s.y) continue;
        auto it = std::find(names.begin(), names.end(), s.cls);
        std::size_t k = static_cast<std::size_t>(it - names.begin());
        if (it == names.end()) { names.push_back(s.cls); hit.push_back(0); tot.push_back(0); }
        ++tot[k];
        if (m.prob(s.x) >= 0.5f) ++hit[k];
    }
    std::vector<std::pair<std::string, double>> out;
    for (std::size_t k = 0; k < names.size(); ++k) out.push_back({names[k], double(hit[k]) / tot[k]});
    std::sort(out.begin(), out.end());
    return out;
}
