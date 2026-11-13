#pragma once
// Collects latency samples (nanoseconds) and reports percentiles.
// p99 matters more than the average in real-time systems: it tells you
// the worst latency 1 in 100 windows will see.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

class LatencyStats {
public:
    void add(uint64_t ns) { s_.push_back(ns); }
    std::size_t count() const { return s_.size(); }
    double percentile_us(double p) const {
        if (s_.empty()) return 0.0;
        std::vector<uint64_t> v = s_;
        std::size_t k = static_cast<std::size_t>(p / 100.0 * (v.size() - 1));
        std::nth_element(v.begin(), v.begin() + k, v.end());   // O(n), no full sort
        return v[k] / 1000.0;
    }
    double max_us() const { return s_.empty() ? 0.0 : *std::max_element(s_.begin(), s_.end()) / 1000.0; }
    void print(const char* name) const {
        std::printf("%-18s n=%-7zu p50=%8.2f us  p99=%8.2f us  max=%8.2f us\n", name, count(),
                    percentile_us(50), percentile_us(99), max_us());
    }
private:
    std::vector<uint64_t> s_;
};
