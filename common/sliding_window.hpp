#pragma once
// Fixed-size ring buffer over the most recent W frames.
// push() is O(1). Every `stride` frames (once full) it reports "window ready";
// snapshot() then copies the W frames out in arrival order (O(W)).
#include "frame_source.hpp"
#include <cstddef>
#include <vector>

class SlidingWindow {
public:
    SlidingWindow(std::size_t size, std::size_t stride)
        : buf_(size), size_(size), stride_(stride) {}

    // Returns true when a new full window is ready to be featurized.
    bool push(const CanFrame& f) {
        buf_[head_] = f;
        head_ = (head_ + 1) % size_;          // wrap around: ring buffer
        if (count_ < size_) ++count_;
        ++since_last_;
        if (count_ == size_ && since_last_ >= stride_) {
            since_last_ = 0;
            return true;
        }
        return false;
    }

    // Oldest -> newest. head_ points at the oldest element once full.
    std::vector<CanFrame> snapshot() const {
        std::vector<CanFrame> out;
        out.reserve(count_);
        std::size_t start = (count_ == size_) ? head_ : 0;
        for (std::size_t i = 0; i < count_; ++i) out.push_back(buf_[(start + i) % size_]);
        return out;
    }

    std::size_t size() const { return size_; }
    std::size_t count() const { return count_; }

private:
    std::vector<CanFrame> buf_;
    std::size_t size_, stride_;
    std::size_t head_ = 0, count_ = 0, since_last_ = 0;
};
