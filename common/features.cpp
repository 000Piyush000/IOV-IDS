#include "features.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace {
double entropy_from_counts(const std::unordered_map<uint32_t, int>& counts, int total) {
    if (total == 0) return 0.0;
    double h = 0.0;
    for (const auto& kv : counts) {
        double p = static_cast<double>(kv.second) / total;
        h -= p * std::log2(p);
    }
    return h;
}
}  // namespace

const std::array<std::string, kNumFeatures>& feature_names() {
    static const std::array<std::string, kNumFeatures> names = {
        "n_unique_ids", "id_entropy", "top_id_ratio", "frac_id_zero",
        "mean_id", "std_id", "min_id", "max_id", "id_change_rate", "mean_dlc",
        "mean_b0", "mean_b1", "mean_b2", "mean_b3", "mean_b4", "mean_b5", "mean_b6", "mean_b7",
        "std_b0", "std_b1", "std_b2", "std_b3", "std_b4", "std_b5", "std_b6", "std_b7",
        "chg_b0", "chg_b1", "chg_b2", "chg_b3", "chg_b4", "chg_b5", "chg_b6", "chg_b7",
        "payload_entropy", "frac_zero_bytes", "frac_ff_bytes", "mean_hamming",
        "max_repeat_run"};
    return names;
}

FeatureVec extract_features(const std::vector<CanFrame>& w) {
    FeatureVec f{};
    const int n = static_cast<int>(w.size());
    if (n == 0) return f;
    constexpr double kMaxId = 2047.0;   // 11-bit ID range

    // ---- ID-based features (DoS & injection change the ID distribution)
    std::unordered_map<uint32_t, int> id_cnt;
    double id_sum = 0, id_sq = 0, dlc_sum = 0;
    uint32_t id_min = w[0].id, id_max = w[0].id;
    int changes = 0, run = 1, max_run = 1;
    for (int i = 0; i < n; ++i) {
        const auto& fr = w[i];
        ++id_cnt[fr.id];
        double x = fr.id / kMaxId;
        id_sum += x; id_sq += x * x; dlc_sum += fr.dlc;
        id_min = std::min(id_min, fr.id); id_max = std::max(id_max, fr.id);
        if (i > 0) {
            if (fr.id != w[i - 1].id) { ++changes; run = 1; }
            else { ++run; max_run = std::max(max_run, run); }
        }
    }
    int top = 0;
    for (const auto& kv : id_cnt) top = std::max(top, kv.second);
    double id_mean = id_sum / n;

    f[0] = static_cast<double>(id_cnt.size());
    f[1] = entropy_from_counts(id_cnt, n);
    f[2] = static_cast<double>(top) / n;
    f[3] = id_cnt.count(0) ? static_cast<double>(id_cnt[0]) / n : 0.0;
    f[4] = id_mean;
    f[5] = std::sqrt(std::max(0.0, id_sq / n - id_mean * id_mean));
    f[6] = id_min / kMaxId;
    f[7] = id_max / kMaxId;
    f[8] = n > 1 ? static_cast<double>(changes) / (n - 1) : 0.0;
    f[9] = dlc_sum / n / 8.0;

    // ---- Per-byte payload statistics (spoofing changes the values)
    for (int b = 0; b < 8; ++b) {
        double s = 0, sq = 0; int cnt = 0;
        for (const auto& fr : w)
            if (fr.dlc > b) { double v = fr.data[b] / 255.0; s += v; sq += v * v; ++cnt; }
        double mean = cnt ? s / cnt : 0.0;
        f[10 + b] = mean;
        f[18 + b] = cnt ? std::sqrt(std::max(0.0, sq / cnt - mean * mean)) : 0.0;
    }

    // ---- Payload volatility: compare each frame with the previous frame of the SAME ID
    std::unordered_map<uint32_t, const CanFrame*> last;
    int comparisons = 0, bit_diffs = 0;
    std::array<int, 8> byte_changes{};
    for (const auto& fr : w) {
        auto it = last.find(fr.id);
        if (it != last.end()) {
            ++comparisons;
            for (int b = 0; b < 8; ++b) {
                uint8_t x = fr.data[b] ^ it->second->data[b];
                if (x) ++byte_changes[b];
                bit_diffs += __builtin_popcount(x);
            }
        }
        last[fr.id] = &fr;
    }
    for (int b = 0; b < 8; ++b)
        f[26 + b] = comparisons ? static_cast<double>(byte_changes[b]) / comparisons : 0.0;

    // ---- Whole-payload byte distribution
    std::unordered_map<uint32_t, int> byte_cnt;
    int nbytes = 0, zeros = 0, ffs = 0;
    for (const auto& fr : w)
        for (int b = 0; b < fr.dlc && b < 8; ++b) {
            ++byte_cnt[fr.data[b]]; ++nbytes;
            if (fr.data[b] == 0x00) ++zeros;
            if (fr.data[b] == 0xFF) ++ffs;
        }
    f[34] = entropy_from_counts(byte_cnt, nbytes);
    f[35] = nbytes ? static_cast<double>(zeros) / nbytes : 0.0;
    f[36] = nbytes ? static_cast<double>(ffs) / nbytes : 0.0;
    f[37] = comparisons ? static_cast<double>(bit_diffs) / (64.0 * comparisons) : 0.0;
    f[38] = static_cast<double>(max_run) / n;
    return f;
}
