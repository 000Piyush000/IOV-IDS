// Step 4/6 benchmark + evaluation, using exactly the agent's C++ code path:
// replay.log -> windows -> features -> ONNX classifier + zero-day scorer.
//
//   ./build/bench_infer data/replay.log models/ids.onnx models/classes.txt [data/features_py.csv]
//
// With the optional features_py.csv (same windows, carries the true labels),
// it prints per-class detection: by the classifier, by the zero-day scorer,
// and by either. Run it on a model trained with --exclude SPEED to see how
// an attack the model has NEVER seen is still caught.
#include "detector.hpp"
#include "features.hpp"
#include "file_source.hpp"
#include "latency.hpp"
#include "sliding_window.hpp"
#include "zero_day.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

using Clock = std::chrono::steady_clock;
static uint64_t ns_since(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t).count();
}

static std::vector<std::string> load_labels(const std::string& path) {
    std::ifstream in(path);
    std::vector<std::string> out;
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();   // CSV written with \r\n
        auto p = line.rfind(',');
        if (p != std::string::npos) out.push_back(line.substr(p + 1));
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: bench_infer <replay.log> <model.onnx> <classes.txt> [features_py.csv]\n";
        return 1;
    }
    try {
        FileSource src(argv[1]);
        Detector det(argv[2], argv[3]);
        std::vector<std::string> labels;
        if (argc > 4) labels = load_labels(argv[4]);
        SlidingWindow win(100, 50);
        ZeroDayScorer zday(200, 5.0);
        LatencyStats feat_lat, infer_lat, total_lat;
        std::map<std::string, std::size_t> hist;
        struct Row { std::size_t n = 0, clf = 0, zd = 0, any = 0; };
        std::map<std::string, Row> table;

        for (int i = 0; i < 50; ++i) det.predict(FeatureVec{});   // warm-up

        std::size_t frames = 0, w = 0;
        auto t_all = Clock::now();
        while (auto f = src.next()) {
            ++frames;
            if (!win.push(*f)) continue;
            auto t0 = Clock::now();
            auto feats = extract_features(win.snapshot());
            uint64_t t_feat = ns_since(t0);
            auto t1 = Clock::now();
            auto d = det.predict(feats);
            uint64_t t_inf = ns_since(t1);
            double z = zday.score(feats);
            feat_lat.add(t_feat); infer_lat.add(t_inf); total_lat.add(t_feat + t_inf);
            bool clf = d.cls != det.benign_index();
            bool zd = !clf && zday.is_anomaly(z);
            ++hist[clf ? det.classes()[d.cls] : (zd ? "ZERO_DAY" : "BENIGN")];
            if (w < labels.size() && zday.calibrated()) {
                auto& r = table[labels[w]];
                ++r.n; r.clf += clf; r.zd += zday.is_anomaly(z); r.any += (clf || zday.is_anomaly(z));
            }
            ++w;
        }
        double secs = ns_since(t_all) / 1e9;

        std::printf("frames=%zu windows=%zu  offline throughput=%.0f frames/s\n\n",
                    frames, total_lat.count(), frames / secs);
        feat_lat.print("feature extract");
        infer_lat.print("ONNX inference");
        total_lat.print("total per window");
        std::printf("\nPredicted windows (classifier, else zero-day):\n");
        for (auto& [c, n] : hist) std::printf("  %-16s %zu\n", c.c_str(), n);

        if (!table.empty()) {
            std::printf("\nDetection rate per TRUE class (after zero-day calibration):\n");
            std::printf("  %-16s %7s %12s %12s %12s\n", "class", "windows", "classifier", "zero-day", "either");
            for (auto& [c, r] : table) {
                bool benign = (c == "BENIGN");
                std::printf("  %-16s %7zu %11.1f%% %11.1f%% %11.1f%%%s\n", c.c_str(), r.n,
                            100.0 * r.clf / r.n, 100.0 * r.zd / r.n, 100.0 * r.any / r.n,
                            benign ? "   <- false-positive rate" : "");
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
