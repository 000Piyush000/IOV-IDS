// Offline tool: replay.log -> features_cpp.csv using the SAME C++ code the
// agent runs. Used for the golden test against tools/features.py.
#include "features.hpp"
#include "file_source.hpp"
#include "sliding_window.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: feature_dump <replay.log> <out.csv> [window=100] [stride=50]\n";
        return 1;
    }
    std::size_t W = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 100;
    std::size_t S = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 50;
    FileSource src(argv[1]);
    SlidingWindow win(W, S);
    FILE* out = std::fopen(argv[2], "w");
    if (!out) { std::cerr << "cannot write " << argv[2] << "\n"; return 1; }

    std::fprintf(out, "end_frame");
    for (const auto& n : feature_names()) std::fprintf(out, ",%s", n.c_str());
    std::fprintf(out, "\n");

    std::size_t idx = 0, windows = 0;
    while (auto f = src.next()) {
        if (win.push(*f)) {
            auto feats = extract_features(win.snapshot());
            std::fprintf(out, "%zu", idx);
            for (double v : feats) std::fprintf(out, ",%.6f", v);
            std::fprintf(out, "\n");
            ++windows;
        }
        ++idx;
    }
    std::fclose(out);
    std::cout << "frames=" << idx << " windows=" << windows << " -> " << argv[2] << "\n";
    return 0;
}
