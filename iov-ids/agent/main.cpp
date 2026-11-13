// Vehicle agent (Step 3): read CAN frames from vcan0, keep a sliding window,
// compute 39 features per window. Prints a live summary every second.
// Next steps add: ONNX inference (Step 4) and alerts to the edge server (Step 5).
#include "features.hpp"
#include "sliding_window.hpp"
#include "socketcan_source.hpp"

#include <csignal>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>

static volatile std::sig_atomic_t g_stop = 0;
static void on_sigint(int) { g_stop = 1; }

int main(int argc, char** argv) {
    std::string iface = "vcan0";
    std::size_t W = 100, S = 50;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-w") && i + 1 < argc) W = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) S = std::strtoul(argv[++i], nullptr, 10);
        else iface = argv[i];
    }
    std::signal(SIGINT, on_sigint);

    try {
        SocketCanSource src(iface);
        SlidingWindow win(W, S);
        std::printf("Listening on %s  window=%zu stride=%zu (Ctrl+C to stop)\n", iface.c_str(), W, S);

        uint64_t total = 0, in_sec = 0, sec_start = 0, windows = 0;
        FeatureVec lastf{};
        while (!g_stop) {
            auto f = src.next();
            if (!f) break;
            ++total; ++in_sec;
            if (sec_start == 0) sec_start = f->ts_us;

            if (win.push(*f)) {
                lastf = extract_features(win.snapshot());
                ++windows;
            }
            if (f->ts_us - sec_start >= 1'000'000) {
                // id_entropy drops and frac_id_zero jumps during a DoS flood
                std::printf("frames/s=%-6llu windows=%-6llu id_entropy=%.2f top_id_ratio=%.2f "
                            "frac_id_zero=%.2f mean_hamming=%.3f\n",
                            (unsigned long long)in_sec, (unsigned long long)windows,
                            lastf[1], lastf[2], lastf[3], lastf[37]);
                in_sec = 0;
                sec_start = f->ts_us;
            }
        }
        std::printf("\nStopped. frames=%llu windows=%llu\n",
                    (unsigned long long)total, (unsigned long long)windows);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
