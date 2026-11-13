// Vehicle agent (Step 5): CAN source -> sliding window -> 39 features ->
// ONNX detector -> alerts to the edge server over TCP.
//
//   ./build/agent [vcanX] -i <vehicle_id> [-r obu|mec] [-S host:port]
//                 [-m models/ids.onnx] [-c models/classes.txt] [-w 100] [-s 50]
//                 [-f replay.log]   (read a log file instead of SocketCAN, for tests)
//                 [-z 5.0]          (zero-day threshold in standard deviations)
//
// Without -S the agent runs standalone (prints alerts locally).
#include "detector.hpp"
#include "features.hpp"
#include "file_source.hpp"
#include "latency.hpp"
#include "protocol.hpp"
#include "sliding_window.hpp"
#include "socketcan_source.hpp"
#include "tcp_client.hpp"
#include "zero_day.hpp"

#include <chrono>
#include <csignal>
#include <signal.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>

static volatile std::sig_atomic_t g_stop = 0;
static void on_sigint(int) { g_stop = 1; }

using Clock = std::chrono::steady_clock;
static uint64_t ns_since(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t).count();
}
static uint64_t wall_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

static uint16_t most_frequent_id(const std::vector<CanFrame>& w) {
    std::unordered_map<uint32_t, int> cnt;
    uint32_t best = 0; int bc = 0;
    for (const auto& f : w) {
        int c = ++cnt[f.id];
        if (c > bc) { bc = c; best = f.id; }
    }
    return static_cast<uint16_t>(best);
}

int main(int argc, char** argv) {
    std::string iface = "vcan0", model = "models/ids.onnx", classes = "models/classes.txt";
    std::string server, file;
    std::size_t W = 100, S = 50;
    double zk = 5.0;
    Hello hello;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (a == "-w") W = std::strtoul(val().c_str(), nullptr, 10);
        else if (a == "-s") S = std::strtoul(val().c_str(), nullptr, 10);
        else if (a == "-m") model = val();
        else if (a == "-c") classes = val();
        else if (a == "-S") server = val();
        else if (a == "-f") file = val();
        else if (a == "-z") zk = std::atof(val().c_str());
        else if (a == "-i") hello.vehicle_id = static_cast<uint16_t>(std::atoi(val().c_str()));
        else if (a == "-r") hello.role = (val() == "mec") ? Role::MEC : Role::OBU;
        else iface = a;
    }
    // sigaction WITHOUT SA_RESTART: Ctrl+C interrupts the blocking read()
    struct sigaction sa{};
    sa.sa_handler = on_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    try {
        Detector det(model, classes);
        std::unique_ptr<IFrameSource> src;
        if (!file.empty()) src = std::make_unique<FileSource>(file);
        else src = std::make_unique<SocketCanSource>(iface);

        std::unique_ptr<TcpClient> net;
        if (!server.empty()) {
            auto colon = server.find(':');
            std::string host = server.substr(0, colon);
            int port = colon == std::string::npos ? 9000 : std::atoi(server.c_str() + colon + 1);
            net = std::make_unique<TcpClient>(host, port, hello);
        }

        SlidingWindow win(W, S);
        ZeroDayScorer zday(200, zk);       // first 200 windows (~2.5 s) learn "normal"
        LatencyStats infer_lat, total_lat;
        for (int i = 0; i < 50; ++i) det.predict(FeatureVec{});   // warm-up
        std::printf("Calibrating zero-day baseline on the first 200 windows...\n");
        std::printf("Vehicle %u (%s) on %s  window=%zu stride=%zu  server=%s\n",
                    hello.vehicle_id, hello.role == Role::MEC ? "MEC" : "OBU",
                    file.empty() ? iface.c_str() : file.c_str(), W, S,
                    server.empty() ? "none" : server.c_str());

        uint64_t total = 0, windows = 0, alerts_total = 0;
        uint32_t frames_sec = 0, windows_sec = 0, alerts_sec_n = 0;
        uint64_t sec_start = 0;
        std::map<std::string, int> alerts_sec;
        while (!g_stop) {
            auto f = src->next();
            if (!f) break;
            ++total; ++frames_sec;
            if (sec_start == 0) sec_start = f->ts_us;

            if (win.push(*f)) {
                auto t0 = Clock::now();
                auto snap = win.snapshot();
                auto feats = extract_features(snap);
                auto t1 = Clock::now();
                auto d = det.predict(feats);
                infer_lat.add(ns_since(t1));
                total_lat.add(ns_since(t0));
                ++windows; ++windows_sec;
                double zscore = zday.score(feats);
                bool known_attack = d.cls != det.benign_index();
                // classifier says BENIGN but traffic is far from the learned normal
                // -> possible attack the model was never trained on
                bool zero_day = !known_attack && zday.is_anomaly(zscore);
                if (known_attack || zero_day) {
                    const std::string cls = known_attack ? det.classes()[d.cls] : "ZERO_DAY";
                    float conf = known_attack ? d.prob : static_cast<float>(zscore);
                    ++alerts_sec[cls]; ++alerts_total; ++alerts_sec_n;
                    if (net) {
                        Alert a;
                        a.vehicle_id = hello.vehicle_id;
                        a.ts_us = wall_us();
                        a.cls = cls;
                        a.prob = conf;          // ZERO_DAY: anomaly score (sigmas)
                        a.top_can_id = most_frequent_id(snap);
                        a.id_entropy = static_cast<float>(feats[1]);
                        net->send(encode(a));
                    }
                }
            }
            if (f->ts_us - sec_start >= 1'000'000) {
                if (net) {
                    Stats st{hello.vehicle_id, frames_sec, windows_sec, alerts_sec_n};
                    net->send(encode(st));
                }
                std::printf("frames/s=%-6u windows=%-6llu ", frames_sec, (unsigned long long)windows);
                if (alerts_sec.empty()) std::printf("status=NORMAL\n");
                else {
                    std::printf("ALERT ");
                    for (auto& [c, n] : alerts_sec) std::printf("%s=%d ", c.c_str(), n);
                    std::printf("\n");
                }
                alerts_sec.clear();
                frames_sec = windows_sec = alerts_sec_n = 0;
                sec_start = f->ts_us;
            }
        }
        if (net && frames_sec) {
            Stats st{hello.vehicle_id, frames_sec, windows_sec, alerts_sec_n};
            net->send(encode(st));
        }
        std::printf("\nStopped. frames=%llu windows=%llu alert_windows=%llu\n",
                    (unsigned long long)total, (unsigned long long)windows,
                    (unsigned long long)alerts_total);
        infer_lat.print("ONNX inference");
        total_lat.print("features+infer");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
