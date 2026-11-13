// Federated-learning client (Step 6). One process = one vehicle.
//
//   ./build/fl_client -i 3 [-r obu|mec] -S 127.0.0.1:9000
//        --train data/fl/client3.csv --test data/fl/test.csv [--epochs 2] [--lr 0.02]
//
// Each round:  receive GLOBAL model -> train locally on OWN data for E epochs
//              -> send updated parameters + sample count to the server.
// Raw CAN data / features never leave this process.
// For comparison it also trains a LOCAL-ONLY model (no collaboration) on the
// same data for the same number of epochs, and reports both on the test set.
#include "logreg.hpp"
#include "protocol.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>

static int connect_to(const std::string& host, int port) {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0) return -1;
    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    for (int attempt = 0; fd >= 0 && attempt < 20; ++attempt) {   // server may start later
        if (connect(fd, res->ai_addr, res->ai_addrlen) == 0) { freeaddrinfo(res); return fd; }
        usleep(250000);
    }
    freeaddrinfo(res);
    if (fd >= 0) close(fd);
    return -1;
}

static bool send_all(int fd, const std::vector<uint8_t>& b) {
    std::size_t off = 0;
    while (off < b.size()) {
        ssize_t n = send(fd, b.data() + off, b.size() - off, MSG_NOSIGNAL);
        if (n > 0) off += static_cast<std::size_t>(n);
        else if (n < 0 && errno == EINTR) continue;
        else return false;
    }
    return true;
}

int main(int argc, char** argv) {
    std::string server = "127.0.0.1:9000", train_path, test_path;
    Hello hello; hello.fl = 1;
    int epochs = 2; float lr = 0.02f;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (a == "-S") server = val();
        else if (a == "-i") hello.vehicle_id = static_cast<uint16_t>(std::atoi(val().c_str()));
        else if (a == "-r") hello.role = (val() == "mec") ? Role::MEC : Role::OBU;
        else if (a == "--train") train_path = val();
        else if (a == "--test") test_path = val();
        else if (a == "--epochs") epochs = std::atoi(val().c_str());
        else if (a == "--lr") lr = std::strtof(val().c_str(), nullptr);
    }
    if (train_path.empty()) { std::cerr << "need --train <client.csv>\n"; return 1; }

    try {
        auto train = load_samples(train_path);
        std::vector<Sample> test;
        if (!test_path.empty()) test = load_samples(test_path);
        std::size_t dim = train.at(0).x.size();
        std::size_t pos = 0;
        for (auto& s : train) pos += s.y;
        // balance classes: an attack sample counts as much as (benign/attack) benign samples
        float pos_weight = pos ? float(train.size() - pos) / float(pos) : 1.f;

        auto colon = server.find(':');
        int fd = connect_to(server.substr(0, colon), std::atoi(server.c_str() + colon + 1));
        if (fd < 0) { std::cerr << "cannot connect to " << server << "\n"; return 1; }
        send_all(fd, encode(hello));
        std::printf("[client %u] %zu local windows (%zu attack), waiting for round 1\n",
                    hello.vehicle_id, train.size(), pos);

        LogReg global_m(dim), local_only(dim);
        std::mt19937 rng(1000 + hello.vehicle_id), rng_local(2000 + hello.vehicle_id);
        StreamDecoder dec;
        uint8_t buf[65536];
        bool done = false;
        int rounds = 0;
        while (!done) {
            ssize_t n = recv(fd, buf, sizeof(buf), 0);
            if (n <= 0) { if (n < 0 && errno == EINTR) continue; break; }
            dec.feed(buf, static_cast<std::size_t>(n));
            while (auto m = dec.next()) {
                if (m->type != MsgType::GlobalModel) continue;
                auto g = decode_global_model(m->payload);
                if (!g || g->params.size() != dim + 1) continue;
                global_m.params() = g->params;
                if (g->round == GlobalModel::kFinal) { done = true; break; }

                for (int e = 0; e < epochs; ++e) {           // LOCAL training on own data
                    global_m.train_epoch(train, lr, pos_weight, rng);
                    local_only.train_epoch(train, lr, pos_weight, rng_local);
                }
                ++rounds;
                ModelUpdate u;
                u.vehicle_id = hello.vehicle_id;
                u.round = g->round;
                u.n_samples = static_cast<uint32_t>(train.size());
                u.params = global_m.params();
                if (!send_all(fd, encode(u))) { done = true; break; }
            }
        }
        close(fd);

        if (!test.empty()) {
            auto mg = evaluate(global_m, test), ml = evaluate(local_only, test);
            std::printf("[client %u] after %d rounds, on the shared test set (all attack types):\n",
                        hello.vehicle_id, rounds);
            std::printf("   local-only model : F1=%.4f  recall=%.4f  precision=%.4f\n", ml.f1, ml.recall, ml.precision);
            std::printf("   federated model  : F1=%.4f  recall=%.4f  precision=%.4f\n", mg.f1, mg.recall, mg.precision);
            std::printf("   per-attack detection (local-only -> federated):\n");
            auto rl = per_class_recall(local_only, test), rg = per_class_recall(global_m, test);
            for (std::size_t k = 0; k < rg.size(); ++k)
                std::printf("     %-16s %.3f -> %.3f\n", rg[k].first.c_str(), rl[k].second, rg[k].second);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
