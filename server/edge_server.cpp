// Edge server (Step 5): one thread, epoll event loop, many vehicle agents.
//
//   ./build/edge_server [-p 9000] [-o reports/alerts.jsonl]
//                       [--fl-clients 10 --fl-rounds 20 --fl-test data/fl/test.csv]
//
// - Non-blocking listen socket + non-blocking client sockets.
// - epoll tells us WHICH sockets are ready, so we never block on one
//   vehicle while others are waiting (no thread per client needed).
// - Each connection has its own StreamDecoder (TCP is a byte stream).
// - Alerts are aggregated per vehicle/class and appended to a JSON-lines
//   file that the LLM reporter (Step 7) reads as evidence.
// - Federated learning coordinator (Step 6): once K FL clients have joined,
//   broadcast the global model, collect K local updates, FedAvg them
//   (weighted by sample count), evaluate, repeat for R rounds.
#include "logreg.hpp"
#include "protocol.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <poll.h>
#include <set>
#include <string>
#include <unordered_map>

static volatile std::sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

struct Conn {
    int fd = -1;
    std::string peer;
    StreamDecoder dec;
    int vehicle_id = -1;                 // known after HELLO
    Role role = Role::OBU;
    uint64_t frames = 0, windows = 0, alerts = 0;
    bool fl = false;                     // federated-learning participant
};

// Small server->client messages. Sockets are non-blocking, so if the kernel
// send buffer is momentarily full we wait (poll) briefly. A production server
// would keep a per-connection output queue and use EPOLLOUT instead.
static bool send_all(int fd, const std::vector<uint8_t>& b) {
    std::size_t off = 0;
    while (off < b.size()) {
        ssize_t n = send(fd, b.data() + off, b.size() - off, MSG_NOSIGNAL);
        if (n > 0) { off += static_cast<std::size_t>(n); continue; }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            pollfd p{fd, POLLOUT, 0};
            if (poll(&p, 1, 1000) <= 0) return false;
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        return false;
    }
    return true;
}

// FedAvg: w_global = sum_k (n_k / N) * w_k
static std::vector<float> fedavg(const std::map<int, ModelUpdate>& ups) {
    std::vector<double> acc;
    double total = 0;
    for (auto& [vid, u] : ups) {
        if (acc.empty()) acc.assign(u.params.size(), 0.0);
        if (u.params.size() != acc.size()) continue;
        for (std::size_t i = 0; i < acc.size(); ++i) acc[i] += double(u.n_samples) * u.params[i];
        total += u.n_samples;
    }
    std::vector<float> w(acc.size());
    for (std::size_t i = 0; i < acc.size(); ++i) w[i] = total > 0 ? float(acc[i] / total) : 0.f;
    return w;
}

static bool set_nonblocking(int fd) {
    int fl = fcntl(fd, F_GETFL, 0);
    return fl >= 0 && fcntl(fd, F_SETFL, fl | O_NONBLOCK) == 0;
}

int main(int argc, char** argv) {
    int port = 9000;
    std::string out_path = "reports/alerts.jsonl", fl_test_path, fl_out = "models/fedavg_head.txt";
    int fl_k = 0, fl_rounds = 20;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-p") && i + 1 < argc) port = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-o") && i + 1 < argc) out_path = argv[++i];
        else if (!std::strcmp(argv[i], "--fl-clients") && i + 1 < argc) fl_k = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--fl-rounds") && i + 1 < argc) fl_rounds = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--fl-test") && i + 1 < argc) fl_test_path = argv[++i];
    }
    // ---- federated-learning state
    std::vector<Sample> fl_test;
    std::size_t fl_dim = 40;
    if (fl_k > 0 && !fl_test_path.empty()) {
        fl_test = load_samples(fl_test_path);
        fl_dim = fl_test.at(0).x.size() + 1;
    }
    std::vector<float> global_w(fl_dim, 0.f);
    uint32_t fl_round = 0;                     // 0 = not started
    std::map<int, ModelUpdate> fl_updates;     // fd -> update for current round
    struct sigaction sa{};
    sa.sa_handler = on_sig;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    signal(SIGPIPE, SIG_IGN);   // writing to a closed socket must not kill the server

    // ---- listening socket
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));  // quick restart
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (bind(lfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0 || listen(lfd, 128) < 0) {
        std::perror("bind/listen");
        return 1;
    }
    set_nonblocking(lfd);

    // ---- epoll instance; level-triggered (default): simple and safe
    int ep = epoll_create1(0);
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.fd = lfd;
    epoll_ctl(ep, EPOLL_CTL_ADD, lfd, &ev);

    FILE* out = std::fopen(out_path.c_str(), "a");
    if (!out) { std::perror(("open " + out_path).c_str()); return 1; }

    std::unordered_map<int, Conn> conns;                  // fd -> connection
    std::map<std::string, uint64_t> alerts_sec;           // class -> count this second
    std::map<int, uint64_t> vehicle_alerts_sec;           // vehicle -> count this second
    uint64_t frames_sec = 0, msgs_total = 0, alerts_total = 0;
    auto last_report = std::chrono::steady_clock::now();
    std::printf("Edge server listening on port %d, alerts -> %s (Ctrl+C to stop)\n",
                port, out_path.c_str());
    if (fl_k > 0)
        std::printf("Federated learning: waiting for %d clients, %d rounds\n", fl_k, fl_rounds);

    std::set<int> fl_fds;                  // connections taking part in FL

    auto broadcast_global = [&](uint32_t round) {
        GlobalModel g; g.round = round; g.params = global_w;
        auto bytes = encode(g);
        for (int fd : fl_fds) send_all(fd, bytes);
    };

    // Called when an update arrives or a participant leaves: if every
    // connected participant has sent its update, aggregate and move on.
    auto maybe_finish_round = [&]() {
        if (fl_round == 0 || fl_fds.empty() || fl_updates.size() < fl_fds.size()) return;
        global_w = fedavg(fl_updates);
        uint64_t n = 0;
        for (auto& [fd, u] : fl_updates) n += u.n_samples;
        std::printf("[FL] round %2u: FedAvg of %zu updates (%llu samples)", fl_round,
                    fl_updates.size(), (unsigned long long)n);
        if (!fl_test.empty()) {
            LogReg m(fl_dim - 1);
            m.params() = global_w;
            auto r = evaluate(m, fl_test);
            std::printf("  global model on test: F1=%.4f recall=%.4f precision=%.4f", r.f1, r.recall, r.precision);
        }
        std::printf("\n");
        fl_updates.clear();
        if (fl_round >= static_cast<uint32_t>(fl_rounds)) {
            broadcast_global(GlobalModel::kFinal);
            std::ofstream mo(fl_out);
            for (float v : global_w) mo << v << "\n";
            std::printf("[FL] training finished after %u rounds -> %s\n", fl_round, fl_out.c_str());
            if (!fl_test.empty()) {
                LogReg m(fl_dim - 1);
                m.params() = global_w;
                std::printf("[FL] per-attack detection of the global model:\n");
                for (auto& [cls, rec] : per_class_recall(m, fl_test))
                    std::printf("       %-16s %.3f\n", cls.c_str(), rec);
            }
            fl_round = 0;
            fl_k = 0;                          // don't start again
        } else {
            broadcast_global(++fl_round);
        }
    };

    auto close_conn = [&](int fd, const char* why) {
        auto& c = conns[fd];
        if (fl_fds.erase(fd)) {
            fl_updates.erase(fd);
            std::printf("[FL] participant %d left; %zu remain\n", c.vehicle_id, fl_fds.size());
        }
        std::printf("[-] vehicle %d (%s) disconnected: %s\n", c.vehicle_id, c.peer.c_str(), why);
        epoll_ctl(ep, EPOLL_CTL_DEL, fd, nullptr);
        close(fd);
        conns.erase(fd);
    };

    constexpr int kMaxEvents = 64;
    epoll_event events[kMaxEvents];
    while (!g_stop) {
        // timeout 200 ms so the 1-second report still prints when nobody sends
        int n = epoll_wait(ep, events, kMaxEvents, 200);
        if (n < 0) {
            if (errno == EINTR) continue;
            std::perror("epoll_wait");
            break;
        }
        for (int i = 0; i < n; ++i) {
            int fd = events[i].data.fd;

            if (fd == lfd) {                                  // new connection(s)
                while (true) {
                    sockaddr_in ca{};
                    socklen_t cl = sizeof(ca);
                    int cfd = accept(lfd, reinterpret_cast<sockaddr*>(&ca), &cl);
                    if (cfd < 0) break;                       // EAGAIN: no more pending
                    set_nonblocking(cfd);
                    setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
                    char ip[INET_ADDRSTRLEN];
                    inet_ntop(AF_INET, &ca.sin_addr, ip, sizeof(ip));
                    Conn c;
                    c.fd = cfd;
                    c.peer = std::string(ip) + ":" + std::to_string(ntohs(ca.sin_port));
                    conns.emplace(cfd, std::move(c));
                    epoll_event cev{};
                    cev.events = EPOLLIN | EPOLLRDHUP;
                    cev.data.fd = cfd;
                    epoll_ctl(ep, EPOLL_CTL_ADD, cfd, &cev);
                }
                continue;
            }

            auto it = conns.find(fd);
            if (it == conns.end()) continue;
            Conn& c = it->second;

            // read everything available right now (non-blocking)
            bool closed = false;
            uint8_t buf[16384];
            while (true) {
                ssize_t r = recv(fd, buf, sizeof(buf), 0);
                if (r > 0) { c.dec.feed(buf, static_cast<std::size_t>(r)); continue; }
                if (r == 0) { closed = true; break; }                       // peer closed
                if (errno == EAGAIN || errno == EWOULDBLOCK) break;         // drained
                if (errno == EINTR) continue;
                closed = true; break;                                        // real error
            }

            // process every COMPLETE message; partial ones stay buffered
            while (auto m = c.dec.next()) {
                ++msgs_total;
                switch (m->type) {
                case MsgType::Hello:
                    if (auto h = decode_hello(m->payload)) {
                        c.vehicle_id = h->vehicle_id;
                        c.role = h->role;
                        c.fl = h->fl != 0;
                        std::printf("[+] vehicle %d connected from %s (%s), %zu online\n",
                                    c.vehicle_id, c.peer.c_str(),
                                    c.role == Role::MEC ? "MEC" : "OBU", conns.size());
                        if (c.fl && fl_k > 0) {
                            fl_fds.insert(fd);
                            if (fl_round == 0 && static_cast<int>(fl_fds.size()) >= fl_k) {
                                std::printf("[FL] %zu clients joined -> starting round 1\n", fl_fds.size());
                                broadcast_global(fl_round = 1);
                            }
                        }
                    }
                    break;
                case MsgType::ModelUpdate:
                    if (auto u = decode_model_update(m->payload)) {
                        if (fl_round != 0 && u->round == fl_round && fl_fds.count(fd) &&
                            u->params.size() == global_w.size()) {
                            fl_updates[fd] = std::move(*u);
                            maybe_finish_round();
                        }
                    }
                    break;
                case MsgType::Alert:
                    if (auto a = decode_alert(m->payload)) {
                        ++c.alerts; ++alerts_total;
                        ++alerts_sec[a->cls];
                        ++vehicle_alerts_sec[a->vehicle_id];
                        std::fprintf(out,
                            "{\"ts_us\":%llu,\"vehicle\":%u,\"role\":\"%s\",\"class\":\"%s\","
                            "\"prob\":%.4f,\"top_can_id\":\"%03X\",\"id_entropy\":%.4f}\n",
                            (unsigned long long)a->ts_us, a->vehicle_id,
                            c.role == Role::MEC ? "MEC" : "OBU", a->cls.c_str(),
                            a->prob, a->top_can_id, a->id_entropy);
                    }
                    break;
                case MsgType::Stats:
                    if (auto s = decode_stats(m->payload)) {
                        c.frames += s->frames; c.windows += s->windows;
                        frames_sec += s->frames;
                    }
                    break;
                default:
                    break;   // ModelUpdate etc. handled in Step 6
                }
            }
            if (c.dec.error()) { close_conn(fd, "protocol error"); continue; }
            if (closed || (events[i].events & (EPOLLRDHUP | EPOLLHUP | EPOLLERR))) {
                close_conn(fd, "connection closed");
                maybe_finish_round();          // a leaver must not stall the round
            }
        }

        // ---- once per second: fleet summary
        auto now = std::chrono::steady_clock::now();
        if (now - last_report >= std::chrono::seconds(1)) {
            last_report = now;
            std::fflush(out);
            std::printf("vehicles=%zu fleet_frames/s=%llu ", conns.size(),
                        (unsigned long long)frames_sec);
            if (alerts_sec.empty()) std::printf("status=NORMAL\n");
            else {
                std::printf("ALERTS ");
                for (auto& [cls, k] : alerts_sec) std::printf("%s=%llu ", cls.c_str(), (unsigned long long)k);
                std::printf("| vehicles:");
                for (auto& [v, k] : vehicle_alerts_sec) std::printf(" v%d=%llu", v, (unsigned long long)k);
                std::printf("\n");
            }
            alerts_sec.clear();
            vehicle_alerts_sec.clear();
            frames_sec = 0;
        }
    }

    std::printf("\nShutting down. messages=%llu alerts=%llu (written to %s)\n",
                (unsigned long long)msgs_total, (unsigned long long)alerts_total, out_path.c_str());
    for (auto& [fd, c] : conns) close(fd);
    close(ep);
    close(lfd);
    std::fclose(out);
    return 0;
}
