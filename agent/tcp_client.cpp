#include "tcp_client.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>

TcpClient::TcpClient(std::string host, int port, Hello hello)
    : host_(std::move(host)), port_(port), hello_(hello) {
    try_connect();
}

TcpClient::~TcpClient() { disconnect(); }

void TcpClient::disconnect() {
    if (fd_ >= 0) { close(fd_); fd_ = -1; }
}

bool TcpClient::try_connect() {
    auto now = std::chrono::steady_clock::now();
    if (now < next_retry_) return false;                     // backoff
    next_retry_ = now + std::chrono::seconds(2);

    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host_.c_str(), std::to_string(port_).c_str(), &hints, &res) != 0) return false;
    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { freeaddrinfo(res); return false; }
    if (connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        freeaddrinfo(res);
        close(fd);
        return false;
    }
    freeaddrinfo(res);
    int one = 1;
    // small alert messages: send immediately, don't wait to batch (Nagle off)
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    fd_ = fd;
    std::printf("[net] connected to edge server %s:%d\n", host_.c_str(), port_);
    if (!send(encode(hello_))) return false;
    return true;
}

bool TcpClient::send(const std::vector<uint8_t>& bytes) {
    if (fd_ < 0 && !try_connect()) return false;
    std::size_t off = 0;
    while (off < bytes.size()) {                     // send() may write only part
        ssize_t n = ::send(fd_, bytes.data() + off, bytes.size() - off, MSG_NOSIGNAL);
        if (n > 0) { off += static_cast<std::size_t>(n); continue; }
        if (n < 0 && errno == EINTR) continue;
        std::printf("[net] lost connection to edge server, will retry\n");
        disconnect();
        return false;
    }
    return true;
}
