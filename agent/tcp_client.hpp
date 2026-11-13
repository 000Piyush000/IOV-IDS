#pragma once
// Agent-side connection to the edge server.
// Detection must NEVER depend on the network: if the server is down,
// send() just fails, the agent keeps detecting locally, and we retry
// connecting at most every 2 seconds (no busy reconnect loop).
#include "protocol.hpp"

#include <chrono>
#include <string>
#include <vector>

class TcpClient {
public:
    TcpClient(std::string host, int port, Hello hello);
    ~TcpClient();
    TcpClient(const TcpClient&) = delete;
    TcpClient& operator=(const TcpClient&) = delete;

    bool send(const std::vector<uint8_t>& bytes);   // false if not connected / failed
    bool connected() const { return fd_ >= 0; }
private:
    bool try_connect();
    void disconnect();
    std::string host_;
    int port_;
    Hello hello_;
    int fd_ = -1;
    std::chrono::steady_clock::time_point next_retry_{};
};
