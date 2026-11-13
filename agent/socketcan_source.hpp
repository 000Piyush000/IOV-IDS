#pragma once
#include "frame_source.hpp"
#include <string>

// Reads frames from a Linux SocketCAN interface (e.g. "vcan0") via a raw socket.
class SocketCanSource : public IFrameSource {
public:
    explicit SocketCanSource(const std::string& ifname);
    ~SocketCanSource() override;
    SocketCanSource(const SocketCanSource&) = delete;            // owns an fd:
    SocketCanSource& operator=(const SocketCanSource&) = delete; // not copyable
    std::optional<CanFrame> next() override;
private:
    int fd_ = -1;
};
