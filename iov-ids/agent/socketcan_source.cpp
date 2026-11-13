#include "socketcan_source.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>

#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

SocketCanSource::SocketCanSource(const std::string& ifname) {
    // 1. Create a raw CAN socket (protocol family PF_CAN, raw frames).
    fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (fd_ < 0) throw std::runtime_error("socket(PF_CAN) failed: " + std::string(strerror(errno)));

    // 2. Translate interface name ("vcan0") -> kernel interface index.
    ifreq ifr{};
    std::strncpy(ifr.ifr_name, ifname.c_str(), IFNAMSIZ - 1);
    if (ioctl(fd_, SIOCGIFINDEX, &ifr) < 0) {
        close(fd_);
        throw std::runtime_error("interface '" + ifname + "' not found (did you run vcan-up.sh?)");
    }

    // 3. Bind the socket to that interface: now read() returns its frames.
    sockaddr_can addr{};
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(fd_);
        throw std::runtime_error("bind() failed: " + std::string(strerror(errno)));
    }
}

SocketCanSource::~SocketCanSource() {
    if (fd_ >= 0) close(fd_);   // RAII: socket closed automatically
}

std::optional<CanFrame> SocketCanSource::next() {
    can_frame raw{};
    // read() is a system call: blocks in the kernel until a frame arrives.
    ssize_t n = read(fd_, &raw, sizeof(raw));
    if (n != static_cast<ssize_t>(sizeof(raw))) return std::nullopt;

    CanFrame f;
    f.id = raw.can_id & CAN_SFF_MASK;   // keep 11-bit standard ID
    f.dlc = raw.can_dlc;
    std::memcpy(f.data, raw.data, 8);
    f.ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now().time_since_epoch()).count();
    return f;
}
