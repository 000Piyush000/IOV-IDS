#pragma once
// Abstraction over "where CAN frames come from".
// Real: SocketCanSource (Linux kernel CAN socket).
// Tests: a mock/fake source, so unit tests don't need vcan or hardware.
#include <cstdint>
#include <optional>

struct CanFrame {
    uint32_t id = 0;        // 11-bit standard ID (e.g. 0x123)
    uint8_t  dlc = 0;       // data length code: 0..8 bytes
    uint8_t  data[8] = {};  // payload
    uint64_t ts_us = 0;     // receive timestamp in microseconds
};

class IFrameSource {
public:
    virtual ~IFrameSource() = default;
    // Blocks until a frame arrives. Returns std::nullopt on error/shutdown.
    virtual std::optional<CanFrame> next() = 0;
};
