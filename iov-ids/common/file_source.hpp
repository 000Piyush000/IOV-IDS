#pragma once
// Reads frames from a candump-format log file: "(1700000000.000250) vcan0 123#DEADBEEF".
// Same IFrameSource interface as SocketCanSource -> the rest of the code
// doesn't care whether frames come from the kernel or from a file.
#include "frame_source.hpp"
#include <fstream>
#include <string>

class FileSource : public IFrameSource {
public:
    explicit FileSource(const std::string& path);
    std::optional<CanFrame> next() override;
private:
    std::ifstream in_;
};
