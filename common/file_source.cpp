#include "file_source.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

FileSource::FileSource(const std::string& path) : in_(path) {
    if (!in_) throw std::runtime_error("cannot open " + path);
}

std::optional<CanFrame> FileSource::next() {
    std::string line;
    while (std::getline(in_, line)) {
        // "(ts) iface ID#HEXDATA"
        auto lp = line.find('('), rp = line.find(')'), hash = line.find('#');
        if (lp == std::string::npos || rp == std::string::npos || hash == std::string::npos) continue;
        auto sp = line.rfind(' ', hash);
        CanFrame f;
        f.ts_us = static_cast<uint64_t>(std::stod(line.substr(lp + 1, rp - lp - 1)) * 1e6);
        f.id = static_cast<uint32_t>(std::strtoul(line.substr(sp + 1, hash - sp - 1).c_str(), nullptr, 16));
        std::string hex = line.substr(hash + 1);
        f.dlc = static_cast<uint8_t>(std::min<std::size_t>(hex.size() / 2, 8));
        for (int i = 0; i < f.dlc; ++i)
            f.data[i] = static_cast<uint8_t>(std::strtoul(hex.substr(2 * i, 2).c_str(), nullptr, 16));
        return f;
    }
    return std::nullopt;
}
