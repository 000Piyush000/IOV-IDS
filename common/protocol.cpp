#include "protocol.hpp"

#include <arpa/inet.h>   // htonl / ntohl / htons / ntohs
#include <algorithm>
#include <cstring>

namespace {
// Writer/Reader keep encoding symmetric and bounds-checked.
struct Writer {
    std::vector<uint8_t> b;
    void u8(uint8_t v) { b.push_back(v); }
    void u16(uint16_t v) { v = htons(v); auto p = reinterpret_cast<uint8_t*>(&v); b.insert(b.end(), p, p + 2); }
    void u32(uint32_t v) { v = htonl(v); auto p = reinterpret_cast<uint8_t*>(&v); b.insert(b.end(), p, p + 4); }
    void u64(uint64_t v) { u32(static_cast<uint32_t>(v >> 32)); u32(static_cast<uint32_t>(v)); }
    void f32(float f) { uint32_t u; std::memcpy(&u, &f, 4); u32(u); }
    void str(const std::string& s) { u8(static_cast<uint8_t>(s.size())); b.insert(b.end(), s.begin(), s.begin() + static_cast<long>(std::min<std::size_t>(s.size(), 255))); }
};

struct Reader {
    const std::vector<uint8_t>& b;
    std::size_t i = 0;
    bool ok = true;
    bool need(std::size_t n) { if (i + n > b.size()) ok = false; return ok; }
    uint8_t u8() { if (!need(1)) return 0; return b[i++]; }
    uint16_t u16() { if (!need(2)) return 0; uint16_t v; std::memcpy(&v, &b[i], 2); i += 2; return ntohs(v); }
    uint32_t u32() { if (!need(4)) return 0; uint32_t v; std::memcpy(&v, &b[i], 4); i += 4; return ntohl(v); }
    uint64_t u64() { uint64_t hi = u32(); return (hi << 32) | u32(); }
    float f32() { uint32_t u = u32(); float f; std::memcpy(&f, &u, 4); return f; }
    std::string str() { uint8_t n = u8(); if (!need(n)) return {}; std::string s(b.begin() + static_cast<long>(i), b.begin() + static_cast<long>(i + n)); i += n; return s; }
};
}  // namespace

std::vector<uint8_t> encode_raw(MsgType type, const std::vector<uint8_t>& payload) {
    Writer w;
    w.u32(static_cast<uint32_t>(payload.size() + 1));
    w.u8(static_cast<uint8_t>(type));
    w.b.insert(w.b.end(), payload.begin(), payload.end());
    return w.b;
}

std::vector<uint8_t> encode(const Hello& m) {
    Writer w; w.u16(m.vehicle_id); w.u8(static_cast<uint8_t>(m.role)); w.u8(m.fl);
    return encode_raw(MsgType::Hello, w.b);
}
std::vector<uint8_t> encode(const Alert& m) {
    Writer w; w.u16(m.vehicle_id); w.u64(m.ts_us); w.str(m.cls); w.f32(m.prob);
    w.u16(m.top_can_id); w.f32(m.id_entropy);
    return encode_raw(MsgType::Alert, w.b);
}
std::vector<uint8_t> encode(const Stats& m) {
    Writer w; w.u16(m.vehicle_id); w.u32(m.frames); w.u32(m.windows); w.u32(m.alerts);
    return encode_raw(MsgType::Stats, w.b);
}

std::optional<Hello> decode_hello(const std::vector<uint8_t>& p) {
    Reader r{p}; Hello m; m.vehicle_id = r.u16(); m.role = static_cast<Role>(r.u8()); m.fl = r.u8();
    if (!r.ok) return std::nullopt;
    return m;
}
std::optional<Alert> decode_alert(const std::vector<uint8_t>& p) {
    Reader r{p}; Alert m; m.vehicle_id = r.u16(); m.ts_us = r.u64(); m.cls = r.str();
    m.prob = r.f32(); m.top_can_id = r.u16(); m.id_entropy = r.f32();
    if (!r.ok) return std::nullopt;
    return m;
}
std::optional<Stats> decode_stats(const std::vector<uint8_t>& p) {
    Reader r{p}; Stats m; m.vehicle_id = r.u16(); m.frames = r.u32(); m.windows = r.u32(); m.alerts = r.u32();
    if (!r.ok) return std::nullopt;
    return m;
}

void StreamDecoder::feed(const uint8_t* data, std::size_t n) {
    // compact occasionally so the buffer doesn't grow forever
    if (pos_ > 4096 && pos_ * 2 > buf_.size()) {
        buf_.erase(buf_.begin(), buf_.begin() + static_cast<long>(pos_));
        pos_ = 0;
    }
    buf_.insert(buf_.end(), data, data + n);
}

std::optional<Message> StreamDecoder::next() {
    if (error_ || buffered() < 4) return std::nullopt;          // need the length field
    uint32_t len;
    std::memcpy(&len, &buf_[pos_], 4);
    len = ntohl(len);
    if (len == 0 || len > kMaxMsg) { error_ = true; return std::nullopt; }   // corrupt stream
    if (buffered() < 4 + static_cast<std::size_t>(len)) return std::nullopt;  // partial message: wait
    Message m;
    m.type = static_cast<MsgType>(buf_[pos_ + 4]);
    m.payload.assign(buf_.begin() + static_cast<long>(pos_ + 5), buf_.begin() + static_cast<long>(pos_ + 4 + len));
    pos_ += 4 + len;
    return m;
}

static constexpr uint32_t kMaxParams = 100000;

std::vector<uint8_t> encode(const ModelUpdate& m) {
    Writer w; w.u16(m.vehicle_id); w.u32(m.round); w.u32(m.n_samples);
    w.u32(static_cast<uint32_t>(m.params.size()));
    for (float v : m.params) w.f32(v);
    return encode_raw(MsgType::ModelUpdate, w.b);
}
std::vector<uint8_t> encode(const GlobalModel& m) {
    Writer w; w.u32(m.round); w.u32(static_cast<uint32_t>(m.params.size()));
    for (float v : m.params) w.f32(v);
    return encode_raw(MsgType::GlobalModel, w.b);
}
std::optional<ModelUpdate> decode_model_update(const std::vector<uint8_t>& p) {
    Reader r{p}; ModelUpdate m; m.vehicle_id = r.u16(); m.round = r.u32(); m.n_samples = r.u32();
    uint32_t n = r.u32();
    if (!r.ok || n > kMaxParams) return std::nullopt;
    m.params.resize(n);
    for (auto& v : m.params) v = r.f32();
    if (!r.ok) return std::nullopt;
    return m;
}
std::optional<GlobalModel> decode_global_model(const std::vector<uint8_t>& p) {
    Reader r{p}; GlobalModel m; m.round = r.u32();
    uint32_t n = r.u32();
    if (!r.ok || n > kMaxParams) return std::nullopt;
    m.params.resize(n);
    for (auto& v : m.params) v = r.f32();
    if (!r.ok) return std::nullopt;
    return m;
}
