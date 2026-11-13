#pragma once
// Agent <-> edge-server wire protocol over TCP.
//
// TCP is a BYTE STREAM: one send() may arrive as several recv()s, and
// several sends may arrive in one recv(). So every message is framed:
//
//   [ uint32 length (big-endian) ][ uint8 type ][ payload ... ]
//     length = 1 + payload size
//
// All integers are big-endian ("network byte order"); floats are sent as
// their IEEE-754 bit pattern in a big-endian uint32.
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

enum class MsgType : uint8_t {
    Hello = 1,        // agent -> server, once after connect
    Alert = 2,        // agent -> server, per attack window
    Stats = 3,        // agent -> server, once per second
    ModelUpdate = 4,  // agent -> server (federated learning, Step 6)
    GlobalModel = 5,  // server -> agent (federated learning, Step 6)
};

enum class Role : uint8_t { OBU = 0, MEC = 1 };   // on-board unit / edge node

struct Hello { uint16_t vehicle_id = 0; Role role = Role::OBU; uint8_t fl = 0; };  // fl=1: takes part in federated learning

struct Alert {
    uint16_t vehicle_id = 0;
    uint64_t ts_us = 0;          // when the window was classified
    std::string cls;             // e.g. "DoS", "RPM"
    float prob = 0.f;            // detector confidence
    uint16_t top_can_id = 0;     // most frequent CAN ID in the window (evidence)
    float id_entropy = 0.f;      // evidence for zero-day scoring / LLM report
};

struct Stats { uint16_t vehicle_id = 0; uint32_t frames = 0; uint32_t windows = 0; uint32_t alerts = 0; };

// Federated learning (Step 6). Only model PARAMETERS travel, never CAN data.
struct ModelUpdate {                 // vehicle -> server after local training
    uint16_t vehicle_id = 0;
    uint32_t round = 0;
    uint32_t n_samples = 0;          // FedAvg weights each update by its data size
    std::vector<float> params;
};
struct GlobalModel {                 // server -> vehicles, start of each round
    static constexpr uint32_t kFinal = 0xFFFFFFFF;   // round value meaning "training done"
    uint32_t round = 0;
    std::vector<float> params;
};

struct Message {
    MsgType type;
    std::vector<uint8_t> payload;
};

// ---- encoding: returns the complete framed bytes, ready for send()
std::vector<uint8_t> encode(const Hello& m);
std::vector<uint8_t> encode(const Alert& m);
std::vector<uint8_t> encode(const Stats& m);
std::vector<uint8_t> encode(const ModelUpdate& m);
std::vector<uint8_t> encode(const GlobalModel& m);
std::vector<uint8_t> encode_raw(MsgType type, const std::vector<uint8_t>& payload);

// ---- decoding a payload (after framing is removed)
std::optional<Hello> decode_hello(const std::vector<uint8_t>& p);
std::optional<Alert> decode_alert(const std::vector<uint8_t>& p);
std::optional<Stats> decode_stats(const std::vector<uint8_t>& p);
std::optional<ModelUpdate> decode_model_update(const std::vector<uint8_t>& p);
std::optional<GlobalModel> decode_global_model(const std::vector<uint8_t>& p);

// Reassembles messages from an arbitrary sequence of recv() chunks.
class StreamDecoder {
public:
    static constexpr uint32_t kMaxMsg = 1 << 20;  // 1 MB: reject garbage/attacks
    void feed(const uint8_t* data, std::size_t n);
    std::optional<Message> next();   // a complete message, or nullopt if more bytes needed
    bool error() const { return error_; }
    std::size_t buffered() const { return buf_.size() - pos_; }
private:
    std::vector<uint8_t> buf_;
    std::size_t pos_ = 0;
    bool error_ = false;
};
