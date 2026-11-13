#include "protocol.hpp"
#include <gtest/gtest.h>

TEST(Protocol, AlertRoundTrip) {
    Alert a; a.vehicle_id = 7; a.ts_us = 1234567890123ULL; a.cls = "DoS";
    a.prob = 0.987f; a.top_can_id = 0x000; a.id_entropy = 0.25f;
    StreamDecoder d;
    auto bytes = encode(a);
    d.feed(bytes.data(), bytes.size());
    auto m = d.next();
    ASSERT_TRUE(m);
    EXPECT_EQ(m->type, MsgType::Alert);
    auto b = decode_alert(m->payload);
    ASSERT_TRUE(b);
    EXPECT_EQ(b->vehicle_id, 7);
    EXPECT_EQ(b->ts_us, 1234567890123ULL);
    EXPECT_EQ(b->cls, "DoS");
    EXPECT_FLOAT_EQ(b->prob, 0.987f);
    EXPECT_FLOAT_EQ(b->id_entropy, 0.25f);
}

TEST(Protocol, PartialReadsByteByByte) {      // TCP may split a message anywhere
    Hello h; h.vehicle_id = 3; h.role = Role::MEC;
    auto bytes = encode(h);
    StreamDecoder d;
    for (std::size_t i = 0; i + 1 < bytes.size(); ++i) {
        d.feed(&bytes[i], 1);
        EXPECT_FALSE(d.next()) << "message must not complete early at byte " << i;
    }
    d.feed(&bytes.back(), 1);
    auto m = d.next();
    ASSERT_TRUE(m);
    auto h2 = decode_hello(m->payload);
    ASSERT_TRUE(h2);
    EXPECT_EQ(h2->vehicle_id, 3);
    EXPECT_EQ(h2->role, Role::MEC);
}

TEST(Protocol, MultipleMessagesInOneChunk) {   // TCP may merge messages
    std::vector<uint8_t> all;
    for (uint16_t i = 0; i < 5; ++i) {
        Stats s; s.vehicle_id = i; s.frames = 4000u + i;
        auto b = encode(s);
        all.insert(all.end(), b.begin(), b.end());
    }
    StreamDecoder d;
    d.feed(all.data(), all.size());
    for (uint16_t i = 0; i < 5; ++i) {
        auto m = d.next();
        ASSERT_TRUE(m);
        EXPECT_EQ(decode_stats(m->payload)->frames, 4000u + i);
    }
    EXPECT_FALSE(d.next());
}

TEST(Protocol, RejectsHugeLength) {            // garbage / malicious input
    uint8_t bad[] = {0xFF, 0xFF, 0xFF, 0xFF, 2};
    StreamDecoder d;
    d.feed(bad, sizeof(bad));
    EXPECT_FALSE(d.next());
    EXPECT_TRUE(d.error());
}

TEST(Protocol, TruncatedPayloadFailsDecode) {
    std::vector<uint8_t> p = {0x00};           // too short for an Alert
    EXPECT_FALSE(decode_alert(p));
}

TEST(Protocol, ModelUpdateRoundTrip) {
    ModelUpdate u; u.vehicle_id = 4; u.round = 12; u.n_samples = 345;
    for (int i = 0; i < 40; ++i) u.params.push_back(0.5f * i - 3.0f);
    auto bytes = encode(u);
    StreamDecoder d; d.feed(bytes.data(), bytes.size());
    auto m = d.next();
    ASSERT_TRUE(m);
    EXPECT_EQ(m->type, MsgType::ModelUpdate);
    auto v = decode_model_update(m->payload);
    ASSERT_TRUE(v);
    EXPECT_EQ(v->round, 12u);
    EXPECT_EQ(v->n_samples, 345u);
    ASSERT_EQ(v->params.size(), 40u);
    for (int i = 0; i < 40; ++i) EXPECT_FLOAT_EQ(v->params[i], 0.5f * i - 3.0f);
}
