#include <gtest/gtest.h>

#include <event_camera_codecs/decoder_factory.h>

#include <mrs_uav_flightforge_simulator/sensor_conversions.h>

using namespace mrs_uav_flightforge_simulator;

namespace
{

struct Ev
{
  uint16_t x, y;
  uint8_t  p;
  uint32_t t_us;
};

struct Decoded
{
  uint64_t t;
  uint16_t x, y;
  uint8_t  p;
};

class Collector : public event_camera_codecs::EventProcessor {
public:
  void eventCD(uint64_t t, uint16_t x, uint16_t y, uint8_t p) override {
    events.push_back({t, x, y, p});
  }
  bool eventExtTrigger(uint64_t, uint8_t, uint8_t) override {
    return true;
  }
  void finished() override {
  }
  void rawData(const char*, size_t) override {
  }

  std::vector<Decoded> events;
};

ueds_connector::CameraEvents events(const std::vector<Ev>& in) {
  ueds_connector::CameraEvents ev;
  ev.width      = 640;
  ev.height     = 480;
  ev.base_stamp = 12.5;
  for (const auto& e : in) {
    ev.xyp.push_back(uint32_t(e.x) | (uint32_t(e.y) << 16) | (e.p ? 0x80000000u : 0u));
    ev.t_us.push_back(e.t_us);
  }
  return ev;
}

std_msgs::msg::Header header(double stamp) {
  std_msgs::msg::Header h;
  h.stamp = rclcpp::Time(int64_t(stamp * 1e9));
  return h;
}

std::vector<Decoded> decodeAll(const event_camera_msgs::msg::EventPacket& pkt) {
  event_camera_codecs::DecoderFactory<event_camera_msgs::msg::EventPacket, Collector> factory;
  auto*                                                                               decoder = factory.getInstance(pkt);
  Collector                                                                           collector;
  EXPECT_NE(decoder, nullptr);
  if (decoder) {
    decoder->decode(pkt, &collector);
  }
  return collector.events;
}

}  // namespace

TEST(Events, RoundTripsThroughEvt3) {
  auto pkt = toEventPacket(events({{10, 20, 1, 0}, {11, 20, 0, 5}, {400, 300, 1, 1500}}), header(100.0));
  EXPECT_EQ(pkt.encoding, "evt3");
  EXPECT_EQ(pkt.width, 640u);
  EXPECT_EQ(pkt.time_base, uint64_t(100e9));
  auto got = decodeAll(pkt);
  ASSERT_EQ(got.size(), 3u);
  EXPECT_EQ(got[0].x, 10);
  EXPECT_EQ(got[2].x, 400);
  EXPECT_EQ(got[2].y, 300);
  EXPECT_EQ(got[0].p, 1);
  EXPECT_EQ(got[1].p, 0);
  EXPECT_EQ(got[2].t - got[0].t, 1500u * 1000u);
}

TEST(Events, EmptyBatchStillHasHeader) {
  auto pkt = toEventPacket(events({}), header(5.0));
  EXPECT_TRUE(decodeAll(pkt).empty());
  EXPECT_EQ(pkt.header.stamp.sec, 5);
  EXPECT_EQ(pkt.width, 640u);
  EXPECT_EQ(pkt.height, 480u);
}

TEST(InstanceMap, KeepsTwentyFourBitIds) {
  Serializable::Drone::InstanceSegMapEntry e;
  e.id       = 0x123456;
  e.actor    = "/Game/A";
  e.class_id = 3;
  auto m     = toInstanceMap({e}, header(1.0));
  ASSERT_EQ(m.ids.size(), 1u);
  EXPECT_EQ(m.ids[0], 0x123456u);
  EXPECT_EQ(m.actors[0], "/Game/A");
  EXPECT_EQ(m.class_ids[0], 3);
  EXPECT_EQ(m.header.stamp.sec, 1);
}

TEST(Events, LongBatchKeepsItsTiming) {
  auto got = decodeAll(toEventPacket(events({{1, 1, 1, 0}, {2, 1, 1, 2500000}, {3, 1, 0, 4500000}}), header(1.0)));
  ASSERT_EQ(got.size(), 3u);
  EXPECT_EQ(got[1].t - got[0].t, 2500000ull * 1000ull);
  EXPECT_EQ(got[2].t - got[0].t, 4500000ull * 1000ull);
}
