#include <gtest/gtest.h>

#include <cstring>

#include <mrs_uav_flightforge_simulator/sensor_conversions.h>

using namespace mrs_uav_flightforge_simulator;

namespace
{

struct Ray
{
  double distance_cm, dx, dy, dz;
};

ueds_connector::LidarScan scan(const std::vector<Ray>& rays) {
  ueds_connector::LidarScan s;
  for (const auto& r : rays) {
    ueds_connector::LidarData d;
    d.distance   = r.distance_cm;
    d.directionX = r.dx;
    d.directionY = r.dy;
    d.directionZ = r.dz;
    s.points.push_back(d);
  }
  return s;
}

ueds_connector::LidarScan withAll(ueds_connector::LidarScan s) {
  s.intensity.assign(s.points.size(), 4);
  s.class_id.assign(s.points.size(), 2);
  s.instance_id.assign(s.points.size(), 9);
  return s;
}

LidarParams params(bool intensity, bool labels) {
  LidarParams p{};
  p.intensity     = intensity;
  p.labels        = labels;
  p.publish_free  = true;
  p.noise.enabled = false;
  return p;
}

std::vector<std::string> names(const sensor_msgs::msg::PointCloud2& c) {
  std::vector<std::string> out;
  for (const auto& f : c.fields) {
    out.push_back(f.name);
  }
  return out;
}

template <class T>
T field(const sensor_msgs::msg::PointCloud2& c, const std::string& name, size_t i) {
  for (const auto& f : c.fields) {
    if (f.name == name) {
      T v;
      std::memcpy(&v, &c.data[i * c.point_step + f.offset], sizeof(T));
      return v;
    }
  }
  ADD_FAILURE() << "no field " << name;
  return T{};
}

std::mt19937 rng(0);

}  // namespace

TEST(Lidar, SplitsHitsAndMisses) {
  auto c = toLidarClouds(scan({{150, 1, 0, 0}, {-1, 0, 1, 0}}), params(false, false), {}, rng);
  ASSERT_TRUE(c);
  EXPECT_EQ(c->points.width, 1u);
  EXPECT_EQ(c->points.height, 1u);
  EXPECT_EQ(c->free.width, 1u);
  EXPECT_FLOAT_EQ(field<float>(c->free, "y", 0), -1.0f);
}

TEST(Lidar, MetresAndRightHanded) {
  auto c = toLidarClouds(scan({{200, 0, 2, 0}}), params(false, false), {}, rng);
  ASSERT_TRUE(c);
  EXPECT_FLOAT_EQ(field<float>(c->points, "y", 0), -2.0f);
  EXPECT_FLOAT_EQ(field<float>(c->points, "x", 0), 0.0f);
}

TEST(Lidar, FieldsFollowFlags) {
  auto all = toLidarClouds(withAll(scan({{100, 1, 0, 0}})), params(true, true), {}, rng);
  ASSERT_TRUE(all);
  EXPECT_EQ(names(all->points), (std::vector<std::string>{"x", "y", "z", "intensity", "class_id", "instance_id"}));
  auto none = toLidarClouds(scan({{100, 1, 0, 0}}), params(false, false), {}, rng);
  ASSERT_TRUE(none);
  EXPECT_EQ(names(none->points), (std::vector<std::string>{"x", "y", "z"}));
  EXPECT_EQ(none->points.point_step, 12u);
}

TEST(Lidar, LabelsCarried) {
  auto s        = withAll(scan({{100, 1, 0, 0}}));
  s.class_id    = {7};
  s.instance_id = {0x010203};
  auto c        = toLidarClouds(s, params(false, true), {}, rng);
  ASSERT_TRUE(c);
  EXPECT_EQ(field<uint16_t>(c->points, "class_id", 0), 7);
  EXPECT_EQ(field<uint32_t>(c->points, "instance_id", 0), 0x010203u);
}

TEST(Lidar, IntensityIsTheMaterialClass) {
  auto s      = withAll(scan({{100, 1, 0, 0}}));
  s.intensity = {5};
  auto c      = toLidarClouds(s, params(true, false), {}, rng);
  ASSERT_TRUE(c);
  EXPECT_FLOAT_EQ(field<float>(c->points, "intensity", 0), 5.0f);
}

TEST(Lidar, ShortChannelIsRejected) {
  auto s        = scan({{100, 1, 0, 0}, {100, 0, 1, 0}});
  s.class_id    = {1};
  s.instance_id = {1};
  EXPECT_FALSE(toLidarClouds(s, params(false, true), {}, rng));
  EXPECT_FALSE(toLidarClouds(scan({{100, 1, 0, 0}}), params(true, false), {}, rng));
}

TEST(Lidar, NoiseOffMeansExact) {
  auto c = toLidarClouds(scan({{1234, 1, 0, 0}}), params(false, false), {}, rng);
  ASSERT_TRUE(c);
  EXPECT_FLOAT_EQ(field<float>(c->points, "x", 0), 12.34f);
}

TEST(Lidar, NoiseOnPerturbs) {
  auto p          = params(false, false);
  p.noise.enabled = true;
  p.noise.std_at_1m = 0.1;
  p.noise.std_slope = 1.0;
  auto c          = toLidarClouds(scan({{1000, 1, 0, 0}}), p, {}, rng);
  ASSERT_TRUE(c);
  EXPECT_NE(field<float>(c->points, "x", 0), 10.0f);
}

TEST(Lidar, ZeroDirectionRayIsDropped) {
  auto c = toLidarClouds(scan({{100, 0, 0, 0}, {-1, 0, 0, 0}, {100, 1, 0, 0}}), params(false, false), {}, rng);
  ASSERT_TRUE(c);
  EXPECT_EQ(c->points.width, 1u);
  EXPECT_EQ(c->free.width, 0u);
}
