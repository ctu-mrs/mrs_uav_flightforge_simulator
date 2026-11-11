#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>

#include <mrs_uav_flightforge_simulator/sensor_conversions.h>

using namespace mrs_uav_flightforge_simulator;

namespace
{

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

flightforge::LidarScan scan(const std::vector<std::array<float, 3>>& points, uint32_t rows = 0, uint32_t cols = 0) {
  flightforge::LidarScan s;
  s.points = points;
  s.rows   = rows;
  s.cols   = cols;
  return s;
}

flightforge::LidarScan withAll(flightforge::LidarScan s) {
  s.intensity.assign(s.points.size(), 4);
  s.class_id.assign(s.points.size(), 2);
  s.instance_id.assign(s.points.size(), 9);
  return s;
}

LidarParams params(bool intensity, bool labels) {
  LidarParams p{};
  p.intensity            = intensity;
  p.labels               = labels;
  p.publish_free         = true;
  p.noise.enabled        = false;
  p.horizontal_rays      = 3;
  p.vertical_rays        = 2;
  p.horizontal_fov_left  = 90;
  p.horizontal_fov_right = 90;
  p.vertical_fov_up      = 30;
  p.vertical_fov_down    = 30;
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
  auto c = toLidarClouds(scan({{1.5f, 0, 0}, {kNaN, kNaN, kNaN}, {0, 2, 0}, {kNaN, kNaN, kNaN}, {0, 0, 1}, {1, 1, 0}}, 2, 3), params(false, false), {}, rng);
  ASSERT_TRUE(c);
  EXPECT_EQ(c->points.width, 4u);
  EXPECT_EQ(c->points.height, 1u);
  EXPECT_EQ(c->free.width, 2u);
}

TEST(Lidar, PointsPassThroughInMetres) {
  auto c = toLidarClouds(scan({{0.5f, -2.0f, 0.25f}}), params(false, false), {}, rng);
  ASSERT_TRUE(c);
  EXPECT_FLOAT_EQ(field<float>(c->points, "x", 0), 0.5f);
  EXPECT_FLOAT_EQ(field<float>(c->points, "y", 0), -2.0f);
  EXPECT_FLOAT_EQ(field<float>(c->points, "z", 0), 0.25f);
}

TEST(Lidar, FreeRaysFollowTheGrid) {
  std::vector<std::array<float, 3>> points(6, {kNaN, kNaN, kNaN});
  auto                              c = toLidarClouds(scan(points, 2, 3), params(false, false), {}, rng);
  ASSERT_TRUE(c);
  ASSERT_EQ(c->free.width, 6u);

  // row 0 is the top beam, column 0 the leftmost
  const float up = std::sin(30.0 * M_PI / 180.0);
  EXPECT_NEAR(field<float>(c->free, "z", 0), up, 1e-6);
  EXPECT_NEAR(field<float>(c->free, "y", 0), std::cos(30.0 * M_PI / 180.0), 1e-6);
  EXPECT_NEAR(field<float>(c->free, "x", 1), std::cos(30.0 * M_PI / 180.0), 1e-6);
  EXPECT_NEAR(field<float>(c->free, "y", 2), -std::cos(30.0 * M_PI / 180.0), 1e-6);
  EXPECT_NEAR(field<float>(c->free, "z", 4), -up, 1e-6);
}

TEST(Lidar, UnorganisedScanHasNoFreeRays) {
  auto c = toLidarClouds(scan({{1, 0, 0}, {kNaN, kNaN, kNaN}}), params(false, false), {}, rng);
  ASSERT_TRUE(c);
  EXPECT_EQ(c->points.width, 1u);
  EXPECT_EQ(c->free.width, 0u);
}

TEST(Lidar, GridOtherThanConfiguredHasNoFreeRays) {
  auto c = toLidarClouds(scan({{1, 0, 0}, {kNaN, kNaN, kNaN}}, 1, 2), params(false, false), {}, rng);
  ASSERT_TRUE(c);
  EXPECT_EQ(c->free.width, 0u);
}

TEST(Lidar, FieldsFollowFlags) {
  auto all = toLidarClouds(withAll(scan({{1, 0, 0}})), params(true, true), {}, rng);
  ASSERT_TRUE(all);
  EXPECT_EQ(names(all->points), (std::vector<std::string>{"x", "y", "z", "intensity", "class_id", "instance_id"}));
  auto none = toLidarClouds(scan({{1, 0, 0}}), params(false, false), {}, rng);
  ASSERT_TRUE(none);
  EXPECT_EQ(names(none->points), (std::vector<std::string>{"x", "y", "z"}));
  EXPECT_EQ(none->points.point_step, 12u);
}

TEST(Lidar, LabelsCarried) {
  auto s        = withAll(scan({{1, 0, 0}}));
  s.class_id    = {7};
  s.instance_id = {0x010203};
  auto c        = toLidarClouds(s, params(false, true), {}, rng);
  ASSERT_TRUE(c);
  EXPECT_EQ(field<uint16_t>(c->points, "class_id", 0), 7);
  EXPECT_EQ(field<uint32_t>(c->points, "instance_id", 0), 0x010203u);
}

TEST(Lidar, LabelsSkipMisses) {
  auto s        = withAll(scan({{kNaN, kNaN, kNaN}, {1, 0, 0}}));
  s.class_id    = {-1, 5};
  s.instance_id = {0, 11};
  auto c        = toLidarClouds(s, params(false, true), {}, rng);
  ASSERT_TRUE(c);
  ASSERT_EQ(c->points.width, 1u);
  EXPECT_EQ(field<uint16_t>(c->points, "class_id", 0), 5);
  EXPECT_EQ(field<uint32_t>(c->points, "instance_id", 0), 11u);
}

TEST(Lidar, IntensityIsTheMaterialClass) {
  auto s      = withAll(scan({{1, 0, 0}}));
  s.intensity = {5};
  auto c      = toLidarClouds(s, params(true, false), {}, rng);
  ASSERT_TRUE(c);
  EXPECT_FLOAT_EQ(field<float>(c->points, "intensity", 0), 5.0f);
}

TEST(Lidar, ShortChannelIsRejected) {
  auto s        = scan({{1, 0, 0}, {0, 1, 0}});
  s.class_id    = {1};
  s.instance_id = {1};
  EXPECT_FALSE(toLidarClouds(s, params(false, true), {}, rng));
  EXPECT_FALSE(toLidarClouds(scan({{1, 0, 0}}), params(true, false), {}, rng));
}

TEST(Lidar, NoiseOffMeansExact) {
  auto c = toLidarClouds(scan({{12.34f, 0, 0}}), params(false, false), {}, rng);
  ASSERT_TRUE(c);
  EXPECT_FLOAT_EQ(field<float>(c->points, "x", 0), 12.34f);
}

TEST(Lidar, NoiseOnPerturbsAlongTheRay) {
  auto p            = params(false, false);
  p.noise.enabled   = true;
  p.noise.std_at_1m = 0.1;
  p.noise.std_slope = 1.0;
  auto c            = toLidarClouds(scan({{6, 8, 0}}), p, {}, rng);
  ASSERT_TRUE(c);
  const float x = field<float>(c->points, "x", 0), y = field<float>(c->points, "y", 0);
  EXPECT_NE(x, 6.0f);
  EXPECT_NEAR(y / x, 8.0 / 6.0, 1e-5);
}
