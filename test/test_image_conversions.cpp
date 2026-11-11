#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>

#include <flight_forge_connector/wire_convert.h>

#include <mrs_uav_flightforge_simulator/sensor_conversions.h>

using namespace mrs_uav_flightforge_simulator;

using flightforge::wire::Compression;
using flightforge::wire::PixelFormat;

namespace
{

cv::Mat solid(int w, int h, const cv::Vec3b& bgr) {
  return cv::Mat(h, w, CV_8UC3, cv::Scalar(bgr[0], bgr[1], bgr[2]));
}

flightforge::Image encoded(const std::string& ext, const cv::Mat& m) {
  flightforge::Image img;
  img.width       = m.cols;
  img.height      = m.rows;
  img.compression = ext == ".png" ? Compression::PNG : Compression::JPEG;
  cv::imencode(ext, m, img.data);
  return img;
}

flightforge::Image raw(int w, int h, PixelFormat format, std::vector<uint8_t> pixel) {
  flightforge::Image img;
  img.width  = w;
  img.height = h;
  img.format = format;
  for (int i = 0; i < w * h; i++) {
    img.data.insert(img.data.end(), pixel.begin(), pixel.end());
  }
  return img;
}

flightforge::wire::CameraInfoT info(flightforge::wire::Eye eye, double fx, double baseline_m) {
  flightforge::wire::CameraInfoT ci;
  ci.eye              = eye;
  ci.width            = 640;
  ci.height           = 480;
  ci.fx               = fx;
  ci.fy               = fx;
  ci.cx               = 320;
  ci.cy               = 240;
  ci.distortion_model = "plumb_bob";
  ci.d                = {0, 0, 0, 0, 0};
  ci.baseline_m       = baseline_m;
  return ci;
}

void expectMount(const Mount& a, const Mount& b) {
  EXPECT_NEAR(a.x, b.x, 1e-9);
  EXPECT_NEAR(a.y, b.y, 1e-9);
  EXPECT_NEAR(a.z, b.z, 1e-9);
  EXPECT_NEAR(a.roll, b.roll, 1e-9);
  EXPECT_NEAR(a.pitch, b.pitch, 1e-9);
  EXPECT_NEAR(a.yaw, b.yaw, 1e-9);
}

}  // namespace

TEST(Images, DecodesJpegToBgr8) {
  auto m = toBgr(encoded(".jpg", solid(64, 48, {255, 0, 0})), 64, 48);
  ASSERT_TRUE(m);
  auto img = toColorImage(*m, {});
  EXPECT_EQ(img.encoding, "bgr8");
  EXPECT_EQ(img.width, 64u);
  EXPECT_EQ(img.step, 64u * 3);
  EXPECT_GT(img.data[0], 200);
}

TEST(Images, WrongSizeIsRejected) {
  EXPECT_FALSE(toBgr(encoded(".jpg", solid(64, 48, {0, 0, 0})), 640, 480));
}

TEST(Images, EmptyIsRejected) {
  EXPECT_FALSE(toBgr(flightforge::Image{}, 64, 48));
}

TEST(Images, GarbageIsRejected) {
  flightforge::Image img;
  img.compression = Compression::JPEG;
  img.data        = {1, 2, 3, 4};
  EXPECT_FALSE(toBgr(img, 64, 48));
}

TEST(Images, RawFormatsBecomeBgr) {
  const auto bgr = toBgr(raw(4, 2, PixelFormat::BGR8, {10, 20, 30}), 4, 2);
  const auto rgb = toBgr(raw(4, 2, PixelFormat::RGB8, {30, 20, 10}), 4, 2);
  const auto rgba = toBgr(raw(4, 2, PixelFormat::RGBA8, {30, 20, 10, 255}), 4, 2);
  const auto bgra = toBgr(raw(4, 2, PixelFormat::BGRA8, {10, 20, 30, 255}), 4, 2);
  for (const auto& m : {bgr, rgb, rgba, bgra}) {
    ASSERT_TRUE(m);
    EXPECT_EQ(m->type(), CV_8UC3);
    EXPECT_EQ(m->at<cv::Vec3b>(1, 3), cv::Vec3b(10, 20, 30));
  }
}

TEST(Images, RawFrameNeedsExactSize) {
  auto img = raw(4, 2, PixelFormat::BGR8, {7, 7, 7});
  EXPECT_TRUE(toBgr(img, 4, 2));
  EXPECT_FALSE(toBgr(img, 4, 3));
  img.data.pop_back();
  EXPECT_FALSE(toBgr(img, 4, 2));
  EXPECT_FALSE(toBgr(raw(4, 2, PixelFormat::UNKNOWN, {7, 7, 7}), 4, 2));
}

TEST(Images, InstanceIdFromBgr) {
  auto img = toInstanceImage(solid(2, 2, {0x03, 0x02, 0x01}), {});
  EXPECT_EQ(img.encoding, "32SC1");
  EXPECT_EQ(img.step, 2u * 4);
  EXPECT_EQ(reinterpret_cast<const int32_t*>(img.data.data())[0], 0x030201);
}

TEST(Images, InstanceIdSurvivesRgbaPng) {
  // R + (G << 8) + (B << 16), the class in alpha
  cv::Mat rgba(4, 4, CV_8UC4, cv::Scalar(0x56, 0x34, 0x12, 7));
  auto    m = toBgr(encoded(".png", rgba), 4, 4);
  ASSERT_TRUE(m);
  EXPECT_EQ(reinterpret_cast<const int32_t*>(toInstanceImage(*m, {}).data.data())[5], 0x563412);
}

TEST(Images, DepthSizeChecked) {
  EXPECT_FALSE(toDepthImage(std::vector<uint16_t>(10), 4, 4, {}));
  auto d = toDepthImage(std::vector<uint16_t>(16, 1500), 4, 4, {});
  ASSERT_TRUE(d);
  EXPECT_EQ(d->encoding, "16UC1");
  EXPECT_EQ(d->step, 4u * 2);
  EXPECT_EQ(reinterpret_cast<const uint16_t*>(d->data.data())[15], 1500);
}

TEST(CameraInfo, PinholeIntrinsics) {
  auto out = toCameraInfo(info(flightforge::wire::Eye::LEFT, 400, 0));
  EXPECT_EQ(out.width, 640u);
  EXPECT_DOUBLE_EQ(out.k[0], 400);
  EXPECT_DOUBLE_EQ(out.k[2], 320);
  EXPECT_DOUBLE_EQ(out.k[8], 1);
  EXPECT_DOUBLE_EQ(out.r[0], 1);
  EXPECT_DOUBLE_EQ(out.p[3], 0);
  EXPECT_EQ(out.distortion_model, "plumb_bob");
}

TEST(CameraInfo, RightEyeCarriesBaseline) {
  EXPECT_DOUBLE_EQ(toCameraInfo(info(flightforge::wire::Eye::RIGHT, 400, 0.075)).p[3], -400 * 0.075);
}

TEST(CameraInfo, FisheyeKeepsEquidistantModel) {
  auto ci             = info(flightforge::wire::Eye::LEFT, 300, 0);
  ci.distortion_model = "equidistant";
  ci.d                = {0.1, 0.2, 0.3, 0.4};
  auto out            = toCameraInfo(ci);
  EXPECT_EQ(out.distortion_model, "equidistant");
  ASSERT_EQ(out.d.size(), 4u);
  EXPECT_DOUBLE_EQ(out.d[3], 0.4);
}

TEST(CameraInfo, EyeChosenByField) {
  using flightforge::wire::Eye;
  auto                                        l = info(Eye::LEFT, 400, 0.075);
  auto                                        r = info(Eye::RIGHT, 400, 0.075);
  std::vector<flightforge::wire::CameraInfoT> both{r, l};
  EXPECT_EQ(findEye(both, Eye::LEFT)->eye, Eye::LEFT);
  EXPECT_EQ(findEye(both, Eye::RIGHT)->eye, Eye::RIGHT);
  EXPECT_EQ(findEye({l}, Eye::RIGHT), nullptr);
}

TEST(Mount, PoseIsRep103) {
  const auto pose = toPose(Mount{1, 2, 3, 0, 0, 90});
  EXPECT_DOUBLE_EQ(pose.position.y, 2);
  const auto forward = pose.orientation.Rotate(flightforge::Vec3{1, 0, 0});
  EXPECT_NEAR(forward.x, 0, 1e-12);
  EXPECT_NEAR(forward.y, 1, 1e-12);

  const auto down = toPose(Mount{0, 0, 0, 0, 90, 0}).orientation.Rotate(flightforge::Vec3{1, 0, 0});
  EXPECT_NEAR(down.z, -1, 1e-12);
}

TEST(Mount, RoundTripsThroughThePose) {
  const Mount m{0.1, -0.2, 0.3, 5.0, 10.0, -20.0};
  const auto  wire = flightforge::ToWire(toPose(m));
  expectMount(toMount(&wire), m);
  expectMount(toMount(nullptr), Mount{});
}

TEST(Mount, StereoEyesAlongTheRigsY) {
  const auto [left, right] = stereoEyes(Mount{0.3, 0, 0, 0, 20, 90}, 0.1);
  EXPECT_NEAR(left.x, 0.25, 1e-12);
  EXPECT_NEAR(right.x, 0.35, 1e-12);
  EXPECT_NEAR(left.y, 0, 1e-12);
  EXPECT_DOUBLE_EQ(left.pitch, 20);
}

TEST(Mount, RangefinderBeamIsTheFramesX) {
  for (const Mount& m : {Mount{0, 0, 0, 0, 90, 0}, Mount{0.1, 0, 0, 0, 45, 30}}) {
    const auto beam = rangefinderPose(m).orientation.Rotate(flightforge::Vec3{0, 0, -1});
    const auto x    = toPose(m).orientation.Rotate(flightforge::Vec3{1, 0, 0});
    EXPECT_NEAR(beam.x, x.x, 1e-12);
    EXPECT_NEAR(beam.y, x.y, 1e-12);
    EXPECT_NEAR(beam.z, x.z, 1e-12);
  }
}

TEST(SensorConfig, CameraCarriesTheMount) {
  CameraParams p{};
  p.width = 320, p.height = 240, p.fov = 90, p.mount = Mount{0.14, 0, 0, 0, 10, 0};
  flightforge::wire::RgbCameraConfigT cfg;
  cfg.capture_rate = 5.0;
  applyCameraParams(p, cfg);
  EXPECT_EQ(cfg.width, 320u);
  ASSERT_TRUE(cfg.mount);
  expectMount(toMount(cfg.mount.get()), p.mount);
  EXPECT_DOUBLE_EQ(cfg.capture_rate, 5.0);
}

TEST(SensorConfig, LidarRangeInMetres) {
  LidarParams p{};
  p.beam_length = 40.0, p.rate = 10.0, p.horizontal_rays = 128, p.vertical_rays = 64, p.livox = true, p.livox_sensor = "mid360";
  flightforge::wire::LidarConfigT cfg;
  applyLidarParams(p, cfg);
  EXPECT_DOUBLE_EQ(cfg.range_m, 40.0);
  EXPECT_EQ(cfg.vertical_rays, 64u);
  EXPECT_EQ(cfg.livox_pattern, "mid360");
  EXPECT_TRUE(cfg.enable);
}

TEST(SensorConfig, StereoEyesMatchTheReadback) {
  StereoParams p{};
  p.baseline = 0.05, p.mount = Mount{0.1, 0, 0, 0, 0, 0};
  flightforge::wire::StereoCameraConfigT cfg;
  applyStereoParams(p, cfg);
  EXPECT_NEAR(toMount(cfg.mount_left.get()).y, 0.025, 1e-12);
  EXPECT_NEAR(toMount(cfg.mount_right.get()).y, -0.025, 1e-12);
}

TEST(DeviceReadback, StereoRecoversBaselineAndCentre) {
  flightforge::wire::StereoCameraConfigT cfg{};
  cfg.mount_left  = std::make_unique<flightforge::wire::Pose>(flightforge::ToWire(toPose(Mount{0.1, 0.025, 0, 0, 0, 0})));
  cfg.mount_right = std::make_unique<flightforge::wire::Pose>(flightforge::ToWire(toPose(Mount{0.1, -0.025, 0, 0, 0, 0})));

  const auto p = toStereoParams(cfg, 30.0);

  EXPECT_NEAR(p.baseline, 0.05, 1e-12);
  EXPECT_NEAR(p.mount.x, 0.1, 1e-12);
  EXPECT_NEAR(p.mount.y, 0.0, 1e-12);
}

TEST(DeviceReadback, LidarKeepsItsRateUnlessOverridden) {
  flightforge::wire::LidarConfigT cfg{};
  cfg.frequency = 10.0;
  cfg.range_m   = 40.0;

  DeviceParams device{};
  device.intensity = true;

  auto p = toLidarParams(cfg, device);

  EXPECT_DOUBLE_EQ(p.rate, 10.0);
  EXPECT_DOUBLE_EQ(p.beam_length, 40.0);
  EXPECT_TRUE(p.intensity);

  device.lidar_rate = 20.0;

  EXPECT_DOUBLE_EQ(toLidarParams(cfg, device).rate, 20.0);
}
