#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>

#include <mrs_uav_flightforge_simulator/sensor_conversions.h>

using namespace mrs_uav_flightforge_simulator;

namespace
{

cv::Mat solid(int w, int h, const cv::Vec3b& bgr) {
  return cv::Mat(h, w, CV_8UC3, cv::Scalar(bgr[0], bgr[1], bgr[2]));
}

std::vector<unsigned char> encode(const std::string& ext, const cv::Mat& m) {
  std::vector<unsigned char> out;
  cv::imencode(ext, m, out);
  return out;
}

Serializable::Drone::CameraInfo info(int eye, double fx, double baseline_cm) {
  Serializable::Drone::CameraInfo ci;
  ci.eye_              = eye;
  ci.width_            = 640;
  ci.height_           = 480;
  ci.fx_               = fx;
  ci.fy_               = fx;
  ci.cx_               = 320;
  ci.cy_               = 240;
  ci.distortion_model_ = "plumb_bob";
  ci.d_                = {0, 0, 0, 0, 0};
  ci.baseline_         = baseline_cm;
  return ci;
}

}  // namespace

TEST(Images, DecodesJpegToBgr8) {
  auto m = decodeFrame(encode(".jpg", solid(64, 48, {255, 0, 0})), 64, 48);
  ASSERT_TRUE(m);
  auto img = toColorImage(*m, {});
  EXPECT_EQ(img.encoding, "bgr8");
  EXPECT_EQ(img.width, 64u);
  EXPECT_EQ(img.step, 64u * 3);
  EXPECT_GT(img.data[0], 200);
}

TEST(Images, WrongSizeIsRejected) {
  EXPECT_FALSE(decodeFrame(encode(".jpg", solid(64, 48, {0, 0, 0})), 640, 480));
}

TEST(Images, EmptyIsRejected) {
  EXPECT_FALSE(decodeFrame({}, 64, 48));
}

TEST(Images, GarbageIsRejected) {
  EXPECT_FALSE(decodeFrame({1, 2, 3, 4}, 64, 48));
}

TEST(Images, InstanceIdFromBgr) {
  auto img = toInstanceImage(solid(2, 2, {0x03, 0x02, 0x01}), {});
  EXPECT_EQ(img.encoding, "32SC1");
  EXPECT_EQ(img.step, 2u * 4);
  EXPECT_EQ(reinterpret_cast<const int32_t*>(img.data.data())[0], 0x030201);
}

TEST(Images, InstanceIdSurvivesPng) {
  auto m = decodeFrame(encode(".png", solid(4, 4, {0x56, 0x34, 0x12})), 4, 4);
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
  auto out = toCameraInfo(info(0, 400, 0));
  EXPECT_EQ(out.width, 640u);
  EXPECT_DOUBLE_EQ(out.k[0], 400);
  EXPECT_DOUBLE_EQ(out.k[2], 320);
  EXPECT_DOUBLE_EQ(out.k[8], 1);
  EXPECT_DOUBLE_EQ(out.r[0], 1);
  EXPECT_DOUBLE_EQ(out.p[3], 0);
  EXPECT_EQ(out.distortion_model, "plumb_bob");
}

TEST(CameraInfo, RightEyeCarriesBaseline) {
  EXPECT_DOUBLE_EQ(toCameraInfo(info(1, 400, 7.5)).p[3], -400 * 0.075);
}

TEST(CameraInfo, FisheyeKeepsEquidistantModel) {
  auto ci              = info(0, 300, 0);
  ci.distortion_model_ = "equidistant";
  ci.d_                = {0.1, 0.2, 0.3, 0.4};
  auto out             = toCameraInfo(ci);
  EXPECT_EQ(out.distortion_model, "equidistant");
  ASSERT_EQ(out.d.size(), 4u);
  EXPECT_DOUBLE_EQ(out.d[3], 0.4);
}

TEST(CameraInfo, EyeChosenByField) {
  auto l = info(0, 400, 7.5);
  auto r = info(1, 400, 7.5);
  std::vector<Serializable::Drone::CameraInfo> both{r, l};
  EXPECT_EQ(findEye(both, 0)->eye_, 0);
  EXPECT_EQ(findEye(both, 1)->eye_, 1);
  EXPECT_EQ(findEye({l}, 1), nullptr);
}

TEST(Mount, ToUe) {
  Mount m{1, 2, 3, 10, 20, 30};
  auto  o = toUeOffset(m);
  EXPECT_DOUBLE_EQ(o.x, 100);
  EXPECT_DOUBLE_EQ(o.y, -200);
  EXPECT_DOUBLE_EQ(o.z, 300);
  auto r = toUeRotation(m);
  EXPECT_DOUBLE_EQ(r.pitch, -20);
  EXPECT_DOUBLE_EQ(r.yaw, -30);
  EXPECT_DOUBLE_EQ(r.roll, 10);
}

TEST(Images, RawFrameNeedsExactSize) {
  std::vector<unsigned char> raw(4 * 2 * 3, 7);
  raw[0]   = 200;
  auto m   = rawFrame(raw, 4, 2);
  ASSERT_TRUE(m);
  EXPECT_EQ(m->cols, 4);
  EXPECT_EQ(m->at<cv::Vec3b>(0, 0)[0], 200);
  EXPECT_FALSE(rawFrame(raw, 4, 3));
  EXPECT_FALSE(rawFrame({}, 4, 2));
}
