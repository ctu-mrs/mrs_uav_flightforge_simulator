#include <mrs_uav_flightforge_simulator/sensor_conversions.h>

#include <cstring>

#include <opencv2/imgcodecs.hpp>

namespace mrs_uav_flightforge_simulator
{

/* images //{ */

std::optional<cv::Mat> decodeFrame(const std::vector<unsigned char>& data, int width, int height) {

  if (data.empty()) {
    return std::nullopt;
  }

  cv::Mat image = cv::imdecode(data, cv::IMREAD_COLOR);

  if (image.empty() || image.cols != width || image.rows != height) {
    return std::nullopt;
  }

  return image;
}

namespace
{

sensor_msgs::msg::Image emptyImage(int width, int height, const std::string& encoding, int bytes_per_pixel, const std_msgs::msg::Header& header) {

  sensor_msgs::msg::Image img;

  img.header       = header;
  img.width        = width;
  img.height       = height;
  img.encoding     = encoding;
  img.is_bigendian = false;
  img.step         = width * bytes_per_pixel;
  img.data.resize(size_t(img.step) * height);

  return img;
}

}  // namespace

sensor_msgs::msg::Image toColorImage(const cv::Mat& bgr, const std_msgs::msg::Header& header) {

  auto img = emptyImage(bgr.cols, bgr.rows, "bgr8", 3, header);

  for (int row = 0; row < bgr.rows; row++) {
    std::memcpy(&img.data[size_t(row) * img.step], bgr.ptr(row), img.step);
  }

  return img;
}

sensor_msgs::msg::Image toInstanceImage(const cv::Mat& bgr, const std_msgs::msg::Header& header) {

  auto img = emptyImage(bgr.cols, bgr.rows, "32SC1", 4, header);

  int32_t* out = reinterpret_cast<int32_t*>(img.data.data());

  for (int row = 0; row < bgr.rows; row++) {
    for (int col = 0; col < bgr.cols; col++) {
      const cv::Vec3b& px          = bgr.at<cv::Vec3b>(row, col);
      out[row * bgr.cols + col] = px[2] | (px[1] << 8) | (px[0] << 16);
    }
  }

  return img;
}

std::optional<sensor_msgs::msg::Image> toDepthImage(const std::vector<uint16_t>& mm, int width, int height, const std_msgs::msg::Header& header) {

  if (width <= 0 || height <= 0 || mm.size() != size_t(width) * height) {
    return std::nullopt;
  }

  auto img = emptyImage(width, height, "16UC1", 2, header);

  std::memcpy(img.data.data(), mm.data(), img.data.size());

  return img;
}

//}

/* calibration and mounting //{ */

sensor_msgs::msg::CameraInfo toCameraInfo(const Serializable::Drone::CameraInfo& ci) {

  sensor_msgs::msg::CameraInfo out;

  out.width            = ci.width_;
  out.height           = ci.height_;
  out.distortion_model = ci.distortion_model_;
  out.d                = ci.d_;

  out.k = {ci.fx_, 0, ci.cx_, 0, ci.fy_, ci.cy_, 0, 0, 1};
  out.r = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  // the baseline is in UE centimetres
  const double tx = ci.eye_ == 1 ? -ci.fx_ * ci.baseline_ / 100.0 : 0.0;

  out.p = {ci.fx_, 0, ci.cx_, tx, 0, ci.fy_, ci.cy_, 0, 0, 0, 1, 0};

  return out;
}

const Serializable::Drone::CameraInfo* findEye(const std::vector<Serializable::Drone::CameraInfo>& infos, int eye) {

  for (const auto& ci : infos) {
    if (ci.eye_ == eye) {
      return &ci;
    }
  }

  return nullptr;
}

ueds_connector::Coordinates toUeOffset(const Mount& m) {
  return ueds_connector::Coordinates(m.x * 100.0, -m.y * 100.0, m.z * 100.0);
}

ueds_connector::Rotation toUeRotation(const Mount& m) {
  return ueds_connector::Rotation(-m.pitch, m.yaw, m.roll);
}

//}

}  // namespace mrs_uav_flightforge_simulator
