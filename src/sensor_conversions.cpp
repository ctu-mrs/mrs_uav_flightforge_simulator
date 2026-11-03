#include <mrs_uav_flightforge_simulator/sensor_conversions.h>

#include <cmath>
#include <cstring>

#include <opencv2/imgcodecs.hpp>

#include <sensor_msgs/point_cloud2_iterator.hpp>

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

/* lidar //{ */

namespace
{

sensor_msgs::msg::PointCloud2 emptyCloud(const std_msgs::msg::Header& header, bool intensity, bool labels, size_t n) {

  sensor_msgs::msg::PointCloud2 cloud;

  cloud.header       = header;
  cloud.is_dense     = true;
  cloud.is_bigendian = false;

  sensor_msgs::PointCloud2Modifier modifier(cloud);

  using PF = sensor_msgs::msg::PointField;

  if (intensity && labels) {
    modifier.setPointCloud2Fields(6, "x", 1, PF::FLOAT32, "y", 1, PF::FLOAT32, "z", 1, PF::FLOAT32, "intensity", 1, PF::FLOAT32, "class_id", 1, PF::UINT16,
                                  "instance_id", 1, PF::UINT32);
  } else if (intensity) {
    modifier.setPointCloud2Fields(4, "x", 1, PF::FLOAT32, "y", 1, PF::FLOAT32, "z", 1, PF::FLOAT32, "intensity", 1, PF::FLOAT32);
  } else if (labels) {
    modifier.setPointCloud2Fields(5, "x", 1, PF::FLOAT32, "y", 1, PF::FLOAT32, "z", 1, PF::FLOAT32, "class_id", 1, PF::UINT16, "instance_id", 1, PF::UINT32);
  } else {
    modifier.setPointCloud2Fields(3, "x", 1, PF::FLOAT32, "y", 1, PF::FLOAT32, "z", 1, PF::FLOAT32);
  }

  modifier.resize(n);

  return cloud;
}

}  // namespace

std::optional<LidarClouds> toLidarClouds(const ueds_connector::LidarScan& scan, const LidarParams& p, const std_msgs::msg::Header& header, std::mt19937& rng) {

  const size_t n = scan.points.size();

  if ((p.intensity && scan.intensity.size() != n) || (p.labels && (scan.class_id.size() != n || scan.instance_id.size() != n))) {
    return std::nullopt;
  }

  size_t n_hits = 0, n_free = 0;

  for (const auto& ray : scan.points) {
    if (ray.directionX != 0 || ray.directionY != 0 || ray.directionZ != 0) {
      (ray.distance > 0 ? n_hits : n_free)++;
    }
  }

  LidarClouds out;

  out.points = emptyCloud(header, p.intensity, p.labels, n_hits);
  out.free   = emptyCloud(header, false, false, n_free);

  sensor_msgs::PointCloud2Iterator<float> hx(out.points, "x"), hy(out.points, "y"), hz(out.points, "z");
  sensor_msgs::PointCloud2Iterator<float> fx(out.free, "x"), fy(out.free, "y"), fz(out.free, "z");

  std::optional<sensor_msgs::PointCloud2Iterator<float>>    hi;
  std::optional<sensor_msgs::PointCloud2Iterator<uint16_t>> hc;
  std::optional<sensor_msgs::PointCloud2Iterator<uint32_t>> hn;

  if (p.intensity) {
    hi.emplace(out.points, "intensity");
  }

  if (p.labels) {
    hc.emplace(out.points, "class_id");
    hn.emplace(out.points, "instance_id");
  }

  for (size_t i = 0; i < n; i++) {

    const auto&  ray  = scan.points[i];
    const double norm = std::sqrt(ray.directionX * ray.directionX + ray.directionY * ray.directionY + ray.directionZ * ray.directionZ);

    if (norm <= 0) {
      continue;
    }

    const double ux = ray.directionX / norm, uy = ray.directionY / norm, uz = ray.directionZ / norm;

    if (ray.distance <= 0) {
      *fx = ux;
      *fy = -uy;  // UE is left-handed
      *fz = uz;
      ++fx, ++fy, ++fz;
      continue;
    }

    double range = ray.distance / 100.0;

    if (p.noise.enabled) {
      range += std::normal_distribution<double>(0.0, range * p.noise.std_slope * p.noise.std_at_1m)(rng);
    }

    *hx = ux * range;
    *hy = -uy * range;
    *hz = uz * range;
    ++hx, ++hy, ++hz;

    if (hi) {
      **hi = scan.intensity[i];
      ++*hi;
    }

    if (hc) {
      **hc = scan.class_id[i];
      **hn = scan.instance_id[i];
      ++*hc, ++*hn;
    }
  }

  return out;
}

//}

}  // namespace mrs_uav_flightforge_simulator
