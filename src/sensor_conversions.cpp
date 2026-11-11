#include <mrs_uav_flightforge_simulator/sensor_conversions.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include <map>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <flight_forge_connector/wire_convert.h>

#include <rclcpp/time.hpp>

#include <event_camera_codecs/encoder.h>
#include <event_camera_codecs/evt3_encoder.h>

namespace mrs_uav_flightforge_simulator
{

/* images //{ */

std::optional<cv::Mat> toBgr(const flightforge::Image& image, int width, int height) {

  using flightforge::wire::Compression;
  using flightforge::wire::PixelFormat;

  if (image.data.empty() || width <= 0 || height <= 0) {
    return std::nullopt;
  }

  cv::Mat bgr;

  if (image.compression != Compression::NONE) {

    bgr = cv::imdecode(image.data, cv::IMREAD_COLOR);

  } else {

    const std::map<PixelFormat, std::pair<int, int>> layouts = {{PixelFormat::BGR8, {CV_8UC3, -1}},
                                                                {PixelFormat::RGB8, {CV_8UC3, cv::COLOR_RGB2BGR}},
                                                                {PixelFormat::BGRA8, {CV_8UC4, cv::COLOR_BGRA2BGR}},
                                                                {PixelFormat::RGBA8, {CV_8UC4, cv::COLOR_RGBA2BGR}},
                                                                {PixelFormat::MONO8, {CV_8UC1, cv::COLOR_GRAY2BGR}}};

    const auto layout = layouts.find(image.format);

    if (layout == layouts.end() || int(image.width) != width || int(image.height) != height) {
      return std::nullopt;
    }

    const cv::Mat raw(height, width, layout->second.first, const_cast<uint8_t*>(image.data.data()));

    if (image.data.size() != raw.total() * raw.elemSize()) {
      return std::nullopt;
    }

    if (layout->second.second < 0) {
      bgr = raw.clone();
    } else {
      cv::cvtColor(raw, bgr, layout->second.second);
    }
  }

  if (bgr.empty() || bgr.cols != width || bgr.rows != height) {
    return std::nullopt;
  }

  return bgr;
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

sensor_msgs::msg::CameraInfo toCameraInfo(const flightforge::wire::CameraInfoT& ci) {

  sensor_msgs::msg::CameraInfo out;

  out.width            = ci.width;
  out.height           = ci.height;
  out.distortion_model = ci.distortion_model;
  out.d                = ci.d;

  out.k = {ci.fx, 0, ci.cx, 0, ci.fy, ci.cy, 0, 0, 1};
  out.r = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  const double tx = ci.eye == flightforge::wire::Eye::RIGHT ? -ci.fx * ci.baseline_m : 0.0;

  out.p = {ci.fx, 0, ci.cx, tx, 0, ci.fy, ci.cy, 0, 0, 0, 1, 0};

  return out;
}

const flightforge::wire::CameraInfoT* findEye(const std::vector<flightforge::wire::CameraInfoT>& infos, flightforge::wire::Eye eye) {

  for (const auto& ci : infos) {
    if (ci.eye == eye) {
      return &ci;
    }
  }

  return nullptr;
}

namespace
{

constexpr double kDeg = M_PI / 180.0;

}  // namespace

flightforge::Pose toPose(const Mount& m) {
  return flightforge::Pose{flightforge::Vec3{m.x, m.y, m.z}, flightforge::Quat::FromRpy(m.roll * kDeg, m.pitch * kDeg, m.yaw * kDeg)};
}

Mount toMount(const flightforge::wire::Pose* pose) {

  if (!pose) {
    return Mount{};
  }

  const auto& q = pose->orientation();
  const auto& t = pose->position();

  const double roll  = std::atan2(2.0 * (q.w() * q.x() + q.y() * q.z()), 1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y()));
  const double pitch = std::asin(std::clamp(2.0 * (q.w() * q.y() - q.z() * q.x()), -1.0, 1.0));
  const double yaw   = std::atan2(2.0 * (q.w() * q.z() + q.x() * q.y()), 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));

  return Mount{t.x(), t.y(), t.z(), roll / kDeg, pitch / kDeg, yaw / kDeg};
}

std::pair<Mount, Mount> stereoEyes(const Mount& rig, double baseline) {

  const flightforge::Vec3 shift = toPose(rig).orientation.Rotate(flightforge::Vec3{0.0, baseline / 2.0, 0.0});

  Mount left = rig, right = rig;

  left.x += shift.x, left.y += shift.y, left.z += shift.z;
  right.x -= shift.x, right.y -= shift.y, right.z -= shift.z;

  return {left, right};
}

flightforge::Pose rangefinderPose(const Mount& m) {
  flightforge::Pose pose = toPose(m);
  pose.orientation       = pose.orientation * flightforge::Quat::FromRpy(0.0, -M_PI / 2.0, 0.0);
  return pose;
}

//}

/* sensor configs //{ */

namespace
{

std::unique_ptr<flightforge::wire::Pose> wirePose(const Mount& m) {
  return std::make_unique<flightforge::wire::Pose>(flightforge::ToWire(toPose(m)));
}

}  // namespace

void applyCameraParams(const CameraParams& p, flightforge::wire::RgbCameraConfigT& cfg) {
  cfg.width                  = p.width;
  cfg.height                 = p.height;
  cfg.fov                    = p.fov;
  cfg.mount                  = wirePose(p.mount);
  cfg.enable_hdr             = p.enable_hdr;
  cfg.enable_temporal_aa     = p.enable_temporal_aa;
  cfg.enable_raytracing      = p.enable_raytracing;
  cfg.enable_motion_blur     = p.enable_motion_blur;
  cfg.motion_blur_amount     = p.motion_blur_amount;
  cfg.motion_blur_distortion = p.motion_blur_distortion;
}

void applyFisheyeParams(const FisheyeParams& p, flightforge::wire::FisheyeCameraConfigT& cfg) {
  cfg.width      = p.width;
  cfg.height     = p.height;
  cfg.fov        = p.fov;
  cfg.lens       = flightforge::wire::FisheyeLens(p.lens_model);
  cfg.full_frame = p.full_frame;
  cfg.mount      = wirePose(p.mount);
}

void applyStereoParams(const StereoParams& p, flightforge::wire::StereoCameraConfigT& cfg) {

  const auto [left, right] = stereoEyes(p.mount, p.baseline);

  cfg.width              = p.width;
  cfg.height             = p.height;
  cfg.fov                = p.fov;
  cfg.mount_left         = wirePose(left);
  cfg.mount_right        = wirePose(right);
  cfg.enable_hdr         = p.enable_hdr;
  cfg.enable_temporal_aa = p.enable_temporal_aa;
  cfg.enable_raytracing  = p.enable_raytracing;
}

void applyEventParams(const EventParams& p, flightforge::wire::EventCameraConfigT& cfg) {
  cfg.width                  = p.width;
  cfg.height                 = p.height;
  cfg.fov                    = p.fov;
  cfg.contrast_threshold_pos = p.contrast_threshold_pos;
  cfg.contrast_threshold_neg = p.contrast_threshold_neg;
  cfg.max_events_per_frame   = p.max_events_per_frame;
  cfg.use_sim_time           = true;
  cfg.mount                  = wirePose(p.mount);
}

void applyLidarParams(const LidarParams& p, flightforge::wire::LidarConfigT& cfg) {
  cfg.enable          = true;
  cfg.show_beams      = false;
  cfg.range_m         = p.beam_length;
  cfg.horizontal_rays = p.horizontal_rays;
  cfg.vertical_rays   = p.vertical_rays;
  cfg.frequency       = p.rate;
  cfg.mount           = wirePose(p.mount);
  cfg.fov_left        = p.horizontal_fov_left;
  cfg.fov_right       = p.horizontal_fov_right;
  cfg.fov_up          = p.vertical_fov_up;
  cfg.fov_down        = p.vertical_fov_down;
  cfg.livox           = p.livox;
  cfg.livox_pattern   = p.livox_sensor;
}

//}

/* device sensors, as the simulator built them //{ */

CameraParams toCameraParams(const flightforge::wire::RgbCameraConfigT& cfg, double rate) {

  CameraParams p{};

  p.rate                   = rate;
  p.width                  = cfg.width;
  p.height                 = cfg.height;
  p.fov                    = cfg.fov;
  p.mount                  = toMount(cfg.mount.get());
  p.enable_hdr             = cfg.enable_hdr;
  p.enable_temporal_aa     = cfg.enable_temporal_aa;
  p.enable_raytracing      = cfg.enable_raytracing;
  p.enable_motion_blur     = cfg.enable_motion_blur;
  p.motion_blur_amount     = cfg.motion_blur_amount;
  p.motion_blur_distortion = cfg.motion_blur_distortion;

  return p;
}

FisheyeParams toFisheyeParams(const flightforge::wire::FisheyeCameraConfigT& cfg, double rate) {

  FisheyeParams p{};

  p.rate       = rate;
  p.width      = cfg.width;
  p.height     = cfg.height;
  p.fov        = cfg.fov;
  p.lens_model = int(cfg.lens);
  p.full_frame = cfg.full_frame;
  p.mount      = toMount(cfg.mount.get());

  return p;
}

StereoParams toStereoParams(const flightforge::wire::StereoCameraConfigT& cfg, double rate) {

  const Mount left = toMount(cfg.mount_left.get()), right = toMount(cfg.mount_right.get());

  StereoParams p{};

  p.rate               = rate;
  p.width              = cfg.width;
  p.height             = cfg.height;
  p.fov                = cfg.fov;
  p.baseline           = std::hypot(left.x - right.x, left.y - right.y, left.z - right.z);
  p.mount              = left;
  p.mount.x            = (left.x + right.x) / 2.0;
  p.mount.y            = (left.y + right.y) / 2.0;
  p.mount.z            = (left.z + right.z) / 2.0;
  p.enable_hdr         = cfg.enable_hdr;
  p.enable_temporal_aa = cfg.enable_temporal_aa;
  p.enable_raytracing  = cfg.enable_raytracing;

  return p;
}

LidarParams toLidarParams(const flightforge::wire::LidarConfigT& cfg, const DeviceParams& device) {

  LidarParams p{};

  p.rate                 = device.lidar_rate > 0 ? device.lidar_rate : cfg.frequency;
  p.livox                = cfg.livox;
  p.livox_sensor         = cfg.livox_pattern;
  p.intensity            = device.intensity;
  p.labels               = device.labels;
  p.publish_free         = device.publish_free;
  p.horizontal_rays      = int(cfg.horizontal_rays);
  p.vertical_rays        = int(cfg.vertical_rays);
  p.horizontal_fov_left  = cfg.fov_left;
  p.horizontal_fov_right = cfg.fov_right;
  p.vertical_fov_up      = cfg.fov_up;
  p.vertical_fov_down    = cfg.fov_down;
  p.beam_length          = cfg.range_m;
  p.mount                = toMount(cfg.mount.get());
  p.noise                = device.noise;

  return p;
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

std::array<float, 3> beamDirection(const LidarParams& p, uint32_t row, uint32_t col) {

  const auto fraction = [](uint32_t i, int n) { return n > 1 ? double(i) / double(n - 1) : 0.0; };

  const double elevation = (-p.vertical_fov_down + (p.vertical_fov_up + p.vertical_fov_down) * fraction(p.vertical_rays - 1 - row, p.vertical_rays)) * kDeg;
  const double azimuth   = (p.horizontal_fov_left - (p.horizontal_fov_left + p.horizontal_fov_right) * fraction(col, p.horizontal_rays)) * kDeg;

  return {float(std::cos(elevation) * std::cos(azimuth)), float(std::cos(elevation) * std::sin(azimuth)), float(std::sin(elevation))};
}

std::optional<LidarClouds> toLidarClouds(const flightforge::LidarScan& scan, const LidarParams& p, const std_msgs::msg::Header& header, std::mt19937& rng) {

  const size_t n = scan.points.size();

  if ((p.intensity && scan.intensity.size() != n) || (p.labels && (scan.class_id.size() != n || scan.instance_id.size() != n))) {
    return std::nullopt;
  }

  const auto hit = [](const std::array<float, 3>& pt) { return std::isfinite(pt[0]) && std::isfinite(pt[1]) && std::isfinite(pt[2]); };

  const bool organised = scan.rows == uint32_t(p.vertical_rays) && scan.cols == uint32_t(p.horizontal_rays) && size_t(scan.rows) * scan.cols == n;

  const size_t n_hits = std::count_if(scan.points.begin(), scan.points.end(), hit);
  const size_t n_free = organised ? n - n_hits : 0;

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

    const auto& pt = scan.points[i];

    if (!hit(pt)) {

      if (organised) {
        const auto dir = beamDirection(p, i / scan.cols, i % scan.cols);
        *fx = dir[0], *fy = dir[1], *fz = dir[2];
        ++fx, ++fy, ++fz;
      }

      continue;
    }

    double scale = 1.0;

    if (p.noise.enabled) {
      const double range = std::sqrt(double(pt[0]) * pt[0] + double(pt[1]) * pt[1] + double(pt[2]) * pt[2]);
      if (range > 0) {
        scale += std::normal_distribution<double>(0.0, range * p.noise.std_slope * p.noise.std_at_1m)(rng) / range;
      }
    }

    *hx = pt[0] * scale;
    *hy = pt[1] * scale;
    *hz = pt[2] * scale;
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

/* events and labels //{ */

event_camera_msgs::msg::EventPacket toEventPacket(const flightforge::CameraEvents& ev, const std_msgs::msg::Header& header) {

  event_camera_msgs::msg::EventPacket pkt;

  pkt.header       = header;
  pkt.width        = ev.width;
  pkt.height       = ev.height;
  pkt.encoding     = "evt3";
  pkt.is_bigendian = false;
  pkt.time_base    = rclcpp::Time(header.stamp).nanoseconds();

  event_camera_codecs::evt3::Encoder encoder;

  const uint64_t base = uint64_t(ev.base_stamp * 1e9);
  uint64_t       anchor = base;

  encoder.setBuffer(&pkt.events);
  encoder.setSensorTime(anchor);

  const size_t n = std::min(ev.xyp.size(), ev.t_us.size());

  for (size_t i = 0; i < n; i++) {

    const uint64_t t = base + uint64_t(ev.t_us[i]) * 1000;

    // the encoder takes an int32 ns offset, which overflows past 2.1 s
    if (t - anchor > 1000000000ull) {
      anchor = t;
      encoder.setSensorTime(anchor);
    }

    const uint32_t xyp = ev.xyp[i];
    encoder.encodeCD(int32_t(t - anchor), xyp & 0xFFFFu, (xyp >> 16) & 0x7FFFu, (xyp >> 31) & 1u);
  }

  encoder.flush();

  return pkt;
}

mrs_uav_flightforge_simulator::msg::InstanceMap toInstanceMap(const std::vector<flightforge::wire::InstanceSegEntryT>& entries,
                                                              const std_msgs::msg::Header&                          header) {

  mrs_uav_flightforge_simulator::msg::InstanceMap map;

  map.header = header;

  for (const auto& e : entries) {
    map.ids.push_back(e.id);
    map.actors.push_back(e.actor);
    map.class_ids.push_back(e.class_id);
  }

  return map;
}

//}

}  // namespace mrs_uav_flightforge_simulator
