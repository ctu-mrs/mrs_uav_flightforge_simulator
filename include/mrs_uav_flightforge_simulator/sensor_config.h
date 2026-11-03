#pragma once

#include <string>
#include <vector>

#include <mrs_lib/param_loader.h>

namespace mrs_uav_flightforge_simulator
{

/* params //{ */

// [m] and [deg]
struct Mount
{
  double x = 0, y = 0, z = 0;
  double roll = 0, pitch = 0, yaw = 0;
};

struct CameraParams
{
  double rate;
  int    width, height;
  double fov;
  Mount  mount;
  bool   enable_hdr, enable_temporal_aa, enable_raytracing, enable_motion_blur;
  double motion_blur_amount, motion_blur_distortion;
};

struct FisheyeParams
{
  double rate;
  int    width, height;
  double fov;
  int    lens_model;
  bool   full_frame;
  Mount  mount;
};

struct StereoParams
{
  double rate;
  int    width, height;
  double fov;
  double baseline;  // [m]
  Mount  mount;
  bool   enable_hdr, enable_temporal_aa, enable_raytracing;
};

struct EventParams
{
  double rate;
  int    width, height;
  double fov;
  double contrast_threshold_pos, contrast_threshold_neg;
  int    max_events_per_frame;
  Mount  mount;
};

struct LidarNoise
{
  bool   enabled;
  double std_at_1m, std_slope;
};

struct LidarParams
{
  double      rate;
  bool        livox;
  std::string livox_sensor;
  bool        intensity, labels, publish_free;
  int         horizontal_rays, vertical_rays;
  double      horizontal_fov_left, horizontal_fov_right, vertical_fov_up, vertical_fov_down;
  double      beam_length;  // [m]
  Mount       mount;
  LidarNoise  noise;
};

struct RangefinderParams
{
  double rate;
  Mount  mount;
};

struct SensorsConfig
{
  std::vector<CameraParams>      rgb, rgb_segmented, depth, instance_seg;
  std::vector<FisheyeParams>     fisheye;
  std::vector<StereoParams>      stereo;
  std::vector<EventParams>       event;
  std::vector<LidarParams>       lidar;
  std::vector<RangefinderParams> rangefinder;

  double highestRate() const;
};

//}

extern const std::vector<std::string> kSensorTypes;

// throws std::runtime_error naming the offending parameter path
SensorsConfig loadSensors(mrs_lib::ParamLoader& loader, const std::string& prefix);

// "" when valid, else a message naming the offending key path
std::string validateSensorKeys(const std::vector<std::string>& yaml_files, const std::string& root_key);

}  // namespace mrs_uav_flightforge_simulator
