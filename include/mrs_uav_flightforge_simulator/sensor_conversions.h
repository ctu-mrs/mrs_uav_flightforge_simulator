#pragma once

#include <array>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

#include <event_camera_msgs/msg/event_packet.hpp>

#include <mrs_uav_flightforge_simulator/msg/instance_map.hpp>

#include <flight_forge_connector/data_types.h>

#include <mrs_uav_flightforge_simulator/sensor_config.h>

namespace mrs_uav_flightforge_simulator
{

/* images //{ */

// BGR8; nullopt when empty, undecodable, or not width x height
std::optional<cv::Mat> toBgr(const flightforge::Image& image, int width, int height);

sensor_msgs::msg::Image toColorImage(const cv::Mat& bgr, const std_msgs::msg::Header& header);

sensor_msgs::msg::Image toInstanceImage(const cv::Mat& bgr, const std_msgs::msg::Header& header);

std::optional<sensor_msgs::msg::Image> toDepthImage(const std::vector<uint16_t>& mm, int width, int height, const std_msgs::msg::Header& header);

//}

/* calibration and mounting //{ */

sensor_msgs::msg::CameraInfo toCameraInfo(const flightforge::wire::CameraInfoT& ci);

const flightforge::wire::CameraInfoT* findEye(const std::vector<flightforge::wire::CameraInfoT>& infos, flightforge::wire::Eye eye);

flightforge::Pose toPose(const Mount& m);

// a missing pose is the identity
Mount toMount(const flightforge::wire::Pose* pose);

// the eyes sit at +-baseline/2 along the rig's y
std::pair<Mount, Mount> stereoEyes(const Mount& rig, double baseline);

// the simulator's rangefinder beam is its mount's -z, sensor_msgs/Range's is +x
flightforge::Pose rangefinderPose(const Mount& m);

//}

/* sensor configs //{ */

void applyCameraParams(const CameraParams& p, flightforge::wire::RgbCameraConfigT& cfg);

void applyFisheyeParams(const FisheyeParams& p, flightforge::wire::FisheyeCameraConfigT& cfg);

void applyStereoParams(const StereoParams& p, flightforge::wire::StereoCameraConfigT& cfg);

void applyEventParams(const EventParams& p, flightforge::wire::EventCameraConfigT& cfg);

void applyLidarParams(const LidarParams& p, flightforge::wire::LidarConfigT& cfg);

//}

/* device sensors, as the simulator built them //{ */

CameraParams toCameraParams(const flightforge::wire::RgbCameraConfigT& cfg, double rate);

FisheyeParams toFisheyeParams(const flightforge::wire::FisheyeCameraConfigT& cfg, double rate);

StereoParams toStereoParams(const flightforge::wire::StereoCameraConfigT& cfg, double rate);

LidarParams toLidarParams(const flightforge::wire::LidarConfigT& cfg, const DeviceParams& device);

//}

/* lidar //{ */

struct LidarClouds
{
  sensor_msgs::msg::PointCloud2 points, free;
};

// unit vector of an organised scan's beam, row 0 the top beam and column 0 the leftmost
std::array<float, 3> beamDirection(const LidarParams& p, uint32_t row, uint32_t col);

// nullopt when a requested channel is not parallel to the points; free stays empty unless the scan is organised as p says
std::optional<LidarClouds> toLidarClouds(const flightforge::LidarScan& scan, const LidarParams& p, const std_msgs::msg::Header& header, std::mt19937& rng);

//}

/* events and labels //{ */

event_camera_msgs::msg::EventPacket toEventPacket(const flightforge::CameraEvents& ev, const std_msgs::msg::Header& header);

mrs_uav_flightforge_simulator::msg::InstanceMap toInstanceMap(const std::vector<flightforge::wire::InstanceSegEntryT>& entries,
                                                              const std_msgs::msg::Header&                          header);

//}

}  // namespace mrs_uav_flightforge_simulator
