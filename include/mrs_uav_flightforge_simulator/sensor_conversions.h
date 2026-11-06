#pragma once

#include <optional>
#include <random>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

#include <event_camera_msgs/msg/event_packet.hpp>

#include <mrs_uav_flightforge_simulator/msg/instance_map.hpp>

#include <flight_forge_connector/data_types.h>
#include <flight_forge_connector/serialization/serializable_shared.h>

#include <mrs_uav_flightforge_simulator/sensor_config.h>

namespace mrs_uav_flightforge_simulator
{

/* images //{ */

// nullopt when empty, undecodable, or not width x height
std::optional<cv::Mat> decodeFrame(const std::vector<unsigned char>& data, int width, int height);

// the segmentation camera sends uncompressed BGR
std::optional<cv::Mat> rawFrame(const std::vector<unsigned char>& data, int width, int height);

sensor_msgs::msg::Image toColorImage(const cv::Mat& bgr, const std_msgs::msg::Header& header);

sensor_msgs::msg::Image toInstanceImage(const cv::Mat& bgr, const std_msgs::msg::Header& header);

std::optional<sensor_msgs::msg::Image> toDepthImage(const std::vector<uint16_t>& mm, int width, int height, const std_msgs::msg::Header& header);

//}

/* calibration and mounting //{ */

sensor_msgs::msg::CameraInfo toCameraInfo(const Serializable::Drone::CameraInfo& ci);

const Serializable::Drone::CameraInfo* findEye(const std::vector<Serializable::Drone::CameraInfo>& infos, int eye);

ueds_connector::Coordinates toUeOffset(const Mount& m);

ueds_connector::Rotation toUeRotation(const Mount& m);

Mount fromUe(const ueds_connector::Coordinates& offset, const ueds_connector::Rotation& rotation);

//}

/* device sensors, as the simulator built them //{ */

CameraParams toCameraParams(const ueds_connector::RgbCameraConfig& cfg, double rate);

FisheyeParams toFisheyeParams(const ueds_connector::FisheyeCameraConfig& cfg, double rate);

StereoParams toStereoParams(const Serializable::Drone::StereoCameraConfig& cfg, double rate);

Mount leftEyeMount(const Serializable::Drone::StereoCameraConfig& cfg);

Mount rightEyeMount(const Serializable::Drone::StereoCameraConfig& cfg);

LidarParams toLidarParams(const ueds_connector::LidarConfig& cfg, const DeviceParams& device);

//}

/* lidar //{ */

struct LidarClouds
{
  sensor_msgs::msg::PointCloud2 points, free;
};

// nullopt when a requested channel is not parallel to the points
std::optional<LidarClouds> toLidarClouds(const ueds_connector::LidarScan& scan, const LidarParams& p, const std_msgs::msg::Header& header, std::mt19937& rng);

//}

/* events and labels //{ */

event_camera_msgs::msg::EventPacket toEventPacket(const ueds_connector::CameraEvents& ev, const std_msgs::msg::Header& header);

mrs_uav_flightforge_simulator::msg::InstanceMap toInstanceMap(const std::vector<Serializable::Drone::InstanceSegMapEntry>& entries,
                                                              const std_msgs::msg::Header&                             header);

//}

}  // namespace mrs_uav_flightforge_simulator
