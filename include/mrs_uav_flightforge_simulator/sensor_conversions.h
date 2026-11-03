#pragma once

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include <flight_forge_connector/data_types.h>
#include <flight_forge_connector/serialization/serializable_shared.h>

#include <mrs_uav_flightforge_simulator/sensor_config.h>

namespace mrs_uav_flightforge_simulator
{

/* images //{ */

// nullopt when empty, undecodable, or not width x height
std::optional<cv::Mat> decodeFrame(const std::vector<unsigned char>& data, int width, int height);

sensor_msgs::msg::Image toColorImage(const cv::Mat& bgr, const std_msgs::msg::Header& header);

sensor_msgs::msg::Image toInstanceImage(const cv::Mat& bgr, const std_msgs::msg::Header& header);

std::optional<sensor_msgs::msg::Image> toDepthImage(const std::vector<uint16_t>& mm, int width, int height, const std_msgs::msg::Header& header);

//}

/* calibration and mounting //{ */

sensor_msgs::msg::CameraInfo toCameraInfo(const Serializable::Drone::CameraInfo& ci);

const Serializable::Drone::CameraInfo* findEye(const std::vector<Serializable::Drone::CameraInfo>& infos, int eye);

ueds_connector::Coordinates toUeOffset(const Mount& m);

ueds_connector::Rotation toUeRotation(const Mount& m);

//}

}  // namespace mrs_uav_flightforge_simulator
