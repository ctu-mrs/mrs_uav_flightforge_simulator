#pragma once

#include <memory>
#include <random>
#include <string>

#include <image_transport/image_transport.hpp>
#include <mrs_lib/publisher_handler.h>
#include <mrs_lib/timer_handler.h>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/range.hpp>

#include <event_camera_msgs/msg/event_packet.hpp>

#include <mrs_uav_flightforge_simulator/msg/instance_map.hpp>
#include <mrs_uav_flightforge_simulator/sensor_config.h>

namespace mrs_uav_flightforge_simulator
{

#if USE_ROS_TIMER == 1
typedef mrs_lib::ROSTimer TimerType;
#else
typedef mrs_lib::ThreadTimer TimerType;
#endif

/* sensor instances //{ */

// rgb, rgb_segmented, depth and instance_seg
struct CameraSensor
{
  CameraParams                                                              params;
  int                                                                       type = 0;
  int                                                                       id   = -1;
  std::string                                                               name;
  std::string                                                               frame;
  image_transport::Publisher                                                ph_image;
  mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>                   ph_info;
  mrs_lib::PublisherHandler<mrs_uav_flightforge_simulator::msg::InstanceMap> ph_map;
  bool                                                                      map_published = false;
  sensor_msgs::msg::CameraInfo                                              info;
  std::shared_ptr<TimerType>                                                timer;
  double                                                                    last_stamp = 0.0;
};

struct FisheyeSensor
{
  FisheyeParams                                           params;
  int                                                     id = -1;
  std::string                                             name;
  std::string                                             frame;
  image_transport::Publisher                              ph_image;
  mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo> ph_info;
  sensor_msgs::msg::CameraInfo                            info;
  std::shared_ptr<TimerType>                              timer;
  double                                                  last_stamp = 0.0;
};

struct StereoSensor
{
  StereoParams                                            params;
  int                                                     id = -1;
  std::string                                             name;
  std::string                                             frame_left, frame_right;
  image_transport::Publisher                              ph_image_left, ph_image_right;
  mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo> ph_info_left, ph_info_right;
  sensor_msgs::msg::CameraInfo                            info_left, info_right;
  std::shared_ptr<TimerType>                              timer;
  double                                                  last_stamp = 0.0;
};

struct EventSensor
{
  EventParams                                                    params;
  int                                                            id = -1;
  std::string                                                    name;
  std::string                                                    frame;
  mrs_lib::PublisherHandler<event_camera_msgs::msg::EventPacket> ph_events;
  std::shared_ptr<TimerType>                                     timer;
  double                                                         last_stamp = 0.0;
};

struct LidarSensor
{
  LidarParams                                              params;
  int                                                      id = -1;
  std::string                                              name;
  std::string                                              frame;
  mrs_lib::PublisherHandler<sensor_msgs::msg::PointCloud2> ph_points, ph_free;
  std::shared_ptr<TimerType>                               timer;
  double                                                   last_stamp = 0.0;
  std::mt19937                                             rng{std::random_device{}()};  // each timer has its own thread
};

struct RangefinderSensor
{
  RangefinderParams                                  params;
  int                                                id = -1;
  std::string                                        name;
  std::string                                        frame;
  mrs_lib::PublisherHandler<sensor_msgs::msg::Range> ph_range;
  std::shared_ptr<TimerType>                         timer;
};

//}

}  // namespace mrs_uav_flightforge_simulator
