/* includes //{ */

#include <algorithm>

#include <rclcpp/rclcpp.hpp>

#include <mrs_multirotor_simulator/uav_system_ros.h>

#include <rosgraph_msgs/msg/clock.hpp>
#include <std_msgs/msg/bool.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>

#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/quaternion_stamped.hpp>

#include <mrs_lib/param_loader.h>
#include <mrs_lib/publisher_handler.h>
#include <mrs_lib/timer_handler.h>
#include <mrs_lib/scope_timer.h>
#include <mrs_lib/transform_broadcaster.h>
#include <mrs_lib/attitude_converter.h>
#include <mrs_lib/dynparam_mgr.h>

#include <tf2_ros/static_transform_broadcaster.hpp>
#include <tf2_ros/transform_broadcaster.hpp>

#include <tf2/LinearMath/Vector3.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2/LinearMath/Matrix3x3.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <image_transport/image_transport.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

#include <flight_forge_connector/robot_client.h>
#include <flight_forge_connector/simulator_client.h>

#include <mrs_uav_flightforge_simulator/world.h>

#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>

#include <cctype>
#include <cmath>
#include <limits>
#include <condition_variable>
#include <random>
#include <thread>
#include <vector>
#include <chrono>

#include <mrs_msgs/srv/float64_srv.hpp>
#include <mrs_uav_flightforge_simulator/srv/set_orientation.hpp>

#include <mrs_uav_flightforge_simulator/ground.h>
#include <mrs_uav_flightforge_simulator/sensor_config.h>
#include <mrs_uav_flightforge_simulator/sensor_conversions.h>
#include <mrs_uav_flightforge_simulator/sensor_scheduler.h>
#include <mrs_uav_flightforge_simulator/sensors.h>

//}

/* defines //{ */

using namespace std::chrono_literals;

using PCLPoint               = pcl::PointXYZ;
using PCLPointCloud          = pcl::PointCloud<PCLPoint>;
using PCLPointCloudColor     = pcl::PointCloud<pcl::PointXYZRGB>;
using PCLPointCloudIntensity = pcl::PointCloud<pcl::PointXYZI>;

//}

namespace mrs_uav_flightforge_simulator
{

/* class FlightforgeSimulator //{ */

class FlightforgeSimulator : public rclcpp::Node {

public:
  FlightforgeSimulator(rclcpp::NodeOptions options);

private:
  rclcpp::CallbackGroup::SharedPtr cbgrp_main_;
  rclcpp::CallbackGroup::SharedPtr cbgrp_status_;
  rclcpp::CallbackGroup::SharedPtr cbgrp_unreal_sync_;

  rclcpp::TimerBase::SharedPtr timer_init_;
  void                         timerInit();
  std::string                  setUpWorld();

  rclcpp::Node::SharedPtr  node_;
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Clock::SharedPtr wall_clock_;
  std::atomic<bool>        is_initialized_ = false;

  std::shared_ptr<mrs_lib::ScopeTimerLogger>       scope_timer_logger_;
  std::shared_ptr<image_transport::ImageTransport> it_;

  // | ------------------------- params ------------------------- |

  double _simulation_rate_;
  double _clock_rate_;

  rclcpp::Time sim_time_;
  double       sim_time_wall_ = 0;
  rclcpp::Time last_step_time_;
  std::mutex   mutex_sim_time_;

  std::string _world_frame_name_;
  std::string _simulator_ip_;
  int         _simulator_port_;
  double      _connect_timeout_;

  // | ------------------------- timers ------------------------- |

  /* timer dynamic */
  rclcpp::TimerBase::SharedPtr timer_main_;
  /* std::shared_ptr<mrs_lib::ThreadTimer> timer_main_ */
  void timerMain();

  rclcpp::TimerBase::SharedPtr timer_status_;
  void                         timerStatus();

  rclcpp::TimerBase::SharedPtr timer_time_sync_;
  void                         timerTimeSync();

  rclcpp::TimerBase::SharedPtr timer_unreal_sync_;
  void                         timerUnrealSync();

  // | ------------------------- sensors ------------------------ |

  SensorsConfig sensors_config_;
  double        highest_rate_ = 0.0;

  std::vector<std::vector<CameraSensor>>      cameras_;
  std::vector<std::vector<FisheyeSensor>>     fisheyes_;
  std::vector<std::vector<StereoSensor>>      stereos_;
  std::vector<std::vector<EventSensor>>       events_;
  std::vector<std::vector<LidarSensor>>       lidars_;
  std::vector<std::vector<RangefinderSensor>> rangefinders_;

  struct SensorWorker
  {
    std::function<void()>   job;
    std::thread             thread;
    std::mutex              mutex;
    std::condition_variable cv;
    bool                    pending = false;
    bool                    stop    = false;
    std::atomic<bool>       busy{false};
    std::string             name;
    std::atomic<double>     started{0.0};  // [s] wall time
  };

  std::vector<std::unique_ptr<SensorWorker>> workers_;
  std::vector<ScheduledSensor>               schedule_;

  FetchBarrier fetches_;

  bool         lockstep_ = false;
  rclcpp::Time step_stamp_;
  int          lockstep_stride_      = 1;
  int          steps_since_lockstep_ = 0;

  double       next_tick_wall_ = 0;
  double       main_timer_rtf_ = 0;

  std::atomic<bool> shutting_down_{false};

  void   onShutdown(void);
  double mainTimerPeriod(double rtf) const;

  void         stepSimulator(const rclcpp::Time& sim_time);
  rclcpp::Time worldStamp(double world_time);

  void setupSensors(size_t uav);
  void setupDevices(size_t uav);
  void startSensorWorkers(void);

  [[noreturn]] void failSetup(size_t uav, const std::string& what);

  std::shared_ptr<flightforge::RobotClient>    connectSensor(size_t uav, const std::string& name);
  std::vector<flightforge::wire::CameraInfoT>  calibration(size_t uav, int id, const std::string& name);
  void                                              warnOnSize(size_t uav, const sensor_msgs::msg::CameraInfo& info, int width, int height, const std::string& name);
  void                                              registerCamera(size_t uav, CameraSensor s, const std::string& topic);
  void                                              registerFisheye(size_t uav, FisheyeSensor s, const std::string& topic);
  void                                              registerStereo(size_t uav, StereoSensor s, const std::string& topic, const Mount& left, const Mount& right);
  void                                              registerLidar(size_t uav, LidarSensor s, const std::string& topic);

  void dispatchDueSensors(double sim_time);
  void sendStaticTf(const std::string& uav_name, const std::string& frame, const Mount& mount, bool optical);

  void fetchCamera(size_t uav, size_t i);
  void fetchFisheye(size_t uav, size_t i);
  void fetchStereo(size_t uav, size_t i);
  void fetchEvent(size_t uav, size_t i);
  void fetchLidar(size_t uav, size_t i);
  void fetchRangefinder(size_t uav, size_t i);

  // free running captures on the simulator's clock, so the frame due for this read can land a tick after it
  template <class Fetch>
  auto fetchFresh(Fetch fetch, double last_stamp, double rate) {

    auto         got      = fetch();
    const double deadline = wall_clock_->now().seconds() + 0.5 / rate;

    while (!lockstep_ && !shutting_down_ && got && got->stamp <= last_stamp && wall_clock_->now().seconds() < deadline) {
      std::this_thread::sleep_for(2ms);
      got = fetch();
    }

    return got;
  }

  // | ------------------------ rtf check ----------------------- |

  double     actual_rtf_ = 1.0;
  std::mutex mutex_actual_rtf_;

  double flightforge_time_dilation_ = 0.0;  // forces the first sync, the sim may still be dilated by a previous run

  rclcpp::Time last_sim_time_status_;

  // | ----------------------- publishers ----------------------- |

  mrs_lib::PublisherHandler<rosgraph_msgs::msg::Clock>                  ph_clock_;
  mrs_lib::PublisherHandler<geometry_msgs::msg::PoseArray>              ph_poses_;

  // | ------------------------- system ------------------------- |

  std::vector<std::unique_ptr<mrs_multirotor_simulator::UavSystemRos>> uavs_;

  // | ------------------------- methods ------------------------ |

  void publishPoses(void);

  // | ------------------------- FlightForge methods ------------------------ |

  std::shared_ptr<flightforge::SimulatorClient> ff_sim_;

  // own connection, so a Step never delays the pose stream
  std::shared_ptr<flightforge::SimulatorClient>          ff_pose_sim_;
  std::vector<std::shared_ptr<flightforge::RobotClient>> ff_robots_;

  std::mutex mutex_flightforge_;

  void updateUnrealPoses(const bool teleport_without_collision);

  flightforge::RobotState robotState(size_t uav);

  void checkForCrash(void);

  // | ------------------------- crashes ------------------------ |

  std::vector<std::string>                                          crash_reasons_;  // empty while the UAV flies
  std::mutex                                                        mutex_crashes_;
  std::vector<mrs_lib::PublisherHandler<std_msgs::msg::Bool>>       ph_crashed_;
  mrs_lib::PublisherHandler<diagnostic_msgs::msg::DiagnosticArray> ph_diagnostics_;

  void crashUav(size_t uav, const std::string& reason, bool hit_obstacle);
  void settleCrashed(size_t uav);
  void publishCrashes(void);

  // | ------------------------- ground ------------------------- |

  bool ground_enabled_        = false;
  bool ground_from_simulator_ = false;
  bool ground_robot_ids_      = false;  // API 1.8: each UAV stands on other robots, never on itself

  std::vector<double> clearance_;  // [m] of the base link above the surface it rests on
  std::vector<double> surface_;    // [m] highest surface under each UAV's footprint, NaN for none
  std::vector<double> applied_ground_z_;

  // own connection, so a game-thread hop never holds the main loop
  std::shared_ptr<flightforge::SimulatorClient> ff_ground_sim_;

  struct GroundPose
  {
    double x, y, z, yaw;
  };

  std::thread                 ground_thread_;
  std::mutex                  mutex_ground_;
  std::condition_variable     cv_ground_;
  std::vector<GroundPose>     ground_poses_;
  std::vector<double>         ground_surfaces_;
  bool                        ground_request_ = false;
  bool                        ground_stop_    = false;
  int                         ground_queries_ = 0;
  double                      ground_latency_ = 0.0;  // [s] summed since the last status
  std::atomic<bool>           first_step_done_{false};

  void        spawnRobot(size_t uav, const std::string& frame);
  void        groundWorker(void);
  void        requestGround(void);
  void        applyGround(void);
  GroundPose  groundPose(size_t uav);

  rclcpp::Time ueWorldTimeToSimtime(const double ue_world_time);

  std::mutex mutex_ue_world_offset_;
  double     ue_world_offset_       = 0;  // ROS sim time minus UE world time
  bool       ue_world_offset_valid_ = false;

  double                   flightforge_fps_ = 0;
  std::string              flightforge_world_level_name_enum_;
  std::string              flightforge_graphics_settings_enum_;
  int                      flightforge_forest_density_     = 5;
  int                      flightforge_forest_hilly_level_ = 3;
  std::string              weather_type_;
  bool                     uavs_mutual_visibility_ = true;
  std::vector<std::string> uav_names;

  int daytime_hour_   = 0;
  int daytime_minute_ = 0;

  // | ------------------------- Transformations ------------------------ |

  std::shared_ptr<mrs_lib::TransformBroadcaster>       tf_broadcaster_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
  std::shared_ptr<mrs_lib::TransformBroadcaster>       dynamic_broadcaster_;

  std::shared_ptr<mrs_lib::DynparamMgr> dynparam_mgr_;

  struct drs_params
  {
    double realtime_factor = 1.0;
    bool   dynamic_rtf     = false;
    bool   paused          = false;

    // TODO implement mutual collisions
    bool   collisions_enabled  = false;
    bool   collisions_crash    = false;
    double collisions_rebounce = 1;

    double max_touchdown_speed = 2.0;  // [m/s]

  };

  drs_params drs_params_;
  std::mutex mutex_drs_params_;

  void callbackRealtimeFactor(const double& param_value);
  void callbackPause(const bool& param_value);

};

//}

/* FlightforgeSimulator::FlightforgeSimulator() //{ */

FlightforgeSimulator::FlightforgeSimulator(rclcpp::NodeOptions options) : Node("mrs_uav_flightforge_simulator", options) {

  timer_init_ = this->create_wall_timer(std::chrono::duration<double>(0.1s), std::bind(&FlightforgeSimulator::timerInit, this));
}

//}

/* enumByName() //{ */

template <typename E, size_t N>
static std::optional<E> enumByName(const E (&values)[N], const char* (*name_of)(E), std::string name) {

  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::toupper(c); });

  for (const E value : values) {
    if (name == name_of(value)) {
      return value;
    }
  }

  return std::nullopt;
}

//}

// | ------------------------- timers ------------------------- |

/* timerInit() //{ */

void FlightforgeSimulator::timerInit() {

  node_       = this->shared_from_this();
  clock_      = node_->get_clock();
  wall_clock_ = std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME);

  srand(time(NULL));

  RCLCPP_INFO(node_->get_logger(), "initializing");

  cbgrp_main_        = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  cbgrp_status_      = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  // the 400 Hz pose sync blocks on a game round trip and would starve the 1 Hz time sync
  cbgrp_unreal_sync_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  mrs_lib::ParamLoader param_loader(node_, node_->get_name());

  dynparam_mgr_ = std::make_shared<mrs_lib::DynparamMgr>(node_, mutex_drs_params_);

  // | ----------------------- load files ----------------------- |

  // load custom config

  std::string custom_config_path;
  param_loader.loadParam("custom_config", custom_config_path);

  if (custom_config_path != "") {
    RCLCPP_INFO(node_->get_logger(), "loading custom config '%s", custom_config_path.c_str());
    param_loader.addYamlFile(custom_config_path);
  }

  // load other configs

  std::vector<std::string> config_files;
  param_loader.loadParam("config_files", config_files);

  for (auto config_file : config_files) {
    RCLCPP_INFO(node_->get_logger(), "loading config file '%s'", config_file.c_str());
    param_loader.addYamlFile(config_file);
  }

  dynparam_mgr_->get_param_provider().copyYamls(param_loader.getParamProvider());

  // | ----------------------- load params ---------------------- |

  std::string yaml_prefix = "mrs_uav_flightforge_simulator/";

  param_loader.loadParam(yaml_prefix + "simulation_rate", _simulation_rate_);
  param_loader.loadParam(yaml_prefix + "clock_rate", _clock_rate_);
  param_loader.loadParam(yaml_prefix + "simulator_ip", _simulator_ip_);
  param_loader.loadParam(yaml_prefix + "simulator_port", _simulator_port_, int(flightforge::Connection::kDefaultPort));

  if (_simulator_ip_.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "simulator_ip parameter is empty!");
    rclcpp::shutdown();
    exit(1);
  }

  if (_simulator_port_ < 1 || _simulator_port_ > 65535) {
    RCLCPP_ERROR(node_->get_logger(), "simulator_port %d is not a valid TCP port!", _simulator_port_);
    rclcpp::shutdown();
    exit(1);
  }

  RCLCPP_INFO(node_->get_logger(), "Using simulator at %s:%d", _simulator_ip_.c_str(), _simulator_port_);

  if (_clock_rate_ < _simulation_rate_) {
    RCLCPP_ERROR(node_->get_logger(), "clock_rate (%.2f Hz) should be higher than simulation rate (%.2f Hz)!", _clock_rate_, _simulation_rate_);
    rclcpp::shutdown();
    exit(1);
  }

  dynparam_mgr_->register_param(yaml_prefix + "realtime_factor", &drs_params_.realtime_factor, mrs_lib::DynparamMgr::range_t<double>(0.01, 10),
                                (std::function<void(const double&)>)std::bind(&FlightforgeSimulator::callbackRealtimeFactor, this, std::placeholders::_1));

  dynparam_mgr_->register_param(yaml_prefix + "dynamic_rtf", &drs_params_.dynamic_rtf);

  dynparam_mgr_->register_param(yaml_prefix + "collisions/enabled", &drs_params_.collisions_enabled);

  dynparam_mgr_->register_param(yaml_prefix + "collisions/crash", &drs_params_.collisions_crash);

  dynparam_mgr_->register_param(yaml_prefix + "collisions/rebounce", &drs_params_.collisions_rebounce, mrs_lib::DynparamMgr::range_t<double>(0.1, 1000));

  dynparam_mgr_->register_param(yaml_prefix + "collisions/max_touchdown_speed", &drs_params_.max_touchdown_speed,
                                mrs_lib::DynparamMgr::range_t<double>(0.0, 100.0));

  dynparam_mgr_->register_param(yaml_prefix + "paused", &drs_params_.paused, false,
                                (std::function<void(const bool&)>)std::bind(&FlightforgeSimulator::callbackPause, this, std::placeholders::_1));

  param_loader.loadParam("frames/world/name", _world_frame_name_);

  bool sim_time_from_wall_time;
  param_loader.loadParam(yaml_prefix + "sim_time_from_wall_time", sim_time_from_wall_time);

  if (sim_time_from_wall_time) {
    sim_time_       = clock_->now();
    last_step_time_ = clock_->now();
  } else {
    sim_time_       = rclcpp::Time(0, 0, RCL_ROS_TIME);
    last_step_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  }

  {
    std::vector<std::string> yaml_files = config_files;

    if (custom_config_path != "") {
      yaml_files.insert(yaml_files.begin(), custom_config_path);
    }

    const std::string key_error = validateSensorKeys(yaml_files, "mrs_uav_flightforge_simulator");

    if (!key_error.empty()) {
      RCLCPP_ERROR(node_->get_logger(), "%s", key_error.c_str());
      rclcpp::shutdown();
      exit(1);
    }

    try {
      sensors_config_ = loadSensors(param_loader, yaml_prefix);
      highest_rate_   = sensors_config_.highestRate();
    }
    catch (const std::exception& e) {
      RCLCPP_ERROR(node_->get_logger(), "%s", e.what());
      rclcpp::shutdown();
      exit(1);
    }
  }

  std::string time_model;
  param_loader.loadParam(yaml_prefix + "time_model", time_model);

  if (time_model != "free_running" && time_model != "lockstep") {
    RCLCPP_ERROR(node_->get_logger(), "time_model must be \"free_running\" or \"lockstep\", got \"%s\"", time_model.c_str());
    rclcpp::shutdown();
    exit(1);
  }

  lockstep_ = time_model == "lockstep";

  if (lockstep_ && !lockstepClockValid(_clock_rate_, _simulation_rate_)) {
    RCLCPP_ERROR(node_->get_logger(), "lockstep needs clock_rate (%.1f Hz) to be a whole multiple of simulation_rate (%.1f Hz)", _clock_rate_, _simulation_rate_);
    rclcpp::shutdown();
    exit(1);
  }

  param_loader.loadParam(yaml_prefix + "weather_type", weather_type_);
  param_loader.loadParam(yaml_prefix + "graphics_settings", flightforge_graphics_settings_enum_);
  param_loader.loadParam(yaml_prefix + "uavs_mutual_visibility", uavs_mutual_visibility_);
  param_loader.loadParam(yaml_prefix + "daytime/hour", daytime_hour_);
  param_loader.loadParam(yaml_prefix + "daytime/minute", daytime_minute_);
  param_loader.loadParam(yaml_prefix + "world_name", flightforge_world_level_name_enum_);
  param_loader.loadParam(yaml_prefix + "connect_timeout", _connect_timeout_);
  param_loader.loadParam(yaml_prefix + "forest_density", flightforge_forest_density_);
  param_loader.loadParam(yaml_prefix + "forest_hilly_level", flightforge_forest_hilly_level_);

  param_loader.loadParam("ground/enabled", ground_enabled_);
  param_loader.loadParam("ground/from_simulator", ground_from_simulator_);

  ground_from_simulator_ = ground_enabled_ && ground_from_simulator_;

  std::vector<std::string> scene_files, libraries;
  param_loader.loadParam(yaml_prefix + "scene_files", scene_files);
  param_loader.loadParam(yaml_prefix + "libraries", libraries);

  if (!param_loader.loadedSuccessfully() || !dynparam_mgr_->loaded_successfully()) {
    RCLCPP_ERROR(node_->get_logger(), "could not load all parameters!");
    rclcpp::shutdown();
    exit(1);
  }

  last_sim_time_status_ = sim_time_;

  drs_params_.paused = false;

  tf_broadcaster_      = std::make_shared<mrs_lib::TransformBroadcaster>(node_);
  static_broadcaster_  = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node_);
  dynamic_broadcaster_ = std::make_shared<mrs_lib::TransformBroadcaster>(node_);

  it_ = std::make_shared<image_transport::ImageTransport>(node_);

  /* std::vector<std::string> uav_names; */
  param_loader.loadParam("uav_names", uav_names);

  for (size_t i = 0; i < uav_names.size(); i++) {

    std::string uav_name = uav_names.at(i);

    RCLCPP_INFO(node_->get_logger(), "initializing '%s'", uav_name.c_str());

    mrs_multirotor_simulator::UavSystemRos_CommonHandlers_t common_handlers;

    common_handlers.node                  = node_;
    common_handlers.uav_name              = uav_name;
    common_handlers.transform_broadcaster = tf_broadcaster_;

    uavs_.push_back(std::make_unique<mrs_multirotor_simulator::UavSystemRos>(common_handlers));
  }

  RCLCPP_INFO(node_->get_logger(), "all uavs initialized");

  // | ----------- initialize the FlightForge connector ---------- |

  const auto weather  = enumByName(flightforge::wire::EnumValuesWeather(), flightforge::wire::EnumNameWeather, weather_type_);
  const auto graphics = enumByName(flightforge::wire::EnumValuesGraphicsQuality(), flightforge::wire::EnumNameGraphicsQuality, flightforge_graphics_settings_enum_);

  if (!weather || !graphics) {
    RCLCPP_ERROR(node_->get_logger(), "unknown %s '%s'", weather ? "graphics_settings" : "weather_type",
                 weather ? flightforge_graphics_settings_enum_.c_str() : weather_type_.c_str());
    rclcpp::shutdown();
    exit(1);
  }

  ff_sim_ = std::make_shared<flightforge::SimulatorClient>(_simulator_ip_, _simulator_port_);

  const auto connect_start = std::chrono::steady_clock::now();

  // a proxy that starts FlightForge on demand accepts before the simulator can answer, so an unanswered GetInfo is retried like a refused connect;
  // the late reply is skipped by its seq
  flightforge::Expected<flightforge::ApiInfo> info;

  while (rclcpp::ok()) {
    if (!ff_sim_->Connect()) {
      RCLCPP_ERROR(node_->get_logger(), "could not connect to FlightForge at %s:%d", _simulator_ip_.c_str(), _simulator_port_);
    } else if (!(info = ff_sim_->GetInfo())) {
      RCLCPP_ERROR(node_->get_logger(), "FlightForge at %s:%d did not answer GetInfo: %s", _simulator_ip_.c_str(), _simulator_port_, info.error.c_str());
    } else {
      break;
    }

    if (_connect_timeout_ > 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - connect_start).count() > _connect_timeout_) {
      RCLCPP_ERROR(node_->get_logger(), "could not connect to FlightForge within %.1f s, is it running and does its API version match?", _connect_timeout_);
      rclcpp::shutdown();
      exit(1);
    }

    std::this_thread::sleep_for(1s);
  }

  {
    if (!info || info->major != flightforge::SimulatorClient::kApiMajor) {
      const std::string theirs = info ? std::to_string(info->major) + "." + std::to_string(info->minor) : "unknown (" + info.error + ")";
      RCLCPP_ERROR(node_->get_logger(), "FlightForge API %s, this node speaks %d.%d; update either the simulator or this package so they agree",
                   theirs.c_str(), flightforge::SimulatorClient::kApiMajor, flightforge::SimulatorClient::kApiMinor);
      rclcpp::shutdown();
      exit(1);
    }

    RCLCPP_INFO(node_->get_logger(), "FlightForge API %d.%d (this node %d.%d)", info->major, info->minor, flightforge::SimulatorClient::kApiMajor,
                flightforge::SimulatorClient::kApiMinor);

    if (ground_from_simulator_ && info->minor < 1) {
      RCLCPP_ERROR(node_->get_logger(), "ground/from_simulator needs FlightForge API 1.1 or newer for get_surface_height; update the simulator or set it false");
      rclcpp::shutdown();
      exit(1);
    }

    ground_robot_ids_ = info->minor >= 8;
  }

  const std::string world = setUpWorld();

  {
    // a node that died in lockstep leaves the world paused, and a loading level does not answer yet
    const auto wait_from = std::chrono::steady_clock::now();
    bool       reset     = false;

    while (!reset && rclcpp::ok() && std::chrono::steady_clock::now() - wait_from < 15min) {

      reset = ff_sim_->SetLockstep(false) && ff_sim_->SetFixedTimestep(0.0) &&
              ff_sim_->SetCaptureMode(lockstep_ ? flightforge::wire::CaptureMode::ON_DEMAND : flightforge::wire::CaptureMode::ALL_FRAMES);

      if (!reset) {
        RCLCPP_INFO_THROTTLE(node_->get_logger(), *wall_clock_, 10000, "waiting for the world to load");
        std::this_thread::sleep_for(1s);
      }
    }

    if (!reset) {
      RCLCPP_ERROR(node_->get_logger(), "could not reset the FlightForge time model");
      rclcpp::shutdown();
      exit(1);
    }
  }

  const auto report = [this](const flightforge::Status& status, const std::string& what) {
    if (status) {
      RCLCPP_INFO(node_->get_logger(), "%s", what.c_str());
    } else if (status.error.find("has no sky system") != std::string::npos) {
      RCLCPP_WARN(node_->get_logger(), "could not %s: %s", what.c_str(), status.error.c_str());
    } else {
      RCLCPP_ERROR(node_->get_logger(), "could not %s: %s", what.c_str(), status.error.c_str());
    }
  };

  report(ff_sim_->SetGraphicsSettings(*graphics), "set the graphics to " + flightforge_graphics_settings_enum_);
  report(ff_sim_->SetMutualVisibility(uavs_mutual_visibility_), std::string("set the mutual visibility to ") + (uavs_mutual_visibility_ ? "true" : "false"));
  report(ff_sim_->SetWeather(*weather), "set the weather to " + weather_type_);
  char daytime[8];
  std::snprintf(daytime, sizeof(daytime), "%02d:%02d", daytime_hour_ % 100, daytime_minute_ % 100);
  report(ff_sim_->SetDaytime(daytime_hour_, daytime_minute_), std::string("set the daytime to ") + daytime);

  if (isWorld(world, "infinite_forest")) {
    report(ff_sim_->SetForestDensity(flightforge_forest_density_), "set the forest density to " + std::to_string(flightforge_forest_density_));
    report(ff_sim_->SetForestHillyLevel(flightforge_forest_hilly_level_), "set the forest hilly level to " + std::to_string(flightforge_forest_hilly_level_));
  }

  for (const auto& library : libraries) {

    const auto loaded = ff_sim_->LoadLibrary(library);

    if (!loaded) {
      RCLCPP_ERROR(node_->get_logger(), "could not load library %s: %s", library.c_str(), loaded.error.c_str());
      rclcpp::shutdown();
      exit(1);
    }

    RCLCPP_INFO(node_->get_logger(), "loaded %d objects from %s", *loaded, library.c_str());
  }

  if (!scene_files.empty()) {

    // objects outlive the node, a restart would double them
    ff_sim_->RemoveAllSpawnedObjects();

    for (const auto& file : scene_files) {

      std::string document;

      try {
        document = loadScene(file);
      }
      catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "%s", e.what());
        rclcpp::shutdown();
        exit(1);
      }

      const auto placed = ff_sim_->SpawnObjectsFromYaml(document);

      if (!placed) {
        RCLCPP_ERROR(node_->get_logger(), "could not place scene %s: %s", file.c_str(), placed.error.c_str());
        rclcpp::shutdown();
        exit(1);
      }

      RCLCPP_INFO(node_->get_logger(), "placed %zu objects from %s", placed->size(), file.c_str());
    }
  }

  // | --------------------- Spawn the UAVs in FlightForge --------------------- |

  ff_pose_sim_ = std::make_shared<flightforge::SimulatorClient>(_simulator_ip_, _simulator_port_);

  if (!ff_pose_sim_->Connect(3)) {
    RCLCPP_ERROR(node_->get_logger(), "could not open the pose connection to FlightForge");
    rclcpp::shutdown();
    exit(1);
  }

  clearance_.assign(uav_names.size(), 0.0);
  surface_.assign(uav_names.size(), std::numeric_limits<double>::quiet_NaN());
  applied_ground_z_.assign(uav_names.size(), std::numeric_limits<double>::quiet_NaN());

  for (size_t i = 0; i < uav_names.size(); i++) {
    spawnRobot(i, "x500");
    setupSensors(i);
  }

  if (ground_from_simulator_) {

    ff_ground_sim_ = std::make_shared<flightforge::SimulatorClient>(_simulator_ip_, _simulator_port_);

    if (!ff_ground_sim_->Connect(3)) {
      RCLCPP_ERROR(node_->get_logger(), "could not open the ground connection to FlightForge");
      rclcpp::shutdown();
      exit(1);
    }

    ground_surfaces_ = surface_;
    applyGround();

    ground_thread_ = std::thread(&FlightforgeSimulator::groundWorker, this);
  }

  // | ----------------------- publishers ----------------------- |

  ph_clock_ = mrs_lib::PublisherHandler<rosgraph_msgs::msg::Clock>(node_, "~/clock_out");

  ph_poses_ = mrs_lib::PublisherHandler<geometry_msgs::msg::PoseArray>(node_, "~/uav_poses_out");

  crash_reasons_.assign(uav_names.size(), "");

  {
    mrs_lib::PublisherHandlerOptions latched(node_);
    latched.qos = rclcpp::QoS(1).transient_local();

    for (const auto& uav_name : uav_names) {
      ph_crashed_.emplace_back(latched, "~/" + uav_name + "/crashed");
      ph_crashed_.back().publish(std_msgs::msg::Bool());
    }
  }

  ph_diagnostics_ = mrs_lib::PublisherHandler<diagnostic_msgs::msg::DiagnosticArray>(node_, "/diagnostics");

  // | ------------------------- timers ------------------------- |

  mrs_lib::TimerHandlerOptions opts;

  opts.node      = node_;
  opts.autostart = true;

  timer_main_ = node_->create_wall_timer(std::chrono::duration<double>(mainTimerPeriod(drs_params_.realtime_factor)),
                                         std::bind(&FlightforgeSimulator::timerMain, this), cbgrp_main_);

  timer_status_ = node_->create_wall_timer(std::chrono::duration<double>(1.0), std::bind(&FlightforgeSimulator::timerStatus, this), cbgrp_status_);

  timer_time_sync_ = node_->create_wall_timer(std::chrono::duration<double>(1.0), std::bind(&FlightforgeSimulator::timerTimeSync, this), cbgrp_status_);

  timer_unreal_sync_ =
      node_->create_wall_timer(std::chrono::duration<double>(1.0 / _clock_rate_), std::bind(&FlightforgeSimulator::timerUnrealSync, this), cbgrp_unreal_sync_);

  startSensorWorkers();

  // | ----------------------- scope timer ---------------------- |

  scope_timer_logger_ = std::make_shared<mrs_lib::ScopeTimerLogger>(node_, "", false);

  // | ------------------------ time model ---------------------- |

  if (lockstep_) {

    lockstep_stride_ = lockstepStride(_simulation_rate_, highest_rate_);

    if (!ff_sim_->SetTimeDilation(1.0) || !ff_sim_->SetFixedTimestep(lockstep_stride_ / _simulation_rate_) || !ff_sim_->SetLockstep(true)) {
      RCLCPP_ERROR(node_->get_logger(), "could not enter lockstep");
      rclcpp::shutdown();
      exit(1);
    }

    flightforge_time_dilation_ = 1.0;

    RCLCPP_INFO(node_->get_logger(), "lockstep at %.1f Hz", _simulation_rate_ / lockstep_stride_);
  }

  node_->get_node_base_interface()->get_context()->add_pre_shutdown_callback([this]() { onShutdown(); });

  // | ----------------------- finish init ---------------------- |

  is_initialized_ = true;

  RCLCPP_INFO(node_->get_logger(), "initialized");

  timer_init_->cancel();
}

//}

/* setUpWorld() //{ */

// the package of the world the UAVs spawn into, once it has begun play
std::string FlightforgeSimulator::setUpWorld() {

  const auto wait_ready = [this]() -> std::optional<flightforge::WorldList> {
    const auto wait_from = std::chrono::steady_clock::now();
    while (rclcpp::ok() && std::chrono::steady_clock::now() - wait_from < 30min) {
      const auto list = ff_sim_->ListWorlds();
      if (!list) {
        RCLCPP_ERROR(node_->get_logger(), "could not list the worlds (FlightForge API 1.7 or newer needed): %s", list.error.c_str());
        return std::nullopt;
      }
      if (worldReady(*list)) {
        return *list;
      }
      RCLCPP_INFO_THROTTLE(node_->get_logger(), *wall_clock_, 10000, "waiting for the simulator to finish loading %s",
                           list->loading.empty() ? "its world" : list->loading.c_str());
      std::this_thread::sleep_for(1s);
    }
    return std::nullopt;
  };

  const auto fail = [this](const std::string& why) {
    RCLCPP_ERROR(node_->get_logger(), "%s", why.c_str());
    rclcpp::shutdown();
    exit(1);
  };

  const auto list = wait_ready();
  if (!list) {
    fail("no world is ready in the simulator");
  }

  const auto target = worldToSwitchTo(flightforge_world_level_name_enum_);
  if (!target) {
    RCLCPP_INFO(node_->get_logger(), "keeping the loaded world %s", list->current.c_str());
    return list->current;
  }

  RCLCPP_INFO(node_->get_logger(), "switching to world '%s'", target->c_str());
  const auto switched = ff_sim_->SwitchWorld(*target);
  if (!switched) {
    fail("could not switch to world '" + *target + "': " + switched.error + "; the worlds in this build: " + worldNames(*list));
  }

  RCLCPP_INFO(node_->get_logger(), "world %s is up, %zu robot(s) went with the old one", switched->package.c_str(), switched->removed_robots.size());
  return switched->package;
}

//}

/* timerMain() //{ */

void FlightforgeSimulator::timerMain() {

  if (!is_initialized_) {
    return;
  }

  if (shutting_down_) {
    return;
  }

  RCLCPP_INFO_ONCE(node_->get_logger(), "Main timer spinning");

  if (lockstep_) {

    const double wall = wall_clock_->now().seconds();

    if (wall < next_tick_wall_) {
      return;
    }

    next_tick_wall_ = std::max(next_tick_wall_ + 1.0 / (_clock_rate_ * mrs_lib::get_mutexed(mutex_drs_params_, drs_params_).realtime_factor), wall - 0.1);
  }

  // | ------------------ make simulation step ------------------ |

  double simulation_step_size = 1.0 / _simulation_rate_;
  double clock_step_size      = 1.0 / _clock_rate_;

  {
    std::scoped_lock lock(mutex_sim_time_);

    sim_time_      = sim_time_ + rclcpp::Duration(std::chrono::duration<double>(clock_step_size));
    sim_time_wall_ = wall_clock_->now().seconds();

    // | ---------------------- publish time ---------------------- |

    rosgraph_msgs::msg::Clock ros_time;

    ros_time.clock = sim_time_;

    ph_clock_.publish(ros_time);
  }

  auto sim_time = mrs_lib::get_mutexed(mutex_sim_time_, sim_time_);

  {
    const double dt_since_last_step = (sim_time - last_step_time_).seconds();

    // ticks are whole nanoseconds, so three 300 Hz ticks fall a hair short of 10 ms
    if (dt_since_last_step >= simulation_step_size - 0.5 * clock_step_size) {

      const auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

      const bool crash_on_collision = drs_params.collisions_enabled && drs_params.collisions_crash;

      if (crash_on_collision) {
        checkForCrash();
      }

      if (ground_from_simulator_) {
        applyGround();
      }

      for (size_t i = 0; i < uavs_.size(); i++) {

        const bool crashed = uavs_[i]->hasCrashed();

        // without a ground a crashed UAV would fall for ever
        if (crashed && !ground_enabled_) {
          continue;
        }

        const auto before = uavs_[i]->getState();

        uavs_.at(i)->makeStep(dt_since_last_step, sim_time.seconds());

        if (crashed) {
          settleCrashed(i);
          continue;
        }

        if (!ground_enabled_ || !crash_on_collision) {
          continue;
        }

        const auto   after    = uavs_[i]->getState();
        const double ground_z = uavs_[i]->getParams().ground_z;

        if (touchedDown(after.x(2), after.v(0), after.v(1), after.v(2), before.v(2), ground_z) && -before.v(2) > drs_params.max_touchdown_speed) {
          char reason[128];
          std::snprintf(reason, sizeof(reason), "hit the ground at %.2f m/s, faster than collisions/max_touchdown_speed (%.2f m/s)", -before.v(2),
                        drs_params.max_touchdown_speed);
          crashUav(i, reason, false);
        }
      }

      first_step_done_ = true;

      if (ground_from_simulator_) {
        requestGround();
      }

      publishPoses();

      if (lockstep_) {
        if (++steps_since_lockstep_ >= lockstep_stride_) {
          steps_since_lockstep_ = 0;
          stepSimulator(sim_time);
        }
      } else {
        dispatchDueSensors(sim_time.seconds());
      }

      last_step_time_ = sim_time;
    }
  }
}

//}

/* timeStatus() //{ */

void FlightforgeSimulator::timerStatus() {

  if (!is_initialized_) {
    return;
  }

  auto sim_time   = mrs_lib::get_mutexed(mutex_sim_time_, sim_time_);
  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);
  auto actual_rtf = mrs_lib::get_mutexed(mutex_actual_rtf_, actual_rtf_);

  rclcpp::Duration last_sec_sim_dt = sim_time - last_sim_time_status_;

  last_sim_time_status_ = sim_time;

  double last_sec_rtf = last_sec_sim_dt.seconds() / 1.0;

  actual_rtf = 0.9 * actual_rtf + 0.1 * last_sec_rtf;

  double fps;

  {
    std::scoped_lock lock(mutex_flightforge_);

    const auto got = ff_sim_->GetFps();

    if (!got) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to get the FPS from FlightForge: %s", got.error.c_str());
      return;
    }

    fps = *got;
  }

  const double fps_filter_const = 0.9;
  flightforge_fps_              = fps_filter_const * flightforge_fps_ + (1.0 - fps_filter_const) * fps;

  const double flightforge_rtf = flightforge_fps_ / highest_rate_;

  const double desired_rtf =
      (!lockstep_ && drs_params.dynamic_rtf && flightforge_rtf < drs_params.realtime_factor) ? flightforge_rtf : drs_params.realtime_factor;

  // the GetFps above can block for a whole Step, long enough to miss a pause
  if (desired_rtf != main_timer_rtf_ && !mrs_lib::get_mutexed(mutex_drs_params_, drs_params_).paused) {

    main_timer_rtf_ = desired_rtf;

    timer_main_->cancel();
    timer_main_ = node_->create_wall_timer(std::chrono::duration<double>(mainTimerPeriod(desired_rtf)), std::bind(&FlightforgeSimulator::timerMain, this),
                                           cbgrp_main_);
  }

  // the sim's own clock drives Livox point counts and motion, so it must run at the ROS rate
  if (!lockstep_ && std::abs(desired_rtf - flightforge_time_dilation_) > 0.01 * desired_rtf) {

    std::scoped_lock lock(mutex_flightforge_);

    if (const auto applied = ff_sim_->SetTimeDilation(desired_rtf)) {
      flightforge_time_dilation_ = *applied;
    } else {
      RCLCPP_ERROR(node_->get_logger(), "Failed to set the FlightForge time dilation to %.3f", desired_rtf);
    }
  }


  std::string ground_status;

  if (ground_from_simulator_) {
    std::scoped_lock lock(mutex_ground_);
    char             text[96];
    std::snprintf(text, sizeof(text), ", ground %d Hz, %.1f ms", ground_queries_, ground_queries_ > 0 ? 1e3 * ground_latency_ / ground_queries_ : 0.0);
    ground_status   = text;
    ground_queries_ = 0;
    ground_latency_ = 0.0;
  }

  std::string crashed;

  {
    std::scoped_lock lock(mutex_crashes_);
    for (size_t i = 0; i < crash_reasons_.size(); i++) {
      if (!crash_reasons_[i].empty()) {
        crashed += (crashed.empty() ? ", crashed: " : " ") + uav_names[i];
      }
    }
  }

  RCLCPP_INFO(node_->get_logger(), "%s, desired RTF = %.2f, actual RTF = %.2f, FlightForge FPS = %.2f, FlightForge RTF = %.2f, time dilation = %.2f%s%s",
              drs_params.paused ? "paused" : "running", drs_params.realtime_factor, actual_rtf, flightforge_fps_, flightforge_rtf, flightforge_time_dilation_,
              ground_status.c_str(), crashed.c_str());

  publishCrashes();

  mrs_lib::set_mutexed(mutex_actual_rtf_, actual_rtf, actual_rtf_);
}

//}

/* timerUnrealSync() //{ */

void FlightforgeSimulator::timerUnrealSync() {

  if (!is_initialized_) {
    return;
  }

  if (!lockstep_) {
    updateUnrealPoses(false);
  }
}

//}

/* timerTimeSync() //{ */

void FlightforgeSimulator::timerTimeSync() {

  if (!is_initialized_ || lockstep_) {
    return;
  }

  flightforge::Expected<flightforge::TimeInfo> time;
  double                                       sync_start;
  double                                       sync_end;

  // stamped inside the lock, so waiting for it is not mistaken for round trip
  {
    std::scoped_lock lock(mutex_flightforge_);

    sync_start = wall_clock_->now().seconds();
    time       = ff_sim_->GetTime();
    sync_end   = wall_clock_->now().seconds();
  }

  if (!time) {
    RCLCPP_ERROR(node_->get_logger(), "Failed to get FlightForge's time: %s", time.error.c_str());
    rclcpp::shutdown();
    return;
  }

  auto [sim_now, sim_now_wall] = mrs_lib::get_mutexed(mutex_sim_time_, sim_time_, sim_time_wall_);
  auto actual_rtf              = mrs_lib::get_mutexed(mutex_actual_rtf_, actual_rtf_);

  const double ros_at_sample = sim_now.seconds() + ((sync_start + sync_end) / 2.0 - sim_now_wall) * actual_rtf;

  std::scoped_lock lock(mutex_ue_world_offset_);

  // round-trip jitter would otherwise step stamps back
  ue_world_offset_       = ue_world_offset_valid_ ? 0.9 * ue_world_offset_ + 0.1 * (ros_at_sample - time->sim_time) : ros_at_sample - time->sim_time;
  ue_world_offset_valid_ = true;
}

//}

// | ------------------------ callbacks ----------------------- |

/* dynamic parameter callbacks //{ */

/* callbackRealtimeFactor() //{ */

void FlightforgeSimulator::callbackRealtimeFactor(const double& param_value) {

  timer_main_->cancel();

  timer_main_ =
      create_wall_timer(std::chrono::duration<double>(mainTimerPeriod(param_value)), std::bind(&FlightforgeSimulator::timerMain, this), cbgrp_main_);

  RCLCPP_INFO(get_logger(), "desired realtime factor updated to %.3f", param_value);
}

//}

/* callbackPause() //{ */

void FlightforgeSimulator::callbackPause(const bool& param_value) {

  RCLCPP_INFO(get_logger(), "callbackPause()");

  if (param_value) {

    timer_main_->cancel();
    timer_status_->cancel();

    RCLCPP_INFO(get_logger(), "paused");

  } else {

    timer_main_ = create_wall_timer(std::chrono::duration<double>(mainTimerPeriod(drs_params_.realtime_factor)),
                                    std::bind(&FlightforgeSimulator::timerMain, this), cbgrp_main_);

    timer_status_ = create_wall_timer(std::chrono::duration<double>(1.0), std::bind(&FlightforgeSimulator::timerStatus, this), cbgrp_status_);

    RCLCPP_INFO(get_logger(), "unpaused");
  }
}

//}

//}

// | ------------------------ routines ------------------------ |

/* publishPoses() //{ */

void FlightforgeSimulator::publishPoses(void) {

  auto sim_time = mrs_lib::get_mutexed(mutex_sim_time_, sim_time_);

  geometry_msgs::msg::PoseArray pose_array;

  pose_array.header.stamp    = sim_time;
  pose_array.header.frame_id = _world_frame_name_;

  for (size_t i = 0; i < uavs_.size(); i++) {

    auto state = uavs_.at(i)->getState();

    geometry_msgs::msg::Pose pose;

    pose.position.x  = state.x(0);
    pose.position.y  = state.x(1);
    pose.position.z  = state.x(2);
    pose.orientation = mrs_lib::AttitudeConverter(state.R);

    pose_array.poses.push_back(pose);
  }

  ph_poses_.publish(pose_array);
}

//}

/* robotState() //{ */

flightforge::RobotState FlightforgeSimulator::robotState(size_t uav) {

  const auto               state = uavs_[uav]->getState();
  const Eigen::Quaterniond q(state.R);

  flightforge::RobotState out;

  out.base                  = flightforge::Pose{flightforge::Vec3{state.x(0), state.x(1), state.x(2)}, flightforge::Quat{q.x(), q.y(), q.z(), q.w()}};
  out.base_linear_velocity  = flightforge::Vec3{state.v(0), state.v(1), state.v(2)};
  out.base_angular_velocity = flightforge::Vec3{state.omega(0), state.omega(1), state.omega(2)};

  return out;
}

//}

/* updateUnrealPoses() //{ */

void FlightforgeSimulator::updateUnrealPoses(const bool teleport_without_collision) {

  // the spawn pose may lie below the model's ground until its first step lifts it
  if (shutting_down_ || (ground_from_simulator_ && !first_step_done_)) {
    return;
  }

  const bool collide = !teleport_without_collision && mrs_lib::get_mutexed(mutex_drs_params_, drs_params_).collisions_enabled;

  // a crashed airframe falls along what it hit, where a sweep would hang it
  if (ff_robots_.size() == 1) {

    if (const auto sent = ff_robots_[0]->SetStateAsync(robotState(0), collide && !uavs_[0]->hasCrashed()); !sent) {
      RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "could not send the pose of %s: %s", uav_names[0].c_str(), sent.error.c_str());
    }

    return;
  }

  std::vector<std::pair<int, flightforge::RobotState>> flying, crashed;

  for (size_t i = 0; i < ff_robots_.size(); i++) {
    (uavs_[i]->hasCrashed() ? crashed : flying).emplace_back(ff_robots_[i]->Id(), robotState(i));
  }

  for (const auto& [states, sweep] : {std::pair{&flying, collide}, std::pair{&crashed, false}}) {
    if (states->empty()) {
      continue;
    }
    if (const auto sent = ff_pose_sim_->SetStates(*states, sweep, false); !sent) {
      RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "could not send the UAV poses: %s", sent.error.c_str());
    }
  }
}

//}

/* checkForCrash() //{ */

void FlightforgeSimulator::checkForCrash(void) {

  const double max_touchdown_speed = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_).max_touchdown_speed;

  for (size_t i = 0; i < ff_robots_.size(); i++) {

    if (uavs_[i]->hasCrashed()) {
      continue;
    }

    const auto collision = ff_robots_[i]->GetCollision(true);

    if (!collision) {
      RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "could not get the collision state of %s: %s", uav_names[i].c_str(),
                            collision.error.c_str());
      continue;
    }

    if (!collision->collided) {
      continue;
    }

    const double speed = uavs_[i]->getState().v.norm();

    if (ground_from_simulator_ && isLanding(collision->impact_point.z, surface_[i], speed, max_touchdown_speed)) {
      RCLCPP_INFO_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s touched the ground at [%.2f, %.2f, %.2f], %.2f m/s", uav_names[i].c_str(),
                           collision->impact_point.x, collision->impact_point.y, collision->impact_point.z, speed);
      continue;
    }

    char reason[128];
    std::snprintf(reason, sizeof(reason), "hit an obstacle at [%.2f, %.2f, %.2f] at %.2f m/s", collision->impact_point.x, collision->impact_point.y,
                  collision->impact_point.z, speed);
    crashUav(i, reason, true);
  }
}

//}

/* crashUav() //{ */

// the motors stop and the dynamics let it fall onto the surface under it, as the ground query sees it
void FlightforgeSimulator::crashUav(const size_t uav, const std::string& reason, const bool hit_obstacle) {

  if (uavs_[uav]->hasCrashed()) {
    return;
  }

  uavs_[uav]->crash();

  auto state  = uavs_[uav]->getState();
  state.omega = Eigen::Vector3d::Zero();

  if (hit_obstacle) {

    // the pose sweep held the airframe at the obstacle while the dynamics flew on into it
    if (const auto held = ff_robots_[uav]->GetState(); held && held->state.base) {
      const auto& p = held->state.base->position;
      state.x       = Eigen::Vector3d(p.x, p.y, p.z);
    }

    state.v(0) = 0.0;
    state.v(1) = 0.0;
  }

  uavs_[uav]->setState(state);

  {
    std::scoped_lock lock(mutex_crashes_);
    crash_reasons_[uav] = reason;
  }

  RCLCPP_ERROR(node_->get_logger(), "%s crashed: %s; it falls to the ground and stays there, the simulation goes on", uav_names[uav].c_str(), reason.c_str());

  std_msgs::msg::Bool crashed;
  crashed.data = true;
  ph_crashed_[uav].publish(crashed);

  publishCrashes();
}

//}

/* settleCrashed() //{ */

// a crashed UAV at rest lies on its landing gear
void FlightforgeSimulator::settleCrashed(const size_t uav) {

  auto         state    = uavs_[uav]->getState();
  const double ground_z = uavs_[uav]->getParams().ground_z;

  if (state.x(2) > ground_z + 1e-3 || state.R(2, 2) > 0.9999) {
    return;
  }

  const double yaw = std::atan2(state.R(1, 0), state.R(0, 0));

  state.R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  state.v.setZero();
  state.omega.setZero();

  uavs_[uav]->setState(state);
}

//}

/* publishCrashes() //{ */

void FlightforgeSimulator::publishCrashes(void) {

  diagnostic_msgs::msg::DiagnosticArray array;

  array.header.stamp = mrs_lib::get_mutexed(mutex_sim_time_, sim_time_);

  std::scoped_lock lock(mutex_crashes_);

  for (size_t i = 0; i < crash_reasons_.size(); i++) {

    const bool crashed = !crash_reasons_[i].empty();

    diagnostic_msgs::msg::DiagnosticStatus status;

    status.name        = "flightforge_simulator: " + uav_names[i];
    status.hardware_id = uav_names[i];
    status.level       = crashed ? diagnostic_msgs::msg::DiagnosticStatus::ERROR : diagnostic_msgs::msg::DiagnosticStatus::OK;
    status.message     = crashed ? "crashed: " + crash_reasons_[i] : "ok";

    diagnostic_msgs::msg::KeyValue value;
    value.key   = "crashed";
    value.value = crashed ? "true" : "false";
    status.values.push_back(value);

    array.status.push_back(status);
  }

  ph_diagnostics_.publish(array);
}

//}

/* spawnRobot() //{ */

// resting the airframe on the surface measures how high its base link sits above the ground
void FlightforgeSimulator::spawnRobot(const size_t uav, const std::string& frame) {

  const std::string& uav_name = uav_names[uav];
  const auto         asked    = *robotState(uav).base;

  double surface = std::numeric_limits<double>::quiet_NaN();

  if (ground_from_simulator_) {

    const auto heights = ff_sim_->GetSurfaceHeight({flightforge::Vec3{asked.position.x, asked.position.y, asked.position.z + 1.0}});

    if (!heights) {
      RCLCPP_ERROR(node_->get_logger(), "could not query the ground under %s: %s", uav_name.c_str(), heights.error.c_str());
      rclcpp::shutdown();
      exit(1);
    }

    surface = (*heights)[0];

    if (!std::isfinite(surface)) {
      RCLCPP_WARN(node_->get_logger(), "there is no ground under %s's spawn, it has none until it flies over some", uav_name.c_str());
    }
  }

  flightforge::Pose spawn = asked;

  if (std::isfinite(surface)) {
    spawn.position.z = surface;
  }

  RCLCPP_INFO(node_->get_logger(), "spawning %s, a %s, at [%.2f, %.2f, %.2f]", uav_name.c_str(), frame.c_str(), spawn.position.x, spawn.position.y,
              spawn.position.z);

  const auto spawned = ff_sim_->SpawnRobot(frame, spawn, uav_name);

  if (!spawned) {
    RCLCPP_ERROR(node_->get_logger(), "could not spawn %s: %s", uav_name.c_str(), spawned.error.c_str());
    rclcpp::shutdown();
    exit(1);
  }

  RCLCPP_INFO(node_->get_logger(), "%s is robot %d", uav_name.c_str(), *spawned);

  ff_robots_.push_back(std::make_shared<flightforge::RobotClient>(ff_pose_sim_->SharedConnection(), *spawned));

  if (!std::isfinite(surface)) {
    return;
  }

  const auto rested = ff_robots_[uav]->GetState();

  if (!rested || !rested->state.base) {
    RCLCPP_ERROR(node_->get_logger(), "could not read where %s came to rest: %s", uav_name.c_str(), rested.error.c_str());
    rclcpp::shutdown();
    exit(1);
  }

  surface_[uav]   = surface;
  clearance_[uav] = std::max(0.0, rested->state.base->position.z - surface);

  RCLCPP_INFO(node_->get_logger(), "%s rests %.3f m above the ground at %.3f m", uav_name.c_str(), clearance_[uav], surface);

  if (asked.position.z > surface + clearance_[uav]) {
    flightforge::RobotState lifted;
    lifted.base = asked;
    if (const auto moved = ff_robots_[uav]->SetState(lifted, false); !moved) {
      RCLCPP_ERROR(node_->get_logger(), "could not lift %s to its spawn height: %s", uav_name.c_str(), moved.error.c_str());
    }
  }
}

//}

/* groundPose() //{ */

FlightforgeSimulator::GroundPose FlightforgeSimulator::groundPose(const size_t uav) {

  const auto state = uavs_[uav]->getState();

  return GroundPose{state.x(0), state.x(1), state.x(2), std::atan2(state.R(1, 0), state.R(0, 0))};
}

//}

/* requestGround() //{ */

void FlightforgeSimulator::requestGround(void) {

  {
    std::scoped_lock lock(mutex_ground_);

    ground_poses_.resize(uavs_.size());
    for (size_t i = 0; i < uavs_.size(); i++) {
      ground_poses_[i] = groundPose(i);
    }

    ground_request_ = true;
  }

  cv_ground_.notify_one();
}

//}

/* applyGround() //{ */

void FlightforgeSimulator::applyGround(void) {

  {
    std::scoped_lock lock(mutex_ground_);
    surface_ = ground_surfaces_;
  }

  for (size_t i = 0; i < uavs_.size(); i++) {

    const double ground_z = std::isfinite(surface_[i]) ? surface_[i] + clearance_[i] : -std::numeric_limits<double>::infinity();

    if (ground_z != applied_ground_z_[i]) {
      uavs_[i]->setGroundZ(ground_z);
      applied_ground_z_[i] = ground_z;
    }
  }
}

//}

/* groundWorker() //{ */

// asks for one step's poses at a time and drops the ones that pile up meanwhile, so it runs at most at the simulation rate
void FlightforgeSimulator::groundWorker(void) {

  std::vector<GroundPose> poses;
  std::vector<double>     radius(uavs_.size());
  std::vector<int32_t>    robot_ids;

  for (size_t i = 0; i < uavs_.size(); i++) {
    radius[i] = uavs_[i]->getParams().arm_length;
    robot_ids.insert(robot_ids.end(), kFootprintPoints, ff_robots_[i]->Id());
  }

  while (true) {

    {
      std::unique_lock lock(mutex_ground_);
      cv_ground_.wait(lock, [this]() { return ground_stop_ || ground_request_; });

      if (ground_stop_) {
        return;
      }

      ground_request_ = false;
      poses           = ground_poses_;
    }

    std::vector<flightforge::Vec3> points;
    points.reserve(poses.size() * kFootprintPoints);

    for (size_t i = 0; i < poses.size(); i++) {
      for (const auto& p : footprintPoints(poses[i].x, poses[i].y, poses[i].z - clearance_[i], poses[i].yaw, radius[i])) {
        points.push_back(flightforge::Vec3{p[0], p[1], p[2]});
      }
    }

    const auto started = std::chrono::steady_clock::now();
    const auto heights = ground_robot_ids_ ? ff_ground_sim_->GetSurfaceHeight(points, robot_ids) : ff_ground_sim_->GetSurfaceHeight(points);
    const double took  = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    if (!heights) {
      RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "could not query the ground: %s", heights.error.c_str());
      continue;
    }

    std::vector<double> surfaces(poses.size());
    for (size_t i = 0; i < poses.size(); i++) {
      surfaces[i] = highestSurface(heights->data() + i * kFootprintPoints, kFootprintPoints);
    }

    std::scoped_lock lock(mutex_ground_);
    ground_surfaces_ = std::move(surfaces);
    ground_queries_++;
    ground_latency_ += took;
  }
}

//}

/* ueWorldTimeToSimtime() //{ */

rclcpp::Time FlightforgeSimulator::ueWorldTimeToSimtime(const double ue_world_time) {

  const double sim_time = ue_world_time + mrs_lib::get_mutexed(mutex_ue_world_offset_, ue_world_offset_);

  return rclcpp::Time(int64_t(std::max(0.0, sim_time) * 1e9), clock_->get_clock_type());
}

//}

// | ------------------------- sensors ------------------------ |

/* sendStaticTf() //{ */

void FlightforgeSimulator::sendStaticTf(const std::string& uav_name, const std::string& frame, const Mount& mount, bool optical) {

  geometry_msgs::msg::TransformStamped tf;

  tf.header.stamp    = clock_->now();
  tf.header.frame_id = uav_name + "/fcu";
  tf.child_frame_id  = frame;

  tf.transform.translation.x = mount.x;
  tf.transform.translation.y = mount.y;
  tf.transform.translation.z = mount.z;

  Eigen::Matrix3d rotation = mrs_lib::AttitudeConverter(M_PI * mount.roll / 180.0, M_PI * mount.pitch / 180.0, M_PI * mount.yaw / 180.0);

  if (optical) {
    rotation = rotation * Eigen::Matrix3d(mrs_lib::AttitudeConverter(Eigen::Quaterniond(-0.5, 0.5, -0.5, 0.5)));
  }

  tf.transform.rotation = mrs_lib::AttitudeConverter(rotation);

  static_broadcaster_->sendTransform(tf);
}

//}

/* failSetup() //{ */

void FlightforgeSimulator::failSetup(size_t uav, const std::string& what) {
  RCLCPP_ERROR(node_->get_logger(), "%s: %s", uav_names[uav].c_str(), what.c_str());
  rclcpp::shutdown();
  exit(1);
}

//}

/* connectSensor() //{ */

std::shared_ptr<flightforge::RobotClient> FlightforgeSimulator::connectSensor(size_t uav, const std::string& name) {

  auto c = std::make_shared<flightforge::RobotClient>(_simulator_ip_, _simulator_port_, ff_robots_[uav]->Id());

  if (!c->Connect(3)) {
    failSetup(uav, "could not open a connection for " + name);
  }

  return c;
}

//}

/* calibration() //{ */

std::vector<flightforge::wire::CameraInfoT> FlightforgeSimulator::calibration(size_t uav, int id, const std::string& name) {

  auto infos = ff_robots_[uav]->GetCameraInfo(id);

  if (!infos || infos->empty()) {
    failSetup(uav, "no camera info for " + name + (infos ? "" : ": " + infos.error));
  }

  return std::move(*infos);
}

//}

/* warnOnSize() //{ */

void FlightforgeSimulator::warnOnSize(size_t uav, const sensor_msgs::msg::CameraInfo& info, int width, int height, const std::string& name) {
  if (int(info.width) != width || int(info.height) != height) {
    RCLCPP_WARN(node_->get_logger(), "%s/%s renders %dx%d, not the configured %dx%d", uav_names[uav].c_str(), name.c_str(), info.width, info.height, width, height);
  }
}

//}

/* registerCamera() //{ */

void FlightforgeSimulator::registerCamera(size_t uav, CameraSensor s, const std::string& topic) {

  s.connector = connectSensor(uav, s.name);
  s.info      = toCameraInfo(calibration(uav, s.id, s.name).front());
  warnOnSize(uav, s.info, s.params.width, s.params.height, s.name);

  s.ph_image = it_->advertise(topic + "image_raw", 1);
  s.ph_info  = mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>(node_, topic + "camera_info");

  if (s.type == flightforge::wire::SensorType::INSTANCE_SEG_CAMERA) {
    mrs_lib::PublisherHandlerOptions latched(node_);
    latched.qos = rclcpp::QoS(1).transient_local();
    s.ph_map    = mrs_lib::PublisherHandler<mrs_uav_flightforge_simulator::msg::InstanceMap>(latched, topic + "labels");
  }

  sendStaticTf(uav_names[uav], s.frame, s.params.mount, true);

  cameras_[uav].push_back(std::move(s));
}

//}

/* registerFisheye() //{ */

void FlightforgeSimulator::registerFisheye(size_t uav, FisheyeSensor s, const std::string& topic) {

  s.connector = connectSensor(uav, s.name);
  s.info      = toCameraInfo(calibration(uav, s.id, s.name).front());
  warnOnSize(uav, s.info, s.params.width, s.params.height, s.name);

  s.ph_image = it_->advertise(topic + "image_raw", 1);
  s.ph_info  = mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>(node_, topic + "camera_info");

  sendStaticTf(uav_names[uav], s.frame, s.params.mount, true);

  fisheyes_[uav].push_back(std::move(s));
}

//}

/* registerStereo() //{ */

void FlightforgeSimulator::registerStereo(size_t uav, StereoSensor s, const std::string& topic, const Mount& left, const Mount& right) {

  s.connector = connectSensor(uav, s.name);

  const auto  infos      = calibration(uav, s.id, s.name);
  const auto* left_info  = findEye(infos, flightforge::wire::Eye::LEFT);
  const auto* right_info = findEye(infos, flightforge::wire::Eye::RIGHT);

  if (!left_info || !right_info) {
    failSetup(uav, s.name + " did not report both eyes");
  }

  s.info_left  = toCameraInfo(*left_info);
  s.info_right = toCameraInfo(*right_info);
  warnOnSize(uav, s.info_left, s.params.width, s.params.height, s.name);

  s.ph_image_left  = it_->advertise(topic + "left/image_raw", 1);
  s.ph_image_right = it_->advertise(topic + "right/image_raw", 1);
  s.ph_info_left   = mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>(node_, topic + "left/camera_info");
  s.ph_info_right  = mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>(node_, topic + "right/camera_info");

  sendStaticTf(uav_names[uav], s.frame_left, left, true);
  sendStaticTf(uav_names[uav], s.frame_right, right, true);

  stereos_[uav].push_back(std::move(s));
}

//}

/* registerLidar() //{ */

void FlightforgeSimulator::registerLidar(size_t uav, LidarSensor s, const std::string& topic) {

  s.connector = connectSensor(uav, s.name);
  s.ph_points = mrs_lib::PublisherHandler<sensor_msgs::msg::PointCloud2>(node_, topic + "points");

  if (s.params.publish_free) {
    s.ph_free = mrs_lib::PublisherHandler<sensor_msgs::msg::PointCloud2>(node_, topic + "points_free");
  }

  if (s.params.publish_free && s.params.livox) {
    RCLCPP_WARN(node_->get_logger(), "%s/%s: a Livox scan is not organised, so its rays without a return have no direction and %spoints_free stays empty",
                uav_names[uav].c_str(), s.name.c_str(), topic.c_str());
  }

  sendStaticTf(uav_names[uav], s.frame, s.params.mount, false);

  lidars_[uav].push_back(std::move(s));
}

//}

/* setupSensors() //{ */

void FlightforgeSimulator::setupSensors(size_t uav) {

  using flightforge::wire::SensorType;

  const std::string uav_name  = uav_names[uav];
  auto&             connector = ff_robots_[uav];

  const auto add = [&](SensorType type, const std::string& name, const std::optional<flightforge::Pose>& pose = std::nullopt) {
    const auto id = connector->AddSensor(type, pose);
    if (!id) {
      failSetup(uav, "could not add " + name + ": " + id.error);
    }
    return *id;
  };

  const auto configure = [&](auto get, auto set, auto apply, const auto& params, int id, const std::string& name) {
    auto cfg = ((*connector).*get)(id);
    if (!cfg) {
      failSetup(uav, "could not read the config of " + name + ": " + cfg.error);
    }
    apply(params, *cfg);
    if (const auto set_ok = ((*connector).*set)(*cfg, id); !set_ok) {
      failSetup(uav, "could not configure " + name + ": " + set_ok.error);
    }
  };

  using flightforge::RobotClient;

  cameras_.emplace_back();
  fisheyes_.emplace_back();
  stereos_.emplace_back();
  events_.emplace_back();
  lidars_.emplace_back();
  rangefinders_.emplace_back();

  // | ------------------------- cameras ------------------------ |

  if (!sensors_config_.rgb_segmented.empty()) {
    RCLCPP_WARN(node_->get_logger(), "[%s] rgb_segmented is retired since FlightForge API 1.5, use instance_seg; skipping %zu camera(s)", uav_name.c_str(),
                sensors_config_.rgb_segmented.size());
  }

  const std::vector<std::tuple<std::string, SensorType, const std::vector<CameraParams>*>> camera_types = {
      {"rgb", SensorType::RGB_CAMERA, &sensors_config_.rgb},
      {"depth", SensorType::DEPTH_CAMERA, &sensors_config_.depth},
      {"instance_seg", SensorType::INSTANCE_SEG_CAMERA, &sensors_config_.instance_seg}};

  for (const auto& [type_name, type, params_list] : camera_types) {
    for (size_t i = 0; i < params_list->size(); i++) {

      CameraSensor s;

      s.params = params_list->at(i);
      s.type   = type;
      s.name   = type_name + "_" + std::to_string(i);
      s.frame  = uav_name + "/" + s.name;
      s.id     = add(type, s.name);

      configure(&RobotClient::GetRgbCameraConfig, &RobotClient::SetRgbCameraConfig, applyCameraParams, s.params, s.id, s.name);

      const std::string topic = "/" + uav_name + "/" + s.name + "/";

      registerCamera(uav, std::move(s), topic);
    }
  }

  // | ------------------------- fisheye ------------------------ |

  for (size_t i = 0; i < sensors_config_.fisheye.size(); i++) {

    FisheyeSensor s;

    s.params = sensors_config_.fisheye[i];
    s.name   = "fisheye_" + std::to_string(i);
    s.frame  = uav_name + "/" + s.name;
    s.id     = add(SensorType::FISHEYE_CAMERA, s.name);

    configure(&RobotClient::GetFisheyeCameraConfig, &RobotClient::SetFisheyeCameraConfig, applyFisheyeParams, s.params, s.id, s.name);

    const std::string topic = "/" + uav_name + "/" + s.name + "/";

    registerFisheye(uav, std::move(s), topic);
  }

  // | ------------------------- stereo ------------------------- |

  for (size_t i = 0; i < sensors_config_.stereo.size(); i++) {

    StereoSensor s;

    s.params      = sensors_config_.stereo[i];
    s.name        = "stereo_" + std::to_string(i);
    s.frame_left  = uav_name + "/" + s.name + "_left";
    s.frame_right = uav_name + "/" + s.name + "_right";
    s.id          = add(SensorType::STEREO_CAMERA, s.name);

    configure(&RobotClient::GetStereoCameraConfig, &RobotClient::SetStereoCameraConfig, applyStereoParams, s.params, s.id, s.name);

    const auto [left, right] = stereoEyes(s.params.mount, s.params.baseline);
    const std::string topic  = "/" + uav_name + "/" + s.name + "/";

    registerStereo(uav, std::move(s), topic, left, right);
  }

  // | ------------------------- events ------------------------- |

  for (size_t i = 0; i < sensors_config_.event.size(); i++) {

    EventSensor s;

    s.params    = sensors_config_.event[i];
    s.name      = "event_" + std::to_string(i);
    s.frame     = uav_name + "/" + s.name;
    s.id        = add(SensorType::EVENT_CAMERA, s.name);
    s.connector = connectSensor(uav, s.name);

    configure(&RobotClient::GetEventCameraConfig, &RobotClient::SetEventCameraConfig, applyEventParams, s.params, s.id, s.name);

    s.ph_events = mrs_lib::PublisherHandler<event_camera_msgs::msg::EventPacket>(node_, "/" + uav_name + "/" + s.name + "/events");

    sendStaticTf(uav_name, s.frame, s.params.mount, true);

    events_[uav].push_back(std::move(s));
  }

  // | ------------------------- lidars ------------------------- |

  for (size_t i = 0; i < sensors_config_.lidar.size(); i++) {

    LidarSensor s;

    s.params = sensors_config_.lidar[i];
    s.name   = "lidar_" + std::to_string(i);
    s.frame  = uav_name + "/" + s.name;
    s.id     = add(SensorType::LIDAR, s.name);

    configure(&RobotClient::GetLidarConfig, &RobotClient::SetLidarConfig, applyLidarParams, s.params, s.id, s.name);

    const std::string topic = "/" + uav_name + "/" + s.name + "/";

    registerLidar(uav, std::move(s), topic);
  }

  // | ---------------------- rangefinders ---------------------- |

  for (size_t i = 0; i < sensors_config_.rangefinder.size(); i++) {

    RangefinderSensor s;

    s.params    = sensors_config_.rangefinder[i];
    s.name      = "rangefinder_" + std::to_string(i);
    s.frame     = uav_name + "/" + s.name;
    s.id        = add(SensorType::RANGEFINDER, s.name, rangefinderPose(s.params.mount));
    s.connector = connectSensor(uav, s.name);
    s.ph_range  = mrs_lib::PublisherHandler<sensor_msgs::msg::Range>(node_, "/" + uav_name + "/" + s.name + "/range");

    sendStaticTf(uav_name, s.frame, s.params.mount, false);

    rangefinders_[uav].push_back(std::move(s));
  }

  setupDevices(uav);
}

//}

/* setupDevices() //{ */

void FlightforgeSimulator::setupDevices(size_t uav) {

  using flightforge::wire::SensorType;

  if (sensors_config_.devices.empty()) {
    return;
  }

  const std::string uav_name  = uav_names[uav];
  auto&             connector = ff_robots_[uav];

  const auto library = connector->ListDevices();

  if (!library) {
    failSetup(uav, "could not list the simulator's devices: " + library.error);
  }

  const auto names = deviceNames(sensors_config_.devices);

  for (size_t d = 0; d < names.size(); d++) {

    const DeviceParams& device = sensors_config_.devices[d];

    if (std::none_of(library->begin(), library->end(), [&](const auto& known) { return known.name == device.type; })) {

      std::string known;

      for (const auto& l : *library) {
        known += " " + l.name;
      }

      failSetup(uav, "unknown device \"" + device.type + "\", the simulator has:" + known);
    }

    const auto created = connector->AddDevice(device.type, toPose(device.mount), device.show_mesh);

    if (!created) {
      failSetup(uav, "could not add device " + names[d] + ": " + created.error);
    }

    for (const auto& sensor : *created) {

      const auto named = [&](auto& s, const std::string& role) {
        s.id   = sensor.id;
        s.name = names[d] + "/" + role;
        return "/" + uav_name + "/" + s.name + "/";
      };

      const auto read = [&](const auto& cfg, const std::string& role) {
        if (!cfg) {
          failSetup(uav, "could not read " + names[d] + "/" + role + ": " + cfg.error);
        }
      };

      const auto capture_at_camera_rate = [&](auto& cfg, auto set, const std::string& role) {
        cfg->capture_rate = device.camera_rate;
        if (const auto set_ok = ((*connector).*set)(*cfg, sensor.id); !set_ok) {
          failSetup(uav, "could not configure " + names[d] + "/" + role + ": " + set_ok.error);
        }
      };

      const std::string frame = uav_name + "/" + names[d] + "_";

      switch (sensor.type) {

        case SensorType::RGB_CAMERA: {

          auto cfg = connector->GetRgbCameraConfig(sensor.id);
          read(cfg, "rgb");
          capture_at_camera_rate(cfg, &flightforge::RobotClient::SetRgbCameraConfig, "rgb");

          CameraSensor s;

          s.params = toCameraParams(*cfg, device.camera_rate);
          s.type   = sensor.type;
          s.frame  = frame + "rgb";

          const std::string topic = named(s, "rgb");

          highest_rate_ = std::max(highest_rate_, s.params.rate);
          registerCamera(uav, std::move(s), topic);
          break;
        }

        case SensorType::FISHEYE_CAMERA: {

          auto cfg = connector->GetFisheyeCameraConfig(sensor.id);
          read(cfg, "fisheye");
          capture_at_camera_rate(cfg, &flightforge::RobotClient::SetFisheyeCameraConfig, "fisheye");

          FisheyeSensor s;

          s.params = toFisheyeParams(*cfg, device.camera_rate);
          s.frame  = frame + "fisheye";

          const std::string topic = named(s, "fisheye");

          highest_rate_ = std::max(highest_rate_, s.params.rate);
          registerFisheye(uav, std::move(s), topic);
          break;
        }

        case SensorType::STEREO_CAMERA: {

          auto cfg = connector->GetStereoCameraConfig(sensor.id);
          read(cfg, "stereo");
          capture_at_camera_rate(cfg, &flightforge::RobotClient::SetStereoCameraConfig, "stereo");

          StereoSensor s;

          s.params      = toStereoParams(*cfg, device.camera_rate);
          s.frame_left  = frame + "stereo_left";
          s.frame_right = frame + "stereo_right";

          const std::string topic = named(s, "stereo");

          highest_rate_ = std::max(highest_rate_, s.params.rate);
          registerStereo(uav, std::move(s), topic, toMount(cfg->mount_left.get()), toMount(cfg->mount_right.get()));
          break;
        }

        case SensorType::LIDAR: {

          auto cfg = connector->GetLidarConfig(sensor.id);
          read(cfg, "lidar");

          LidarSensor s;

          s.params = toLidarParams(*cfg, device);
          s.frame  = frame + "lidar";

          cfg->show_beams = false;
          cfg->frequency  = s.params.rate;

          if (const auto set = connector->SetLidarConfig(*cfg, sensor.id); !set) {
            failSetup(uav, "could not configure " + names[d] + "/lidar: " + set.error);
          }

          const std::string topic = named(s, "lidar");

          highest_rate_ = std::max(highest_rate_, s.params.rate);
          registerLidar(uav, std::move(s), topic);
          break;
        }

        default:
          failSetup(uav, "device " + device.type + " has a " + flightforge::wire::EnumNameSensorType(sensor.type) + ", which the node does not publish yet");
      }
    }

    RCLCPP_INFO(node_->get_logger(), "%s: added %s with %zu sensors", uav_name.c_str(), names[d].c_str(), created->size());
  }
}

//}

/* startSensorWorkers() //{ */

void FlightforgeSimulator::startSensorWorkers(void) {

  const auto add = [this](auto& sensors, auto fetch) {
    for (size_t uav = 0; uav < sensors.size(); uav++) {
      for (size_t i = 0; i < sensors[uav].size(); i++) {

        auto w = std::make_unique<SensorWorker>();
        w->job  = [this, fetch, uav, i]() { (this->*fetch)(uav, i); };
        w->name = uav_names[uav] + "/" + sensors[uav][i].name;

        schedule_.push_back(ScheduledSensor{sensors[uav][i].params.rate, 0.0, false});
        workers_.push_back(std::move(w));
      }
    }
  };

  add(cameras_, &FlightforgeSimulator::fetchCamera);
  add(fisheyes_, &FlightforgeSimulator::fetchFisheye);
  add(stereos_, &FlightforgeSimulator::fetchStereo);
  add(events_, &FlightforgeSimulator::fetchEvent);
  add(lidars_, &FlightforgeSimulator::fetchLidar);
  add(rangefinders_, &FlightforgeSimulator::fetchRangefinder);

  for (auto& w : workers_) {

    SensorWorker* worker = w.get();

    worker->thread = std::thread([this, worker]() {
      while (true) {
        {
          std::unique_lock lock(worker->mutex);
          worker->cv.wait(lock, [worker]() { return worker->pending || worker->stop; });
          if (worker->stop) {
            return;
          }
          worker->pending = false;
        }

        worker->job();

        worker->busy = false;
        fetches_.done();
      }
    });
  }
}

//}

/* dispatchDueSensors() //{ */

void FlightforgeSimulator::dispatchDueSensors(double sim_time) {

  const double wall_now = wall_clock_->now().seconds();

  for (size_t i = 0; i < workers_.size(); i++) {

    schedule_[i].busy = workers_[i]->busy.load();

    if (schedule_[i].busy && wall_now - workers_[i]->started > 15.0) {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *wall_clock_, 5000, "%s: its fetch has been running for %.0f s", workers_[i]->name.c_str(), wall_now - workers_[i]->started.load());
    }
  }

  for (size_t i : dueSensors(sim_time, schedule_)) {

    SensorWorker& w = *workers_[i];

    w.started = wall_now;
    w.busy    = true;
    fetches_.add();
    {
      std::scoped_lock lock(w.mutex);
      w.pending = true;
    }
    w.cv.notify_one();
  }
}

//}

/* onShutdown() //{ */

void FlightforgeSimulator::onShutdown(void) {

  shutting_down_ = true;

  {
    std::scoped_lock lock(mutex_ground_);
    ground_stop_ = true;
  }
  cv_ground_.notify_one();

  if (ground_thread_.joinable()) {
    ground_thread_.join();
  }

  for (auto& w : workers_) {
    {
      std::scoped_lock lock(w->mutex);
      w->stop = true;
    }
    w->cv.notify_one();
  }

  for (auto& w : workers_) {
    if (w->thread.joinable()) {
      w->thread.join();
    }
  }

  std::scoped_lock lock(mutex_flightforge_);

  if (lockstep_) {
    ff_sim_->SetLockstep(false);
    ff_sim_->SetFixedTimestep(0.0);
    ff_sim_->SetCaptureMode(flightforge::wire::CaptureMode::ALL_FRAMES);
  }

  for (size_t i = 0; i < ff_robots_.size(); i++) {
    if (const auto removed = ff_sim_->RemoveRobot(ff_robots_[i]->Id()); !removed) {
      RCLCPP_WARN(node_->get_logger(), "could not remove %s from the simulator: %s", uav_names[i].c_str(), removed.error.c_str());
    }
  }
}

//}

/* mainTimerPeriod() //{ */

// lockstep polls faster and paces itself on next_tick_wall_, as a Step already spends the wall time
double FlightforgeSimulator::mainTimerPeriod(double rtf) const {
  return 1.0 / (_clock_rate_ * rtf * (lockstep_ ? 4.0 : 1.0));
}

//}

/* stepSimulator() //{ */

void FlightforgeSimulator::stepSimulator(const rclcpp::Time& sim_time) {

  updateUnrealPoses(false);

  flightforge::Expected<flightforge::StepResult> step;

  {
    std::scoped_lock lock(mutex_flightforge_);
    step = ff_sim_->Step(1);
  }

  if (!step) {
    RCLCPP_ERROR(node_->get_logger(), "FlightForge did not complete a lockstep step (%s), is the simulator still running?", step.error.c_str());
    rclcpp::shutdown();
    exit(1);
  }

  step_stamp_ = sim_time;

  {
    std::scoped_lock lock(mutex_ue_world_offset_);
    ue_world_offset_       = sim_time.seconds() - step->sim_time;
    ue_world_offset_valid_ = true;
  }

  dispatchDueSensors(sim_time.seconds());

  // longer than the sensor sockets' timeout, so no fetch outlives its step
  if (!fetches_.wait(15.0)) {
    RCLCPP_WARN_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "sensor fetches did not return within 15 s of a lockstep step");
  }
}

//}

/* worldStamp() //{ */

rclcpp::Time FlightforgeSimulator::worldStamp(double world_time) {
  return lockstep_ ? step_stamp_ : ueWorldTimeToSimtime(world_time);
}

//}

/* fetchCamera() //{ */

void FlightforgeSimulator::fetchCamera(size_t uav, size_t i) {

  using flightforge::wire::SensorType;

  if (!is_initialized_) {
    return;
  }

  CameraSensor& s         = cameras_[uav][i];
  auto&         connector = s.connector;

  flightforge::Expected<flightforge::Image>      frame;
  flightforge::Expected<flightforge::DepthImage> depth;
  flightforge::Status                            got;
  double                                         stamp = 0;

  if (s.type == SensorType::DEPTH_CAMERA) {
    depth = fetchFresh([&]() { return connector->GetDepthCameraData(s.id); }, s.last_stamp, s.params.rate);
  } else {
    frame = fetchFresh(
        [&]() {
          switch (s.type) {
            case SensorType::INSTANCE_SEG_CAMERA:
              return connector->GetInstanceSegData(s.id);
            default:
              return connector->GetRgbCameraData(s.id);
          }
        },
        s.last_stamp, s.params.rate);
  }

  if (s.type == SensorType::DEPTH_CAMERA) {
    got   = depth;
    stamp = depth->stamp;
  } else {
    got   = frame;
    stamp = frame->stamp;
  }

  if (!got) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: could not get %s: %s", uav_names[uav].c_str(), s.name.c_str(), got.error.c_str());
    return;
  }

  if (stamp <= s.last_stamp || (!lockstep_ && !mrs_lib::get_mutexed(mutex_ue_world_offset_, ue_world_offset_valid_))) {
    if (lockstep_) {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: %s returned no new frame", uav_names[uav].c_str(), s.name.c_str());
    }
    return;
  }

  s.last_stamp = stamp;

  const rclcpp::Time published_stamp = worldStamp(stamp);
  if (published_stamp <= s.last_published_stamp) {
    return;
  }
  s.last_published_stamp = published_stamp;

  std_msgs::msg::Header header;
  header.frame_id = s.frame;
  header.stamp    = published_stamp;

  std::optional<sensor_msgs::msg::Image> image;

  if (s.type == SensorType::DEPTH_CAMERA) {
    image = depth->width == s.info.width && depth->height == s.info.height ? toDepthImage(depth->mm, s.info.width, s.info.height, header) : std::nullopt;
  } else if (const auto bgr = toBgr(*frame, s.info.width, s.info.height)) {
    image = s.type == SensorType::INSTANCE_SEG_CAMERA ? toInstanceImage(*bgr, header) : toColorImage(*bgr, header);
  }

  if (!image) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: %s frame is not %dx%d, dropped", uav_names[uav].c_str(), s.name.c_str(),
                          s.info.width, s.info.height);
    return;
  }

  s.ph_image.publish(*image);

  auto info   = s.info;
  info.header = header;
  s.ph_info.publish(info);

  if (s.type == SensorType::INSTANCE_SEG_CAMERA && !s.map_published) {
    if (const auto entries = connector->GetInstanceSegMap(s.id)) {
      s.ph_map.publish(toInstanceMap(*entries, header));
      s.map_published = true;
    }
  }
}

//}

/* fetchFisheye() //{ */

void FlightforgeSimulator::fetchFisheye(size_t uav, size_t i) {

  if (!is_initialized_) {
    return;
  }

  FisheyeSensor& s = fisheyes_[uav][i];

  const auto got = fetchFresh([&]() { return s.connector->GetFisheyeCameraData(s.id); }, s.last_stamp, s.params.rate);

  if (!got) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: could not get %s: %s", uav_names[uav].c_str(), s.name.c_str(), got.error.c_str());
    return;
  }

  if (got->stamp <= s.last_stamp || (!lockstep_ && !mrs_lib::get_mutexed(mutex_ue_world_offset_, ue_world_offset_valid_))) {
    if (lockstep_) {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: %s returned no new frame", uav_names[uav].c_str(), s.name.c_str());
    }
    return;
  }

  s.last_stamp = got->stamp;

  const rclcpp::Time published_stamp = worldStamp(got->stamp);
  if (published_stamp <= s.last_published_stamp) {
    return;
  }
  s.last_published_stamp = published_stamp;

  const auto frame = toBgr(*got, s.info.width, s.info.height);

  if (!frame) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: %s frame is not %dx%d, dropped", uav_names[uav].c_str(), s.name.c_str(),
                          s.info.width, s.info.height);
    return;
  }

  std_msgs::msg::Header header;
  header.frame_id = s.frame;
  header.stamp    = published_stamp;

  s.ph_image.publish(toColorImage(*frame, header));

  auto info   = s.info;
  info.header = header;
  s.ph_info.publish(info);
}

//}

/* fetchStereo() //{ */

void FlightforgeSimulator::fetchStereo(size_t uav, size_t i) {

  if (!is_initialized_) {
    return;
  }

  StereoSensor& s = stereos_[uav][i];

  const auto got = fetchFresh([&]() { return s.connector->GetStereoCameraData(s.id); }, s.last_stamp, s.params.rate);

  if (!got) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: could not get %s: %s", uav_names[uav].c_str(), s.name.c_str(), got.error.c_str());
    return;
  }

  if (got->stamp <= s.last_stamp || (!lockstep_ && !mrs_lib::get_mutexed(mutex_ue_world_offset_, ue_world_offset_valid_))) {
    if (lockstep_) {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: %s returned no new frame", uav_names[uav].c_str(), s.name.c_str());
    }
    return;
  }

  s.last_stamp = got->stamp;

  const rclcpp::Time published_stamp = worldStamp(got->stamp);
  if (published_stamp <= s.last_published_stamp) {
    return;
  }
  s.last_published_stamp = published_stamp;

  const auto left  = toBgr(got->left, s.info_left.width, s.info_left.height);
  const auto right = toBgr(got->right, s.info_right.width, s.info_right.height);

  if (!left || !right) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: %s frame is not %dx%d, dropped", uav_names[uav].c_str(), s.name.c_str(),
                          s.info_left.width, s.info_left.height);
    return;
  }

  std_msgs::msg::Header header_left, header_right;

  header_left.stamp     = published_stamp;
  header_left.frame_id  = s.frame_left;
  header_right.stamp    = header_left.stamp;
  header_right.frame_id = s.frame_right;

  s.ph_image_left.publish(toColorImage(*left, header_left));
  s.ph_image_right.publish(toColorImage(*right, header_right));

  auto info_left   = s.info_left;
  info_left.header = header_left;
  s.ph_info_left.publish(info_left);

  auto info_right   = s.info_right;
  info_right.header = header_right;
  s.ph_info_right.publish(info_right);
}

//}

/* fetchEvent() //{ */

void FlightforgeSimulator::fetchEvent(size_t uav, size_t i) {

  if (!is_initialized_) {
    return;
  }

  EventSensor& s = events_[uav][i];

  const auto got = s.connector->GetEventCameraData(s.id);

  if (!got) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: could not get %s: %s", uav_names[uav].c_str(), s.name.c_str(), got.error.c_str());
    return;
  }

  const flightforge::CameraEvents& events = *got;

  if (events.dropped > 0) {
    RCLCPP_WARN_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: %s dropped %u events", uav_names[uav].c_str(), s.name.c_str(), events.dropped);
  }

  // without the first time sync the stamp would be raw UE world time
  if (events.stamp <= s.last_stamp || !mrs_lib::get_mutexed(mutex_ue_world_offset_, ue_world_offset_valid_)) {
    return;
  }

  s.last_stamp = events.stamp;

  std_msgs::msg::Header header;
  header.frame_id = s.frame;
  header.stamp    = ueWorldTimeToSimtime(events.stamp);

  s.ph_events.publish(toEventPacket(events, header));
}

//}

/* fetchLidar() //{ */

void FlightforgeSimulator::fetchLidar(size_t uav, size_t i) {

  if (!is_initialized_) {
    return;
  }

  LidarSensor& s = lidars_[uav][i];

  const auto scan = s.connector->GetLidarScan(s.params.intensity, s.params.labels, s.id);

  if (!scan) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: could not get %s: %s", uav_names[uav].c_str(), s.name.c_str(), scan.error.c_str());
    return;
  }

  // without the first time sync the stamp would be raw world time
  if (scan->stamp == s.last_stamp || (!lockstep_ && !mrs_lib::get_mutexed(mutex_ue_world_offset_, ue_world_offset_valid_))) {
    return;
  }

  s.last_stamp = scan->stamp;

  const rclcpp::Time published_stamp = worldStamp(scan->stamp);

  if (published_stamp <= s.last_published_stamp) {
    RCLCPP_WARN_THROTTLE(node_->get_logger(), *wall_clock_, 5000, "%s: %s dropped a scan stamped %.3f, the last was %.3f", uav_names[uav].c_str(), s.name.c_str(),
                         published_stamp.seconds(), s.last_published_stamp.seconds());
    return;
  }

  s.last_published_stamp = published_stamp;

  std_msgs::msg::Header header;
  header.frame_id = s.frame;
  header.stamp    = published_stamp;

  const auto clouds = toLidarClouds(*scan, s.params, header, s.rng);

  if (!clouds) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: %s returned channels not parallel to its points, dropped", uav_names[uav].c_str(),
                          s.name.c_str());
    return;
  }

  s.ph_points.publish(clouds->points);

  if (s.params.publish_free) {
    s.ph_free.publish(clouds->free);
  }
}

//}

/* fetchRangefinder() //{ */

void FlightforgeSimulator::fetchRangefinder(size_t uav, size_t i) {

  if (!is_initialized_) {
    return;
  }

  RangefinderSensor& s = rangefinders_[uav][i];

  const auto reading = s.connector->GetRangefinderData(s.id);

  if (!reading) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: could not get %s: %s", uav_names[uav].c_str(), s.name.c_str(), reading.error.c_str());
    return;
  }

  if (!lockstep_ && !mrs_lib::get_mutexed(mutex_ue_world_offset_, ue_world_offset_valid_)) {
    return;
  }

  const rclcpp::Time published_stamp = worldStamp(reading->stamp);

  if (published_stamp <= s.last_published_stamp) {
    return;
  }

  s.last_published_stamp = published_stamp;

  sensor_msgs::msg::Range msg;

  msg.header.stamp    = published_stamp;
  msg.header.frame_id = s.frame;
  msg.radiation_type  = sensor_msgs::msg::Range::INFRARED;
  msg.min_range       = 0.1;
  msg.max_range       = 30.0;
  msg.range           = std::isnan(reading->range) ? std::numeric_limits<float>::infinity() : float(reading->range);  // REP 117: +inf is no return

  s.ph_range.publish(msg);
}

//}

}  // namespace mrs_uav_flightforge_simulator

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(mrs_uav_flightforge_simulator::FlightforgeSimulator)
