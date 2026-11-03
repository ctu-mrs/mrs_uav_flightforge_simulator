/* includes //{ */

#include <algorithm>

#include <rclcpp/rclcpp.hpp>

#include <mrs_multirotor_simulator/uav_system_ros.h>

#include <rosgraph_msgs/msg/clock.hpp>

#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/quaternion_stamped.hpp>

#include <mrs_lib/param_loader.h>
#include <mrs_lib/publisher_handler.h>
#include <mrs_lib/timer_handler.h>
#include <mrs_lib/scope_timer.h>
#include <mrs_lib/transform_broadcaster.h>
#include <mrs_lib/attitude_converter.h>
#include <mrs_lib/dynparam_mgr.h>

#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

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

#include <flight_forge_connector/flight_forge_connector.h>
#include <flight_forge_connector/game_mode_controller.h>
#include <flight_forge_connector/serialization/serializable_shared.h>
#include <flight_forge_connector/serialization/serializable_schema.h>

#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>

#include <random>
#include <vector>
#include <chrono>

#include <mrs_msgs/srv/float64_srv.hpp>
#include <mrs_uav_flightforge_simulator/srv/set_orientation.hpp>

#include <mrs_uav_flightforge_simulator/sensor_config.h>
#include <mrs_uav_flightforge_simulator/sensor_conversions.h>
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
  rclcpp::CallbackGroup::SharedPtr cbgrp_sensors_;
  rclcpp::CallbackGroup::SharedPtr cbgrp_status_;
  rclcpp::CallbackGroup::SharedPtr cbgrp_unreal_sync_;

  rclcpp::TimerBase::SharedPtr timer_init_;
  void                         timerInit();

  rclcpp::Node::SharedPtr  node_;
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Clock::SharedPtr wall_clock_;
  std::atomic<bool>        is_initialized_ = false;

  std::shared_ptr<mrs_lib::ScopeTimerLogger>       scope_timer_logger_;
  std::shared_ptr<image_transport::ImageTransport> it_;

  // | ------------------------- params ------------------------- |

  double _simulation_rate_;
  double _clock_rate_;
  bool   _collisions_ = false;

  rclcpp::Time sim_time_;
  double       sim_time_wall_ = 0;
  rclcpp::Time last_step_time_;
  rclcpp::Time last_step_wall_time_;
  std::mutex   mutex_sim_time_;

  std::string _world_frame_name_;
  std::string _simulator_ip_;
  int         _simulator_port_;

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

  std::vector<std::vector<CameraSensor>>      cameras_;
  std::vector<std::vector<FisheyeSensor>>     fisheyes_;
  std::vector<std::vector<StereoSensor>>      stereos_;
  std::vector<std::vector<EventSensor>>       events_;
  std::vector<std::vector<LidarSensor>>       lidars_;
  std::vector<std::vector<RangefinderSensor>> rangefinders_;

  void setupSensors(size_t uav);
  void startSensorTimers(void);
  void sendStaticTf(const std::string& uav_name, const std::string& frame, const Mount& mount, bool optical, double y_shift = 0.0);

  void timerCamera(size_t uav, size_t i);
  void timerFisheye(size_t uav, size_t i);
  void timerStereo(size_t uav, size_t i);
  void timerEvent(size_t uav, size_t i);
  void timerLidar(size_t uav, size_t i);
  void timerRangefinder(size_t uav, size_t i);

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

  std::shared_ptr<ueds_connector::GameModeController> ueds_game_controller_;

  std::vector<std::shared_ptr<ueds_connector::UedsConnector>> ueds_connectors_;

  std::mutex mutex_flightforge_;

  ueds_connector::Coordinates ueds_world_origin_;

  std::vector<ueds_connector::Coordinates> ueds_world_origins_;


  void updateUnrealPoses(const bool teleport_without_collision);

  ueds_connector::Coordinates position2ue(const Eigen::Vector3d& pos, const ueds_connector::Coordinates& ueds_world_origin);

  void checkForCrash(void);

  rclcpp::Time flightforgeTimeToSimtime(const double flightforge_time);

  rclcpp::Time ueWorldTimeToSimtime(const double ue_world_time);

  // how much to add to unreal time to get to our wall time
  double       wall_time_offset_             = 0;
  double       wall_time_offset_drift_slope_ = 0;
  std::mutex   mutex_wall_time_offset_;
  double       ue_world_offset_ = 0;  // ROS sim time minus UE world time
  bool         ue_world_offset_valid_ = false;
  rclcpp::Time last_real_;

  double                   flightforge_fps_ = 0;
  std::string              flightforge_world_level_name_enum_;
  std::string              flightforge_graphics_settings_enum_;
  int                      flightforge_forest_density_     = 5;
  int                      flightforge_forest_hilly_level_ = 3;
  std::string              weather_type_;
  ueds_connector::Daytime  daytime_;
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

    // TODO load collisions params
    // TODO implement mutual collisions
    bool   collisions_enabled  = false;
    bool   collisions_crash    = false;
    double collisions_rebounce = 1;

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

/* op compatibility //{ */

static const std::vector<std::string> kGameModeOps = {"get_api_version",  "get_time",           "get_fps",
                                                      "get_world_origin", "switch_world_level", "spawn_drone_at_location",
                                                      "set_weather",      "set_daytime",        "set_graphics_settings",
                                                      "set_forest_density", "set_forest_hilly_level", "set_mutual_visibility"};

static const std::vector<std::string> kDroneOps = {"set_location_and_rotation_async", "get_crash_state",           "add_sensor",
                                                   "get_camera_info",                 "set_rgb_camera_config",     "set_stereo_camera_config",
                                                   "set_fisheye_camera_config",       "set_event_camera_config",   "set_lidar_config",
                                                   "get_rgb_camera_data",             "get_rgb_seg_camera_data",   "get_stereo_camera_data",
                                                   "get_fisheye_camera_data",         "get_depth_camera_data",     "get_instance_seg_data",
                                                   "get_instance_seg_map",            "get_event_camera_data",     "get_lidar_scan",
                                                   "get_rangefinder_data"};

static bool checkOpCompatibility(ueds_connector::SocketClient& client, const std::vector<std::string>& ops, const std::string& who,
                                 const rclcpp::Logger& logger) {

  // Incompatibilities() skips names the local registry lacks, so a typo here would pass silently
  const auto local_ops = Serializable::Schema::BuildMetaResponse().ops_;
  for (const auto& op : ops) {
    if (std::none_of(local_ops.begin(), local_ops.end(), [&](const auto& m) { return m.name_ == op; })) {
      RCLCPP_ERROR(logger, "%s: op '%s' is not in the connector's registry, fix the op list in this node", who.c_str(), op.c_str());
      return false;
    }
  }

  const auto [answered, problems] = client.CheckCompatibility(ops);

  if (answered && problems.empty()) {
    return true;
  }

  RCLCPP_ERROR(logger, "%s: the simulator does not serve %zu op(s) this node needs:", who.c_str(), problems.size());
  for (const auto& problem : problems) {
    RCLCPP_ERROR(logger, "    %s", problem.c_str());
  }
  RCLCPP_ERROR(logger, "%s: update either the simulator binary or this package so they agree", who.c_str());

  return false;
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
  param_loader.loadParam(yaml_prefix + "simulator_port", _simulator_port_, 8551);

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

  dynparam_mgr_->register_param(yaml_prefix + "collisions/crash", &drs_params_.collisions_crash);

  dynparam_mgr_->register_param(yaml_prefix + "collisions/rebounce", &drs_params_.collisions_rebounce, mrs_lib::DynparamMgr::range_t<double>(0.1, 1000));

  dynparam_mgr_->register_param(yaml_prefix + "paused", &drs_params_.paused, false,
                                (std::function<void(const bool&)>)std::bind(&FlightforgeSimulator::callbackPause, this, std::placeholders::_1));

  param_loader.loadParam("frames/world/name", _world_frame_name_);

  bool sim_time_from_wall_time;
  param_loader.loadParam(yaml_prefix + "sim_time_from_wall_time", sim_time_from_wall_time);

  if (sim_time_from_wall_time) {
    sim_time_            = clock_->now();
    last_step_time_      = clock_->now();
    last_step_wall_time_ = wall_clock_->now();
  } else {
    sim_time_            = rclcpp::Time(0, 0, RCL_ROS_TIME);
    last_step_time_      = rclcpp::Time(0, 0, RCL_ROS_TIME);
    last_step_wall_time_ = rclcpp::Time(0, 0, RCL_SYSTEM_TIME);
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
    }
    catch (const std::exception& e) {
      RCLCPP_ERROR(node_->get_logger(), "%s", e.what());
      rclcpp::shutdown();
      exit(1);
    }
  }

  param_loader.loadParam(yaml_prefix + "weather_type", weather_type_);
  param_loader.loadParam(yaml_prefix + "graphics_settings", flightforge_graphics_settings_enum_);
  param_loader.loadParam(yaml_prefix + "uavs_mutual_visibility", uavs_mutual_visibility_);
  param_loader.loadParam(yaml_prefix + "daytime/hour", daytime_hour_);
  param_loader.loadParam(yaml_prefix + "daytime/minute", daytime_minute_);
  param_loader.loadParam(yaml_prefix + "world_name", flightforge_world_level_name_enum_);

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

  ueds_game_controller_ = std::make_shared<ueds_connector::GameModeController>(_simulator_ip_, _simulator_port_);

  while (rclcpp::ok()) {

    if (ueds_game_controller_->Connect()) {
      break;
    }

    RCLCPP_ERROR(node_->get_logger(), "[FlightforgeSimulator]: Error connecting to Unreal's game mode controller");

    std::this_thread::sleep_for(1s);
  }

  auto [res, api_version]                     = ueds_game_controller_->GetApiVersion();
  auto [api_version_major, api_version_minor] = api_version;

  if (res) {
    RCLCPP_INFO(node_->get_logger(), "FlightForge API %d.%d (ROS side built against %d.%d)", api_version_major, api_version_minor, API_VERSION_MAJOR,
                API_VERSION_MINOR);
  }

  if (!checkOpCompatibility(*ueds_game_controller_, kGameModeOps, "game mode", node_->get_logger())) {
    rclcpp::shutdown();
    return;
  }

  const auto& name_to_level = ueds_connector::WorldName::Name2Level();
  const auto  it            = name_to_level.find(flightforge_world_level_name_enum_);
  if (it == name_to_level.end()) {
    std::string error_msg = "Unknown world_name: '" + flightforge_world_level_name_enum_ + "'. Available worlds: ";
    for (const auto& pair : name_to_level) {
      error_msg += "'" + pair.first + "' ";
    }
    throw std::invalid_argument(error_msg);
  }
  res = ueds_game_controller_->SwitchWorldLevel(it->second);

  if (res) {
    RCLCPP_INFO(node_->get_logger(), "World was switched succesfully.");
  } else {
    RCLCPP_ERROR(node_->get_logger(), "World was not switched succesfully");
  }

  res = ueds_game_controller_->Disconnect();
  if (res) {
    RCLCPP_INFO(node_->get_logger(), "ueds_game_controller_ was Disconnected succesfully.");
  } else {
    RCLCPP_ERROR(node_->get_logger(), "ueds_game_controller_ was not Disconnected succesfully.");
  }

  std::this_thread::sleep_for(std::chrono::seconds(5));

  while (true) {
    bool connect_result = ueds_game_controller_->Connect();
    if (connect_result != 1) {
      RCLCPP_ERROR(node_->get_logger(), "Error connecting to Unreal's game mode controller, connect_result was %d", connect_result);
    } else {
      break;
    }
    /* TODO: This should be probably shomehow be dynamic and set to a machine/world combo */
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  res = ueds_game_controller_->SetGraphicsSettings(ueds_connector::GraphicsSettings::Name2Id().at(flightforge_graphics_settings_enum_));

  if (res) {
    RCLCPP_INFO(node_->get_logger(), "Graphical Settings was set succesfully to '%s'", flightforge_graphics_settings_enum_.c_str());
  } else {
    RCLCPP_ERROR(node_->get_logger(), "Graphical Settings was set succesfully to '%s'", flightforge_graphics_settings_enum_.c_str());
  }

  res = ueds_game_controller_->SetMutualDroneVisibility(uavs_mutual_visibility_);

  if (res) {
    RCLCPP_INFO(node_->get_logger(), "Mutual Drone Visibility was succesfully set to true");
  } else {
    RCLCPP_ERROR(node_->get_logger(), "Set Mutual Drone Visibility was NOT succesfull.");
  }

  res = ueds_game_controller_->SetWeather(ueds_connector::WeatherType::Type2Id().at(weather_type_));

  if (res) {
    RCLCPP_INFO(node_->get_logger(), "SetWeather successful.");
  } else {
    RCLCPP_ERROR(node_->get_logger(), "SetWeather error");
  }

  res = ueds_game_controller_->SetDatetime(daytime_hour_, daytime_minute_);

  if (res) {
    RCLCPP_INFO(node_->get_logger(), "Setdaytime successful.");
  } else {
    RCLCPP_ERROR(node_->get_logger(), "Setdaytime error");
  }


  /* // | --------------------- These graphical settings influence only Forest Game World --------------------- | */

  /* res = ueds_game_controller_->SetForestDensity(ueds_forest_density_); */
  /* if (res) { */
  /*   ROS_INFO("[UnrealSimulator]: Forest Density was set succesfully to '%d'", ueds_forest_density_); */
  /* } else { */
  /*   ROS_ERROR("[UnrealSimulator]: Forest Density wasn't set succesfully to '%d'", ueds_forest_density_); */
  /* } */

  /* res = ueds_game_controller_->SetForestHillyLevel(ueds_forest_hilly_level_); */
  /* if (res) { */
  /*   ROS_INFO("[UnrealSimulator]: Forest Hilly Level was set succesfully to '%d'", ueds_forest_hilly_level_); */
  /* } else { */
  /*   ROS_ERROR("[UnrealSimulator]: Forest Hilly Level wasn't set succesfully to '%d'", ueds_forest_hilly_level_); */
  /* } */

  /* std::this_thread::sleep_for(std::chrono::seconds(1)); */

  // | --------------------- Spawn the UAVs in FlightForge --------------------- |

  const auto [result, world_origin] = ueds_game_controller_->GetWorldOrigin();

  if (!result) {
    RCLCPP_ERROR(node_->get_logger(), "getting world origin");
    rclcpp::shutdown();
  } else {
    ueds_world_origin_ = world_origin;
  }

  for (size_t i = 0; i < uav_names.size(); i++) {

    const std::string uav_name = uav_names[i];

    mrs_multirotor_simulator::MultirotorModel::State uav_state = uavs_[i]->getState();

    ueds_connector::Coordinates pos = position2ue(uav_state.x, ueds_world_origin_);

    RCLCPP_INFO(node_->get_logger(), "%s spawning at [%.2lf, %.2lf, %.2lf] ...", uav_name.c_str(), uav_state.x(0), uav_state.x(1), uav_state.x(2));

    std::string uav_frame = "x500";
    /* TODO: frame param */
    /* param_loader.loadParam(uav_names[i] + "/frame", uav_frame); */

    RCLCPP_INFO(node_->get_logger(), "Frame type to spawn is %s", uav_frame.c_str());

    const auto& known_frames = ueds_connector::UavFrameType::KnownNames();
    if (std::find(known_frames.begin(), known_frames.end(), uav_frame) == known_frames.end()) {
      RCLCPP_WARN(node_->get_logger(), "Frame '%s' is not a packed mesh; the simulator will look for a .glb of that name", uav_frame.c_str());
    }

    auto [resSpawn, port] = ueds_game_controller_->SpawnDroneAtLocation(pos, uav_frame);

    if (!resSpawn) {
      RCLCPP_ERROR(node_->get_logger(), "failed to spawn %s", uav_names[i].c_str());
      rclcpp::shutdown();
    }

    RCLCPP_INFO(node_->get_logger(), "%s spawned", uav_name.c_str());

    std::shared_ptr<ueds_connector::UedsConnector> ueds_connector = std::make_shared<ueds_connector::UedsConnector>(_simulator_ip_, port);

    ueds_connectors_.push_back(ueds_connector);

    auto connect_result = ueds_connector->Connect();

    if (connect_result != 1) {

      RCLCPP_ERROR(node_->get_logger(), "%s - Error connecting to drone controller, connect_result was %d", uav_name.c_str(), connect_result);
      rclcpp::shutdown();

    } else {
      RCLCPP_INFO(node_->get_logger(), "%s - Connection succeed: %d", uav_name.c_str(), connect_result);

      if (!checkOpCompatibility(*ueds_connector, kDroneOps, uav_name, node_->get_logger())) {
        rclcpp::shutdown();
        return;
      }

      // ROS_INFO("[UnrealSimulator]: wait until UAV fall on the ground ... && uptade their world origin");

      // std::this_thread::sleep_for(std::chrono::seconds(3));

      // const auto [res, location] = ueds_connector->GetLocation();

      // if (!res) {
      //   ROS_ERROR("[UnrealSimulator]: %s - DroneError: getting location", uav_name.c_str());
      //   ros::shutdown();
      // } else {
      //   ueds_world_origins_.push_back(location);
      // }
    }

    setupSensors(i);
  }

  // | ----------------------- publishers ----------------------- |

  ph_clock_ = mrs_lib::PublisherHandler<rosgraph_msgs::msg::Clock>(node_, "~/clock_out");

  ph_poses_ = mrs_lib::PublisherHandler<geometry_msgs::msg::PoseArray>(node_, "~/uav_poses_out");

  // | ------------------------- timers ------------------------- |

  mrs_lib::TimerHandlerOptions opts;

  opts.node      = node_;
  opts.autostart = true;

  timer_main_ = node_->create_wall_timer(std::chrono::duration<double>(1.0 / (_clock_rate_ * drs_params_.realtime_factor)),
                                         std::bind(&FlightforgeSimulator::timerMain, this), cbgrp_main_);

  timer_status_ = node_->create_wall_timer(std::chrono::duration<double>(1.0), std::bind(&FlightforgeSimulator::timerStatus, this), cbgrp_status_);

  timer_time_sync_ = node_->create_wall_timer(std::chrono::duration<double>(1.0), std::bind(&FlightforgeSimulator::timerTimeSync, this), cbgrp_status_);

  timer_unreal_sync_ =
      node_->create_wall_timer(std::chrono::duration<double>(1.0 / _clock_rate_), std::bind(&FlightforgeSimulator::timerUnrealSync, this), cbgrp_unreal_sync_);

  startSensorTimers();

  // | ----------------------- scope timer ---------------------- |

  scope_timer_logger_ = std::make_shared<mrs_lib::ScopeTimerLogger>(node_, "", false);

  // | -------------------- finishing methods ------------------- |

  // | ----------------------- finish init ---------------------- |

  is_initialized_ = true;

  RCLCPP_INFO(node_->get_logger(), "initialized");

  timer_init_->cancel();
}

//}//}

/* timerMain() //{ */

void FlightforgeSimulator::timerMain() {

  if (!is_initialized_) {
    return;
  }

  RCLCPP_INFO_ONCE(node_->get_logger(), "Main timer spinning");

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

  // fast propagation of the wall time offset using the offset drift
  {
    std::scoped_lock lock(mutex_wall_time_offset_);

    rclcpp::Time now_wall = wall_clock_->now();

    const double wall_dt_since_last_step = (now_wall - last_step_wall_time_).seconds();

    if (wall_dt_since_last_step > 1e-5) {

      wall_time_offset_ += wall_time_offset_drift_slope_ * wall_dt_since_last_step;
    }

    last_step_wall_time_ = now_wall;
  }

  // step the sim
  {
    const double dt_since_last_step = (sim_time - last_step_time_).seconds();

    if (dt_since_last_step >= simulation_step_size) {

      for (size_t i = 0; i < uavs_.size(); i++) {

        uavs_.at(i)->makeStep(dt_since_last_step, sim_time.seconds());
      }

      publishPoses();

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

    bool res;

    std::tie(res, fps) = ueds_game_controller_->GetFps();

    if (!res) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to get the FPS from FlightForge");
      return;
    }
  }

  const double fps_filter_const = 0.9;
  flightforge_fps_              = fps_filter_const * flightforge_fps_ + (1.0 - fps_filter_const) * fps;

  const double highest_fps = sensors_config_.highestRate();

  const double flightforge_rtf = flightforge_fps_ / highest_fps;

  const double desired_rtf = (drs_params.dynamic_rtf && flightforge_rtf < drs_params.realtime_factor) ? flightforge_rtf : drs_params.realtime_factor;

  timer_main_->cancel();
  timer_main_ = node_->create_wall_timer(std::chrono::duration<double>(1.0 / (_clock_rate_ * desired_rtf)), std::bind(&FlightforgeSimulator::timerMain, this),
                                         cbgrp_main_);

  // the sim's own clock drives Livox point counts and motion, so it must run at the ROS rate
  if (std::abs(desired_rtf - flightforge_time_dilation_) > 0.01 * desired_rtf) {

    std::scoped_lock lock(mutex_flightforge_);

    double applied;

    if (ueds_game_controller_->SetTimeDilation(desired_rtf, applied)) {
      flightforge_time_dilation_ = applied;
    } else {
      RCLCPP_ERROR(node_->get_logger(), "Failed to set the FlightForge time dilation to %.3f", desired_rtf);
    }
  }

  if (_collisions_) {
    checkForCrash();
  }

  RCLCPP_INFO(node_->get_logger(), "%s, desired RTF = %.2f, actual RTF = %.2f, FlightForge FPS = %.2f, FlightForge RTF = %.2f, time dilation = %.2f",
              drs_params.paused ? "paused" : "running", drs_params.realtime_factor, actual_rtf, flightforge_fps_, flightforge_rtf, flightforge_time_dilation_);

  mrs_lib::set_mutexed(mutex_actual_rtf_, actual_rtf, actual_rtf_);
}

//}

/* timerUnrealSync() //{ */

void FlightforgeSimulator::timerUnrealSync() {

  if (!is_initialized_) {
    return;
  }

  updateUnrealPoses(false);
}

//}

/* timerTimeSync() //{ */

void FlightforgeSimulator::timerTimeSync() {

  rclcpp::Time current_real = wall_clock_->now();

  if (!is_initialized_) {
    return;
  }

  auto wall_time_offset = mrs_lib::get_mutexed(mutex_wall_time_offset_, wall_time_offset_);

  bool   res;
  double flightforge_time;
  double sync_start;
  double sync_end;

  bool   res_ue;
  double ue_world_time;
  double ue_dilation;
  double ue_sync_start;
  double ue_sync_end;

  // stamped inside the lock, so waiting for it is not mistaken for round trip
  {
    std::scoped_lock lock(mutex_flightforge_);

    sync_start                      = wall_clock_->now().seconds();
    std::tie(res, flightforge_time) = ueds_game_controller_->GetTime();
    sync_end                        = wall_clock_->now().seconds();

    ue_sync_start = wall_clock_->now().seconds();
    res_ue        = ueds_game_controller_->GetSimTime(ue_world_time, ue_dilation);
    ue_sync_end   = wall_clock_->now().seconds();
  }

  if (res_ue) {

    auto [sim_now, sim_now_wall] = mrs_lib::get_mutexed(mutex_sim_time_, sim_time_, sim_time_wall_);
    auto actual_rtf              = mrs_lib::get_mutexed(mutex_actual_rtf_, actual_rtf_);

    const double ros_at_sample = sim_now.seconds() + ((ue_sync_start + ue_sync_end) / 2.0 - sim_now_wall) * actual_rtf;

    std::scoped_lock lock(mutex_wall_time_offset_);

    // round-trip jitter would otherwise step event stamps back
    ue_world_offset_ = ue_world_offset_valid_ ? 0.9 * ue_world_offset_ + 0.1 * (ros_at_sample - ue_world_time) : ros_at_sample - ue_world_time;
    ue_world_offset_valid_ = true;
  }

  if (!res) {
    RCLCPP_ERROR(node_->get_logger(), "Failed to get FlightForge's time");
    rclcpp::shutdown();
  }

  const double true_flightforge_time = flightforge_time - (sync_end - sync_start) / 2.0;

  const double new_wall_time_offset = sync_start - true_flightforge_time;

  // | --------------- time drift slope estimation -------------- |

  if (current_real.seconds() > 0 && last_real_.seconds() > 0) {

    const double wall_dt = (current_real - last_real_).seconds();

    if (wall_dt > 0) {

      double drift_estimate = (new_wall_time_offset - wall_time_offset) / wall_dt;

      {
        std::scoped_lock lock(mutex_wall_time_offset_);

        wall_time_offset_drift_slope_ += drift_estimate;
      }
    }
  }

  // | ------------------------- finish ------------------------- |

  {
    std::scoped_lock lock(mutex_wall_time_offset_);

    wall_time_offset_ = new_wall_time_offset;
  }

  last_real_ = current_real;

  RCLCPP_INFO(node_->get_logger(), "wall time %f flightforge %f time offset: %f, offset slope %f s/s", sync_start, flightforge_time, wall_time_offset_,
              wall_time_offset_drift_slope_);
}

//}

// | ------------------------ callbacks ----------------------- |

/* dynamic parameter callbacks //{ */

/* callbackRealtimeFactor() //{ */

void FlightforgeSimulator::callbackRealtimeFactor(const double& param_value) {

  timer_main_->cancel();

  timer_main_ =
      create_wall_timer(std::chrono::duration<double>(1.0 / (_clock_rate_ * param_value)), std::bind(&FlightforgeSimulator::timerMain, this), cbgrp_main_);

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

    timer_main_ = create_wall_timer(std::chrono::duration<double>(1.0 / (_clock_rate_ * drs_params_.realtime_factor)),
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

/* updateUnrealPoses() //{ */

void FlightforgeSimulator::updateUnrealPoses(const bool teleport_without_collision) {

  // | ------------ set each UAV's position in unreal ----------- |

  {
    std::scoped_lock lock(mutex_flightforge_);

    for (size_t i = 0; i < uavs_.size(); i++) {

      mrs_multirotor_simulator::MultirotorModel::State state = uavs_[i]->getState();

      auto [roll, pitch, yaw] = mrs_lib::AttitudeConverter(state.R).getExtrinsicRPY();

      ueds_connector::Coordinates pos;

      pos = position2ue(state.x, ueds_world_origin_);

      ueds_connector::Rotation rot;
      rot.pitch = 180.0 * (-pitch / M_PI);
      rot.roll  = 180.0 * (roll / M_PI);
      rot.yaw   = 180.0 * (-yaw / M_PI);

      ueds_connectors_[i]->SetLocationAndRotationAsync(pos, rot, !teleport_without_collision && _collisions_);
    }
  }
}
//}

/* position2ue() //{ */

ueds_connector::Coordinates FlightforgeSimulator::position2ue(const Eigen::Vector3d& pos, const ueds_connector::Coordinates& ueds_world_origin) {
  ueds_connector::Coordinates pos_ue;

  float PlayerStartOffset = 92.12;

  pos_ue.x = ueds_world_origin.x + pos.x() * 100.0;
  pos_ue.y = ueds_world_origin.y - pos.y() * 100.0;
  pos_ue.z = ueds_world_origin.z + pos.z() * 100.0 - PlayerStartOffset;

  return pos_ue;
}

//}

/* checkForCrash() //{ */

void FlightforgeSimulator::checkForCrash(void) {

  // | ------------ set each UAV's position in unreal ----------- |

  {
    std::scoped_lock lock(mutex_flightforge_);

    for (size_t i = 0; i < uavs_.size(); i++) {

      auto [res, crashed] = ueds_connectors_[i]->GetCrashState();

      /* if the uav has crashed check the rangefinder data to determine if its not just a landing */
      if (crashed) {
        bool   res_range;
        double range;
        {
          std::scoped_lock lock(mutex_flightforge_);

          std::tie(res_range, range) = ueds_connectors_[i]->GetRangefinderData();
        }

        /* if (res_range && range < 0.1) { */
        /*   crashed = false; */
        /*   if (set_ground_z_clients_[i].exists()) { */
        /*     mrs_msgs::Float64Srv srv; */
        /*     // set the ground_z to the current position */
        /*     srv.request.value = uavs_[i]->getState().x.z(); */
        /*     if (set_ground_z_clients_[i].call(srv)) { */
        /*       if (srv.response.success) { */
        /*         RCLCPP_INFO(node_->get_logger(), "Successfully set ground_z for uav%lu to ", i + 1 ); */
        /*       } else { */
        /*         RCLCPP_ERROR(node_->get_logger(), "Failed to set ground_z for uav%lu: %s", i + 1, srv.response.message.c_str()); */
        /*       } */
        /*     } else { */
        /*       RCLCPP_ERROR(node_->get_logger(), "Failed to call set_ground_z service for uav%lu", i + 1); */
        /*     } */
        /*   } else { */
        /*     RCLCPP_WARN(node_->get_logger(), "Set_ground_z service for uav%lu does not exist", i + 1); */
        /*   } */
        /* } */
      }

      if (!res) {
        RCLCPP_ERROR_THROTTLE(node_->get_logger(), *clock_, 1000, "Failed to obtain crash state for uav%lu", i + 1);
      }

      if (res && crashed && !uavs_[i]->hasCrashed()) {
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *clock_, 1000, "Uav%lu crashed", i + 1);
        uavs_[i]->crash();
      }
    }
  }
}

//}

/* flightforgeTimeToSimtime() //{ */

rclcpp::Time FlightforgeSimulator::flightforgeTimeToSimtime(const double flightforge_time) {

  auto wall_time_offset = mrs_lib::get_mutexed(mutex_wall_time_offset_, wall_time_offset_);
  auto actual_rtf       = mrs_lib::get_mutexed(mutex_actual_rtf_, actual_rtf_);

  auto [sim_now, sim_now_wall] = mrs_lib::get_mutexed(mutex_sim_time_, sim_time_, sim_time_wall_);

  // interpolated between clock ticks, else two captures within one tick can swap order
  const double capture_wall = flightforge_time + wall_time_offset;
  const double sim_time     = sim_now.seconds() - std::max(0.0, sim_now_wall - capture_wall) * actual_rtf;

  if (sim_time < 0) {
    return rclcpp::Time(0, 0, clock_->get_clock_type());
  }

  const double secs     = floor(sim_time);
  const double nanosecs = (sim_time - secs) * 1e9;

  return rclcpp::Time(secs, nanosecs, clock_->get_clock_type());
}

//}

/* ueWorldTimeToSimtime() //{ */

rclcpp::Time FlightforgeSimulator::ueWorldTimeToSimtime(const double ue_world_time) {

  const double sim_time = ue_world_time + mrs_lib::get_mutexed(mutex_wall_time_offset_, ue_world_offset_);

  return rclcpp::Time(int64_t(std::max(0.0, sim_time) * 1e9), clock_->get_clock_type());
}

//}

// | ------------------------- sensors ------------------------ |

/* sendStaticTf() //{ */

void FlightforgeSimulator::sendStaticTf(const std::string& uav_name, const std::string& frame, const Mount& mount, bool optical, double y_shift) {

  geometry_msgs::msg::TransformStamped tf;

  tf.header.stamp    = clock_->now();
  tf.header.frame_id = uav_name + "/fcu";
  tf.child_frame_id  = frame;

  tf.transform.translation.x = mount.x;
  tf.transform.translation.y = mount.y + y_shift;
  tf.transform.translation.z = mount.z;

  Eigen::Matrix3d rotation = mrs_lib::AttitudeConverter(M_PI * mount.roll / 180.0, M_PI * mount.pitch / 180.0, M_PI * mount.yaw / 180.0);

  if (optical) {
    rotation = rotation * Eigen::Matrix3d(mrs_lib::AttitudeConverter(Eigen::Quaterniond(-0.5, 0.5, -0.5, 0.5)));
  }

  tf.transform.rotation = mrs_lib::AttitudeConverter(rotation);

  static_broadcaster_->sendTransform(tf);
}

//}

/* setupSensors() //{ */

void FlightforgeSimulator::setupSensors(size_t uav) {

  const std::string uav_name  = uav_names[uav];
  auto&             connector = ueds_connectors_[uav];

  const auto fail = [&](const std::string& what) {
    RCLCPP_ERROR(node_->get_logger(), "%s: %s", uav_name.c_str(), what.c_str());
    rclcpp::shutdown();
    exit(1);
  };

  const auto add = [&](int type, const std::string& name) {
    const auto [ok, id] = connector->AddSensor(type);
    if (!ok) {
      fail("could not add " + name);
    }
    return id;
  };

  const auto calibration = [&](int id, const std::string& name) {
    const auto [ok, infos] = connector->GetCameraInfo(id);
    if (!ok || infos.empty()) {
      fail("no camera info for " + name);
    }
    return infos;
  };

  const auto checkSize = [&](const sensor_msgs::msg::CameraInfo& info, int width, int height, const std::string& name) {
    if (int(info.width) != width || int(info.height) != height) {
      RCLCPP_WARN(node_->get_logger(), "%s/%s renders %dx%d, not the configured %dx%d", uav_name.c_str(), name.c_str(), info.width, info.height, width, height);
    }
  };

  mrs_lib::PublisherHandlerOptions latched(node_);
  latched.qos = rclcpp::QoS(1).transient_local();

  // | ------------------------- cameras ------------------------ |

  cameras_.emplace_back();

  const std::vector<std::tuple<std::string, int, const std::vector<CameraParams>*>> camera_types = {
      {"rgb", ueds_connector::SENSOR_RGB_CAMERA, &sensors_config_.rgb},
      {"rgb_segmented", ueds_connector::SENSOR_RGB_SEG_CAMERA, &sensors_config_.rgb_segmented},
      {"depth", ueds_connector::SENSOR_DEPTH_CAMERA, &sensors_config_.depth},
      {"instance_seg", ueds_connector::SENSOR_INSTANCE_SEG_CAMERA, &sensors_config_.instance_seg}};

  for (const auto& [type_name, type, params_list] : camera_types) {
    for (size_t i = 0; i < params_list->size(); i++) {

      CameraSensor s;

      s.params = params_list->at(i);
      s.type   = type;
      s.name   = type_name + "_" + std::to_string(i);
      s.frame  = uav_name + "/" + s.name;
      s.id     = add(type, s.name);

      ueds_connector::RgbCameraConfig cfg{};

      cfg.width_                  = s.params.width;
      cfg.height_                 = s.params.height;
      cfg.fov_                    = s.params.fov;
      cfg.offset_                 = toUeOffset(s.params.mount);
      cfg.orientation_            = toUeRotation(s.params.mount);
      cfg.enable_hdr_             = s.params.enable_hdr;
      cfg.enable_temporal_aa_     = s.params.enable_temporal_aa;
      cfg.enable_raytracing_      = s.params.enable_raytracing;
      cfg.enable_motion_blur_     = s.params.enable_motion_blur;
      cfg.motion_blur_amount_     = s.params.motion_blur_amount;
      cfg.motion_blur_distortion_ = s.params.motion_blur_distortion;

      if (!connector->SetRgbCameraConfig(cfg, s.id)) {
        fail("could not configure " + s.name);
      }

      s.info = toCameraInfo(calibration(s.id, s.name).front());
      checkSize(s.info, s.params.width, s.params.height, s.name);

      const std::string topic = "/" + uav_name + "/" + s.name + "/";

      s.ph_image = it_->advertise(topic + "image_raw", 1);
      s.ph_info  = mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>(node_, topic + "camera_info");

      if (type == ueds_connector::SENSOR_INSTANCE_SEG_CAMERA) {
        s.ph_map = mrs_lib::PublisherHandler<mrs_uav_flightforge_simulator::msg::InstanceMap>(latched, topic + "labels");
      }

      sendStaticTf(uav_name, s.frame, s.params.mount, true);

      cameras_[uav].push_back(std::move(s));
    }
  }

  // | ------------------------- fisheye ------------------------ |

  fisheyes_.emplace_back();

  for (size_t i = 0; i < sensors_config_.fisheye.size(); i++) {

    FisheyeSensor s;

    s.params = sensors_config_.fisheye[i];
    s.name   = "fisheye_" + std::to_string(i);
    s.frame  = uav_name + "/" + s.name;
    s.id     = add(ueds_connector::SENSOR_FISHEYE_CAMERA, s.name);

    ueds_connector::FisheyeCameraConfig cfg{};

    cfg.width_       = s.params.width;
    cfg.height_      = s.params.height;
    cfg.fov_         = s.params.fov;
    cfg.lens_model_  = s.params.lens_model;
    cfg.full_frame_  = s.params.full_frame;
    cfg.offset_      = toUeOffset(s.params.mount);
    cfg.orientation_ = toUeRotation(s.params.mount);

    if (!connector->SetFisheyeCameraConfig(cfg, s.id)) {
      fail("could not configure " + s.name);
    }

    s.info = toCameraInfo(calibration(s.id, s.name).front());
    checkSize(s.info, s.params.width, s.params.height, s.name);

    const std::string topic = "/" + uav_name + "/" + s.name + "/";

    s.ph_image = it_->advertise(topic + "image_raw", 1);
    s.ph_info  = mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>(node_, topic + "camera_info");

    sendStaticTf(uav_name, s.frame, s.params.mount, true);

    fisheyes_[uav].push_back(std::move(s));
  }

  // | ------------------------- stereo ------------------------- |

  stereos_.emplace_back();

  for (size_t i = 0; i < sensors_config_.stereo.size(); i++) {

    StereoSensor s;

    s.params      = sensors_config_.stereo[i];
    s.name        = "stereo_" + std::to_string(i);
    s.frame_left  = uav_name + "/" + s.name + "_left";
    s.frame_right = uav_name + "/" + s.name + "_right";
    s.id          = add(ueds_connector::SENSOR_STEREO_CAMERA, s.name);

    const Mount& m          = s.params.mount;
    const double half_shift = s.params.baseline / 2.0;

    Serializable::Drone::StereoCameraConfig cfg{};

    cfg.width_  = s.params.width;
    cfg.height_ = s.params.height;
    cfg.fov_    = s.params.fov;

    cfg.offset_x_left_  = m.x * 100.0;
    cfg.offset_y_left_  = -(m.y + half_shift) * 100.0;
    cfg.offset_z_left_  = m.z * 100.0;
    cfg.offset_x_right_ = m.x * 100.0;
    cfg.offset_y_right_ = -(m.y - half_shift) * 100.0;
    cfg.offset_z_right_ = m.z * 100.0;

    cfg.rotation_pitch_left_  = -m.pitch;
    cfg.rotation_yaw_left_    = -m.yaw;
    cfg.rotation_roll_left_   = m.roll;
    cfg.rotation_pitch_right_ = -m.pitch;
    cfg.rotation_yaw_right_   = -m.yaw;
    cfg.rotation_roll_right_  = m.roll;

    cfg.enable_hdr_         = s.params.enable_hdr;
    cfg.enable_temporal_aa_ = s.params.enable_temporal_aa;
    cfg.enable_raytracing_  = s.params.enable_raytracing;

    if (!connector->SetStereoCameraConfig(cfg, s.id)) {
      fail("could not configure " + s.name);
    }

    const auto  infos = calibration(s.id, s.name);
    const auto* left  = findEye(infos, 0);
    const auto* right = findEye(infos, 1);

    if (!left || !right) {
      fail(s.name + " did not report both eyes");
    }

    s.info_left  = toCameraInfo(*left);
    s.info_right = toCameraInfo(*right);
    checkSize(s.info_left, s.params.width, s.params.height, s.name);

    const std::string topic = "/" + uav_name + "/" + s.name + "/";

    s.ph_image_left  = it_->advertise(topic + "left/image_raw", 1);
    s.ph_image_right = it_->advertise(topic + "right/image_raw", 1);
    s.ph_info_left   = mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>(node_, topic + "left/camera_info");
    s.ph_info_right  = mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>(node_, topic + "right/camera_info");

    sendStaticTf(uav_name, s.frame_left, m, true, half_shift);
    sendStaticTf(uav_name, s.frame_right, m, true, -half_shift);

    stereos_[uav].push_back(std::move(s));
  }

  // | ------------------------- events ------------------------- |

  events_.emplace_back();

  for (size_t i = 0; i < sensors_config_.event.size(); i++) {

    EventSensor s;

    s.params = sensors_config_.event[i];
    s.name   = "event_" + std::to_string(i);
    s.frame  = uav_name + "/" + s.name;
    s.id     = add(ueds_connector::SENSOR_EVENT_CAMERA, s.name);

    ueds_connector::EventCameraConfig cfg{};

    cfg.width_                  = s.params.width;
    cfg.height_                 = s.params.height;
    cfg.fov_                    = s.params.fov;
    cfg.contrast_threshold_pos_ = s.params.contrast_threshold_pos;
    cfg.contrast_threshold_neg_ = s.params.contrast_threshold_neg;
    cfg.max_events_per_frame_   = s.params.max_events_per_frame;
    cfg.use_sim_time_           = true;
    cfg.offset_                 = toUeOffset(s.params.mount);
    cfg.orientation_            = toUeRotation(s.params.mount);

    if (!connector->SetEventCameraConfig(cfg, s.id)) {
      fail("could not configure " + s.name);
    }

    s.ph_events = mrs_lib::PublisherHandler<event_camera_msgs::msg::EventPacket>(node_, "/" + uav_name + "/" + s.name + "/events");

    sendStaticTf(uav_name, s.frame, s.params.mount, true);

    events_[uav].push_back(std::move(s));
  }

  // | ------------------------- lidars ------------------------- |

  lidars_.emplace_back();

  for (size_t i = 0; i < sensors_config_.lidar.size(); i++) {

    LidarSensor s;

    s.params = sensors_config_.lidar[i];
    s.name   = "lidar_" + std::to_string(i);
    s.frame  = uav_name + "/" + s.name;
    s.id     = add(s.params.livox ? ueds_connector::SENSOR_LIDAR_LIVOX : ueds_connector::SENSOR_LIDAR, s.name);

    ueds_connector::LidarConfig cfg{};

    cfg.Enable       = true;
    cfg.showBeams    = false;
    cfg.beamLength   = s.params.beam_length * 100.0;
    cfg.BeamHorRays  = s.params.horizontal_rays;
    cfg.BeamVertRays = s.params.vertical_rays;
    cfg.Frequency    = s.params.rate;
    cfg.offset       = toUeOffset(s.params.mount);
    cfg.orientation  = toUeRotation(s.params.mount);
    cfg.FOVHorLeft   = s.params.horizontal_fov_left;
    cfg.FOVHorRight  = s.params.horizontal_fov_right;
    cfg.FOVVertUp    = s.params.vertical_fov_up;
    cfg.FOVVertDown  = s.params.vertical_fov_down;
    cfg.Livox        = s.params.livox;
    cfg.LivoxSensor  = s.params.livox_sensor;

    if (!connector->SetLidarConfig(cfg, s.id)) {
      fail("could not configure " + s.name);
    }

    const std::string topic = "/" + uav_name + "/" + s.name + "/";

    s.ph_points = mrs_lib::PublisherHandler<sensor_msgs::msg::PointCloud2>(node_, topic + "points");

    if (s.params.publish_free) {
      s.ph_free = mrs_lib::PublisherHandler<sensor_msgs::msg::PointCloud2>(node_, topic + "points_free");
    }

    sendStaticTf(uav_name, s.frame, s.params.mount, false);

    lidars_[uav].push_back(std::move(s));
  }

  // | ---------------------- rangefinders ---------------------- |

  rangefinders_.emplace_back();

  for (size_t i = 0; i < sensors_config_.rangefinder.size(); i++) {

    RangefinderSensor s;

    s.params   = sensors_config_.rangefinder[i];
    s.name     = "rangefinder_" + std::to_string(i);
    s.frame    = uav_name + "/" + s.name;
    s.id       = add(ueds_connector::SENSOR_RANGEFINDER, s.name);
    s.ph_range = mrs_lib::PublisherHandler<sensor_msgs::msg::Range>(node_, "/" + uav_name + "/" + s.name + "/range");

    sendStaticTf(uav_name, s.frame, s.params.mount, false);

    rangefinders_[uav].push_back(std::move(s));
  }
}

//}

/* startSensorTimers() //{ */

void FlightforgeSimulator::startSensorTimers(void) {

  mrs_lib::TimerHandlerOptions opts;

  opts.node           = node_;
  opts.callback_group = cbgrp_sensors_;
  opts.autostart      = true;

  const auto start = [&](auto& sensors, auto callback) {
    for (size_t uav = 0; uav < sensors.size(); uav++) {
      for (size_t i = 0; i < sensors[uav].size(); i++) {
        std::function<void()> fn = [this, callback, uav, i]() { (this->*callback)(uav, i); };
        sensors[uav][i].timer    = std::make_shared<TimerType>(opts, rclcpp::Rate(sensors[uav][i].params.rate, clock_), fn);
      }
    }
  };

  start(cameras_, &FlightforgeSimulator::timerCamera);
  start(fisheyes_, &FlightforgeSimulator::timerFisheye);
  start(stereos_, &FlightforgeSimulator::timerStereo);
  start(events_, &FlightforgeSimulator::timerEvent);
  start(lidars_, &FlightforgeSimulator::timerLidar);
  start(rangefinders_, &FlightforgeSimulator::timerRangefinder);
}

//}

/* timerCamera() //{ */

void FlightforgeSimulator::timerCamera(size_t uav, size_t i) {

  if (!is_initialized_) {
    return;
  }

  CameraSensor& s         = cameras_[uav][i];
  auto&         connector = ueds_connectors_[uav];

  bool                                   res   = false;
  double                                 stamp = 0;
  std::vector<unsigned char>             frame_data;
  std::vector<uint16_t>                  depth;
  int                                    depth_width = 0, depth_height = 0;
  std::optional<sensor_msgs::msg::Image> image;

  {
    std::scoped_lock lock(mutex_flightforge_);

    switch (s.type) {
      case ueds_connector::SENSOR_RGB_CAMERA:
        std::tie(res, frame_data, stamp, std::ignore) = connector->GetRgbCameraData(s.id);
        break;
      case ueds_connector::SENSOR_RGB_SEG_CAMERA:
        std::tie(res, frame_data, stamp, std::ignore) = connector->GetRgbSegmented(s.id);
        break;
      case ueds_connector::SENSOR_INSTANCE_SEG_CAMERA:
        res = connector->GetInstanceSegData(frame_data, stamp, s.id);
        break;
      default:
        res = connector->GetDepthCameraData(depth, depth_width, depth_height, stamp, s.id);
        break;
    }
  }

  if (!res) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: could not get %s", uav_names[uav].c_str(), s.name.c_str());
    return;
  }

  if (stamp == s.last_stamp) {
    return;
  }

  s.last_stamp = stamp;

  std_msgs::msg::Header header;
  header.frame_id = s.frame;
  header.stamp    = flightforgeTimeToSimtime(stamp);

  if (s.type == ueds_connector::SENSOR_DEPTH_CAMERA) {
    image = toDepthImage(depth, s.info.width, s.info.height, header);
  } else if (const auto frame = s.type == ueds_connector::SENSOR_RGB_SEG_CAMERA ? rawFrame(frame_data, s.info.width, s.info.height)
                                                                                : decodeFrame(frame_data, s.info.width, s.info.height)) {
    image = s.type == ueds_connector::SENSOR_INSTANCE_SEG_CAMERA ? toInstanceImage(*frame, header) : toColorImage(*frame, header);
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

  if (s.type == ueds_connector::SENSOR_INSTANCE_SEG_CAMERA && !s.map_published) {

    bool                                                ok;
    std::vector<Serializable::Drone::InstanceSegMapEntry> entries;

    {
      std::scoped_lock lock(mutex_flightforge_);
      std::tie(ok, entries) = connector->GetInstanceSegMap(s.id);
    }

    if (ok) {
      s.ph_map.publish(toInstanceMap(entries, header));
      s.map_published = true;
    }
  }
}

//}

/* timerFisheye() //{ */

void FlightforgeSimulator::timerFisheye(size_t uav, size_t i) {

  if (!is_initialized_) {
    return;
  }

  FisheyeSensor& s = fisheyes_[uav][i];

  bool                       res;
  double                     stamp;
  std::vector<unsigned char> compressed;

  {
    std::scoped_lock lock(mutex_flightforge_);
    std::tie(res, compressed, stamp, std::ignore) = ueds_connectors_[uav]->GetFisheyeCameraData(s.id);
  }

  if (!res) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: could not get %s", uav_names[uav].c_str(), s.name.c_str());
    return;
  }

  if (stamp == s.last_stamp) {
    return;
  }

  s.last_stamp = stamp;

  const auto frame = decodeFrame(compressed, s.info.width, s.info.height);

  if (!frame) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: %s frame is not %dx%d, dropped", uav_names[uav].c_str(), s.name.c_str(),
                          s.info.width, s.info.height);
    return;
  }

  std_msgs::msg::Header header;
  header.frame_id = s.frame;
  header.stamp    = flightforgeTimeToSimtime(stamp);

  s.ph_image.publish(toColorImage(*frame, header));

  auto info   = s.info;
  info.header = header;
  s.ph_info.publish(info);
}

//}

/* timerStereo() //{ */

void FlightforgeSimulator::timerStereo(size_t uav, size_t i) {

  if (!is_initialized_) {
    return;
  }

  StereoSensor& s = stereos_[uav][i];

  bool                       res;
  double                     stamp;
  std::vector<unsigned char> compressed_left, compressed_right;

  {
    std::scoped_lock lock(mutex_flightforge_);
    std::tie(res, compressed_left, compressed_right, stamp) = ueds_connectors_[uav]->GetStereoCameraData(s.id);
  }

  if (!res) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: could not get %s", uav_names[uav].c_str(), s.name.c_str());
    return;
  }

  if (stamp == s.last_stamp) {
    return;
  }

  s.last_stamp = stamp;

  const auto left  = decodeFrame(compressed_left, s.info_left.width, s.info_left.height);
  const auto right = decodeFrame(compressed_right, s.info_right.width, s.info_right.height);

  if (!left || !right) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: %s frame is not %dx%d, dropped", uav_names[uav].c_str(), s.name.c_str(),
                          s.info_left.width, s.info_left.height);
    return;
  }

  std_msgs::msg::Header header_left, header_right;

  header_left.stamp     = flightforgeTimeToSimtime(stamp);
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

/* timerEvent() //{ */

void FlightforgeSimulator::timerEvent(size_t uav, size_t i) {

  if (!is_initialized_) {
    return;
  }

  EventSensor& s = events_[uav][i];

  bool                         res;
  ueds_connector::CameraEvents events;

  {
    std::scoped_lock lock(mutex_flightforge_);
    std::tie(res, events) = ueds_connectors_[uav]->GetEventCameraData(s.id);
  }

  if (!res) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: could not get %s", uav_names[uav].c_str(), s.name.c_str());
    return;
  }

  if (events.dropped > 0) {
    RCLCPP_WARN_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: %s dropped %u events", uav_names[uav].c_str(), s.name.c_str(), events.dropped);
  }

  // without the first time sync the stamp would be raw UE world time
  if (events.stamp <= s.last_stamp || !mrs_lib::get_mutexed(mutex_wall_time_offset_, ue_world_offset_valid_)) {
    return;
  }

  s.last_stamp = events.stamp;

  std_msgs::msg::Header header;
  header.frame_id = s.frame;
  header.stamp    = ueWorldTimeToSimtime(events.stamp);

  s.ph_events.publish(toEventPacket(events, header));
}

//}

/* timerLidar() //{ */

void FlightforgeSimulator::timerLidar(size_t uav, size_t i) {

  if (!is_initialized_) {
    return;
  }

  LidarSensor& s = lidars_[uav][i];

  bool                      res;
  ueds_connector::LidarScan scan;

  {
    std::scoped_lock lock(mutex_flightforge_);
    std::tie(res, scan) = ueds_connectors_[uav]->GetLidarScan(s.params.labels, s.params.intensity, s.id);
  }

  if (!res) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: could not get %s", uav_names[uav].c_str(), s.name.c_str());
    return;
  }

  if (scan.stamp == s.last_stamp) {
    return;
  }

  s.last_stamp = scan.stamp;

  std_msgs::msg::Header header;
  header.frame_id = s.frame;
  header.stamp    = flightforgeTimeToSimtime(scan.stamp);

  const auto clouds = toLidarClouds(scan, s.params, header, s.rng);

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

/* timerRangefinder() //{ */

void FlightforgeSimulator::timerRangefinder(size_t uav, size_t i) {

  if (!is_initialized_) {
    return;
  }

  RangefinderSensor& s = rangefinders_[uav][i];

  bool   res;
  double range;

  {
    std::scoped_lock lock(mutex_flightforge_);
    std::tie(res, range) = ueds_connectors_[uav]->GetRangefinderData(s.id);
  }

  if (!res) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *wall_clock_, 1000, "%s: could not get %s", uav_names[uav].c_str(), s.name.c_str());
    return;
  }

  sensor_msgs::msg::Range msg;

  msg.header.stamp    = mrs_lib::get_mutexed(mutex_sim_time_, sim_time_);
  msg.header.frame_id = s.frame;
  msg.radiation_type  = sensor_msgs::msg::Range::INFRARED;
  msg.min_range       = 0.1;
  msg.max_range       = 30.0;
  msg.range           = range / 100.0;

  s.ph_range.publish(msg);
}

//}

}  // namespace mrs_uav_flightforge_simulator

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(mrs_uav_flightforge_simulator::FlightforgeSimulator)
