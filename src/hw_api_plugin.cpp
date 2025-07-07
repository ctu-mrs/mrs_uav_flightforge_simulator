/* includes //{ */

#include <rclcpp/rclcpp.hpp>

#include <mrs_uav_hw_api/api.h>

#include <nav_msgs/msg/odometry.hpp>

#include <mrs_lib/param_loader.h>
#include <mrs_lib/attitude_converter.h>
#include <mrs_lib/mutex.h>
#include <mrs_lib/publisher_handler.h>
#include <mrs_lib/subscriber_handler.h>
#include <mrs_lib/service_client_handler.h>

/* #include <std_msgs/Float64.h> */
/* #include <std_srvs/SetBool.h> */

#include <mrs_lib/gps_conversions.h>

//}

/* typedefs //{ */

#if USE_ROS_TIMER == 1
typedef mrs_lib::ROSTimer TimerType;
#else
typedef mrs_lib::ThreadTimer TimerType;
#endif

//}

namespace mrs_uav_flightforge_hw_api_plugin
{

/* class Api //{ */

class Api : public mrs_uav_hw_api::MrsUavHwApi {

public:
  ~Api(){};

  void initialize(const rclcpp::Node::SharedPtr& node, std::shared_ptr<mrs_uav_hw_api::CommonHandlers_t> common_handlers);

  void destroy();

  rclcpp::Node::SharedPtr  node_;
  rclcpp::Clock::SharedPtr clock_;

  // | ------------------------- params ------------------------- |

  mrs_msgs::msg::HwApiCapabilities _capabilities_;

  bool _feedforward_enabled_;

  double      _utm_x_;
  double      _utm_y_;
  std::string _utm_zone_;
  double      _amsl_;

  std::string _uav_name_;
  std::string _world_frame_name_;
  std::string _body_frame_name_;

  double _input_timeout_;

  // | --------------------- status methods --------------------- |

  mrs_msgs::msg::HwApiStatus       getStatus();
  mrs_msgs::msg::HwApiCapabilities getCapabilities();

  // | --------------------- topic callbacks -------------------- |

  bool callbackActuatorCmd(const mrs_msgs::msg::HwApiActuatorCmd::ConstSharedPtr msg);
  bool callbackControlGroupCmd(const mrs_msgs::msg::HwApiControlGroupCmd::ConstSharedPtr msg);
  bool callbackAttitudeRateCmd(const mrs_msgs::msg::HwApiAttitudeRateCmd::ConstSharedPtr msg);
  bool callbackAttitudeCmd(const mrs_msgs::msg::HwApiAttitudeCmd::ConstSharedPtr msg);
  bool callbackAccelerationHdgRateCmd(const mrs_msgs::msg::HwApiAccelerationHdgRateCmd::ConstSharedPtr msg);
  bool callbackAccelerationHdgCmd(const mrs_msgs::msg::HwApiAccelerationHdgCmd::ConstSharedPtr msg);
  bool callbackVelocityHdgRateCmd(const mrs_msgs::msg::HwApiVelocityHdgRateCmd::ConstSharedPtr msg);
  bool callbackVelocityHdgCmd(const mrs_msgs::msg::HwApiVelocityHdgCmd::ConstSharedPtr msg);
  bool callbackPositionCmd(const mrs_msgs::msg::HwApiPositionCmd::ConstSharedPtr msg);

  void callbackTrackerCmd(const mrs_msgs::msg::TrackerCommand::ConstSharedPtr msg);

  // | -------------------- service callbacks ------------------- |

  std::tuple<bool, std::string> callbackArming(const bool& request);
  std::tuple<bool, std::string> callbackOffboard(void);

private:
  bool is_initialized_ = false;

  std::shared_ptr<mrs_uav_hw_api::CommonHandlers_t> common_handlers_;

  rclcpp::Time last_cmd_time_;
  std::mutex   mutex_last_cmd_time_;

  // | ----------------------- subscribers ---------------------- |

  mrs_lib::SubscriberHandler<nav_msgs::msg::Odometry> sh_odom_;
  mrs_lib::SubscriberHandler<sensor_msgs::msg::Imu>   sh_imu_;
  mrs_lib::SubscriberHandler<sensor_msgs::msg::Range> sh_range_;

  void callbackOdom(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
  void callbackImu(const sensor_msgs::msg::Imu::ConstSharedPtr msg);
  void callbackRangefinder(const sensor_msgs::msg::Range::ConstSharedPtr msg);

  // | ----------------------- publishers ----------------------- |

  mrs_lib::PublisherHandler<mrs_msgs::msg::HwApiActuatorCmd>            ph_actuators_cmd_;
  mrs_lib::PublisherHandler<mrs_msgs::msg::HwApiControlGroupCmd>        ph_control_group_cmd_;
  mrs_lib::PublisherHandler<mrs_msgs::msg::HwApiAttitudeRateCmd>        ph_attitude_rate_cmd_;
  mrs_lib::PublisherHandler<mrs_msgs::msg::HwApiAttitudeCmd>            ph_attitude_cmd_;
  mrs_lib::PublisherHandler<mrs_msgs::msg::HwApiAccelerationHdgRateCmd> ph_acceleration_hdg_rate_cmd_;
  mrs_lib::PublisherHandler<mrs_msgs::msg::HwApiAccelerationHdgCmd>     ph_acceleration_hdg_cmd_;
  mrs_lib::PublisherHandler<mrs_msgs::msg::HwApiVelocityHdgRateCmd>     ph_velocity_hdg_rate_cmd_;
  mrs_lib::PublisherHandler<mrs_msgs::msg::HwApiVelocityHdgCmd>         ph_velocity_hdg_cmd_;
  mrs_lib::PublisherHandler<mrs_msgs::msg::HwApiPositionCmd>            ph_position_cmd_;
  mrs_lib::PublisherHandler<mrs_msgs::msg::TrackerCommand>              ph_tracker_cmd_;

  // | ------------------------- timers ------------------------- |

  std::shared_ptr<TimerType> timer_main_;

  void timerMain();

  // | ------------------------ variables ----------------------- |

  std::atomic<bool> offboard_ = false;
  std::string       mode_;
  std::atomic<bool> armed_     = false;
  std::atomic<bool> connected_ = false;
  std::mutex        mutex_status_;

  // | ------------------------- methods ------------------------ |

  void publishBatteryState(void);

  void publishRC(void);

  void timeoutInputs(void);
};

//}

// --------------------------------------------------------------
// |                   controller's interface                   |
// --------------------------------------------------------------

/* initialize() //{ */

void Api::initialize(const rclcpp::Node::SharedPtr& node, std::shared_ptr<mrs_uav_hw_api::CommonHandlers_t> common_handlers) {

  node_  = node;
  clock_ = node_->get_clock();

  common_handlers_ = common_handlers;

  _capabilities_.api_name = "MrsSimulator";

  _uav_name_         = common_handlers->getUavName();
  _body_frame_name_  = common_handlers->getBodyFrameName();
  _world_frame_name_ = common_handlers->getWorldFrameName();

  last_cmd_time_ = rclcpp::Time(0, 0, clock_->get_clock_type());

  // | ------------------- loading parameters ------------------- |

  mrs_lib::ParamLoader local_param_loader(node_, "MultirotorSimulatorHwApi");

  std::string custom_config_path;

  common_handlers_->main_param_loader->loadParam("custom_config", custom_config_path);

  if (custom_config_path != "") {
    local_param_loader.addYamlFile(custom_config_path);
  }

  std::vector<std::string> config_files;
  common_handlers_->main_param_loader->loadParamReusable("configs", config_files);

  if (!common_handlers_->main_param_loader->loadedSuccessfully()) {
    RCLCPP_ERROR(node_->get_logger(), "Could not load all parameters!");
    rclcpp::shutdown();
    exit(1);
  }

  for (auto config_file : config_files) {
    RCLCPP_INFO(node_->get_logger(), "loading config file '%s'", config_file.c_str());
    local_param_loader.addYamlFile(config_file);
  }

  local_param_loader.loadParam("input_timeout", _input_timeout_);

  local_param_loader.loadParam("gnss/utm_x", _utm_x_);
  local_param_loader.loadParam("gnss/utm_y", _utm_y_);
  local_param_loader.loadParam("gnss/utm_zone", _utm_zone_);
  local_param_loader.loadParam("gnss/amsl", _amsl_);

  local_param_loader.loadParam("input_mode/actuators", (bool&)_capabilities_.accepts_actuator_cmd);
  local_param_loader.loadParam("input_mode/control_group", (bool&)_capabilities_.accepts_control_group_cmd);
  local_param_loader.loadParam("input_mode/attitude_rate", (bool&)_capabilities_.accepts_attitude_rate_cmd);
  local_param_loader.loadParam("input_mode/attitude", (bool&)_capabilities_.accepts_attitude_cmd);
  local_param_loader.loadParam("input_mode/acceleration_hdg_rate", (bool&)_capabilities_.accepts_acceleration_hdg_rate_cmd);
  local_param_loader.loadParam("input_mode/acceleration_hdg", (bool&)_capabilities_.accepts_acceleration_hdg_cmd);
  local_param_loader.loadParam("input_mode/velocity_hdg_rate", (bool&)_capabilities_.accepts_velocity_hdg_rate_cmd);
  local_param_loader.loadParam("input_mode/velocity_hdg", (bool&)_capabilities_.accepts_velocity_hdg_cmd);
  local_param_loader.loadParam("input_mode/position", (bool&)_capabilities_.accepts_position_cmd);
  local_param_loader.loadParam("input_mode/feedforward", _feedforward_enabled_);

  local_param_loader.loadParam("outputs/distance_sensor", (bool&)_capabilities_.produces_distance_sensor);
  local_param_loader.loadParam("outputs/gnss", (bool&)_capabilities_.produces_gnss);
  local_param_loader.loadParam("outputs/rtk", (bool&)_capabilities_.produces_rtk);
  local_param_loader.loadParam("outputs/imu", (bool&)_capabilities_.produces_imu);
  local_param_loader.loadParam("outputs/altitude", (bool&)_capabilities_.produces_altitude);
  local_param_loader.loadParam("outputs/magnetometer_heading", (bool&)_capabilities_.produces_magnetometer_heading);
  local_param_loader.loadParam("outputs/rc_channels", (bool&)_capabilities_.produces_rc_channels);
  local_param_loader.loadParam("outputs/battery_state", (bool&)_capabilities_.produces_battery_state);
  local_param_loader.loadParam("outputs/position", (bool&)_capabilities_.produces_position);
  local_param_loader.loadParam("outputs/orientation", (bool&)_capabilities_.produces_orientation);
  local_param_loader.loadParam("outputs/velocity", (bool&)_capabilities_.produces_velocity);
  local_param_loader.loadParam("outputs/angular_velocity", (bool&)_capabilities_.produces_angular_velocity);
  local_param_loader.loadParam("outputs/odometry", (bool&)_capabilities_.produces_odometry);
  local_param_loader.loadParam("outputs/ground_truth", (bool&)_capabilities_.produces_ground_truth);

  _capabilities_.produces_magnetic_field = false;

  if (!local_param_loader.loadedSuccessfully()) {
    RCLCPP_ERROR(node_->get_logger(), "Could not load all parameters!");
    rclcpp::shutdown();
  }

  // | ----------------------- subscribers ---------------------- |

  mrs_lib::SubscriberHandlerOptions shopts;

  shopts.node               = node_;
  shopts.node_name          = "MultirotorSimulatorHwApi";
  shopts.no_message_timeout = mrs_lib::no_timeout;
  shopts.threadsafe         = true;
  shopts.autostart          = true;

  sh_odom_ = mrs_lib::SubscriberHandler<nav_msgs::msg::Odometry>(shopts, "~/simulator_odom_in", &Api::callbackOdom, this);

  sh_imu_ = mrs_lib::SubscriberHandler<sensor_msgs::msg::Imu>(shopts, "~/simulator_imu_in", &Api::callbackImu, this);

  sh_range_ = mrs_lib::SubscriberHandler<sensor_msgs::msg::Range>(shopts, "~/simulator_rangefinder_in", &Api::callbackRangefinder, this);

  // | ----------------------- publishers ----------------------- |

  if (_capabilities_.accepts_actuator_cmd) {
    ph_actuators_cmd_ = mrs_lib::PublisherHandler<mrs_msgs::msg::HwApiActuatorCmd>(node_, "~/simulator_actuators_cmd_out");
  }

  if (_capabilities_.accepts_control_group_cmd) {
    ph_control_group_cmd_ = mrs_lib::PublisherHandler<mrs_msgs::msg::HwApiControlGroupCmd>(node_, "~/simulator_control_group_cmd_out");
  }

  if (_capabilities_.accepts_attitude_rate_cmd) {
    ph_attitude_rate_cmd_ = mrs_lib::PublisherHandler<mrs_msgs::msg::HwApiAttitudeRateCmd>(node_, "~/simulator_attitude_rate_cmd_out");
  }

  if (_capabilities_.accepts_attitude_cmd) {
    ph_attitude_cmd_ = mrs_lib::PublisherHandler<mrs_msgs::msg::HwApiAttitudeCmd>(node_, "~/simulator_attitude_cmd_out");
  }

  if (_capabilities_.accepts_acceleration_hdg_rate_cmd) {
    ph_acceleration_hdg_rate_cmd_ = mrs_lib::PublisherHandler<mrs_msgs::msg::HwApiAccelerationHdgRateCmd>(node_, "~/simulator_acceleration_hdg_rate_cmd_out");
  }

  if (_capabilities_.accepts_acceleration_hdg_cmd) {
    ph_acceleration_hdg_cmd_ = mrs_lib::PublisherHandler<mrs_msgs::msg::HwApiAccelerationHdgCmd>(node_, "~/simulator_acceleration_hdg_cmd_out");
  }

  if (_capabilities_.accepts_velocity_hdg_rate_cmd) {
    ph_velocity_hdg_rate_cmd_ = mrs_lib::PublisherHandler<mrs_msgs::msg::HwApiVelocityHdgRateCmd>(node_, "~/simulator_velocity_hdg_rate_cmd_out");
  }

  if (_capabilities_.accepts_velocity_hdg_cmd) {
    ph_velocity_hdg_cmd_ = mrs_lib::PublisherHandler<mrs_msgs::msg::HwApiVelocityHdgCmd>(node_, "~/simulator_velocity_hdg_cmd_out");
  }

  if (_capabilities_.accepts_position_cmd) {
    ph_position_cmd_ = mrs_lib::PublisherHandler<mrs_msgs::msg::HwApiPositionCmd>(node_, "~/simulator_position_cmd_out");
  }

  if (_feedforward_enabled_) {
    ph_tracker_cmd_ = mrs_lib::PublisherHandler<mrs_msgs::msg::TrackerCommand>(node_, "~/simulator_tracker_cmd_out");
  }

  // | ------------------------- timers ------------------------- |

  {
    std::function<void()> callback_fcn = std::bind(&Api::timerMain, this);

    mrs_lib::TimerHandlerOptions opts;

    opts.node      = node_;
    opts.autostart = true;

    timer_main_ = std::make_shared<TimerType>(opts, rclcpp::Rate(10.0, clock_), callback_fcn);
  }

  // | ----------------------- finish init ---------------------- |

  RCLCPP_INFO(node_->get_logger(), "initialized");

  is_initialized_ = true;
}

//}

/* destroy() //{ */

void Api::destroy() {

  timer_main_->stop();
}

//}

/* getStatus() //{ */

mrs_msgs::msg::HwApiStatus Api::getStatus() {

  mrs_msgs::msg::HwApiStatus status;

  status.stamp = clock_->now();

  bool has_odom = sh_odom_.hasMsg();

  {
    std::scoped_lock lock(mutex_status_);

    status.armed     = armed_;
    status.offboard  = offboard_;
    status.connected = has_odom;
    status.mode      = mode_;
  }

  return status;
}

//}

/* getCapabilities() //{ */

mrs_msgs::msg::HwApiCapabilities Api::getCapabilities() {

  _capabilities_.stamp = clock_->now();

  return _capabilities_;
}

//}

/* callbackArming() //{ */

std::tuple<bool, std::string> Api::callbackArming([[maybe_unused]] const bool& request) {

  std::stringstream ss;

  if (request) {

    armed_ = true;

    ss << "armed";
    RCLCPP_INFO_STREAM(node_->get_logger(), "" << ss.str());
    return std::tuple(true, ss.str());

  } else {

    armed_ = false;

    ss << "disarmed";
    RCLCPP_INFO_STREAM(node_->get_logger(), "" << ss.str());
    return std::tuple(true, ss.str());
  }
}

//}

/* callbackOffboard() //{ */

std::tuple<bool, std::string> Api::callbackOffboard(void) {

  std::stringstream ss;

  if (!armed_) {
    ss << "Cannot switch to offboard, not armed.";
    RCLCPP_INFO(node_->get_logger(), "%s", ss.str().c_str());
    return {false, ss.str()};
  }

  auto last_cmd_time = mrs_lib::get_mutexed(mutex_last_cmd_time_, last_cmd_time_);

  if ((clock_->now() - last_cmd_time).seconds() > _input_timeout_) {
    ss << "Cannot switch to offboard, missing control input.";
    RCLCPP_INFO(node_->get_logger(), "%s", ss.str().c_str());
    return {false, ss.str()};
  }

  offboard_ = true;

  ss << "Offboard set";
  RCLCPP_INFO(node_->get_logger(), "%s", ss.str().c_str());
  return {true, ss.str()};
}

//}

// | --------------------- input callbacks -------------------- |

/* callbackActuatorCmd() //{ */

bool Api::callbackActuatorCmd(const mrs_msgs::msg::HwApiActuatorCmd::ConstSharedPtr msg) {

  if (!_capabilities_.accepts_actuator_cmd) {
    return false;
  }

  RCLCPP_INFO_ONCE(node_->get_logger(), "getting actuators cmd");

  if (offboard_) {
    ph_actuators_cmd_.publish(*msg);
  }

  {
    std::scoped_lock lock(mutex_last_cmd_time_);

    last_cmd_time_ = clock_->now();
  }

  return true;
}

//}

/* callbackControlGroupCmd() //{ */

bool Api::callbackControlGroupCmd(const mrs_msgs::msg::HwApiControlGroupCmd::ConstSharedPtr msg) {

  if (!_capabilities_.accepts_control_group_cmd) {
    return false;
  }

  RCLCPP_INFO_ONCE(node_->get_logger(), "getting control group cmd");

  if (offboard_) {
    ph_control_group_cmd_.publish(*msg);
  }

  {
    std::scoped_lock lock(mutex_last_cmd_time_);

    last_cmd_time_ = clock_->now();
  }

  return true;
}

//}

/* callbackAttitudeRateCmd() //{ */

bool Api::callbackAttitudeRateCmd(const mrs_msgs::msg::HwApiAttitudeRateCmd::ConstSharedPtr msg) {

  if (!_capabilities_.accepts_attitude_rate_cmd) {
    return false;
  }

  RCLCPP_INFO_ONCE(node_->get_logger(), "getting attitude rate cmd");

  if (offboard_) {
    ph_attitude_rate_cmd_.publish(*msg);
  }

  {
    std::scoped_lock lock(mutex_last_cmd_time_);

    last_cmd_time_ = clock_->now();
  }

  return true;
}

//}

/* callbackAttitudeCmd() //{ */

bool Api::callbackAttitudeCmd(const mrs_msgs::msg::HwApiAttitudeCmd::ConstSharedPtr msg) {

  if (!_capabilities_.accepts_attitude_cmd) {
    return false;
  }

  RCLCPP_INFO_ONCE(node_->get_logger(), "getting attitude cmd");

  if (offboard_) {
    ph_attitude_cmd_.publish(*msg);
  }

  {
    std::scoped_lock lock(mutex_last_cmd_time_);

    last_cmd_time_ = clock_->now();
  }

  return true;
}

//}

/* callbackAccelerationHdgRateCmd() //{ */

bool Api::callbackAccelerationHdgRateCmd(const mrs_msgs::msg::HwApiAccelerationHdgRateCmd::ConstSharedPtr msg) {

  if (!_capabilities_.accepts_acceleration_hdg_rate_cmd) {
    return false;
  }

  RCLCPP_INFO_ONCE(node_->get_logger(), "getting acceleration+hdg rate cmd");

  if (offboard_) {
    ph_acceleration_hdg_rate_cmd_.publish(*msg);
  }

  {
    std::scoped_lock lock(mutex_last_cmd_time_);

    last_cmd_time_ = clock_->now();
  }

  return true;
}

//}

/* callbackAccelerationHdgCmd() //{ */

bool Api::callbackAccelerationHdgCmd(const mrs_msgs::msg::HwApiAccelerationHdgCmd::ConstSharedPtr msg) {

  if (!_capabilities_.accepts_acceleration_hdg_cmd) {
    return false;
  }

  RCLCPP_INFO_ONCE(node_->get_logger(), "getting acceleration+hdg cmd");

  if (offboard_) {
    ph_acceleration_hdg_cmd_.publish(*msg);
  }

  {
    std::scoped_lock lock(mutex_last_cmd_time_);

    last_cmd_time_ = clock_->now();
  }

  return true;
}

//}

/* callbackVelocityHdgRateCmd() //{ */

bool Api::callbackVelocityHdgRateCmd(const mrs_msgs::msg::HwApiVelocityHdgRateCmd::ConstSharedPtr msg) {

  if (!_capabilities_.accepts_velocity_hdg_rate_cmd) {
    return false;
  }

  RCLCPP_INFO_ONCE(node_->get_logger(), "getting velocity+hdg rate cmd");

  if (offboard_) {
    ph_velocity_hdg_rate_cmd_.publish(*msg);
  }

  {
    std::scoped_lock lock(mutex_last_cmd_time_);

    last_cmd_time_ = clock_->now();
  }

  return true;
}

//}

/* callbackVelocityHdgCmd() //{ */

bool Api::callbackVelocityHdgCmd(const mrs_msgs::msg::HwApiVelocityHdgCmd::ConstSharedPtr msg) {

  if (!_capabilities_.accepts_velocity_hdg_cmd) {
    return false;
  }

  RCLCPP_INFO_ONCE(node_->get_logger(), "getting velocity+hdg cmd");

  if (offboard_) {
    ph_velocity_hdg_cmd_.publish(*msg);
  }

  {
    std::scoped_lock lock(mutex_last_cmd_time_);

    last_cmd_time_ = clock_->now();
  }

  return true;
}

//}

/* callbackPositionCmd() //{ */

bool Api::callbackPositionCmd(const mrs_msgs::msg::HwApiPositionCmd::ConstSharedPtr msg) {

  if (!_capabilities_.accepts_position_cmd) {
    return false;
  }

  RCLCPP_INFO_ONCE(node_->get_logger(), "getting position cmd");

  if (offboard_) {
    ph_position_cmd_.publish(*msg);
  }

  {
    std::scoped_lock lock(mutex_last_cmd_time_);

    last_cmd_time_ = clock_->now();
  }

  return true;
}

//}

/* callbackTrackerCmd() //{ */

void Api::callbackTrackerCmd(const mrs_msgs::msg::TrackerCommand::ConstSharedPtr msg) {

  RCLCPP_INFO_ONCE(node_->get_logger(), "getting tracker cmd");

  if (offboard_) {
    ph_tracker_cmd_.publish(*msg);
  }
}

//}

// | ------------------------ callbacks ----------------------- |

/* //{ callbackOdom() */

void Api::callbackOdom(const nav_msgs::msg::Odometry::ConstSharedPtr msg) {

  if (!is_initialized_) {
    return;
  }

  RCLCPP_INFO_ONCE(node_->get_logger(), "getting simulator odometry");

  auto odom = msg;

  {
    std::scoped_lock lock(mutex_status_);

    connected_ = true;
  }

  // | ----------------- publish the diagnostics ---------------- |

  mrs_msgs::msg::HwApiStatus status;

  {
    std::scoped_lock lock(mutex_status_);

    status.stamp     = clock_->now();
    status.armed     = armed_;
    status.offboard  = offboard_;
    status.connected = connected_;
    status.mode      = mode_;
  }

  common_handlers_->publishers.publishStatus(status);

  // | -------------------- publish position -------------------- |

  if (_capabilities_.produces_position) {

    geometry_msgs::msg::PointStamped position;

    position.header.stamp    = odom->header.stamp;
    position.header.frame_id = _uav_name_ + "/" + _world_frame_name_;
    position.point           = odom->pose.pose.position;

    common_handlers_->publishers.publishPosition(position);
  }

  // | ------------------- publish orientation ------------------ |

  if (_capabilities_.produces_orientation) {

    geometry_msgs::msg::QuaternionStamped orientation;

    orientation.header.stamp    = odom->header.stamp;
    orientation.header.frame_id = _uav_name_ + "/" + _world_frame_name_;
    orientation.quaternion      = odom->pose.pose.orientation;

    common_handlers_->publishers.publishOrientation(orientation);
  }

  // | -------------------- publish velocity -------------------- |

  if (_capabilities_.produces_velocity) {

    geometry_msgs::msg::Vector3Stamped velocity;

    velocity.header.stamp    = odom->header.stamp;
    velocity.header.frame_id = _uav_name_ + "/" + _body_frame_name_;
    velocity.vector          = odom->twist.twist.linear;

    common_handlers_->publishers.publishVelocity(velocity);
  }

  // | ---------------- publish angular velocity ---------------- |

  if (_capabilities_.produces_angular_velocity) {

    geometry_msgs::msg::Vector3Stamped angular_velocity;

    angular_velocity.header.stamp    = odom->header.stamp;
    angular_velocity.header.frame_id = _uav_name_ + "/" + _body_frame_name_;
    angular_velocity.vector          = odom->twist.twist.angular;

    common_handlers_->publishers.publishAngularVelocity(angular_velocity);
  }

  // | -------------------- publish odometry -------------------- |

  if (_capabilities_.produces_odometry) {
    common_handlers_->publishers.publishOdometry(*odom);
  }

  // | ------------------ publish ground truth ------------------ |

  if (_capabilities_.produces_ground_truth) {
    common_handlers_->publishers.publishGroundTruth(*odom);
  }

  // | ---------------------- publish gnss ---------------------- |

  if (_capabilities_.produces_gnss) {

    double lat;
    double lon;

    mrs_lib::UTMtoLL(odom->pose.pose.position.y + _utm_y_, odom->pose.pose.position.x + _utm_x_, _utm_zone_, lat, lon);

    sensor_msgs::msg::NavSatFix gnss;

    gnss.header.stamp    = odom->header.stamp;
    gnss.header.frame_id = _uav_name_ + "/" + _body_frame_name_;

    gnss.latitude  = lat;
    gnss.longitude = lon;
    gnss.altitude  = odom->pose.pose.position.z + _amsl_;

    common_handlers_->publishers.publishGNSS(gnss);
  }

  // | ----------------------- publish rtk ---------------------- |

  if (_capabilities_.produces_rtk) {

    double lat;
    double lon;

    mrs_lib::UTMtoLL(odom->pose.pose.position.y + _utm_y_, odom->pose.pose.position.x + _utm_x_, _utm_zone_, lat, lon);

    mrs_msgs::msg::RtkGps rtk;

    rtk.header.stamp = odom->header.stamp;

    rtk.gps.latitude  = lat;
    rtk.gps.longitude = lon;
    rtk.gps.altitude  = odom->pose.pose.position.z + _amsl_;

    rtk.fix_type.fix_type = mrs_msgs::msg::RtkFixType::RTK_FIX;

    common_handlers_->publishers.publishRTK(rtk);
  }

  // | ------------------ publish amsl altitude ----------------- |

  if (_capabilities_.produces_altitude) {

    mrs_msgs::msg::HwApiAltitude altitude;

    altitude.stamp = odom->header.stamp;

    altitude.amsl = odom->pose.pose.position.z + _amsl_;

    common_handlers_->publishers.publishAltitude(altitude);
  }

  // | --------------------- publish heading -------------------- |

  if (_capabilities_.produces_magnetometer_heading) {

    double heading = mrs_lib::AttitudeConverter(odom->pose.pose.orientation).getHeading();

    mrs_msgs::msg::Float64Stamped hdg;

    hdg.header.stamp = clock_->now();
    hdg.value        = heading;

    common_handlers_->publishers.publishMagnetometerHeading(hdg);
  }
}

//}

/* callbackImu() //{ */

void Api::callbackImu(const sensor_msgs::msg::Imu::ConstSharedPtr msg) {

  if (!is_initialized_) {
    return;
  }

  RCLCPP_INFO_ONCE(node_->get_logger(), "getting IMU");

  if (_capabilities_.produces_imu) {

    common_handlers_->publishers.publishIMU(*msg);
  }
}

//}

/* callbackRangefinder() //{ */

void Api::callbackRangefinder(const sensor_msgs::msg::Range::ConstSharedPtr msg) {

  if (!is_initialized_) {
    return;
  }

  RCLCPP_INFO_ONCE(node_->get_logger(), "getting rangefinder");

  if (_capabilities_.produces_distance_sensor) {

    common_handlers_->publishers.publishDistanceSensor(*msg);
  }
}

//}

// | ------------------------- timers ------------------------- |

/* timerMain() //{ */

void Api::timerMain() {

  if (!is_initialized_) {
    return;
  }

  RCLCPP_INFO_ONCE(node_->get_logger(), "main timer spinning");

  publishBatteryState();

  publishRC();

  timeoutInputs();
}

//}

// | ------------------------- methods ------------------------ |

/* publishBatteryState() //{ */

void Api::publishBatteryState(void) {

  if (_capabilities_.produces_battery_state) {

    sensor_msgs::msg::BatteryState msg;

    msg.capacity = 100;
    msg.current  = 10.0;
    msg.voltage  = 15.8;
    msg.charge   = 0.8;

    common_handlers_->publishers.publishBatteryState(msg);
  }
}

//}

/* publishRC() //{ */

void Api::publishRC(void) {

  if (_capabilities_.produces_rc_channels) {

    mrs_msgs::msg::HwApiRcChannels rc;

    rc.stamp = clock_->now();

    rc.channels.push_back(0);
    rc.channels.push_back(0);
    rc.channels.push_back(0);
    rc.channels.push_back(0);
    rc.channels.push_back(0);
    rc.channels.push_back(0);
    rc.channels.push_back(0);
    rc.channels.push_back(0);

    common_handlers_->publishers.publishRcChannels(rc);
  }
}

//}

/* MrsUavHwApi() //{ */

void Api::timeoutInputs(void) {

  auto last_cmd_time = mrs_lib::get_mutexed(mutex_last_cmd_time_, last_cmd_time_);

  if (last_cmd_time_.seconds() > 0 && (clock_->now() - last_cmd_time).seconds() > _input_timeout_) {
    offboard_ = false;
  }
}

//}

}  // namespace mrs_uav_flightforge_hw_api_plugin

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(mrs_uav_flightforge_hw_api_plugin::Api, mrs_uav_hw_api::MrsUavHwApi)
