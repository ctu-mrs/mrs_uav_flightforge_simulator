/* includes //{ */

#include <ros/ros.h>

#include <mrs_uav_hw_api/api.h>

#include <nav_msgs/Odometry.h>

#include <mrs_lib/param_loader.h>
#include <mrs_lib/attitude_converter.h>
#include <mrs_lib/mutex.h>
#include <mrs_lib/publisher_handler.h>
#include <mrs_lib/subscribe_handler.h>
#include <mrs_lib/service_client_handler.h>

#include <std_msgs/Float64.h>
#include <std_srvs/SetBool.h>

#include <mrs_lib/gps_conversions.h>

//}

namespace mrs_uav_unreal_simulation
{

/* class Api //{ */

class Api : public mrs_uav_hw_api::MrsUavHwApi {

public:
  ~Api(){};

  void initialize(const ros::NodeHandle& parent_nh, std::shared_ptr<mrs_uav_hw_api::CommonHandlers_t> common_handlers);

  // | ------------------------- params ------------------------- |

  mrs_msgs::HwApiCapabilities _capabilities_;

  bool _feedforward_enabled_;

  double      _utm_x_;
  double      _utm_y_;
  std::string _utm_zone_;
  double      _amsl_;

  std::string _uav_name_;
  std::string _world_frame_name_;
  std::string _body_frame_name_;

  // | --------------------- status methods --------------------- |

  mrs_msgs::HwApiStatus       getStatus();
  mrs_msgs::HwApiCapabilities getCapabilities();

  // | --------------------- topic callbacks -------------------- |

  bool callbackActuatorCmd(const mrs_msgs::HwApiActuatorCmd::ConstPtr msg);
  bool callbackControlGroupCmd(const mrs_msgs::HwApiControlGroupCmd::ConstPtr msg);
  bool callbackAttitudeRateCmd(const mrs_msgs::HwApiAttitudeRateCmd::ConstPtr msg);
  bool callbackAttitudeCmd(const mrs_msgs::HwApiAttitudeCmd::ConstPtr msg);
  bool callbackAccelerationHdgRateCmd(const mrs_msgs::HwApiAccelerationHdgRateCmd::ConstPtr msg);
  bool callbackAccelerationHdgCmd(const mrs_msgs::HwApiAccelerationHdgCmd::ConstPtr msg);
  bool callbackVelocityHdgRateCmd(const mrs_msgs::HwApiVelocityHdgRateCmd::ConstPtr msg);
  bool callbackVelocityHdgCmd(const mrs_msgs::HwApiVelocityHdgCmd::ConstPtr msg);
  bool callbackPositionCmd(const mrs_msgs::HwApiPositionCmd::ConstPtr msg);

  void callbackTrackerCmd(const mrs_msgs::TrackerCommand::ConstPtr msg);

  // | -------------------- service callbacks ------------------- |

  std::tuple<bool, std::string> callbackArming(const bool& request);
  std::tuple<bool, std::string> callbackOffboard(void);

private:
  bool is_initialized_ = false;

  std::shared_ptr<mrs_uav_hw_api::CommonHandlers_t> common_handlers_;

  // | ----------------------- subscribers ---------------------- |

  mrs_lib::SubscribeHandler<nav_msgs::Odometry> sh_odom_;
  mrs_lib::SubscribeHandler<sensor_msgs::Imu>   sh_imu_;
  mrs_lib::SubscribeHandler<sensor_msgs::Range> sh_range_;
  mrs_lib::SubscribeHandler<sensor_msgs::Range> sh_distance_sensor_;

  void callbackOdom(const nav_msgs::Odometry::ConstPtr msg);
  void callbackImu(const sensor_msgs::Imu::ConstPtr msg);
  void callbackDistanceSensor(const sensor_msgs::Range::ConstPtr msg);

  // | ----------------------- publishers ----------------------- |

  mrs_lib::PublisherHandler<mrs_msgs::HwApiActuatorCmd>            ph_actuators_cmd_;
  mrs_lib::PublisherHandler<mrs_msgs::HwApiControlGroupCmd>        ph_control_group_cmd_;
  mrs_lib::PublisherHandler<mrs_msgs::HwApiAttitudeRateCmd>        ph_attitude_rate_cmd_;
  mrs_lib::PublisherHandler<mrs_msgs::HwApiAttitudeCmd>            ph_attitude_cmd_;
  mrs_lib::PublisherHandler<mrs_msgs::HwApiAccelerationHdgRateCmd> ph_acceleration_hdg_rate_cmd_;
  mrs_lib::PublisherHandler<mrs_msgs::HwApiAccelerationHdgCmd>     ph_acceleration_hdg_cmd_;
  mrs_lib::PublisherHandler<mrs_msgs::HwApiVelocityHdgRateCmd>     ph_velocity_hdg_rate_cmd_;
  mrs_lib::PublisherHandler<mrs_msgs::HwApiVelocityHdgCmd>         ph_velocity_hdg_cmd_;
  mrs_lib::PublisherHandler<mrs_msgs::HwApiPositionCmd>            ph_position_cmd_;
  mrs_lib::PublisherHandler<mrs_msgs::TrackerCommand>              ph_tracker_cmd_;


  mrs_lib::PublisherHandler<sensor_msgs::Range> ph_range_;
  // | ------------------------- timers ------------------------- |

  ros::Timer timer_main_;

  void timerMain(const ros::TimerEvent& event);

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

void Api::initialize(const ros::NodeHandle& parent_nh, std::shared_ptr<mrs_uav_hw_api::CommonHandlers_t> common_handlers) {

  ros::NodeHandle nh_(parent_nh);

  common_handlers_ = common_handlers;

  _capabilities_.api_name = "MrsSimulator";

  _uav_name_         = common_handlers->getUavName();
  _body_frame_name_  = common_handlers->getBodyFrameName();
  _world_frame_name_ = common_handlers->getWorldFrameName();

  // | ------------------- loading parameters ------------------- |

  mrs_lib::ParamLoader param_loader(nh_, "MrsUavHwApi");

  param_loader.loadParam("gnss/utm_x", _utm_x_);
  param_loader.loadParam("gnss/utm_y", _utm_y_);
  param_loader.loadParam("gnss/utm_zone", _utm_zone_);
  param_loader.loadParam("gnss/amsl", _amsl_);

  param_loader.loadParam("input_mode/actuators", (bool&)_capabilities_.accepts_actuator_cmd);
  param_loader.loadParam("input_mode/control_group", (bool&)_capabilities_.accepts_control_group_cmd);
  param_loader.loadParam("input_mode/attitude_rate", (bool&)_capabilities_.accepts_attitude_rate_cmd);
  param_loader.loadParam("input_mode/attitude", (bool&)_capabilities_.accepts_attitude_cmd);
  param_loader.loadParam("input_mode/acceleration_hdg_rate", (bool&)_capabilities_.accepts_acceleration_hdg_rate_cmd);
  param_loader.loadParam("input_mode/acceleration_hdg", (bool&)_capabilities_.accepts_acceleration_hdg_cmd);
  param_loader.loadParam("input_mode/velocity_hdg_rate", (bool&)_capabilities_.accepts_velocity_hdg_rate_cmd);
  param_loader.loadParam("input_mode/velocity_hdg", (bool&)_capabilities_.accepts_velocity_hdg_cmd);
  param_loader.loadParam("input_mode/position", (bool&)_capabilities_.accepts_position_cmd);
  param_loader.loadParam("input_mode/feedforward", _feedforward_enabled_);

  param_loader.loadParam("outputs/gnss", (bool&)_capabilities_.produces_gnss);
  param_loader.loadParam("outputs/rtk", (bool&)_capabilities_.produces_rtk);
  param_loader.loadParam("outputs/imu", (bool&)_capabilities_.produces_imu);
  param_loader.loadParam("outputs/altitude", (bool&)_capabilities_.produces_altitude);
  param_loader.loadParam("outputs/magnetometer_heading", (bool&)_capabilities_.produces_magnetometer_heading);
  param_loader.loadParam("outputs/rc_channels", (bool&)_capabilities_.produces_rc_channels);
  param_loader.loadParam("outputs/battery_state", (bool&)_capabilities_.produces_battery_state);
  param_loader.loadParam("outputs/position", (bool&)_capabilities_.produces_position);
  param_loader.loadParam("outputs/orientation", (bool&)_capabilities_.produces_orientation);
  param_loader.loadParam("outputs/velocity", (bool&)_capabilities_.produces_velocity);
  param_loader.loadParam("outputs/angular_velocity", (bool&)_capabilities_.produces_angular_velocity);
  param_loader.loadParam("outputs/odometry", (bool&)_capabilities_.produces_odometry);
  param_loader.loadParam("outputs/ground_truth", (bool&)_capabilities_.produces_ground_truth);

  _capabilities_.produces_magnetic_field = false;
  _capabilities_.produces_distance_sensor = true;

  if (!param_loader.loadedSuccessfully()) {
    ROS_ERROR("[MrsUavHwDummyApi]: Could not load all parameters!");
    ros::shutdown();
  }

  // | ----------------------- subscribers ---------------------- |

  mrs_lib::SubscribeHandlerOptions shopts;
  shopts.nh                 = nh_;
  shopts.node_name          = "MrsSimulatorHwApi";
  shopts.no_message_timeout = mrs_lib::no_timeout;
  shopts.threadsafe         = true;
  shopts.autostart          = true;
  shopts.queue_size         = 10;
  shopts.transport_hints    = ros::TransportHints().tcpNoDelay();

  sh_odom_ = mrs_lib::SubscribeHandler<nav_msgs::Odometry>(shopts, "simulator_odom_in", &Api::callbackOdom, this);

  sh_imu_ = mrs_lib::SubscribeHandler<sensor_msgs::Imu>(shopts, "simulator_imu_in", &Api::callbackImu, this);

  sh_distance_sensor_ = mrs_lib::SubscribeHandler<sensor_msgs::Range>(shopts, "simulator_rangefinder_in", &Api::callbackDistanceSensor, this);

  // | ----------------------- publishers ----------------------- |
ph_range_ = mrs_lib::PublisherHandler<sensor_msgs::Range>(nh_, "mrdka_out", 10);
  if (_capabilities_.accepts_actuator_cmd) {
    ph_actuators_cmd_ = mrs_lib::PublisherHandler<mrs_msgs::HwApiActuatorCmd>(nh_, "simulator_actuators_cmd_out", 1);
  }

  if (_capabilities_.accepts_control_group_cmd) {
    ph_control_group_cmd_ = mrs_lib::PublisherHandler<mrs_msgs::HwApiControlGroupCmd>(nh_, "simulator_control_group_cmd_out", 1);
  }

  if (_capabilities_.accepts_attitude_rate_cmd) {
    ph_attitude_rate_cmd_ = mrs_lib::PublisherHandler<mrs_msgs::HwApiAttitudeRateCmd>(nh_, "simulator_attitude_rate_cmd_out", 1);
  }

  if (_capabilities_.accepts_attitude_cmd) {
    ph_attitude_cmd_ = mrs_lib::PublisherHandler<mrs_msgs::HwApiAttitudeCmd>(nh_, "simulator_attitude_cmd_out", 1);
  }

  if (_capabilities_.accepts_acceleration_hdg_rate_cmd) {
    ph_acceleration_hdg_rate_cmd_ = mrs_lib::PublisherHandler<mrs_msgs::HwApiAccelerationHdgRateCmd>(nh_, "simulator_acceleration_hdg_rate_cmd_out", 1);
  }

  if (_capabilities_.accepts_acceleration_hdg_cmd) {
    ph_acceleration_hdg_cmd_ = mrs_lib::PublisherHandler<mrs_msgs::HwApiAccelerationHdgCmd>(nh_, "simulator_acceleration_hdg_cmd_out", 1);
  }

  if (_capabilities_.accepts_velocity_hdg_rate_cmd) {
    ph_velocity_hdg_rate_cmd_ = mrs_lib::PublisherHandler<mrs_msgs::HwApiVelocityHdgRateCmd>(nh_, "simulator_velocity_hdg_rate_cmd_out", 1);
  }

  if (_capabilities_.accepts_velocity_hdg_cmd) {
    ph_velocity_hdg_cmd_ = mrs_lib::PublisherHandler<mrs_msgs::HwApiVelocityHdgCmd>(nh_, "simulator_velocity_hdg_cmd_out", 1);
  }

  if (_capabilities_.accepts_position_cmd) {
    ph_position_cmd_ = mrs_lib::PublisherHandler<mrs_msgs::HwApiPositionCmd>(nh_, "simulator_position_cmd_out", 1);
  }

  if (_feedforward_enabled_) {
    ph_tracker_cmd_ = mrs_lib::PublisherHandler<mrs_msgs::TrackerCommand>(nh_, "simulator_tracker_cmd_out", 1);
  }

  // | ------------------------- timers ------------------------- |

  timer_main_ = nh_.createTimer(ros::Rate(10.0), &Api::timerMain, this);

  // | ----------------------- finish init ---------------------- |

  ROS_INFO("[MrsUavHwDummyApi]: initialized");

  is_initialized_ = true;
}

//}

/* getStatus() //{ */

mrs_msgs::HwApiStatus Api::getStatus() {

  mrs_msgs::HwApiStatus status;

  status.stamp = ros::Time::now();

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

mrs_msgs::HwApiCapabilities Api::getCapabilities() {

  _capabilities_.stamp = ros::Time::now();

  return _capabilities_;
}

//}

/* callbackArming() //{ */

std::tuple<bool, std::string> Api::callbackArming([[maybe_unused]] const bool& request) {

  std::stringstream ss;

  if (request) {

    armed_ = true;

    ss << "armed";
    ROS_INFO_STREAM_THROTTLE(1.0, "[MrsSimulatorHwApi]: " << ss.str());
    return std::tuple(true, ss.str());

  } else {

    armed_ = false;

    ss << "disarmed";
    ROS_INFO_STREAM_THROTTLE(1.0, "[MrsSimulatorHwApi]: " << ss.str());
    return std::tuple(true, ss.str());
  }
}

//}

/* callbackOffboard() //{ */

std::tuple<bool, std::string> Api::callbackOffboard(void) {

  std::stringstream ss;

  if (!armed_) {
    ss << "Cannot switch to offboard, not armed.";
    ROS_INFO_THROTTLE(1.0, "[MrsSimulatorHwApi]: %s", ss.str().c_str());
    return {false, ss.str()};
  }

  offboard_ = true;

  ss << "Offboard set";
  ROS_INFO_THROTTLE(1.0, "[MrsSimulatorHwApi]: %s", ss.str().c_str());
  return {true, ss.str()};
}

//}

// | --------------------- input callbacks -------------------- |

/* callbackActuatorCmd() //{ */

bool Api::callbackActuatorCmd(const mrs_msgs::HwApiActuatorCmd::ConstPtr msg) {

  if (!_capabilities_.accepts_actuator_cmd) {
    return false;
  }

  ROS_INFO_ONCE("[Api]: getting actuators cmd");

  ph_actuators_cmd_.publish(msg);

  return true;
}

//}

/* callbackControlGroupCmd() //{ */

bool Api::callbackControlGroupCmd(const mrs_msgs::HwApiControlGroupCmd::ConstPtr msg) {

  if (!_capabilities_.accepts_control_group_cmd) {
    return false;
  }

  ROS_INFO_ONCE("[Api]: getting control group cmd");

  ph_control_group_cmd_.publish(msg);

  return true;
}

//}

/* callbackAttitudeRateCmd() //{ */

bool Api::callbackAttitudeRateCmd(const mrs_msgs::HwApiAttitudeRateCmd::ConstPtr msg) {

  if (!_capabilities_.accepts_attitude_rate_cmd) {
    return false;
  }

  ROS_INFO_ONCE("[Api]: getting attitude rate cmd");

  ph_attitude_rate_cmd_.publish(msg);

  return true;
}

//}

/* callbackAttitudeCmd() //{ */

bool Api::callbackAttitudeCmd(const mrs_msgs::HwApiAttitudeCmd::ConstPtr msg) {

  if (!_capabilities_.accepts_attitude_cmd) {
    return false;
  }

  ROS_INFO_ONCE("[Api]: getting attitude cmd");

  ph_attitude_cmd_.publish(msg);

  return true;
}

//}

/* callbackAccelerationHdgRateCmd() //{ */

bool Api::callbackAccelerationHdgRateCmd(const mrs_msgs::HwApiAccelerationHdgRateCmd::ConstPtr msg) {

  if (!_capabilities_.accepts_acceleration_hdg_rate_cmd) {
    return false;
  }

  ROS_INFO_ONCE("[Api]: getting acceleration+hdg rate cmd");

  ph_acceleration_hdg_rate_cmd_.publish(msg);

  return true;
}

//}

/* callbackAccelerationHdgCmd() //{ */

bool Api::callbackAccelerationHdgCmd(const mrs_msgs::HwApiAccelerationHdgCmd::ConstPtr msg) {

  if (!_capabilities_.accepts_acceleration_hdg_cmd) {

    return false;
  }

  ROS_INFO_ONCE("[Api]: getting acceleration+hdg cmd");

  ph_acceleration_hdg_cmd_.publish(msg);

  return true;
}

//}

/* callbackVelocityHdgRateCmd() //{ */

bool Api::callbackVelocityHdgRateCmd(const mrs_msgs::HwApiVelocityHdgRateCmd::ConstPtr msg) {

  if (!_capabilities_.accepts_velocity_hdg_rate_cmd) {
    return false;
  }

  ROS_INFO_ONCE("[Api]: getting velocity+hdg rate cmd");

  ph_velocity_hdg_rate_cmd_.publish(msg);

  return true;
}

//}

/* callbackVelocityHdgCmd() //{ */

bool Api::callbackVelocityHdgCmd(const mrs_msgs::HwApiVelocityHdgCmd::ConstPtr msg) {

  if (!_capabilities_.accepts_velocity_hdg_cmd) {
    return false;
  }

  ROS_INFO_ONCE("[Api]: getting velocity+hdg cmd");

  ph_velocity_hdg_cmd_.publish(msg);

  return true;
}

//}

/* callbackPositionCmd() //{ */

bool Api::callbackPositionCmd(const mrs_msgs::HwApiPositionCmd::ConstPtr msg) {

  if (!_capabilities_.accepts_position_cmd) {
    return false;
  }

  ROS_INFO_ONCE("[Api]: getting position cmd");

  ph_position_cmd_.publish(msg);

  return true;
}

//}

/* callbackTrackerCmd() //{ */

void Api::callbackTrackerCmd(const mrs_msgs::TrackerCommand::ConstPtr msg) {

  ROS_INFO_ONCE("[Api]: getting tracker cmd");

  ph_tracker_cmd_.publish(msg);
}

//}

// | ------------------------ callbacks ----------------------- |

/* //{ callbackOdom() */

void Api::callbackOdom(const nav_msgs::Odometry::ConstPtr msg) {

  if (!is_initialized_) {
    return;
  }

  ROS_INFO_ONCE("[Api]: getting simulator odometry");

  auto odom = msg;

  {
    std::scoped_lock lock(mutex_status_);

    connected_ = true;
  }

  // | ----------------- publish the diagnostics ---------------- |

  mrs_msgs::HwApiStatus status;

  {
    std::scoped_lock lock(mutex_status_);

    status.stamp     = ros::Time::now();
    status.armed     = armed_;
    status.offboard  = offboard_;
    status.connected = connected_;
    status.mode      = mode_;
  }

  common_handlers_->publishers.publishStatus(status);

  // | -------------------- publish position -------------------- |

  if (_capabilities_.produces_position) {

    geometry_msgs::PointStamped position;

    position.header.stamp    = odom->header.stamp;
    position.header.frame_id = _uav_name_ + "/" + _world_frame_name_;
    position.point           = odom->pose.pose.position;

    common_handlers_->publishers.publishPosition(position);
  }

  // | ------------------- publish orientation ------------------ |

  if (_capabilities_.produces_orientation) {

    geometry_msgs::QuaternionStamped orientation;

    orientation.header.stamp    = odom->header.stamp;
    orientation.header.frame_id = _uav_name_ + "/" + _world_frame_name_;
    orientation.quaternion      = odom->pose.pose.orientation;

    common_handlers_->publishers.publishOrientation(orientation);
  }

  // | -------------------- publish velocity -------------------- |

  if (_capabilities_.produces_velocity) {

    geometry_msgs::Vector3Stamped velocity;

    velocity.header.stamp    = odom->header.stamp;
    velocity.header.frame_id = _uav_name_ + "/" + _body_frame_name_;
    velocity.vector          = odom->twist.twist.linear;

    common_handlers_->publishers.publishVelocity(velocity);
  }

  // | ---------------- publish angular velocity ---------------- |

  if (_capabilities_.produces_angular_velocity) {

    geometry_msgs::Vector3Stamped angular_velocity;

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

    sensor_msgs::NavSatFix gnss;

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

    mrs_msgs::RtkGps rtk;

    rtk.header.stamp = odom->header.stamp;

    rtk.gps.latitude  = lat;
    rtk.gps.longitude = lon;
    rtk.gps.altitude  = odom->pose.pose.position.z + _amsl_;

    rtk.fix_type.fix_type = mrs_msgs::RtkFixType::RTK_FIX;

    common_handlers_->publishers.publishRTK(rtk);
  }

  // | ------------------ publish amsl altitude ----------------- |

  if (_capabilities_.produces_altitude) {

    mrs_msgs::HwApiAltitude altitude;

    altitude.stamp = odom->header.stamp;

    altitude.amsl = odom->pose.pose.position.z + _amsl_;

    common_handlers_->publishers.publishAltitude(altitude);
  }

  // | --------------------- publish heading -------------------- |

  if (_capabilities_.produces_magnetometer_heading) {

    double heading = mrs_lib::AttitudeConverter(odom->pose.pose.orientation).getHeading();

    mrs_msgs::Float64Stamped hdg;

    hdg.header.stamp = ros::Time::now();
    hdg.value        = heading;

    common_handlers_->publishers.publishMagnetometerHeading(hdg);
  }
}

//}

/* callbackImu() //{ */

void Api::callbackImu(const sensor_msgs::Imu::ConstPtr msg) {

  if (!is_initialized_) {
    return;
  }

  ROS_INFO_ONCE("[Api]: getting IMU");

  if (_capabilities_.produces_imu) {

    common_handlers_->publishers.publishIMU(*msg);
  }
}

//}

/* callbackDistanceSensor() //{ */
void Api::callbackDistanceSensor(const sensor_msgs::Range::ConstPtr msg) {

  if (!is_initialized_) {
    return;
  }

  ROS_INFO_ONCE("[Api]: getting distance sensor");

  if (_capabilities_.produces_distance_sensor) {
// ph_range_.publish(*msg);
    common_handlers_->publishers.publishDistanceSensor(*msg);
  }
}
//}

// | ------------------------- timers ------------------------- |

/* timerMain() //{ */

void Api::timerMain([[maybe_unused]] const ros::TimerEvent& event) {

  if (!is_initialized_) {
    return;
  }

  ROS_INFO_ONCE("[Api]: main timer spinning");

  publishBatteryState();

  publishRC();
}

//}

// | ------------------------- methods ------------------------ |

/* publishBatteryState() //{ */

void Api::publishBatteryState(void) {

  if (_capabilities_.produces_battery_state) {

    sensor_msgs::BatteryState msg;

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

    mrs_msgs::HwApiRcChannels rc;

    rc.stamp = ros::Time::now();

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
}

//}

}  // namespace mrs_uav_unreal_simulation

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(mrs_uav_unreal_simulation::Api, mrs_uav_hw_api::MrsUavHwApi)
