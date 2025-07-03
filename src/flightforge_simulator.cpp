/* includes //{ */

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
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <Eigen/Dense>

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

#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>

#include <random>
#include <vector>
#include <chrono>

#include <mrs_msgs/srv/float64_srv.hpp>
#include <mrs_uav_flightforge_simulation/srv/set_orientation.hpp>

//}

/* defines //{ */

using namespace std::chrono_literals;

using PCLPoint               = pcl::PointXYZ;
using PCLPointCloud          = pcl::PointCloud<PCLPoint>;
using PCLPointCloudColor     = pcl::PointCloud<pcl::PointXYZRGB>;
using PCLPointCloudIntensity = pcl::PointCloud<pcl::PointXYZI>;

//}

namespace mrs_uav_flightforge_simulation
{

/* class FlightforgeSimulator //{ */

class FlightforgeSimulator : public rclcpp::Node {

public:
  FlightforgeSimulator(rclcpp::NodeOptions options);

private:
  rclcpp::CallbackGroup::SharedPtr cbgrp_main_;
  rclcpp::CallbackGroup::SharedPtr cbgrp_sensors_;
  rclcpp::CallbackGroup::SharedPtr cbgrp_status_;

  rclcpp::TimerBase::SharedPtr timer_init_;
  void                         timerInit();

  rclcpp::Node::SharedPtr  node_;
  rclcpp::Clock::SharedPtr clock_;
  std::atomic<bool>        is_initialized_ = false;

  std::shared_ptr<mrs_lib::ScopeTimerLogger> scope_timer_logger_;
  std::shared_ptr<image_transport::ImageTransport> it_;

  // | ------------------------- params ------------------------- |

  double _simulation_rate_;
  double _clock_rate_;
  bool   _collisions_ = false;

  rclcpp::Time sim_time_;
  rclcpp::Time last_step_time_;
  std::mutex   mutex_sim_time_;

  std::string _world_frame_name_;

  // | ------------------------- timers ------------------------- |

  /* timer dynamic */
  rclcpp::TimerBase::SharedPtr timer_main_;
  /* std::shared_ptr<mrs_lib::ThreadTimer> timer_main_ */
  void                         timerMain();

  rclcpp::TimerBase::SharedPtr timer_status_;
  void                         timerStatus();
  
  rclcpp::TimerBase::SharedPtr timer_time_sync_;
  void                         timerTimeSync();

  rclcpp::TimerBase::SharedPtr timer_unreal_sync_;
  void                         timerUnrealSync();

  rclcpp::TimerBase::SharedPtr timer_rangefinder_;
  void                         timerRangefinder();

  rclcpp::TimerBase::SharedPtr timer_lidar_;
  void                         timerLidar();

  rclcpp::TimerBase::SharedPtr timer_seg_lidar_;
  void                         timerSegLidar();

  rclcpp::TimerBase::SharedPtr timer_int_lidar_;
  void                         timerIntLidar();

  rclcpp::TimerBase::SharedPtr timer_rgb_;
  void                         timerRgb();

  rclcpp::TimerBase::SharedPtr timer_rgb_segmented_;
  void                         timerRgbSegmented();

  rclcpp::TimerBase::SharedPtr timer_depth_;
  void                         timerDepth();

  rclcpp::TimerBase::SharedPtr timer_stereo_;
  void                         timerStereo();

  // | ------------------------ rtf check ----------------------- |

  double       actual_rtf_ = 1.0;
  rclcpp::Time last_sim_time_status_;

  // | ----------------------- publishers ----------------------- |

  mrs_lib::PublisherHandler<rosgraph_msgs::msg::Clock>     ph_clock_;
  mrs_lib::PublisherHandler<geometry_msgs::msg::PoseArray> ph_poses_;

  std::vector<mrs_lib::PublisherHandler<sensor_msgs::msg::Range>> ph_rangefinders_;
  std::vector<mrs_lib::PublisherHandler<sensor_msgs::msg::PointCloud2>> ph_lidars_;
  std::vector<mrs_lib::PublisherHandler<sensor_msgs::msg::PointCloud2>> ph_seg_lidars_;
  std::vector<mrs_lib::PublisherHandler<sensor_msgs::msg::PointCloud2>> ph_int_lidars_;

  std::vector<image_transport::Publisher> imp_rgb_;
  std::vector<image_transport::Publisher> imp_stereo_left_;
  std::vector<image_transport::Publisher> imp_stereo_right_;
  std::vector<image_transport::Publisher> imp_rgbd_segmented_;
  std::vector<image_transport::Publisher> imp_depth_;
  


  std::vector<mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>> ph_rgb_camera_info_;
  std::vector<mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>> ph_rgb_seg_camera_info_;
  std::vector<mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>> ph_stereo_left_camera_info_;
  std::vector<mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>> ph_stereo_right_camera_info_;
  std::vector<mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>> ph_depth_camera_info_;

  // | ------------------------- system ------------------------- |

  std::vector<std::unique_ptr<mrs_multirotor_simulator::UavSystemRos>> uavs_;

  // | ------------------------- methods ------------------------ |

  void handleCollisions(void);

  void publishPoses(void);
  
  // | ------------------------- FlightForge methods ------------------------ |
  
  std::unique_ptr<ueds_connector::GameModeController> ueds_game_controller_;

  std::vector<std::shared_ptr<ueds_connector::UedsConnector>> ueds_connectors_;
  
  std::mutex mutex_flightforge_;

  ueds_connector::Coordinates ueds_world_origin_;

  std::vector<ueds_connector::Coordinates> ueds_world_origins_;

  
  void updateUnrealPoses(const bool teleport_without_collision);
  
  ueds_connector::Coordinates position2ue(const Eigen::Vector3d& pos, const ueds_connector::Coordinates& ueds_world_origin);

  void fabricateCamInfo(void);

  void publishStaticTfs(void);
  
  void publishCameraTf(const int& uav_index);
  
  void checkForCrash(void);

  double flightforgeToWallTime(const double flightforge_time);
  
  // how much to add to unreal time to get to our wall time
  double        wall_time_offset_             = 0;
  double        wall_time_offset_drift_slope_ = 0;
  rclcpp::Time last_sync_time_;
  std::mutex    mutex_wall_time_offset_;
  rclcpp::Time last_real_;

  double                  flightforge_fps_ = 0;
  std::string             flightforge_world_level_name_enum_;
  std::string             flightforge_graphics_settings_enum_;
  int                     flightforge_forest_density_     = 5;
  int                     flightforge_forest_hilly_level_ = 3;
  std::string             weather_type_;
  ueds_connector::Daytime daytime_;
  bool                    uavs_mutual_visibility_ = true;

  // | ------------------------- Transformations ------------------------ |


  std::shared_ptr<mrs_lib::TransformBroadcaster> tf_broadcaster_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
  std::shared_ptr<mrs_lib::TransformBroadcaster> dynamic_broadcaster_;

  OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

  rcl_interfaces::msg::SetParametersResult callbackParameters(std::vector<rclcpp::Parameter> parameters);

  struct drs_params
  {
    double realtime_factor     = 1.0;
    bool   dynamic_rtf = false;
    bool   paused              = false;
    bool   collisions_enabled  = false;
    bool   collisions_crash    = false;
    double collisions_rebounce = 1;

    bool   rangefinder_enabled = false;
    double rangefinder_rate    = 0.0;

    bool   lidar_enabled      = true;
    double lidar_rate         = 10.0;
    bool   lidar_noise_enabled = false;
    double lidar_std_at_1m    = 0.0;
    double lidar_std_slope    = 0.0;
    
    bool   lidar_seg_enabled  = true;
    double lidar_seg_rate     = 10.0;
    
    bool   lidar_int_enabled  = false;
    double lidar_int_rate     = 0.0;
    double lidar_int_value_grass    = 0.0;
    double lidar_int_value_road     = 0.0;
    double lidar_int_value_tree     = 0.0;
    double lidar_int_value_building = 0.0;
    double lidar_int_value_fence    = 0.0;
    double lidar_int_value_dirt_road = 0.0;
    double lidar_int_value_other    = 0.0;
    bool   lidar_int_noise_enabled = false;
    double lidar_int_std_at_1m    = 0.0;
    double lidar_int_std_slope    = 0.0;

    bool   rgb_enabled        = true;
    double rgb_rate           = 10.0;
    bool   rgb_enable_hdr = false;
    bool   rgb_enable_temporal_aa = false;
    bool   rgb_enable_raytracing = false;
    bool   rgb_enable_motion_blur = false;
    double rgb_motion_blur_amount = 0.0;
    double rgb_motion_blur_distortion = 0.0;
    
    bool   rgb_segmented_enabled = true;
    double rgb_segmented_rate = 10.0;
    
    bool   rgb_depth_enabled  = false;
    double rgb_depth_rate     = 0.0;

    bool   stereo_enabled = true;
    double stereo_rate    = 10.0;
    bool   stereo_enable_hdr = false;
    bool   stereo_enable_temporal_aa = false;
    bool   stereo_enable_raytracing = false;

  };

  drs_params drs_params_;
  std::mutex mutex_drs_params_;

  /*segmentation decode array//{*/
  // clang-format off
  uint8_t seg_rgb_[256][3] = {
      255, 255, 255,
      153, 108, 6 ,
      112, 105, 191 ,
      89, 121, 72 ,
      190, 225, 64 ,
      206, 190, 59 ,
      81, 13, 36 ,
      115, 176, 195 ,
      161, 171, 27 ,
      135, 169, 180 ,
      29, 26, 199 ,
      102, 16, 239 ,
      242, 107, 146 ,
      156, 198, 23 ,
      49, 89, 160 ,
      68, 218, 116 ,
      11, 236, 9 ,
      196, 30, 8 ,
      121, 67, 28 ,
      0, 53, 65 ,
      146, 52, 70 ,
      226, 149, 143 ,
      151, 126, 171 ,
      194, 39, 7 ,
      205, 120, 161 ,
      212, 51, 60 ,
      211, 80, 208 ,
      189, 135, 188 ,
      54, 72, 205 ,
      103, 252, 157 ,
      124, 21, 123 ,
      19, 132, 69 ,
      195, 237, 132 ,
      94, 253, 175 ,
      182, 251, 87 ,
      90, 162, 242 ,
      199, 29, 1 ,
      254, 12, 229 ,
      35, 196, 244 ,
      220, 163, 49 ,
      86, 254, 214 ,
      152, 3, 129 ,
      92, 31, 106 ,
      207, 229, 90 ,
      125, 75, 48 ,
      98, 55, 74 ,
      126, 129, 238 ,
      222, 153, 109 ,
      85, 152, 34 ,
      173, 69, 31 ,
      37, 128, 125 ,
      58, 19, 33 ,
      134, 57, 119 ,
      218, 124, 115 ,
      120, 0, 200 ,
      225, 131, 92 ,
      246, 90, 16 ,
      51, 155, 241 ,
      202, 97, 155 ,
      184, 145, 182 ,
      96, 232, 44 ,
      133, 244, 133 ,
      180, 191, 29 ,
      1, 222, 192 ,
      99, 242, 104 ,
      91, 168, 219 ,
      65, 54, 217 ,
      148, 66, 130 ,
      203, 102, 204 ,
      216, 78, 75 ,
      234, 20, 250 ,
      109, 206, 24 ,
      164, 194, 17 ,
      157, 23, 236 ,
      158, 114, 88 ,
      245, 22, 110 ,
      67, 17, 35 ,
      181, 213, 93 ,
      170, 179, 42 ,
      52, 187, 148 ,
      247, 200, 111 ,
      25, 62, 174 ,
      100, 25, 240 ,
      191, 195, 144 ,
      252, 36, 67 ,
      241, 77, 149 ,
      237, 33, 141 ,
      119, 230, 85 ,
      28, 34, 108 ,
      78, 98, 254 ,
      114, 161, 30 ,
      75, 50, 243 ,
      66, 226, 253 ,
      46, 104, 76 ,
      8, 234, 216 ,
      15, 241, 102 ,
      93, 14, 71 ,
      192, 255, 193 ,
      253, 41, 164 ,
      24, 175, 120 ,
      185, 243, 231 ,
      169, 233, 97 ,
      243, 215, 145 ,
      72, 137, 21 ,
      160, 113, 101 ,
      214, 92, 13 ,
      167, 140, 147 ,
      101, 109, 181 ,
      53, 118, 126 ,
      3, 177, 32 ,
      40, 63, 99 ,
      186, 139, 153 ,
      88, 207, 100 ,
      71, 146, 227 ,
      236, 38, 187 ,
      215, 4, 215 ,
      18, 211, 66 ,
      113, 49, 134 ,
      47, 42, 63 ,
      219, 103, 127 ,
      57, 240, 137 ,
      227, 133, 211 ,
      145, 71, 201 ,
      217, 173, 183 ,
      250, 40, 113 ,
      208, 125, 68 ,
      224, 186, 249 ,
      69, 148, 46 ,
      239, 85, 20 ,
      108, 116, 224 ,
      56, 214, 26 ,
      179, 147, 43 ,
      48, 188, 172 ,
      221, 83, 47 ,
      155, 166, 218 ,
      62, 217, 189 ,
      198, 180, 122 ,
      201, 144, 169 ,
      132, 2, 14 ,
      128, 189, 114 ,
      163, 227, 112 ,
      45, 157, 177 ,
      64, 86, 142 ,
      118, 193, 163 ,
      14, 32, 79 ,
      200, 45, 170 ,
      74, 81, 2 ,
      59, 37, 212 ,
      73, 35, 225 ,
      95, 224, 39 ,
      84, 170, 220 ,
      159, 58, 173 ,
      17, 91, 237 ,
      31, 95, 84 ,
      34, 201, 248 ,
      63, 73, 209 ,
      129, 235, 107 ,
      231, 115, 40 ,
      36, 74, 95 ,
      238, 228, 154 ,
      61, 212, 54 ,
      13, 94, 165 ,
      141, 174, 0 ,
      140, 167, 255 ,
      117, 93, 91 ,
      183, 10, 186 ,
      165, 28, 61 ,
      144, 238, 194 ,
      12, 158, 41 ,
      76, 110, 234 ,
      150, 9, 121 ,
      142, 1, 246 ,
      230, 136, 198 ,
      5, 60, 233 ,
      232, 250, 80 ,
      143, 112, 56 ,
      187, 70, 156 ,
      2, 185, 62 ,
      138, 223, 226 ,
      122, 183, 222 ,
      166, 245, 3 ,
      175, 6, 140 ,
      240, 59, 210 ,
      248, 44, 10 ,
      83, 82, 52 ,
      223, 248, 167 ,
      87, 15, 150 ,
      111, 178, 117 ,
      197, 84, 22 ,
      235, 208, 124 ,
      9, 76, 45 ,
      176, 24, 50 ,
      154, 159, 251 ,
      149, 111, 207 ,
      168, 231, 15 ,
      209, 247, 202 ,
      80, 205, 152 ,
      178, 221, 213 ,
      27, 8, 38 ,
      244, 117, 51 ,
      107, 68, 190 ,
      23, 199, 139 ,
      171, 88, 168 ,
      136, 202, 58 ,
      6, 46, 86 ,
      105, 127, 176 ,
      174, 249, 197 ,
      172, 172, 138 ,
      228, 142, 81 ,
      7, 204, 185 ,
      22, 61, 247 ,
      233, 100, 78 ,
      127, 65, 105 ,
      33, 87, 158 ,
      139, 156, 252 ,
      42, 7, 136 ,
      20, 99, 179 ,
      79, 150, 223 ,
      131, 182, 184 ,
      110, 123, 37 ,
      60, 138, 96 ,
      210, 96, 94 ,
      123, 48, 18 ,
      137, 197, 162 ,
      188, 18, 5 ,
      39, 219, 151 ,
      204, 143, 135 ,
      249, 79, 73 ,
      77, 64, 178 ,
      41, 246, 77 ,
      16, 154, 4 ,
      116, 134, 19 ,
      4, 122, 235 ,
      177, 106, 230 ,
      21, 119, 12 ,
      104, 5, 98 ,
      50, 130, 53 ,
      30, 192, 25 ,
      26, 165, 166 ,
      10, 160, 82 ,
      106, 43, 131 ,
      44, 216, 103 ,
      255, 101, 221 ,
      32, 151, 196 ,
      213, 220, 89 ,
      70, 209, 228 ,
      97, 184, 83 ,
      82, 239, 232 ,
      251, 164, 128 ,
      193, 11, 245 ,
      38, 27, 159 ,
      229, 141, 203 ,
      130, 56, 55 ,
      147, 210, 11 ,
      162, 203, 118 ,
      0, 0, 0 
  };
  // clang-format on
/*//}*/

  // | ----------------------- camera info ---------------------- |
  sensor_msgs::msg::CameraInfo rgb_camera_info_;
  sensor_msgs::msg::CameraInfo stereo_camera_info_;

  geometry_msgs::msg::TransformStamped rgb_camera_tf_;
  geometry_msgs::msg::TransformStamped stereo_camera_tf_;
  
  std::vector<double> last_rgb_ue_stamp_;
  std::vector<double> last_rgb_seg_ue_stamp_;
  std::vector<double> last_stereo_ue_stamp_;

  // | --------- store current camera orientation -------- |
  std::vector<Eigen::Quaterniond> rgb_camera_orientations_;

  std::default_random_engine rng;



  // | ----------------------- Dynamic params TODO: fix this mess ---------------------- |

  double lidar_horizontal_fov_left_ = 180.0;
  double lidar_horizontal_fov_right_ = 180.0;
  double lidar_vertical_fov_up_ = 45.0;
  double lidar_vertical_fov_down_ = 45.0;
  int    lidar_horizontal_rays_ = 128;
  int    lidar_vertical_rays_ = 256;
  double lidar_offset_x_ = 0.0;
  double lidar_offset_y_ = 0.0;
  double lidar_offset_z_ = 1.0;
  double lidar_rotation_pitch_ = 0.0;
  double lidar_rotation_roll_ = 0.0;
  double lidar_rotation_yaw_ = 0.0;
  double lidar_beam_length_ = 40;
  bool   lidar_show_beams_ = false;
  bool   lidar_livox_ = false;


  int    rgb_width_ = 640;
  int    rgb_height_ = 480;
  double rgb_fov_ = 120.0;
  double rgb_offset_x_ = 0.14;
  double rgb_offset_y_ = 0.0;
  double rgb_offset_z_ = 0.0;
  double rgb_rotation_pitch_ = 0.0;
  double rgb_rotation_yaw_ = 0.0;
  double rgb_rotation_roll_ = 0.0;
  bool   rgb_enable_hdr_ = false;
  bool   rgb_enable_temporal_aa_ = true;
  bool   rgb_enable_raytracing_ = true;
  bool   rgb_enable_motion_blur_ = true;
  double rgb_motion_blur_amount_ = 0.5;
  double rgb_motion_blur_distortion_ = 50.0;
    
  double stereo_baseline_ = 0.1;
  int    stereo_width_ = 640;
  int    stereo_height_ = 480;
  double stereo_fov_ = 90.0;
  double stereo_offset_x_ = 0.14;
  double stereo_offset_y_ = 0.0;
  double stereo_offset_z_ = 0.0;
  double stereo_rotation_pitch_ = 20.0;
  double stereo_rotation_yaw_ = 0.0;
  double stereo_rotation_roll_ = 0.0;
  bool   stereo_enable_hdr_ = true;
  bool   stereo_enable_temporal_aa_ = true;
  bool   stereo_enable_raytracing_ = true;
};

//}

/* FlightforgeSimulator::FlightforgeSimulator() //{ */

FlightforgeSimulator::FlightforgeSimulator(rclcpp::NodeOptions options) : Node("mrs_uav_flightforge_simulation", options) {

  {
    auto param_desc = rcl_interfaces::msg::ParameterDescriptor{};

    rcl_interfaces::msg::FloatingPointRange range;

    range.from_value = 0.01;
    range.to_value   = 10.0;

    param_desc.floating_point_range = {range};

    param_desc.type = rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE;

    this->declare_parameter("dynamic/realtime_factor", 1.0, param_desc);
  }

  {
    auto param_desc = rcl_interfaces::msg::ParameterDescriptor{};

    param_desc.type = rcl_interfaces::msg::ParameterType::PARAMETER_BOOL;

    this->declare_parameter("dynamic/paused", false, param_desc);
  }

  {
    auto param_desc = rcl_interfaces::msg::ParameterDescriptor{};

    param_desc.type = rcl_interfaces::msg::ParameterType::PARAMETER_BOOL;

    this->declare_parameter("dynamic/collisions_enabled", false, param_desc);
  }

  {
    auto param_desc = rcl_interfaces::msg::ParameterDescriptor{};

    param_desc.type = rcl_interfaces::msg::ParameterType::PARAMETER_BOOL;

    this->declare_parameter("dynamic/collisions_crash", false, param_desc);
  }

  {
    auto param_desc = rcl_interfaces::msg::ParameterDescriptor{};

    param_desc.type = rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE;

    rcl_interfaces::msg::FloatingPointRange range;

    range.from_value = 0.0;
    range.to_value   = 500.0;

    param_desc.floating_point_range = {range};

    this->declare_parameter("dynamic/collisions_rebounce", 1.0, param_desc);
  }

  timer_init_ = create_wall_timer(std::chrono::duration<double>(0.1s), std::bind(&FlightforgeSimulator::timerInit, this));
}

//}

// | ------------------------- timers ------------------------- |

/* timerInit() //{ */

void FlightforgeSimulator::timerInit() {

  node_  = this->shared_from_this();
  clock_ = node_->get_clock();

  srand(time(NULL));

  RCLCPP_INFO(node_->get_logger(), "initializing");

  cbgrp_main_   = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  cbgrp_status_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  mrs_lib::ParamLoader param_loader(node_, this->get_name());

  // load custom config

  std::string custom_config_path;
  param_loader.loadParam("custom_config", custom_config_path);

  if (custom_config_path != "") {
    RCLCPP_INFO(node_->get_logger(), "loading custom config '%s", custom_config_path.c_str());
    param_loader.addYamlFile(custom_config_path);
  }

  // load other configs

  std::vector<std::string> config_files;
  param_loader.loadParam("simulator_configs", config_files);

  for (auto config_file : config_files) {
    RCLCPP_INFO(node_->get_logger(), "loading config file '%s'", config_file.c_str());
    param_loader.addYamlFile(config_file);
  }

  param_loader.loadParam("simulation_rate", _simulation_rate_);
  param_loader.loadParam("dynamic_rtf", drs_params_.dynamic_rtf);
  param_loader.loadParam("clock_rate", _clock_rate_);

  if (_clock_rate_ < _simulation_rate_) {
    RCLCPP_ERROR(node_->get_logger(), "clock_rate (%.2f Hz) should be higher than simulation rate (%.2f Hz)!", _clock_rate_, _simulation_rate_);
    rclcpp::shutdown();
    exit(1);
  }

  param_loader.loadParam("realtime_factor", drs_params_.realtime_factor);
  this->set_parameter(rclcpp::Parameter("dynamic/realtime_factor", drs_params_.realtime_factor));

  param_loader.loadParam("collisions/enabled", drs_params_.collisions_enabled);
  this->set_parameter(rclcpp::Parameter("dynamic/collisions_enabled", drs_params_.collisions_enabled));

  param_loader.loadParam("collisions/crash", drs_params_.collisions_crash);
  this->set_parameter(rclcpp::Parameter("dynamic/collisions_crash", drs_params_.collisions_crash));

  param_loader.loadParam("collisions/rebounce", drs_params_.collisions_rebounce);
  this->set_parameter(rclcpp::Parameter("dynamic/collisions_rebounce", drs_params_.collisions_rebounce));

  param_loader.loadParam("frames/world/name", _world_frame_name_);

  bool sim_time_from_wall_time;
  param_loader.loadParam("sim_time_from_wall_time", sim_time_from_wall_time);

  if (sim_time_from_wall_time) {
    sim_time_       = clock_->now();
    last_step_time_ = clock_->now();
  } else {
    sim_time_       = rclcpp::Time(0, 0, RCL_ROS_TIME);
    last_step_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  }

  last_sim_time_status_ = sim_time_;

  drs_params_.paused = false;

  tf_broadcaster_ = std::make_shared<mrs_lib::TransformBroadcaster>(node_);
  static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node_);
  dynamic_broadcaster_ = std::make_shared<mrs_lib::TransformBroadcaster>(node_);

  it_                  = std::make_shared<image_transport::ImageTransport>(node_);

  std::vector<std::string> uav_names;

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
  
  ueds_game_controller_ = std::make_unique<ueds_connector::GameModeController>(LOCALHOST, 8551);
  while (rclcpp::ok()) {
    if (ueds_game_controller_->Connect() == 1) {
      break;
    }
    RCLCPP_ERROR(node_->get_logger(), "[FlightforgeSimulator]: Error connecting to Unreal's game mode controller");
    std::this_thread::sleep_for(1s);

  }
  
  auto [res, api_version] = ueds_game_controller_->GetApiVersion();
  auto [api_version_major, api_version_minor] = api_version;

  if (!res || api_version_major != API_VERSION_MAJOR || api_version_minor != API_VERSION_MINOR) {

    RCLCPP_ERROR(node_->get_logger(), "The API versions don't match! (ROS side '%d.%d' != FlightForge binary side '%d.%d')", API_VERSION_MAJOR, API_VERSION_MINOR, api_version_major, api_version_minor);
    RCLCPP_ERROR(node_->get_logger(), "     ");
    RCLCPP_ERROR(node_->get_logger(), " Solution:");
    RCLCPP_ERROR(node_->get_logger(), "           1. make sure the mrs_uav_unreal_simulation package is up to date");
    RCLCPP_ERROR(node_->get_logger(), "              sudo apt update && sudo apt upgrade");
    RCLCPP_ERROR(node_->get_logger(), "     ");
    RCLCPP_ERROR(node_->get_logger(), "           2. make sure you have the right version of the FlightForge Simulator binary 'game'");
    RCLCPP_ERROR(node_->get_logger(), "              download at: https://github.com/ctu-mrs/mrs_uav_unreal_simulation");
    RCLCPP_ERROR(node_->get_logger(), "     ");

    rclcpp::shutdown();
  }

  std::string world_name;

  param_loader.loadParam("world_name", world_name);

  res = ueds_game_controller_->SwitchWorldLevel(ueds_connector::WorldName::Name2Id().at(world_name));

  if (res) {
    RCLCPP_INFO(node_->get_logger(), "World was switched succesfully.");
  } else {
    RCLCPP_ERROR(get_logger(), "World was not switched succesfully");  
  }
  
  res = ueds_game_controller_->Disconnect();
  if (res) {
    RCLCPP_INFO(get_logger(), "ueds_game_controller_ was Disconnected succesfully.");
  } else {
    RCLCPP_ERROR(get_logger(), "ueds_game_controller_ was not Disconnected succesfully.");
  }

  std::this_thread::sleep_for(std::chrono::seconds(5));
  
  while (true) {
    bool connect_result = ueds_game_controller_->Connect();
    if (connect_result != 1) {
      RCLCPP_ERROR(get_logger(), "Error connecting to Unreal's game mode controller, connect_result was %d", connect_result);
    } else {
      break;
    }
    /* TODO: This should be probably shomehow be dynamic and set to a machine/world combo */  
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  

  res = ueds_game_controller_->SetGraphicsSettings(ueds_connector::GraphicsSettings::Name2Id().at("low"));

  if (res) {
    RCLCPP_INFO(get_logger(), "Graphical Settings was set succesfully to low");

    /* RCLCPP_INFO(get_logger(), "Graphical Settings was set succesfully to '%s'", ueds_graphics_settings_enum_.c_str()); */
  } else {

    /* RCLCPP_ERROR(get_logger(), "Graphical Settings was not set succesfully to '%s'", ueds_graphics_settings_enum_.c_str()); */
    RCLCPP_ERROR(get_logger(), "Graphical Settings was not set succesfully to low");
  }

  res = ueds_game_controller_->SetMutualDroneVisibility(true); 
  if (res) {
    /* RCLCPP_INFO(get_logger(), "Mutual Drone Visibility was succesfully set to %i.", uavs_mutual_visibility_); */
    RCLCPP_INFO(get_logger(), "Mutual Drone Visibility was succesfully set to true");
  } else {
    RCLCPP_ERROR(get_logger(), "Set Mutual Drone Visibility was NOT succesfull.");
    
  }

  /* res = ueds_game_controller_->SetWeather(ueds_connector::WeatherType::Type2Id().at(weather_type_)); */
  /* if (res) { */
  /*   RCLCPP_INFO(get_logger(), "SetWeather successful."); */
  /* } else { */
  /*   RCLCPP_ERROR(get_logger(), "SetWeather error"); */
  /* } */

  res = ueds_game_controller_->SetDatetime(12, 0);
  if (res) {
    RCLCPP_INFO(get_logger(), "SetDatetime successful.");
  } else {
    RCLCPP_ERROR(get_logger(), "SetDatetime error");
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
    RCLCPP_ERROR(get_logger(), "getting world origin");
    rclcpp::shutdown();
  } else {
    ueds_world_origin_ = world_origin;
  }

  for (size_t i = 0; i < uav_names.size(); i++) {

    const std::string uav_name = uav_names[i];

    mrs_multirotor_simulator::MultirotorModel::State uav_state = uavs_[i]->getState();

    ueds_connector::Coordinates pos = position2ue(uav_state.x, ueds_world_origin_);

    RCLCPP_INFO(get_logger(), "%s spawning at [%.2lf, %.2lf, %.2lf] ...", uav_name.c_str(), uav_state.x.x(), uav_state.x.y(), uav_state.x.z());

    std::string uav_frame = "x500";
    /* TODO: frame param */
    /* param_loader.loadParam(uav_names[i] + "/frame", uav_frame); */

    RCLCPP_INFO(get_logger(), "Frame type to spawn is %s", uav_frame.c_str());

    int uav_frame_id = ueds_connector::UavFrameType::Type2IdMesh().at(uav_frame);

    auto [resSpawn, port] = ueds_game_controller_->SpawnDroneAtLocation(pos, uav_frame_id);

    if (!resSpawn) {
      RCLCPP_ERROR(get_logger(), "failed to spawn %s", uav_names[i].c_str());
      rclcpp::shutdown();
    }

    RCLCPP_INFO(get_logger(), "%s spawned", uav_name.c_str());

    std::shared_ptr<ueds_connector::UedsConnector> ueds_connector = std::make_shared<ueds_connector::UedsConnector>(LOCALHOST, port);

    ueds_connectors_.push_back(ueds_connector);


    auto connect_result = ueds_connector->Connect();

    if (connect_result != 1) {

      RCLCPP_ERROR(get_logger(), "%s - Error connecting to drone controller, connect_result was %d", uav_name.c_str(), connect_result);
      rclcpp::shutdown();

    } else {
      RCLCPP_INFO(get_logger(), "%s - Connection succeed: %d", uav_name.c_str(), connect_result);

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

    ph_rangefinders_.push_back(mrs_lib::PublisherHandler<sensor_msgs::msg::Range>(node_, "/" + uav_name + "/rangefinder"));
    ph_lidars_.push_back(mrs_lib::PublisherHandler<sensor_msgs::msg::PointCloud2>(node_, "/" + uav_name + "/lidar/points"));
    ph_seg_lidars_.push_back(mrs_lib::PublisherHandler<sensor_msgs::msg::PointCloud2>(node_, "/" + uav_name + "/lidar_segmented/points"));
    ph_int_lidars_.push_back(mrs_lib::PublisherHandler<sensor_msgs::msg::PointCloud2>(node_, "/" + uav_name + "/lidar_intensity/points"));
  
    imp_rgb_.push_back(it_->advertise("/" + uav_name + "/rgb/image_raw", 1));
    imp_stereo_left_.push_back(it_->advertise("/" + uav_name + "/stereo/left/image_raw", 1));
    imp_stereo_right_.push_back(it_->advertise("/" + uav_name + "/stereo/right/image_raw", 1));
    imp_rgbd_segmented_.push_back(it_->advertise("/" + uav_name + "/rgb_segmented/image_raw", 1));
    imp_depth_.push_back(it_->advertise("/" + uav_name + "/depth/image_raw", 1));
    


    ph_rgb_camera_info_.push_back(mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>(node_, "/" + uav_name + "/rgb/camera_info"));
    ph_rgb_seg_camera_info_.push_back(mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>(node_, "/" + uav_name + "/rgb_segmented/camera_info"));
    ph_stereo_left_camera_info_.push_back(
        mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>(node_, "/" + uav_name + "/stereo/left/camera_info"));
    ph_stereo_right_camera_info_.push_back(
        mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>(node_, "/" + uav_name + "/stereo/right/camera_info"));

    // | -------------------- set LiDAR config -------------------- |
    {
      ueds_connector::LidarConfig lidarConfig{};

      lidarConfig.BeamHorRays  = lidar_horizontal_rays_;
      lidarConfig.BeamVertRays = lidar_vertical_rays_;
      lidarConfig.FOVHorLeft   = lidar_horizontal_fov_left_;
      lidarConfig.FOVHorRight  = lidar_horizontal_fov_right_;
      lidarConfig.FOVVertUp    = lidar_vertical_fov_up_;
      lidarConfig.FOVVertDown  = lidar_vertical_fov_down_;
      lidarConfig.beamLength   = lidar_beam_length_ * 100.0;
      lidarConfig.offset       = ueds_connector::Coordinates(lidar_offset_x_ * 100.0, lidar_offset_y_ * 100.0, lidar_offset_z_ * 100.0);
      lidarConfig.orientation  = ueds_connector::Rotation(-lidar_rotation_pitch_, lidar_rotation_yaw_, lidar_rotation_roll_);
      lidarConfig.showBeams    = lidar_show_beams_;
      lidarConfig.Livox       = lidar_livox_;

      const auto res = ueds_connectors_[i]->SetLidarConfig(lidarConfig);

      if (!res) {
        RCLCPP_ERROR(get_logger(), "failed to set lidar config for uav %lu", i + 1);
      } else {
        RCLCPP_INFO(get_logger(), "lidar config set for uav%lu", i + 1);
      }
    }

    // | ------------------ set RGB camera config ----------------- |

    {
      ueds_connector::RgbCameraConfig cameraConfig{};

      cameraConfig.width_                  = rgb_width_;
      cameraConfig.height_                 = rgb_height_;
      cameraConfig.fov_                    = rgb_fov_;
      cameraConfig.offset_                 = ueds_connector::Coordinates(rgb_offset_x_ * 100.0, rgb_offset_y_ * 100.0, rgb_offset_z_ * 100.0);
      cameraConfig.orientation_            = ueds_connector::Rotation(-rgb_rotation_pitch_, rgb_rotation_yaw_, rgb_rotation_roll_);
      cameraConfig.enable_raytracing_      = rgb_enable_raytracing_;
      cameraConfig.enable_hdr_             = rgb_enable_hdr_;
      cameraConfig.enable_temporal_aa_     = rgb_enable_temporal_aa_;
      cameraConfig.enable_motion_blur_     = rgb_enable_motion_blur_;
      cameraConfig.motion_blur_amount_     = rgb_motion_blur_amount_;
      cameraConfig.motion_blur_distortion_ = rgb_motion_blur_distortion_;

      const auto res = ueds_connectors_[i]->SetRgbCameraConfig(cameraConfig);

      last_rgb_ue_stamp_.push_back(0.0);
      last_rgb_seg_ue_stamp_.push_back(0.0);

      if (!res) {
        RCLCPP_ERROR(get_logger(), "failed to set camera config for uav %lu", i + 1);
      } else {
        RCLCPP_INFO(get_logger(), "camera config set for uav%lu", i + 1);
      }
    }
    
    // | ---------------- set Stereo camera config ---------------- |

    {
      ueds_connector::StereoCameraConfig cameraConfig{};

      cameraConfig.width_              = stereo_width_;
      cameraConfig.height_             = stereo_height_;
      cameraConfig.fov_                = stereo_fov_;
      cameraConfig.baseline_           = stereo_baseline_;
      cameraConfig.offset_             = ueds_connector::Coordinates(stereo_offset_x_ * 100.0, stereo_offset_y_ * 100.0, stereo_offset_z_ * 100.0);
      cameraConfig.orientation_        = ueds_connector::Rotation(-stereo_rotation_pitch_, stereo_rotation_yaw_, stereo_rotation_roll_);
      cameraConfig.enable_raytracing_  = stereo_enable_raytracing_;
      cameraConfig.enable_hdr_         = stereo_enable_hdr_;
      cameraConfig.enable_temporal_aa_ = stereo_enable_temporal_aa_;

      const auto res = ueds_connectors_[i]->SetStereoCameraConfig(cameraConfig);

      last_stereo_ue_stamp_.push_back(0.0);

      if (!res) {
        RCLCPP_ERROR(get_logger(), "failed to set camera config for uav %lu", i + 1);
      } else {
        RCLCPP_INFO(get_logger(), "camera config set for uav%lu", i + 1);
      }
    }


  }


  // | --------------- dynamic reconfigure server --------------- |

  if (!param_loader.loadedSuccessfully()) {
    RCLCPP_ERROR(node_->get_logger(), "could not load all parameters!");
    rclcpp::shutdown();
  }

  // | ---------------- bind param server callback -------------- |

  param_callback_handle_ = add_on_set_parameters_callback(std::bind(&FlightforgeSimulator::callbackParameters, this, std::placeholders::_1));

  // | ----------------------- publishers ----------------------- |

  ph_clock_ = mrs_lib::PublisherHandler<rosgraph_msgs::msg::Clock>(node_, "~/clock_out");

  ph_poses_ = mrs_lib::PublisherHandler<geometry_msgs::msg::PoseArray>(node_, "~/uav_poses_out");

  // | ------------------------- timers ------------------------- |
  
  mrs_lib::TimerHandlerOptions opts;

  opts.node      = node_;
  opts.autostart = true;

  timer_main_ = create_wall_timer(std::chrono::duration<double>(1.0 / (_clock_rate_ * drs_params_.realtime_factor)), std::bind(&FlightforgeSimulator::timerMain, this), cbgrp_main_);

  timer_status_ = create_wall_timer(std::chrono::duration<double>(1.0), std::bind(&FlightforgeSimulator::timerStatus, this), cbgrp_status_);

  timer_time_sync_ =  create_wall_timer(std::chrono::duration<double>(1.0), std::bind(&FlightforgeSimulator::timerTimeSync, this), cbgrp_status_);
  
  timer_unreal_sync_ =  create_wall_timer(std::chrono::duration<double>(1.0 / _clock_rate_), std::bind(&FlightforgeSimulator::timerUnrealSync, this), cbgrp_status_);
    
  if (drs_params_.rangefinder_rate > 0) {
    timer_rangefinder_ = create_wall_timer(std::chrono::duration<double>(1.0 / drs_params_.rangefinder_rate), std::bind(&FlightforgeSimulator::timerRangefinder, this), cbgrp_sensors_);
  }
  if (drs_params_.lidar_rate > 0) {
    timer_lidar_ = create_wall_timer(std::chrono::duration<double>(1.0 / drs_params_.lidar_rate), std::bind(&FlightforgeSimulator::timerLidar, this), cbgrp_sensors_);
  }
  if (drs_params_.lidar_seg_rate > 0) {
    timer_seg_lidar_ = create_wall_timer(std::chrono::duration<double>(1.0 / drs_params_.lidar_seg_rate), std::bind(&FlightforgeSimulator::timerSegLidar, this), cbgrp_sensors_);
  }
  if (drs_params_.lidar_int_rate > 0) {
    timer_int_lidar_ = create_wall_timer(std::chrono::duration<double>(1.0 / drs_params_.lidar_int_rate), std::bind(&FlightforgeSimulator::timerIntLidar, this), cbgrp_sensors_);
  }
  if (drs_params_.rgb_rate > 0) {
    timer_rgb_ = create_wall_timer(std::chrono::duration<double>(1.0 / drs_params_.rgb_rate), std::bind(&FlightforgeSimulator::timerRgb, this), cbgrp_sensors_);
  }
  if (drs_params_.stereo_rate > 0) {
    timer_stereo_ = create_wall_timer(std::chrono::duration<double>(1.0 / drs_params_.stereo_rate), std::bind(&FlightforgeSimulator::timerStereo, this), cbgrp_sensors_);
  }
  if (drs_params_.rgb_segmented_rate > 0) {
    timer_rgb_segmented_ = create_wall_timer(std::chrono::duration<double>(1.0 / drs_params_.rgb_segmented_rate), std::bind(&FlightforgeSimulator::timerRgbSegmented, this), cbgrp_sensors_);
  }
  /* if (drs_params_.rgb_depth_rate > 0) { */
  /*   timer_depth_ = create_wall_timer(std::chrono::duration<double>(1.0 / drs_params_.rgb_depth_rate), std::bind(&FlightforgeSimulator::timerDepth, this), cbgrp_sensors_); */
  /* } */




  // | ----------------------- scope timer ---------------------- |

  scope_timer_logger_ = std::make_shared<mrs_lib::ScopeTimerLogger>(node_, "", false);

  // | -------------------- finishing methods ------------------- |

  fabricateCamInfo();

  publishStaticTfs();
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

  RCLCPP_INFO_ONCE(node_->get_logger(),"Main timer spinning");

  // | ------------------ make simulation step ------------------ |

  double simulation_step_size = 1.0 / _simulation_rate_;
  double clock_step_size      = 1.0 / _clock_rate_;

  auto sim_time = mrs_lib::get_mutexed(mutex_sim_time_, sim_time_);

  sim_time = sim_time + rclcpp::Duration(std::chrono::duration<double>(clock_step_size));

  mrs_lib::set_mutexed(mutex_sim_time_, sim_time, sim_time_);

  const double dt_since_last_step = (sim_time - last_step_time_).seconds();

  if (dt_since_last_step >= simulation_step_size) {

    for (size_t i = 0; i < uavs_.size(); i++) {

      uavs_.at(i)->makeStep(dt_since_last_step, sim_time_.seconds());
    }

    publishPoses();

    /* handleCollisions(); */

    last_step_time_ = sim_time;
  }

  // | ---------------------- publish time ---------------------- |

  rosgraph_msgs::msg::Clock ros_time;

  ros_time.clock = sim_time;

  ph_clock_.publish(ros_time);
}

//}

/* timeStatus() //{ */

void FlightforgeSimulator::timerStatus() {

  if (!is_initialized_) {
    return;
  }

  auto sim_time   = mrs_lib::get_mutexed(mutex_sim_time_, sim_time_);
  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  rclcpp::Duration last_sec_sim_dt = sim_time - last_sim_time_status_;

  last_sim_time_status_ = sim_time;

  double last_sec_rtf = last_sec_sim_dt.seconds() / 1.0;

  actual_rtf_ = 0.9 * actual_rtf_ + 0.1 * last_sec_rtf;
  
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
  flightforge_fps_                     = fps_filter_const * flightforge_fps_ + (1.0 - fps_filter_const) * fps;

  // get the currently requires highest sensor rate
  double highest_fps = 0;

  if (drs_params.rgb_rate > highest_fps) {
    highest_fps = drs_params.rgb_rate;
  }

  if (drs_params.stereo_rate > highest_fps) {
    highest_fps = drs_params.stereo_rate;
  }

  if (drs_params.lidar_rate > highest_fps) {
    highest_fps = drs_params.lidar_rate;
  }

  if (drs_params.rgb_segmented_rate > highest_fps) {
    highest_fps = drs_params.rgb_segmented_rate;
  }

  if (drs_params.lidar_seg_rate > highest_fps) {
    highest_fps = drs_params.lidar_seg_rate;
  }

  if (drs_params.lidar_int_rate > highest_fps) {
    highest_fps = drs_params.lidar_int_rate;
  }

  const double flightforge_rtf = flightforge_fps_ / highest_fps;

  const double desired_rtf = (drs_params.dynamic_rtf && flightforge_rtf < drs_params.realtime_factor) ? flightforge_rtf : drs_params.realtime_factor;
  
  timer_main_->cancel();
  timer_main_ = create_wall_timer(std::chrono::duration<double>(1.0 / (_clock_rate_ * desired_rtf)), std::bind(&FlightforgeSimulator::timerMain, this), cbgrp_main_);
  
  if (_collisions_) {
    checkForCrash();
  }

  RCLCPP_INFO(node_->get_logger(), "%s, desired RTF = %.2f, actual RTF = %.2f, FlightForge FPS = %.2f", drs_params.paused ? "paused" : "running", drs_params.realtime_factor, actual_rtf_, flightforge_fps_);
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

  rclcpp::Time current_real = clock_->now();

  if (!is_initialized_) {
    return;
  }

  auto wall_time_offset = mrs_lib::get_mutexed(mutex_wall_time_offset_, wall_time_offset_);

  const double sync_start = clock_->now().seconds();

  bool   res;
  double flightforge_time;

  {
    std::scoped_lock lock(mutex_flightforge_);

    std::tie(res, flightforge_time) = ueds_game_controller_->GetTime();
  }

  const double sync_end = clock_->now().seconds();

  if (!res) {
    RCLCPP_ERROR(get_logger(), "Failed to get FlightForge's time");
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

    last_sync_time_ = clock_->now();
  }
  
  last_real_ = current_real;

  RCLCPP_DEBUG(get_logger(), "wall time %f flightforge %f time offset: %f, offset slope %f s/s", sync_start, flightforge_time, wall_time_offset_,
            wall_time_offset_drift_slope_);
}

//}

/*timerRangefinder()//{*/
void FlightforgeSimulator::timerRangefinder() {
  if (!is_initialized_) {
    return;
  }

  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  if (!drs_params.rangefinder_enabled) {
    return;
  }

  for (size_t i = 0; i < uavs_.size(); i++) {
    bool   res;
    double range;
    {
      std::scoped_lock lock(mutex_flightforge_);

      std::tie(res, range) = ueds_connectors_[i]->GetRangefinderData();
    }

    if (!res) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *node_->get_clock(), 1e9, "[uav%d] - ERROR GetRangefinderData", int(i) + 1);
      continue;
    }

    sensor_msgs::msg::Range msg_range;
    auto last_step_time = mrs_lib::get_mutexed(mutex_sim_time_, last_step_time_);
    msg_range.header.stamp    = last_step_time;
    msg_range.header.frame_id = "uav" + std::to_string(i + 1) + "/fcu";
    msg_range.radiation_type  = 1;
    msg_range.min_range       = 0.1;
    msg_range.max_range       = 30;
    msg_range.range           = range / 100;

    ph_rangefinders_[i].publish(msg_range);
  }
}
/*//}*/

/* timerLidar() //{ */

void FlightforgeSimulator::timerLidar() {

  if (!is_initialized_) {
    return;
  }

  /* mrs_lib::ScopeTimer timer = mrs_lib::ScopeTimer("timerLidar()"); */

  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  if (!drs_params_.lidar_enabled) {
    return;
  }

  for (size_t i = 0; i < uavs_.size(); i++) {

    // mrs_multirotor_simulator::MultirotorModel::State state = uavs_[i]->getState();

    bool                                   res;
    std::vector<ueds_connector::LidarData> lidarData;
    ueds_connector::Coordinates            start;

    {
      std::scoped_lock lock(mutex_flightforge_);

      std::tie(res, lidarData, start) = ueds_connectors_[i]->GetLidarData();
    }

    if (!res) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *node_->get_clock(), 1e9, "[uav%d] - ERROR getLidarData", int(i) + 1);
      continue;
    }

    sensor_msgs::msg::PointCloud2 pcl_msg;

    // Modifier to describe what the fields are.
    sensor_msgs::PointCloud2Modifier modifier(pcl_msg);
    modifier.setPointCloud2Fields(4, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1, sensor_msgs::msg::PointField::FLOAT32, "z", 1,
                                  sensor_msgs::msg::PointField::FLOAT32, "intensity", 1, sensor_msgs::msg::PointField::FLOAT32);
    // Msg header
    auto sim_time = mrs_lib::get_mutexed(mutex_sim_time_, sim_time_);
    auto last_step_time = mrs_lib::get_mutexed(mutex_sim_time_, last_step_time_);
    pcl_msg.header.stamp    = last_step_time;
    pcl_msg.header.frame_id = "uav" + std::to_string(i + 1) + "/lidar";
    pcl_msg.height   = lidar_vertical_rays_;
    pcl_msg.width    = lidar_horizontal_rays_;
    pcl_msg.is_dense = true;

    // Total number of bytes per point
    pcl_msg.point_step = 16;
    pcl_msg.row_step   = pcl_msg.point_step * pcl_msg.width;
    pcl_msg.data.resize(pcl_msg.row_step * pcl_msg.height);

    sensor_msgs::PointCloud2Iterator<float> iterX(pcl_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iterY(pcl_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iterZ(pcl_msg, "z");
    sensor_msgs::PointCloud2Iterator<float> iterIntensity(pcl_msg, "intensity");

    for (const ueds_connector::LidarData& ray : lidarData) {

      tf2::Vector3 dir = tf2::Vector3(ray.directionX, ray.directionY, ray.directionZ);

      double ray_distance = ray.distance / 100.0;

      if (drs_params.lidar_noise_enabled && ray_distance > 0) {

        const double std = ray_distance * drs_params.lidar_std_slope * drs_params.lidar_std_at_1m;

        std::normal_distribution<double> distribution(0, std);

        ray_distance += distribution(rng);
      }

      dir = dir.normalized() * ray_distance;

      *iterX         = dir.x();
      *iterY         = -dir.y();  // convert left-hand to right-hand coordinates
      *iterZ         = dir.z();
      *iterIntensity = ray.distance;

      ++iterX;
      ++iterY;
      ++iterZ;
      ++iterIntensity;
    }

    ph_lidars_[i].publish(pcl_msg);
  }
}

//}

/* timerSegLidar() //{ */

void FlightforgeSimulator::timerSegLidar() {

  if (!is_initialized_) {
    return;
  }

  /* mrs_lib::ScopeTimer timer = mrs_lib::ScopeTimer("timerSegLidar()"); */

  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  if (!drs_params_.lidar_seg_enabled) {
    return;
  }

  for (size_t i = 0; i < uavs_.size(); i++) {

    // mrs_multirotor_simulator::MultirotorModel::State state = uavs_[i]->getState();

    bool                                   res;
    std::vector<ueds_connector::LidarSegData> lidarSegData;
    ueds_connector::Coordinates            start;

    {
      std::scoped_lock lock(mutex_flightforge_);

      std::tie(res, lidarSegData, start) = ueds_connectors_[i]->GetLidarSegData();
    }

    if (!res) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *node_->get_clock(), 1e9, "[uav%d] - ERROR getLidarSegData", int(i) + 1);
      continue;
    }
    
    PCLPointCloudColor pcl_cloud;

    for (const ueds_connector::LidarSegData& ray : lidarSegData) {

      pcl::PointXYZRGB point;
      tf2::Vector3 dir = tf2::Vector3(ray.directionX, ray.directionY, ray.directionZ);

      double ray_distance = ray.distance / 100.0;

      if (drs_params.lidar_noise_enabled && ray_distance > 0) {

        const double std = ray_distance * drs_params.lidar_std_slope * drs_params.lidar_std_at_1m;

        std::normal_distribution<double> distribution(0, std);

        ray_distance += distribution(rng);
      }

      dir = dir.normalized() * ray_distance;
      point.x = dir.x();
      point.y = -dir.y();
      point.z = dir.z();
      point.r = seg_rgb_[ray.segmentation][0];
      point.g = seg_rgb_[ray.segmentation][1];
      point.b = seg_rgb_[ray.segmentation][2];
      pcl_cloud.push_back(point);
    }

    sensor_msgs::msg::PointCloud2 pcl_msg;
    pcl::toROSMsg(pcl_cloud, pcl_msg);
    auto last_step_time = mrs_lib::get_mutexed(mutex_sim_time_, last_step_time_);
    pcl_msg.header.stamp    = last_step_time;
    pcl_msg.header.frame_id = "uav" + std::to_string(i + 1) + "/lidar";
    ph_seg_lidars_[i].publish(pcl_msg);
  }
}

//}

/* timerIntLidar() //{ */

void FlightforgeSimulator::timerIntLidar() {

  if (!is_initialized_) {
    return;
  }

  /* mrs_lib::ScopeTimer timer = mrs_lib::ScopeTimer("timerIntLidar()"); */

  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  if (!drs_params_.lidar_enabled) {
    return;
  }

  for (size_t i = 0; i < uavs_.size(); i++) {

    // mrs_multirotor_simulator::MultirotorModel::State state = uavs_[i]->getState();

    bool                                   res;
    std::vector<ueds_connector::LidarData> lidarData;
    ueds_connector::Coordinates            start;

    {
      std::scoped_lock lock(mutex_flightforge_);

      std::tie(res, lidarData, start) = ueds_connectors_[i]->GetLidarData();
    }

    if (!res) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *node_->get_clock(), 1e9, "[uav%d] - ERROR getLidarData", int(i) + 1);
      continue;
    }

    sensor_msgs::msg::PointCloud2 pcl_msg;

    // Modifier to describe what the fields are.
    sensor_msgs::PointCloud2Modifier modifier(pcl_msg);
    modifier.setPointCloud2Fields(4, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1, sensor_msgs::msg::PointField::FLOAT32, "z", 1,
                                  sensor_msgs::msg::PointField::FLOAT32, "intensity", 1, sensor_msgs::msg::PointField::FLOAT32);
    // Msg header
    auto last_step_time = mrs_lib::get_mutexed(mutex_sim_time_, last_step_time_);
    pcl_msg.header.stamp    = last_step_time;
    pcl_msg.header.frame_id = "uav" + std::to_string(i + 1) + "/lidar";
    pcl_msg.height   = lidar_vertical_rays_;
    pcl_msg.width    = lidar_horizontal_rays_;
    pcl_msg.is_dense = true;

    // Total number of bytes per point
    pcl_msg.point_step = 16;
    pcl_msg.row_step   = pcl_msg.point_step * pcl_msg.width;
    pcl_msg.data.resize(pcl_msg.row_step * pcl_msg.height);

    sensor_msgs::PointCloud2Iterator<float> iterX(pcl_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iterY(pcl_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iterZ(pcl_msg, "z");
    sensor_msgs::PointCloud2Iterator<float> iterIntensity(pcl_msg, "intensity");

    for (const ueds_connector::LidarData& ray : lidarData) {

      tf2::Vector3 dir = tf2::Vector3(ray.directionX, ray.directionY, ray.directionZ);

      double ray_distance = ray.distance / 100.0;

      if (drs_params.lidar_noise_enabled && ray_distance > 0) {

        const double std = ray_distance * drs_params.lidar_std_slope * drs_params.lidar_std_at_1m;

        std::normal_distribution<double> distribution(0, std);

        ray_distance += distribution(rng);
      }

      dir = dir.normalized() * ray_distance;

      *iterX         = dir.x();
      *iterY         = -dir.y();  // convert left-hand to right-hand coordinates
      *iterZ         = dir.z();
      *iterIntensity = ray.distance;

      ++iterX;
      ++iterY;
      ++iterZ;
      ++iterIntensity;
    }

    ph_lidars_[i].publish(pcl_msg);
  }
}

//}

/* timerRgb() //{ */

void FlightforgeSimulator::timerRgb() {

  if (!is_initialized_) {
    return;
  }

  /* mrs_lib::ScopeTimer timer = mrs_lib::ScopeTimer("timerRgb()"); */

  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  if (!drs_params.rgb_enabled) {
    return;
  }

  for (size_t i = 0; i < uavs_.size(); i++) {

    bool                       res;
    std::vector<unsigned char> cameraData;
    uint32_t                   size;
    double                     stamp;

    {
      std::scoped_lock lock(mutex_flightforge_);

      std::tie(res, cameraData, stamp, size) = ueds_connectors_[i]->GetRgbCameraData();
    }

    if (abs(stamp - last_rgb_ue_stamp_.at(i)) < 0.001) {
      return;
    }

    last_rgb_ue_stamp_.at(i) = stamp;

    if (!res) {
      RCLCPP_WARN(get_logger(), "failed to obtain rgb camera from uav%lu", i + 1);
      continue;
    }

    if (cameraData.empty()) {
      RCLCPP_WARN(get_logger(), "rgb camera from uav%lu is empty!", i + 1);
      continue;
    }

    cv::Mat image = cv::imdecode(cameraData, cv::IMREAD_COLOR);

    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image).toImageMsg();

    msg->header.frame_id = "uav" + std::to_string(i + 1) + "/rgb";

    const double relative_wall_age = clock_->now().seconds() - flightforgeToWallTime(stamp);

    if (abs(relative_wall_age) < 1.0) {
      rclcpp::Time shifted_time_stamp = rclcpp::Time(clock_->now().seconds() - (relative_wall_age * actual_rtf_));
      msg->header.stamp = shifted_time_stamp; 
    }

    imp_rgb_[i].publish(msg);

    auto camera_info = rgb_camera_info_;

    camera_info.header = msg->header;

    ph_rgb_camera_info_[i].publish(camera_info);
  }
}

//}

/* timerStereo() //{ */

void FlightforgeSimulator::timerStereo() {

  if (!is_initialized_) {
    return;
  }

  /* mrs_lib::ScopeTimer timer = mrs_lib::ScopeTimer("timerStereo()"); */

  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  if (!drs_params.rgb_enabled) {
    return;
  }

  for (size_t i = 0; i < uavs_.size(); i++) {

    bool                       res;
    std::vector<unsigned char> image_left;
    std::vector<unsigned char> image_right;
    double                     stamp;

    {
      std::scoped_lock lock(mutex_flightforge_);

      std::tie(res, image_left, image_right, stamp) = ueds_connectors_[i]->GetStereoCameraData();
    }

    if (abs(stamp - last_stereo_ue_stamp_.at(i)) < 0.001) {
      return;
    }

    last_stereo_ue_stamp_.at(i) = stamp;

    if (!res) {
      RCLCPP_WARN(get_logger(), "failed to obtain stereo camera from uav%lu", i + 1);
      continue;
    }

    if (image_left.empty() || image_right.empty()) {
      RCLCPP_WARN(get_logger(), "stereo camera from uav%lu is empty!", i + 1);
      continue;
    }

    cv::Mat cv_left  = cv::imdecode(image_left, cv::IMREAD_COLOR);
    cv::Mat cv_right = cv::imdecode(image_right, cv::IMREAD_COLOR);

    auto msg_left = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", cv_left).toImageMsg();
    auto msg_right = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", cv_right).toImageMsg();

    msg_left->header.frame_id = "uav" + std::to_string(i + 1) + "/stereo_left";

    msg_right->header.frame_id = "uav" + std::to_string(i + 1) + "/stereo_right";

    const double relative_wall_age = clock_->now().seconds() - flightforgeToWallTime(stamp);

    if (abs(relative_wall_age) < 1.0) {
      rclcpp::Time shifted_time_stamp = rclcpp::Time(clock_->now().seconds() - (relative_wall_age * actual_rtf_));
      msg_right->header.stamp = shifted_time_stamp; 
    }
    msg_left->header.stamp    = msg_right->header.stamp;

    imp_stereo_left_[i].publish(msg_left);
    imp_stereo_right_[i].publish(msg_right);


    {
      auto camera_info = stereo_camera_info_;

      camera_info.header = msg_left->header;

      ph_stereo_left_camera_info_[i].publish(camera_info);
    }

    {
      auto camera_info = stereo_camera_info_;

      camera_info.header = msg_right->header;

      camera_info.p[3] = -camera_info.p[0] * stereo_baseline_;

      ph_stereo_right_camera_info_[i].publish(camera_info);
    }
  }
}

//}

/* timerRgbSegmented() //{ */

void FlightforgeSimulator::timerRgbSegmented() {

  if (!is_initialized_) {
    return;
  }

  /* mrs_lib::ScopeTimer timer = mrs_lib::ScopeTimer("timerRgbSegmented()"); */

  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  if (!drs_params.rgb_segmented_enabled) {
    return;
  }

  for (size_t i = 0; i < uavs_.size(); i++) {

    bool                       res;
    std::vector<unsigned char> cameraData;
    uint32_t                   size;
    double                     stamp;

    {
      std::scoped_lock lock(mutex_flightforge_);

      std::tie(res, cameraData, stamp, size) = ueds_connectors_[i]->GetRgbSegmented();
    }

    if (abs(stamp - last_rgb_seg_ue_stamp_.at(i)) < 0.001) {
      return;
    }

    last_rgb_seg_ue_stamp_.at(i) = stamp;

    if (!res) {
      RCLCPP_WARN(get_logger(), "failed to obtain segmented camera from uav%lu", i + 1);
      continue;
    }

    if (cameraData.empty()) {
      RCLCPP_WARN(get_logger(), "segmented camera from uav%lu is empty!", i + 1);
      continue;
    }

    cv::Mat image = cv::Mat(rgb_height_, rgb_width_, CV_8UC3, cameraData.data());

    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image).toImageMsg();

    msg->header.frame_id = "uav" + std::to_string(i + 1) + "/rgb";

    /* const double relative_wall_age = clock_->now().seconds() - flightforgeToWallTime(stamp); */

    /* if (abs(relative_wall_age) < 1.0) { */
    /*   rclcpp::Time shifted_time_stamp = rclcpp::Time(clock_->now().seconds() - (relative_wall_age * actual_rtf_)); */
    /*   msg->header.stamp = shifted_time_stamp; */ 
    /* } */

    msg->header.stamp = clock_->now();

    imp_rgbd_segmented_[i].publish(msg);

    auto camera_info = rgb_camera_info_;

    camera_info.header = msg->header;

    ph_rgb_seg_camera_info_[i].publish(camera_info);
  }
}

//}

/* timerDepth() //{ */

void FlightforgeSimulator::timerDepth() {

  if (!is_initialized_) {
    return;
  }

  /* mrs_lib::ScopeTimer timer = mrs_lib::ScopeTimer("timerDepth()"); */

  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  if (!drs_params.rgb_depth_enabled) {
    return;
  }

  for (size_t i = 0; i < uavs_.size(); i++) {

    bool                       res;
    std::vector<uint16_t>      cameraData;
    uint32_t                   size;
    double                     stamp;

    {
      std::scoped_lock lock(mutex_flightforge_);

      std::tie(res, cameraData, stamp, size) = ueds_connectors_[i]->GetDepthCameraData();
    }

    if (abs(stamp - last_rgb_ue_stamp_.at(i)) < 0.001) {
      return;
    }

    last_rgb_ue_stamp_.at(i) = stamp;

    if (!res) {
      RCLCPP_WARN(get_logger(), "failed to obtain depth camera from uav%lu", i + 1);
      continue;
    }

    if (cameraData.empty()) {
      RCLCPP_WARN(get_logger(), "depth camera from uav%lu is empty!", i + 1);
      continue;
    }

    cv::Mat image = cv::Mat(rgb_height_, rgb_width_, CV_16UC1, cameraData.data());

    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono16", image).toImageMsg();

    msg->header.frame_id = "uav" + std::to_string(i + 1) + "/rgb";

    const double relative_wall_age = clock_->now().seconds() - flightforgeToWallTime(stamp);

    if (abs(relative_wall_age) < 1.0) {
      rclcpp::Time shifted_time_stamp = rclcpp::Time(clock_->now().seconds() - (relative_wall_age * actual_rtf_));
      msg->header.stamp = shifted_time_stamp; 
    }

    msg->header.stamp = clock_->now();

    imp_rgb_[i].publish(msg);

    auto camera_info = rgb_camera_info_;

    camera_info.header = msg->header;

    ph_depth_camera_info_[i].publish(camera_info);
  }
}

//}

// | ------------------------ callbacks ----------------------- |

/* callbackParameters() //{ */

rcl_interfaces::msg::SetParametersResult FlightforgeSimulator::callbackParameters(std::vector<rclcpp::Parameter> parameters) {

  rcl_interfaces::msg::SetParametersResult result;

  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  // Note that setting a parameter to a nonsensical value (such as setting the `param_namespace.floating_number` parameter to `hello`)
  // doesn't have any effect - it doesn't even call this callback.
  for (auto& param : parameters) {

    RCLCPP_INFO_STREAM(node_->get_logger(), "got parameter: '" << param.get_name() << "' with value '" << param.value_to_string() << "'");

    if (param.get_name() == "dynamic/paused") {

      if (drs_params.paused && !param.as_bool()) {

        timer_main_ = create_wall_timer(std::chrono::duration<double>(1.0 / (_clock_rate_ * drs_params.realtime_factor)), std::bind(&FlightforgeSimulator::timerMain, this), cbgrp_main_);

        timer_status_ = create_wall_timer(std::chrono::duration<double>(1.0), std::bind(&FlightforgeSimulator::timerStatus, this), cbgrp_status_);

      } else if (!drs_params.paused && param.as_bool()) {
        timer_main_->cancel();
        timer_status_->cancel();
      }

      drs_params.paused = param.as_bool();

    } else if (param.get_name() == "dynamic/realtime_factor") {

      drs_params.realtime_factor = param.as_double();

      timer_main_->cancel();

      timer_main_ = create_wall_timer(std::chrono::duration<double>(1.0 / (_clock_rate_ * drs_params.realtime_factor)), std::bind(&FlightforgeSimulator::timerMain, this), cbgrp_main_);

    } else if (param.get_name() == "dynamic/collisions_crash") {

      drs_params.collisions_crash = param.as_bool();

    } else if (param.get_name() == "dynamic/collisions_enabled") {

      drs_params.collisions_enabled = param.as_bool();

    } else if (param.get_name() == "dynamic/collisions_rebounce") {

      drs_params.collisions_rebounce = param.as_double();

    } else {

      RCLCPP_WARN_STREAM(node_->get_logger(), "parameter: '" << param.get_name() << "' is not dynamically reconfigurable!");
      result.successful = false;
      result.reason     = "Parameter '" + param.get_name() + "' is not dynamically reconfigurable!";
      return result;
    }
  }

  RCLCPP_INFO(node_->get_logger(), "params updated");
  result.successful = true;
  result.reason     = "OK";

  mrs_lib::set_mutexed(mutex_drs_params_, drs_params, drs_params_);

  return result;
}

//}

// | ------------------------ routines ------------------------ |

/* /1* handleCollisions() //{ *1/ */

/* void FlightforgeSimulator::handleCollisions(void) { */

/*   auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_); */

/*   if (!(drs_params.collisions_crash || drs_params.collisions_enabled)) { */
/*     return; */
/*   } */

/*   std::vector<Eigen::VectorXd> poses; */

/*   for (size_t i = 0; i < uavs_.size(); i++) { */
/*     poses.push_back(uavs_.at(i)->getPose()); */
/*   } */

/*   typedef std::vector<Eigen::VectorXd> my_vector_of_vectors_t; */

/*   typedef KDTreeVectorOfVectorsAdaptor<my_vector_of_vectors_t, double> my_kd_tree_t; */

/*   my_kd_tree_t mat_index(3, poses, 10); */

/*   std::vector<nanoflann::ResultItem<int, double>> indices_dists; */

/*   std::vector<Eigen::Vector3d> forces; */

/*   for (size_t i = 0; i < uavs_.size(); i++) { */
/*     forces.push_back(Eigen::Vector3d::Zero()); */
/*   } */

/*   for (size_t i = 0; i < uavs_.size(); i++) { */

/*     MultirotorModel::State       state_1  = uavs_.at(i)->getState(); */
/*     MultirotorModel::ModelParams params_1 = uavs_.at(i)->getParams(); */

/*     nanoflann::RadiusResultSet<double, int> resultSet(3.0, indices_dists); */

/*     mat_index.index->findNeighbors(resultSet, &state_1.x(0)); */

/*     for (size_t j = 0; j < resultSet.m_indices_dists.size(); j++) { */

/*       const size_t idx  = resultSet.m_indices_dists.at(j).first; */
/*       const double dist = resultSet.m_indices_dists.at(j).second; */

/*       if (idx == i) { */
/*         continue; */
/*       } */

/*       MultirotorModel::State       state_2  = uavs_.at(idx)->getState(); */
/*       MultirotorModel::ModelParams params_2 = uavs_.at(idx)->getParams(); */

/*       const double crit_dist = params_1.arm_length + params_1.prop_radius + params_2.arm_length + params_2.prop_radius; */

/*       const Eigen::Vector3d rel_pos = state_1.x - state_2.x; */

/*       if (dist < crit_dist) { */
/*         if (drs_params.collisions_crash && !uavs_.at(idx)->hasCrashed()) { */

/*           RCLCPP_WARN(get_logger(), "uav%u crashed", int(idx+1)); */

/*           uavs_.at(idx)->crash(); */

/*         } else { */
/*           forces.at(i) += drs_params.collisions_rebounce * rel_pos.normalized() * params_1.mass * (params_2.mass / (params_1.mass + params_2.mass)); */
/*         } */
/*       } */
/*     } */
/*   } */

/*   for (size_t i = 0; i < uavs_.size(); i++) { */
/*     uavs_.at(i)->applyForce(forces.at(i)); */
/*   } */
/* } */

/* //} */

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
        /*         RCLCPP_INFO(get_logger(), "Successfully set ground_z for uav%lu to ", i + 1 ); */
        /*       } else { */
        /*         RCLCPP_ERROR(get_logger(), "Failed to set ground_z for uav%lu: %s", i + 1, srv.response.message.c_str()); */
        /*       } */
        /*     } else { */
        /*       RCLCPP_ERROR(get_logger(), "Failed to call set_ground_z service for uav%lu", i + 1); */
        /*     } */
        /*   } else { */
        /*     RCLCPP_WARN(get_logger(), "Set_ground_z service for uav%lu does not exist", i + 1); */
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

/* publishStaticTfs() //{ */

void FlightforgeSimulator::publishStaticTfs(void) {

  for (size_t i = 0; i < uavs_.size(); i++) {

    geometry_msgs::msg::TransformStamped tf;

    /* // | ------------------------- rgb tf ------------------------- | */

    {
      tf.header.stamp = clock_->now(); 

      tf.header.frame_id = "uav" + std::to_string(i + 1) + "/fcu";
      tf.child_frame_id  = "uav" + std::to_string(i + 1) + "/rgb";

      tf.transform.translation.x = rgb_offset_x_;
      tf.transform.translation.y = rgb_offset_y_;
      tf.transform.translation.z = rgb_offset_z_;

      Eigen::Matrix3d initial_tf = mrs_lib::AttitudeConverter(Eigen::Quaterniond(-0.5, 0.5, -0.5, 0.5));

      Eigen::Matrix3d dynamic_tf =
          mrs_lib::AttitudeConverter(M_PI * (rgb_rotation_roll_ / 180.0), M_PI * (rgb_rotation_pitch_ / 180.0), M_PI * (rgb_rotation_yaw_ / 180.0));

      Eigen::Matrix3d final_tf = dynamic_tf * initial_tf;

      tf.transform.rotation = mrs_lib::AttitudeConverter(final_tf);

      try {
        static_broadcaster_->sendTransform(tf);
      }
      catch (...) {
        RCLCPP_ERROR(node_->get_logger(), "Could not publish RGB tf");
      }

      // | ------------- print the tf matrix for kalibr ------------- |

      RCLCPP_INFO(node_->get_logger(), "RGB camera-imu chain for kalibr config:");
      printf("cam0:\n");
      printf("  T_imu_cam:\n");
      printf("    - [%f, %f, %f, %f]\n", final_tf(0, 0), final_tf(0, 1), final_tf(0, 2), tf.transform.translation.x);
      printf("    - [%f, %f, %f, %f]\n", final_tf(1, 0), final_tf(1, 1), final_tf(1, 2), tf.transform.translation.y);
      printf("    - [%f, %f, %f, %f]\n", final_tf(2, 0), final_tf(2, 1), final_tf(2, 2), tf.transform.translation.z);
      printf("    - [%f, %f, %f, %f]\n", 0.0, 0.0, 0.0, 1.0);
      printf("  cam_overlaps: [0]\n");
      printf("  camera_model: pinhole\n");
      printf("  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n");
      printf("  distortion_model: radtan\n");
      printf("  intrinsics: [%f, %f, %f, %f]\n", rgb_camera_info_.k[0], rgb_camera_info_.k[4], rgb_camera_info_.k[2], rgb_camera_info_.k[5]);
      printf("  resolution: [%d, %d]\n", rgb_width_, rgb_height_);
    }

    /* // | ----------------------- stereo left ---------------------- | */

    {
      tf.header.stamp = clock_->now();

      tf.header.frame_id = "uav" + std::to_string(i + 1) + "/fcu";
      tf.child_frame_id  = "uav" + std::to_string(i + 1) + "/stereo_left";

      tf.transform.translation.x = stereo_offset_x_;
      tf.transform.translation.y = stereo_offset_y_;
      tf.transform.translation.z = stereo_offset_z_;

      Eigen::Matrix3d initial_tf = mrs_lib::AttitudeConverter(Eigen::Quaterniond(-0.5, 0.5, -0.5, 0.5));

      Eigen::Matrix3d dynamic_tf =
          mrs_lib::AttitudeConverter(M_PI * (stereo_rotation_roll_ / 180.0), M_PI * (stereo_rotation_pitch_ / 180.0), M_PI * (stereo_rotation_yaw_ / 180.0));

      Eigen::Matrix3d final_tf = dynamic_tf * initial_tf;

      tf.transform.rotation = mrs_lib::AttitudeConverter(final_tf);

      try {
        static_broadcaster_->sendTransform(tf);
      }
      catch (...) {
        RCLCPP_ERROR(node_->get_logger(), "Could not publish stereo left tf");
      }

      // | ------------- print the tf matrix for kalibr ------------- |

      RCLCPP_INFO(node_->get_logger(), "Stereo left camera-imu chain for kalibr config:");
      printf("cam0:\n");
      printf("  T_imu_cam:\n");
      printf("    - [%f, %f, %f, %f]\n", final_tf(0, 0), final_tf(0, 1), final_tf(0, 2), tf.transform.translation.x);
      printf("    - [%f, %f, %f, %f]\n", final_tf(1, 0), final_tf(1, 1), final_tf(1, 2), tf.transform.translation.y);
      printf("    - [%f, %f, %f, %f]\n", final_tf(2, 0), final_tf(2, 1), final_tf(2, 2), tf.transform.translation.z);
      printf("    - [%f, %f, %f, %f]\n", 0.0, 0.0, 0.0, 1.0);
      printf("  cam_overlaps: [0]\n");
      printf("  camera_model: pinhole\n");
      printf("  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n");
      printf("  distortion_model: radtan\n");
      printf("  intrinsics: [%f, %f, %f, %f]\n", stereo_camera_info_.k[0], stereo_camera_info_.k[4], stereo_camera_info_.k[2], stereo_camera_info_.k[5]);
      printf("  resolution: [%d, %d]\n", stereo_width_, stereo_height_);
    }

    {
      tf.header.stamp = clock_->now();

      tf.header.frame_id = "uav" + std::to_string(i + 1) + "/fcu";
      tf.child_frame_id  = "uav" + std::to_string(i + 1) + "/stereo_right";

      tf.transform.translation.x = stereo_offset_x_;
      tf.transform.translation.y = stereo_offset_y_ - stereo_baseline_;
      tf.transform.translation.z = stereo_offset_z_;

      Eigen::Matrix3d initial_tf = mrs_lib::AttitudeConverter(Eigen::Quaterniond(-0.5, 0.5, -0.5, 0.5));

      Eigen::Matrix3d dynamic_tf =
          mrs_lib::AttitudeConverter(M_PI * (stereo_rotation_roll_ / 180.0), M_PI * (stereo_rotation_pitch_ / 180.0), M_PI * (stereo_rotation_yaw_ / 180.0));

      Eigen::Matrix3d final_tf = dynamic_tf * initial_tf;

      tf.transform.rotation = mrs_lib::AttitudeConverter(final_tf);

      try {
        static_broadcaster_->sendTransform(tf);
      }
      catch (...) {
        RCLCPP_ERROR(node_->get_logger(), "Could not publish stereo right tf");
      }

      // | ------------- print the tf matrix for kalibr ------------- |

      RCLCPP_INFO(node_->get_logger(), "Stereo right camera-imu chain for kalibr config:");
      printf("cam0:\n");
      printf("  T_imu_cam:\n");
      printf("    - [%f, %f, %f, %f]\n", final_tf(0, 0), final_tf(0, 1), final_tf(0, 2), tf.transform.translation.x);
      printf("    - [%f, %f, %f, %f]\n", final_tf(1, 0), final_tf(1, 1), final_tf(1, 2), tf.transform.translation.y);
      printf("    - [%f, %f, %f, %f]\n", final_tf(2, 0), final_tf(2, 1), final_tf(2, 2), tf.transform.translation.z);
      printf("    - [%f, %f, %f, %f]\n", 0.0, 0.0, 0.0, 1.0);
      printf("  cam_overlaps: [0]\n");
      printf("  camera_model: pinhole\n");
      printf("  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n");
      printf("  distortion_model: radtan\n");
      printf("  intrinsics: [%f, %f, %f, %f]\n", stereo_camera_info_.k[0], stereo_camera_info_.k[4], stereo_camera_info_.k[2], stereo_camera_info_.k[5]);
      printf("  resolution: [%d, %d]\n", stereo_width_, stereo_height_);
    }


    // | ------------------------- lidar tf ------------------------- |
    {
      tf.header.stamp = clock_->now();
      tf.header.frame_id = "uav" + std::to_string(i + 1) + "/fcu";
      tf.child_frame_id  = "uav" + std::to_string(i + 1) + "/lidar";

      tf.transform.translation.x = lidar_offset_x_;
      tf.transform.translation.y = lidar_offset_y_;
      tf.transform.translation.z = lidar_offset_z_;

      mrs_lib::AttitudeConverter ac(M_PI * (lidar_rotation_roll_ / 180.0), M_PI * (lidar_rotation_pitch_ / 180.0), M_PI * (lidar_rotation_yaw_ / 180.0));
      tf.transform.rotation = ac;
      static_broadcaster_->sendTransform(tf);

      try {
        static_broadcaster_->sendTransform(tf);
      }
      catch (...) {
        RCLCPP_ERROR(node_->get_logger(), "Could not publish lidar tf");
      }
    }
  }
}

//}

/* flightforgeToWallTime() //{ */

double FlightforgeSimulator::flightforgeToWallTime(const double flightforge_time) {

  auto wall_time_offset = mrs_lib::get_mutexed(mutex_wall_time_offset_, wall_time_offset_);

  return flightforge_time + wall_time_offset;
}

//}

// | ------------------------ camera routines ------------------------ |

/* fabricateCamInfo() //{ */

void FlightforgeSimulator::fabricateCamInfo(void) {

  // | --------------------------- RGB -------------------------- |

  rgb_camera_info_.height = rgb_width_;
  rgb_camera_info_.width  = rgb_height_;

  // distortion
  rgb_camera_info_.distortion_model = "plumb_bob";

  rgb_camera_info_.d.resize(5);
  rgb_camera_info_.d[0] = 0;
  rgb_camera_info_.d[1] = 0;
  rgb_camera_info_.d[2] = 0;
  rgb_camera_info_.d[3] = 0;
  rgb_camera_info_.d[4] = 0;

  // original camera matrix
  rgb_camera_info_.k[0] = rgb_width_ / (2.0 * tan(0.5 * M_PI * (rgb_fov_ / 180.0)));
  rgb_camera_info_.k[1] = 0.0;
  rgb_camera_info_.k[2] = rgb_width_ / 2.0;
  rgb_camera_info_.k[3] = 0.0;
  rgb_camera_info_.k[4] = rgb_width_ / (2.0 * tan(0.5 * M_PI * (rgb_fov_ / 180.0)));
  rgb_camera_info_.k[5] = rgb_height_ / 2.0;
  rgb_camera_info_.k[6] = 0.0;
  rgb_camera_info_.k[7] = 0.0;
  rgb_camera_info_.k[8] = 1.0;

  // rectification
  rgb_camera_info_.r[0] = 1.0;
  rgb_camera_info_.r[1] = 0.0;
  rgb_camera_info_.r[2] = 0.0;
  rgb_camera_info_.r[3] = 0.0;
  rgb_camera_info_.r[4] = 1.0;
  rgb_camera_info_.r[5] = 0.0;
  rgb_camera_info_.r[6] = 0.0;
  rgb_camera_info_.r[7] = 0.0;
  rgb_camera_info_.r[8] = 1.0;

  // camera projection matrix (same as camera matrix due to lack of distortion/rectification) (is this generated?)
  rgb_camera_info_.p[0]  = rgb_width_ / (2.0 * tan(0.5 * M_PI * (rgb_fov_ / 180.0)));
  rgb_camera_info_.p[1]  = 0.0;
  rgb_camera_info_.p[2]  = rgb_width_ / 2.0;
  rgb_camera_info_.p[3]  = 0.0;
  rgb_camera_info_.p[4]  = 0.0;
  rgb_camera_info_.p[5]  = rgb_width_ / (2.0 * tan(0.5 * M_PI * (rgb_fov_ / 180.0)));
  rgb_camera_info_.p[6]  = rgb_height_ / 2.0;
  rgb_camera_info_.p[7]  = 0.0;
  rgb_camera_info_.p[8]  = 0.0;
  rgb_camera_info_.p[9]  = 0.0;
  rgb_camera_info_.p[10] = 1.0;
  rgb_camera_info_.p[11] = 0.0;

  // | ------------------------- stereo ------------------------- |

  stereo_camera_info_.width  = stereo_width_;
  stereo_camera_info_.height = stereo_height_;

  // distortion
  stereo_camera_info_.distortion_model = "plumb_bob";

  stereo_camera_info_.d.resize(5);
  stereo_camera_info_.d[0] = 0;
  stereo_camera_info_.d[1] = 0;
  stereo_camera_info_.d[2] = 0;
  stereo_camera_info_.d[3] = 0;
  stereo_camera_info_.d[4] = 0;

  // original camera matrix
  stereo_camera_info_.k[0] = stereo_width_ / (2.0 * tan(0.5 * M_PI * (stereo_fov_ / 180.0)));
  stereo_camera_info_.k[1] = 0.0;
  stereo_camera_info_.k[2] = stereo_width_ / 2.0;
  stereo_camera_info_.k[3] = 0.0;
  stereo_camera_info_.k[4] = stereo_width_ / (2.0 * tan(0.5 * M_PI * (stereo_fov_ / 180.0)));
  stereo_camera_info_.k[5] = stereo_height_ / 2.0;
  stereo_camera_info_.k[6] = 0.0;
  stereo_camera_info_.k[7] = 0.0;
  stereo_camera_info_.k[8] = 1.0;

  // rectification
  stereo_camera_info_.r[0] = 1.0;
  stereo_camera_info_.r[1] = 0.0;
  stereo_camera_info_.r[2] = 0.0;
  stereo_camera_info_.r[3] = 0.0;
  stereo_camera_info_.r[4] = 1.0;
  stereo_camera_info_.r[5] = 0.0;
  stereo_camera_info_.r[6] = 0.0;
  stereo_camera_info_.r[7] = 0.0;
  stereo_camera_info_.r[8] = 1.0;

  // camera projection matrix (same as camera matrix due to lack of distortion/rectification) (is this generated?)
  stereo_camera_info_.p[0]  = stereo_width_ / (2.0 * tan(0.5 * M_PI * (stereo_fov_ / 180.0)));
  stereo_camera_info_.p[1]  = 0.0;
  stereo_camera_info_.p[2]  = stereo_width_ / 2.0;
  stereo_camera_info_.p[3]  = 0.0;
  stereo_camera_info_.p[4]  = 0.0;
  stereo_camera_info_.p[5]  = stereo_width_ / (2.0 * tan(0.5 * M_PI * (stereo_fov_ / 180.0)));
  stereo_camera_info_.p[6]  = stereo_height_ / 2.0;
  stereo_camera_info_.p[7]  = 0.0;
  stereo_camera_info_.p[8]  = 0.0;
  stereo_camera_info_.p[9]  = 0.0;
  stereo_camera_info_.p[10] = 1.0;
  stereo_camera_info_.p[11] = 0.0;
}

//}

/* publishCameraTf() //{ */

/* void FlightforgeSimulator::publishCameraTf(const int& uav_index) { */
/*   geometry_msgs::msg::TransformStamped tf; */
/*   // | ------------------------- rgb tf ------------------------- | */

/*   { */
/*     tf.header.stamp = ros::Time::now(); */

/*     tf.header.frame_id = "uav" + std::to_string(uav_index + 1) + "/fcu"; */
/*     tf.child_frame_id  = "uav" + std::to_string(uav_index + 1) + "/rgb"; */

/*     tf.transform.translation.x = rgb_offset_x_; */
/*     tf.transform.translation.y = rgb_offset_y_; */
/*     tf.transform.translation.z = rgb_offset_z_; */

/*     Eigen::Matrix3d initial_tf = mrs_lib::AttitudeConverter(Eigen::Quaterniond(-0.5, 0.5, -0.5, 0.5)); */
/*     Eigen::Matrix3d dynamic_tf = mrs_lib::AttitudeConverter(rgb_camera_orientations_[uav_index]); */
/*     Eigen::Matrix3d final_tf   = dynamic_tf * initial_tf; */
/*     tf.transform.rotation      = mrs_lib::AttitudeConverter(final_tf); */

/*     try { */
/*       dynamic_broadcaster_->sendTransform(tf); */
/*     } */
/*     catch (...) { */
/*       RCLCPP_ERROR(node_->get_logger(), "could not publish rgb tf"); */
/*     } */
/*   } */
/* } */

//}


}  // namespace mrs_uav_flightforge_simulation

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(mrs_uav_flightforge_simulation::FlightforgeSimulator)
