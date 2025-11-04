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

//}

/* defines //{ */

using namespace std::chrono_literals;

using PCLPoint               = pcl::PointXYZ;
using PCLPointCloud          = pcl::PointCloud<PCLPoint>;
using PCLPointCloudColor     = pcl::PointCloud<pcl::PointXYZRGB>;
using PCLPointCloudIntensity = pcl::PointCloud<pcl::PointXYZI>;

#if USE_ROS_TIMER == 1
typedef mrs_lib::ROSTimer TimerType;
#else
typedef mrs_lib::ThreadTimer TimerType;
#endif

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
  rclcpp::Time last_step_time_;
  rclcpp::Time last_step_wall_time_;
  std::mutex   mutex_sim_time_;

  std::string _world_frame_name_;

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

  std::shared_ptr<TimerType> timer_rangefinder_;
  void                       timerRangefinder();

  std::shared_ptr<TimerType> timer_lidar_;
  void                       timerLidar();

  std::shared_ptr<TimerType> timer_seg_lidar_;
  void                       timerSegLidar();

  std::shared_ptr<TimerType> timer_int_lidar_;
  void                       timerIntLidar();

  std::shared_ptr<TimerType> timer_rgb_;
  void                       timerRgb();

  std::shared_ptr<TimerType> timer_rgb_segmented_;
  void                       timerRgbSegmented();

  std::shared_ptr<TimerType> timer_depth_;
  void                       timerDepth();

  std::shared_ptr<TimerType> timer_stereo_;
  void                       timerStereo();

  // | ------------------------ rtf check ----------------------- |

  double     actual_rtf_ = 1.0;
  std::mutex mutex_actual_rtf_;

  rclcpp::Time last_sim_time_status_;

  // | ----------------------- publishers ----------------------- |

  mrs_lib::PublisherHandler<rosgraph_msgs::msg::Clock>                  ph_clock_;
  mrs_lib::PublisherHandler<geometry_msgs::msg::PoseArray>              ph_poses_;
  std::vector<mrs_lib::PublisherHandler<sensor_msgs::msg::Range>>       ph_rangefinders_;
  std::vector<mrs_lib::PublisherHandler<sensor_msgs::msg::PointCloud2>> ph_lidars_;
  std::vector<mrs_lib::PublisherHandler<sensor_msgs::msg::PointCloud2>> ph_seg_lidars_;
  std::vector<mrs_lib::PublisherHandler<sensor_msgs::msg::PointCloud2>> ph_int_lidars_;

  std::vector<image_transport::Publisher> ph_img_rgb_;
  std::vector<image_transport::Publisher> ph_imp_stereo_left_;
  std::vector<image_transport::Publisher> ph_imp_stereo_right_;
  std::vector<image_transport::Publisher> ph_imp_rgbd_segmented_;
  std::vector<image_transport::Publisher> ph_imp_depth_;


  std::vector<mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>> ph_rgb_camera_info_;
  std::vector<mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>> ph_rgb_seg_camera_info_;
  std::vector<mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>> ph_stereo_left_camera_info_;
  std::vector<mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>> ph_stereo_right_camera_info_;
  std::vector<mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>> ph_depth_camera_info_;

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

  void fabricateCamInfo(void);

  void publishStaticTfs(void);

  void checkForCrash(void);

  rclcpp::Time flightforgeTimeToSimtime(const double flightforge_time);

  // how much to add to unreal time to get to our wall time
  double       wall_time_offset_             = 0;
  double       wall_time_offset_drift_slope_ = 0;
  std::mutex   mutex_wall_time_offset_;
  rclcpp::Time last_real_;

  double                  flightforge_fps_ = 0;
  std::string             flightforge_world_level_name_enum_;
  std::string             flightforge_graphics_settings_enum_;
  int                     flightforge_forest_density_     = 5;
  int                     flightforge_forest_hilly_level_ = 3;
  std::string             weather_type_;
  ueds_connector::Daytime daytime_;
  bool                    uavs_mutual_visibility_ = true;

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

    bool   rangefinder_enabled = false;
    double rangefinder_rate    = 0.0;

    bool   lidar_enabled       = false;
    double lidar_rate          = 10.0;
    bool   lidar_noise_enabled = false;
    double lidar_std_at_1m     = 0.0;
    double lidar_std_slope     = 0.0;

    bool   lidar_seg_enabled = false;
    double lidar_seg_rate    = 10.0;

    bool   lidar_int_enabled         = false;
    double lidar_int_rate            = 0.0;
    double lidar_int_value_grass     = 0.0;
    double lidar_int_value_road      = 0.0;
    double lidar_int_value_tree      = 0.0;
    double lidar_int_value_building  = 0.0;
    double lidar_int_value_fence     = 0.0;
    double lidar_int_value_dirt_road = 0.0;
    double lidar_int_value_other     = 0.0;
    bool   lidar_int_noise_enabled   = false;
    double lidar_int_std_at_1m       = 0.0;
    double lidar_int_std_slope       = 0.0;

    bool   rgb_enabled                = false;
    double rgb_rate                   = 10.0;
    bool   rgb_enable_hdr             = false;
    bool   rgb_enable_temporal_aa     = false;
    bool   rgb_enable_raytracing      = false;
    bool   rgb_enable_motion_blur     = false;
    double rgb_motion_blur_amount     = 0.0;
    double rgb_motion_blur_distortion = 0.0;

    bool   rgb_segmented_enabled = false;
    double rgb_segmented_rate    = 10.0;

    bool   rgb_depth_enabled = false;
    double rgb_depth_rate    = 0.0;

    bool   stereo_enabled            = false;
    double stereo_rate               = 10.0;
    bool   stereo_enable_hdr         = false;
    bool   stereo_enable_temporal_aa = false;
    bool   stereo_enable_raytracing  = false;
  };

  drs_params drs_params_;
  std::mutex mutex_drs_params_;

  void callbackRealtimeFactor(const double& param_value);
  void callbackPause(const bool& param_value);

  void callbackRangefinderRate(const double& param_value);
  void callbackRangefinderEnable(const bool& param_value);

  void callbackLidarRate(const double& param_value);
  void callbackLidarEnable(const bool& param_value);

  void callbackLidarSegEnable(const bool& param_value);
  void callbackLidarSegRate(const double& param_value);

  void callbackLidarIntEnable(const bool& param_value);
  void callbackLidarIntRate(const double& param_value);

  void callbackRgbEnable(const bool& param_value);
  void callbackRgbRate(const double& param_value);

  void callbackRgbSegEnable(const bool& param_value);
  void callbackRgbSegRate(const double& param_value);

  void callbackDepthEnable(const bool& param_value);
  void callbackDepthRate(const double& param_value);

  void callbackStereoEnable(const bool& param_value);
  void callbackStereoRate(const double& param_value);

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

  double lidar_horizontal_fov_left_  = 180.0;
  double lidar_horizontal_fov_right_ = 180.0;
  double lidar_vertical_fov_up_      = 45.0;
  double lidar_vertical_fov_down_    = 45.0;
  int    lidar_horizontal_rays_      = 128;
  int    lidar_vertical_rays_        = 256;
  double lidar_offset_x_             = 0.0;
  double lidar_offset_y_             = 0.0;
  double lidar_offset_z_             = 0.0;
  double lidar_rotation_pitch_       = 0.0;
  double lidar_rotation_roll_        = 0.0;
  double lidar_rotation_yaw_         = 0.0;
  double lidar_beam_length_          = 40;
  bool   lidar_show_beams_           = false;
  bool   lidar_livox_                = false;

  int    rgb_width_                  = 640;
  int    rgb_height_                 = 480;
  double rgb_fov_                    = 120.0;
  double rgb_offset_x_               = 0.0;
  double rgb_offset_y_               = 0.0;
  double rgb_offset_z_               = 0.0;
  double rgb_rotation_pitch_         = 0.0;
  double rgb_rotation_yaw_           = 0.0;
  double rgb_rotation_roll_          = 0.0;
  bool   rgb_enable_hdr_             = false;
  bool   rgb_enable_temporal_aa_     = true;
  bool   rgb_enable_raytracing_      = true;
  bool   rgb_enable_motion_blur_     = true;
  double rgb_motion_blur_amount_     = 0.5;
  double rgb_motion_blur_distortion_ = 50.0;

  double stereo_baseline_           = 0.1;
  int    stereo_width_              = 640;
  int    stereo_height_             = 480;
  double stereo_fov_                = 90.0;
  double stereo_offset_x_           = 0.0;
  double stereo_offset_y_           = 0.0;
  double stereo_offset_z_           = 0.0;
  double stereo_rotation_pitch_     = 0.0;
  double stereo_rotation_yaw_       = 0.0;
  double stereo_rotation_roll_      = 0.0;
  bool   stereo_enable_hdr_         = true;
  bool   stereo_enable_temporal_aa_ = true;
  bool   stereo_enable_raytracing_  = true;
};

//}

/* FlightforgeSimulator::FlightforgeSimulator() //{ */

FlightforgeSimulator::FlightforgeSimulator(rclcpp::NodeOptions options) : Node("mrs_uav_flightforge_simulator", options) {

  timer_init_ = this->create_wall_timer(std::chrono::duration<double>(0.1s), std::bind(&FlightforgeSimulator::timerInit, this));
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

  cbgrp_main_   = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  cbgrp_status_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

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

  if (_clock_rate_ < _simulation_rate_) {
    RCLCPP_ERROR(node_->get_logger(), "clock_rate (%.2f Hz) should be higher than simulation rate (%.2f Hz)!", _clock_rate_, _simulation_rate_);
    rclcpp::shutdown();
    exit(1);
  }

  dynparam_mgr_->register_param(yaml_prefix + "realtime_factor", &drs_params_.realtime_factor, mrs_lib::DynparamMgr::range_t<double>(0.01, 10), (std::function<void(const double&)>)std::bind(&FlightforgeSimulator::callbackRealtimeFactor, this, std::placeholders::_1));

  dynparam_mgr_->register_param(yaml_prefix + "dynamic_rtf", &drs_params_.dynamic_rtf);

  dynparam_mgr_->register_param(yaml_prefix + "collisions/enabled", &drs_params_.collisions_enabled);

  dynparam_mgr_->register_param(yaml_prefix + "collisions/crash", &drs_params_.collisions_crash);

  dynparam_mgr_->register_param(yaml_prefix + "collisions/crash", &drs_params_.collisions_crash);

  dynparam_mgr_->register_param(yaml_prefix + "collisions/rebounce", &drs_params_.collisions_rebounce, mrs_lib::DynparamMgr::range_t<double>(0.1, 1000));

  dynparam_mgr_->register_param(yaml_prefix + "paused", &drs_params_.paused, false, (std::function<void(const bool&)>)std::bind(&FlightforgeSimulator::callbackPause, this, std::placeholders::_1));

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

  dynparam_mgr_->register_param(yaml_prefix + "sensors/rangefinder/enabled", &drs_params_.rangefinder_enabled, (std::function<void(const bool&)>)std::bind(&FlightforgeSimulator::callbackRangefinderEnable, this, std::placeholders::_1));
  dynparam_mgr_->register_param(yaml_prefix + "sensors/rangefinder/rate", &drs_params_.rangefinder_rate, mrs_lib::DynparamMgr::range_t<double>(1.0, 100.0), (std::function<void(const double&)>)std::bind(&FlightforgeSimulator::callbackRangefinderRate, this, std::placeholders::_1));

  dynparam_mgr_->register_param(yaml_prefix + "sensors/lidar/enabled", &drs_params_.lidar_enabled, (std::function<void(const bool&)>)std::bind(&FlightforgeSimulator::callbackLidarEnable, this, std::placeholders::_1));
  dynparam_mgr_->register_param(yaml_prefix + "sensors/lidar/rate", &drs_params_.lidar_rate, mrs_lib::DynparamMgr::range_t<double>(1.0, 20.0), (std::function<void(const double&)>)std::bind(&FlightforgeSimulator::callbackLidarRate, this, std::placeholders::_1));

  param_loader.loadParam(yaml_prefix + "sensors/lidar/horizontal_fov_left", lidar_horizontal_fov_left_);
  param_loader.loadParam(yaml_prefix + "sensors/lidar/horizontal_fov_right", lidar_horizontal_fov_right_);

  param_loader.loadParam(yaml_prefix + "sensors/lidar/vertical_fov_up", lidar_vertical_fov_up_);
  param_loader.loadParam(yaml_prefix + "sensors/lidar/vertical_fov_down", lidar_vertical_fov_down_);

  param_loader.loadParam(yaml_prefix + "sensors/lidar/horizontal_rays", lidar_horizontal_rays_);
  param_loader.loadParam(yaml_prefix + "sensors/lidar/vertical_rays", lidar_vertical_rays_);

  param_loader.loadParam(yaml_prefix + "sensors/lidar/offset/x", lidar_offset_x_);
  param_loader.loadParam(yaml_prefix + "sensors/lidar/offset/y", lidar_offset_y_);
  param_loader.loadParam(yaml_prefix + "sensors/lidar/offset/z", lidar_offset_z_);

  param_loader.loadParam(yaml_prefix + "sensors/lidar/rotation/pitch", lidar_rotation_pitch_);
  param_loader.loadParam(yaml_prefix + "sensors/lidar/rotation/roll", lidar_rotation_roll_);
  param_loader.loadParam(yaml_prefix + "sensors/lidar/rotation/yaw", lidar_rotation_yaw_);

  param_loader.loadParam(yaml_prefix + "sensors/lidar/beam_length", lidar_beam_length_);
  param_loader.loadParam(yaml_prefix + "sensors/lidar/livox", lidar_livox_);
  param_loader.loadParam(yaml_prefix + "sensors/lidar/show_beams", lidar_show_beams_);

  dynparam_mgr_->register_param(yaml_prefix + "sensors/lidar/noise/enabled", &drs_params_.lidar_noise_enabled);
  param_loader.loadParam(yaml_prefix + "sensors/lidar/noise/std_at_1m", drs_params_.lidar_std_at_1m);
  param_loader.loadParam(yaml_prefix + "sensors/lidar/noise/std_slope", drs_params_.lidar_std_slope);

  dynparam_mgr_->register_param(yaml_prefix + "sensors/lidar/lidar_segmented/enabled", &drs_params_.lidar_seg_enabled, (std::function<void(const bool&)>)std::bind(&FlightforgeSimulator::callbackLidarSegEnable, this, std::placeholders::_1));
  dynparam_mgr_->register_param(yaml_prefix + "sensors/lidar/lidar_segmented/rate", &drs_params_.lidar_seg_rate, mrs_lib::DynparamMgr::range_t<double>(1.0, 20.0), (std::function<void(const double&)>)std::bind(&FlightforgeSimulator::callbackLidarSegRate, this, std::placeholders::_1));

  dynparam_mgr_->register_param(yaml_prefix + "sensors/lidar/lidar_intensity/enabled", &drs_params_.lidar_int_enabled, (std::function<void(const bool&)>)std::bind(&FlightforgeSimulator::callbackLidarIntEnable, this, std::placeholders::_1));
  dynparam_mgr_->register_param(yaml_prefix + "sensors/lidar/lidar_intensity/rate", &drs_params_.lidar_int_rate, mrs_lib::DynparamMgr::range_t<double>(1.0, 20.0), (std::function<void(const double&)>)std::bind(&FlightforgeSimulator::callbackLidarIntRate, this, std::placeholders::_1));
  param_loader.loadParam(yaml_prefix + "sensors/lidar/lidar_intensity/values/grass", drs_params_.lidar_int_value_grass);
  param_loader.loadParam(yaml_prefix + "sensors/lidar/lidar_intensity/values/road", drs_params_.lidar_int_value_road);
  param_loader.loadParam(yaml_prefix + "sensors/lidar/lidar_intensity/values/tree", drs_params_.lidar_int_value_tree);
  param_loader.loadParam(yaml_prefix + "sensors/lidar/lidar_intensity/values/building", drs_params_.lidar_int_value_building);
  param_loader.loadParam(yaml_prefix + "sensors/lidar/lidar_intensity/values/fence", drs_params_.lidar_int_value_fence);
  param_loader.loadParam(yaml_prefix + "sensors/lidar/lidar_intensity/values/dirt_road", drs_params_.lidar_int_value_dirt_road);
  param_loader.loadParam(yaml_prefix + "sensors/lidar/lidar_intensity/values/other", drs_params_.lidar_int_value_other);
  dynparam_mgr_->register_param(yaml_prefix + "sensors/lidar/lidar_intensity/noise/enabled", &drs_params_.lidar_int_noise_enabled);
  param_loader.loadParam(yaml_prefix + "sensors/lidar/lidar_intensity/noise/std_at_1m", drs_params_.lidar_int_std_at_1m);
  param_loader.loadParam(yaml_prefix + "sensors/lidar/lidar_intensity/noise/std_slope", drs_params_.lidar_int_std_slope);

  dynparam_mgr_->register_param(yaml_prefix + "sensors/rgb/enabled", &drs_params_.rgb_enabled, (std::function<void(const bool&)>)std::bind(&FlightforgeSimulator::callbackRgbEnable, this, std::placeholders::_1));
  dynparam_mgr_->register_param(yaml_prefix + "sensors/rgb/rate", &drs_params_.rgb_rate, mrs_lib::DynparamMgr::range_t<double>(1.0, 100.0), (std::function<void(const double&)>)std::bind(&FlightforgeSimulator::callbackRgbRate, this, std::placeholders::_1));

  param_loader.loadParam(yaml_prefix + "sensors/rgb/width", rgb_width_);
  param_loader.loadParam(yaml_prefix + "sensors/rgb/height", rgb_height_);
  param_loader.loadParam(yaml_prefix + "sensors/rgb/fov", rgb_fov_);

  param_loader.loadParam(yaml_prefix + "sensors/rgb/offset/x", rgb_offset_x_);
  param_loader.loadParam(yaml_prefix + "sensors/rgb/offset/y", rgb_offset_y_);
  param_loader.loadParam(yaml_prefix + "sensors/rgb/offset/z", rgb_offset_z_);

  param_loader.loadParam(yaml_prefix + "sensors/rgb/rotation/pitch", rgb_rotation_pitch_);
  param_loader.loadParam(yaml_prefix + "sensors/rgb/rotation/roll", rgb_rotation_roll_);
  param_loader.loadParam(yaml_prefix + "sensors/rgb/rotation/yaw", rgb_rotation_yaw_);

  dynparam_mgr_->register_param(yaml_prefix + "sensors/rgb/enable_hdr", &drs_params_.rgb_enable_hdr);
  dynparam_mgr_->register_param(yaml_prefix + "sensors/rgb/enable_temporal_aa", &drs_params_.rgb_enable_temporal_aa);
  dynparam_mgr_->register_param(yaml_prefix + "sensors/rgb/enable_raytracing", &drs_params_.rgb_enable_raytracing);
  dynparam_mgr_->register_param(yaml_prefix + "sensors/rgb/enable_motion_blur", &drs_params_.rgb_enable_motion_blur);
  dynparam_mgr_->register_param(yaml_prefix + "sensors/rgb/motion_blur_amount", &drs_params_.rgb_motion_blur_amount);
  dynparam_mgr_->register_param(yaml_prefix + "sensors/rgb/motion_blur_distortion", &drs_params_.rgb_motion_blur_distortion);

  dynparam_mgr_->register_param(yaml_prefix + "sensors/rgb/rgb_segmented/enabled", &drs_params_.rgb_segmented_enabled, (std::function<void(const bool&)>)std::bind(&FlightforgeSimulator::callbackRgbSegEnable, this, std::placeholders::_1));
  dynparam_mgr_->register_param(yaml_prefix + "sensors/rgb/rgb_segmented/rate", &drs_params_.rgb_segmented_rate, mrs_lib::DynparamMgr::range_t<double>(1.0, 100.0), (std::function<void(const double&)>)std::bind(&FlightforgeSimulator::callbackRgbSegRate, this, std::placeholders::_1));

  dynparam_mgr_->register_param(yaml_prefix + "sensors/stereo/enabled", &drs_params_.stereo_enabled, (std::function<void(const bool&)>)std::bind(&FlightforgeSimulator::callbackStereoEnable, this, std::placeholders::_1));
  dynparam_mgr_->register_param(yaml_prefix + "sensors/stereo/rate", &drs_params_.stereo_rate, mrs_lib::DynparamMgr::range_t<double>(1.0, 30.0), (std::function<void(const double&)>)std::bind(&FlightforgeSimulator::callbackStereoRate, this, std::placeholders::_1));
  dynparam_mgr_->register_param(yaml_prefix + "sensors/stereo/enable_hdr", &drs_params_.stereo_enable_hdr);
  dynparam_mgr_->register_param(yaml_prefix + "sensors/stereo/enable_temporal_aa", &drs_params_.stereo_enable_temporal_aa);
  dynparam_mgr_->register_param(yaml_prefix + "sensors/stereo/enable_raytracing", &drs_params_.stereo_enable_raytracing);

  param_loader.loadParam(yaml_prefix + "sensors/stereo/baseline", stereo_baseline_);

  param_loader.loadParam(yaml_prefix + "sensors/stereo/width", stereo_width_);
  param_loader.loadParam(yaml_prefix + "sensors/stereo/height", stereo_height_);
  param_loader.loadParam(yaml_prefix + "sensors/stereo/fov", stereo_fov_);

  param_loader.loadParam(yaml_prefix + "sensors/stereo/offset/x", stereo_offset_x_);
  param_loader.loadParam(yaml_prefix + "sensors/stereo/offset/y", stereo_offset_y_);
  param_loader.loadParam(yaml_prefix + "sensors/stereo/offset/z", stereo_offset_z_);

  param_loader.loadParam(yaml_prefix + "sensors/stereo/rotation/pitch", stereo_rotation_pitch_);
  param_loader.loadParam(yaml_prefix + "sensors/stereo/rotation/roll", stereo_rotation_roll_);
  param_loader.loadParam(yaml_prefix + "sensors/stereo/rotation/yaw", stereo_rotation_yaw_);

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

  ueds_game_controller_ = std::make_shared<ueds_connector::GameModeController>(LOCALHOST, 8551);

  while (rclcpp::ok()) {

    if (ueds_game_controller_->Connect()) {
      break;
    }

    RCLCPP_ERROR(node_->get_logger(), "[FlightforgeSimulator]: Error connecting to Unreal's game mode controller");

    std::this_thread::sleep_for(1s);
  }

  auto [res, api_version]                     = ueds_game_controller_->GetApiVersion();
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

  auto name_to_id_map = ueds_connector::WorldName::Name2Id();
  auto it             = name_to_id_map.find(flightforge_world_level_name_enum_);
  if (it == name_to_id_map.end()) {
    std::string error_msg = "Unknown world_name: '" + flightforge_world_level_name_enum_ + "'. Available worlds: ";
    for (const auto& pair : name_to_id_map) {
      error_msg += "'" + pair.first + "' ";
    }
    throw std::invalid_argument(error_msg);
  }
  res = ueds_game_controller_->SwitchWorldLevel(ueds_connector::WorldName::Name2Id().at(flightforge_world_level_name_enum_));

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

    int uav_frame_id = ueds_connector::UavFrameType::Type2IdMesh().at(uav_frame);

    auto [resSpawn, port] = ueds_game_controller_->SpawnDroneAtLocation(pos, uav_frame_id);

    if (!resSpawn) {
      RCLCPP_ERROR(node_->get_logger(), "failed to spawn %s", uav_names[i].c_str());
      rclcpp::shutdown();
    }

    RCLCPP_INFO(node_->get_logger(), "%s spawned", uav_name.c_str());

    std::shared_ptr<ueds_connector::UedsConnector> ueds_connector = std::make_shared<ueds_connector::UedsConnector>(LOCALHOST, port);

    ueds_connectors_.push_back(ueds_connector);

    auto connect_result = ueds_connector->Connect();

    if (connect_result != 1) {

      RCLCPP_ERROR(node_->get_logger(), "%s - Error connecting to drone controller, connect_result was %d", uav_name.c_str(), connect_result);
      rclcpp::shutdown();

    } else {
      RCLCPP_INFO(node_->get_logger(), "%s - Connection succeed: %d", uav_name.c_str(), connect_result);

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

    ph_img_rgb_.push_back(it_->advertise("/" + uav_name + "/rgb/image_raw", 1));
    ph_imp_stereo_left_.push_back(it_->advertise("/" + uav_name + "/stereo/left/image_raw", 1));
    ph_imp_stereo_right_.push_back(it_->advertise("/" + uav_name + "/stereo/right/image_raw", 1));
    ph_imp_rgbd_segmented_.push_back(it_->advertise("/" + uav_name + "/rgb_segmented/image_raw", 1));
    ph_imp_depth_.push_back(it_->advertise("/" + uav_name + "/depth/image_raw", 1));


    ph_rgb_camera_info_.push_back(mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>(node_, "/" + uav_name + "/rgb/camera_info"));
    ph_rgb_seg_camera_info_.push_back(mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>(node_, "/" + uav_name + "/rgb_segmented/camera_info"));
    ph_stereo_left_camera_info_.push_back(mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>(node_, "/" + uav_name + "/stereo/left/camera_info"));
    ph_stereo_right_camera_info_.push_back(mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo>(node_, "/" + uav_name + "/stereo/right/camera_info"));

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
      lidarConfig.Livox        = lidar_livox_;

      const auto res = ueds_connectors_[i]->SetLidarConfig(lidarConfig);

      if (!res) {
        RCLCPP_ERROR(node_->get_logger(), "failed to set lidar config for uav %lu", i + 1);
      } else {
        RCLCPP_INFO(node_->get_logger(), "lidar config set for uav%lu", i + 1);
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
        RCLCPP_ERROR(node_->get_logger(), "failed to set camera config for uav %lu", i + 1);
      } else {
        RCLCPP_INFO(node_->get_logger(), "camera config set for uav%lu", i + 1);
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
        RCLCPP_ERROR(node_->get_logger(), "failed to set camera config for uav %lu", i + 1);
      } else {
        RCLCPP_INFO(node_->get_logger(), "camera config set for uav%lu", i + 1);
      }
    }
  }

  if (!param_loader.loadedSuccessfully()) {
    RCLCPP_ERROR(node_->get_logger(), "could not load all parameters!");
    rclcpp::shutdown();
    exit(1);
  }

  // | ----------------------- publishers ----------------------- |

  ph_clock_ = mrs_lib::PublisherHandler<rosgraph_msgs::msg::Clock>(node_, "~/clock_out");

  ph_poses_ = mrs_lib::PublisherHandler<geometry_msgs::msg::PoseArray>(node_, "~/uav_poses_out");

  // | ------------------------- timers ------------------------- |

  mrs_lib::TimerHandlerOptions opts;

  opts.node      = node_;
  opts.autostart = true;

  timer_main_ = node_->create_wall_timer(std::chrono::duration<double>(1.0 / (_clock_rate_ * drs_params_.realtime_factor)), std::bind(&FlightforgeSimulator::timerMain, this), cbgrp_main_);

  timer_status_ = node_->create_wall_timer(std::chrono::duration<double>(1.0), std::bind(&FlightforgeSimulator::timerStatus, this), cbgrp_status_);

  timer_time_sync_ = node_->create_wall_timer(std::chrono::duration<double>(1.0), std::bind(&FlightforgeSimulator::timerTimeSync, this), cbgrp_status_);

  timer_unreal_sync_ = node_->create_wall_timer(std::chrono::duration<double>(1.0 / _clock_rate_), std::bind(&FlightforgeSimulator::timerUnrealSync, this), cbgrp_status_);

  mrs_lib::TimerHandlerOptions timer_opts_sensors;

  timer_opts_sensors.node           = node_;
  timer_opts_sensors.callback_group = cbgrp_sensors_;
  timer_opts_sensors.autostart      = false;

  std::function<void()> callback_fcn = std::bind(&FlightforgeSimulator::timerRangefinder, this);

  timer_rangefinder_ = std::make_shared<TimerType>(timer_opts_sensors, rclcpp::Rate(drs_params_.rangefinder_rate, clock_), callback_fcn);

  if (drs_params_.rangefinder_enabled) {
    timer_rangefinder_->start();
  }

  {
    std::function<void()> callback_fcn = std::bind(&FlightforgeSimulator::timerLidar, this);

    timer_lidar_ = std::make_shared<TimerType>(timer_opts_sensors, rclcpp::Rate(drs_params_.lidar_rate, clock_), callback_fcn);
  }

  if (drs_params_.lidar_enabled) {
    timer_lidar_->start();
  }

  {
    std::function<void()> callback_fcn = std::bind(&FlightforgeSimulator::timerSegLidar, this);

    timer_seg_lidar_ = std::make_shared<TimerType>(timer_opts_sensors, rclcpp::Rate(drs_params_.lidar_seg_rate, clock_), callback_fcn);
  }

  if (drs_params_.lidar_seg_enabled) {
    timer_seg_lidar_->start();
  }

  {
    std::function<void()> callback_fcn = std::bind(&FlightforgeSimulator::timerIntLidar, this);

    timer_int_lidar_ = std::make_shared<TimerType>(timer_opts_sensors, rclcpp::Rate(drs_params_.lidar_int_rate, clock_), callback_fcn);
  }

  if (drs_params_.lidar_int_enabled) {
    timer_int_lidar_->start();
  }

  {
    std::function<void()> callback_fcn = std::bind(&FlightforgeSimulator::timerRgb, this);

    timer_rgb_ = std::make_shared<TimerType>(timer_opts_sensors, rclcpp::Rate(drs_params_.rgb_rate, clock_), callback_fcn);
  }

  if (drs_params_.rgb_enabled) {
    timer_rgb_->start();
  }

  {
    std::function<void()> callback_fcn = std::bind(&FlightforgeSimulator::timerStereo, this);

    timer_stereo_ = std::make_shared<TimerType>(timer_opts_sensors, rclcpp::Rate(drs_params_.stereo_rate, clock_), callback_fcn);
  }

  if (drs_params_.stereo_enabled) {
    timer_stereo_->start();
  }

  {
    std::function<void()> callback_fcn = std::bind(&FlightforgeSimulator::timerRgbSegmented, this);

    timer_rgb_segmented_ = std::make_shared<TimerType>(timer_opts_sensors, rclcpp::Rate(drs_params_.rgb_segmented_rate, clock_), callback_fcn);
  }

  if (drs_params_.rgb_segmented_enabled) {
    timer_rgb_segmented_->start();
  }

  /* if (drs_params_.rgb_depth_rate > 0) { */
  /*   timer_depth_ = node_->create_wall_timer(std::chrono::duration<double>(1.0 / drs_params_.rgb_depth_rate), std::bind(&FlightforgeSimulator::timerDepth, this), cbgrp_sensors_); */
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

  RCLCPP_INFO_ONCE(node_->get_logger(), "Main timer spinning");

  // | ------------------ make simulation step ------------------ |

  double simulation_step_size = 1.0 / _simulation_rate_;
  double clock_step_size      = 1.0 / _clock_rate_;

  {
    std::scoped_lock lock(mutex_sim_time_);

    sim_time_ = sim_time_ + rclcpp::Duration(std::chrono::duration<double>(clock_step_size));

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
  timer_main_ = node_->create_wall_timer(std::chrono::duration<double>(1.0 / (_clock_rate_ * desired_rtf)), std::bind(&FlightforgeSimulator::timerMain, this), cbgrp_main_);

  if (_collisions_) {
    checkForCrash();
  }

  RCLCPP_INFO(node_->get_logger(), "%s, desired RTF = %.2f, actual RTF = %.2f, FlightForge FPS = %.2f, FlightForge RTF = %.2f", drs_params.paused ? "paused" : "running", drs_params.realtime_factor, actual_rtf, flightforge_fps_, flightforge_rtf);

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

  const double sync_start = wall_clock_->now().seconds();

  bool   res;
  double flightforge_time;

  {
    std::scoped_lock lock(mutex_flightforge_);

    std::tie(res, flightforge_time) = ueds_game_controller_->GetTime();
  }

  const double sync_end = wall_clock_->now().seconds();

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

  RCLCPP_INFO(node_->get_logger(), "wall time %f flightforge %f time offset: %f, offset slope %f s/s", sync_start, flightforge_time, wall_time_offset_, wall_time_offset_drift_slope_);
}

//}

/*timerRangefinder()//{*/

void FlightforgeSimulator::timerRangefinder() {

  if (!is_initialized_) {
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
      RCLCPP_ERROR_THROTTLE(node_->get_logger(), *node_->get_clock(), 1e9, "[uav%d] - ERROR GetRangefinderData", int(i) + 1);
      continue;
    }

    sensor_msgs::msg::Range msg_range;
    auto                    last_step_time = mrs_lib::get_mutexed(mutex_sim_time_, last_step_time_);
    msg_range.header.stamp                 = last_step_time;
    msg_range.header.frame_id              = "uav" + std::to_string(i + 1) + "/fcu";
    msg_range.radiation_type               = 1;
    msg_range.min_range                    = 0.1;
    msg_range.max_range                    = 30;
    msg_range.range                        = range / 100;

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
      RCLCPP_ERROR_THROTTLE(node_->get_logger(), *node_->get_clock(), 1e9, "[uav%d] - ERROR getLidarData", int(i) + 1);
      continue;
    }

    sensor_msgs::msg::PointCloud2 pcl_msg;

    // Modifier to describe what the fields are.
    sensor_msgs::PointCloud2Modifier modifier(pcl_msg);
    modifier.setPointCloud2Fields(4, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1, sensor_msgs::msg::PointField::FLOAT32, "z", 1, sensor_msgs::msg::PointField::FLOAT32, "intensity", 1, sensor_msgs::msg::PointField::FLOAT32);

    auto last_step_time = mrs_lib::get_mutexed(mutex_sim_time_, last_step_time_);

    // TODO we should publish the actual stamp from the unreal sim (transformed to the simtime)
    pcl_msg.header.stamp = last_step_time - rclcpp::Duration(std::chrono::duration<double>(0.01));

    pcl_msg.header.frame_id = "uav" + std::to_string(i + 1) + "/lidar";
    pcl_msg.height          = lidar_vertical_rays_;
    pcl_msg.width           = lidar_horizontal_rays_;
    pcl_msg.is_dense        = true;

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

  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  /* mrs_lib::ScopeTimer timer = mrs_lib::ScopeTimer("timerSegLidar()"); */

  for (size_t i = 0; i < uavs_.size(); i++) {

    // mrs_multirotor_simulator::MultirotorModel::State state = uavs_[i]->getState();

    bool                                      res;
    std::vector<ueds_connector::LidarSegData> lidarSegData;
    ueds_connector::Coordinates               start;

    {
      std::scoped_lock lock(mutex_flightforge_);

      std::tie(res, lidarSegData, start) = ueds_connectors_[i]->GetLidarSegData();
    }

    if (!res) {
      RCLCPP_ERROR_THROTTLE(node_->get_logger(), *node_->get_clock(), 1e9, "[uav%d] - ERROR getLidarSegData", int(i) + 1);
      continue;
    }

    PCLPointCloudColor pcl_cloud;

    for (const ueds_connector::LidarSegData& ray : lidarSegData) {

      pcl::PointXYZRGB point;
      tf2::Vector3     dir = tf2::Vector3(ray.directionX, ray.directionY, ray.directionZ);

      double ray_distance = ray.distance / 100.0;

      if (drs_params.lidar_noise_enabled && ray_distance > 0) {

        const double std = ray_distance * drs_params.lidar_std_slope * drs_params.lidar_std_at_1m;

        std::normal_distribution<double> distribution(0, std);

        ray_distance += distribution(rng);
      }

      dir     = dir.normalized() * ray_distance;
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

    // TODO we should publish the actual stamp from the unreal sim (transformed to the simtime)
    pcl_msg.header.stamp = last_step_time - rclcpp::Duration(std::chrono::duration<double>(0.01));

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

  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  /* mrs_lib::ScopeTimer timer = mrs_lib::ScopeTimer("timerIntLidar()"); */

  for (size_t i = 0; i < uavs_.size(); i++) {

    // mrs_multirotor_simulator::MultirotorModel::State state = uavs_[i]->getState();

    bool                                      res;
    std::vector<ueds_connector::LidarIntData> lidarData;
    ueds_connector::Coordinates               start;

    {
      std::scoped_lock lock(mutex_flightforge_);

      std::tie(res, lidarData, start) = ueds_connectors_[i]->GetLidarIntData();
    }

    if (!res) {
      RCLCPP_ERROR_THROTTLE(node_->get_logger(), *node_->get_clock(), 1e9, "[uav%d] - ERROR getLidarIntData", int(i) + 1);
      continue;
    }

    sensor_msgs::msg::PointCloud2 pcl_msg;

    // Modifier to describe what the fields are.
    sensor_msgs::PointCloud2Modifier modifier(pcl_msg);
    modifier.setPointCloud2Fields(4, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1, sensor_msgs::msg::PointField::FLOAT32, "z", 1, sensor_msgs::msg::PointField::FLOAT32, "intensity", 1, sensor_msgs::msg::PointField::FLOAT32);
    // Msg header
    auto last_step_time = mrs_lib::get_mutexed(mutex_sim_time_, last_step_time_);

    // TODO we should publish the actual stamp from the unreal sim (transformed to the simtime)
    pcl_msg.header.stamp = last_step_time - rclcpp::Duration(std::chrono::duration<double>(0.01));

    pcl_msg.header.frame_id = "uav" + std::to_string(i + 1) + "/lidar";
    pcl_msg.height          = lidar_vertical_rays_;
    pcl_msg.width           = lidar_horizontal_rays_;
    pcl_msg.is_dense        = true;

    // Total number of bytes per point
    pcl_msg.point_step = 16;
    pcl_msg.row_step   = pcl_msg.point_step * pcl_msg.width;
    pcl_msg.data.resize(pcl_msg.row_step * pcl_msg.height);

    sensor_msgs::PointCloud2Iterator<float> iterX(pcl_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iterY(pcl_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iterZ(pcl_msg, "z");
    sensor_msgs::PointCloud2Iterator<float> iterIntensity(pcl_msg, "intensity");

    for (const ueds_connector::LidarIntData& ray : lidarData) {

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

    ph_int_lidars_[i].publish(pcl_msg);
  }
}

//}

/* timerRgb() //{ */

void FlightforgeSimulator::timerRgb() {

  if (!is_initialized_) {
    return;
  }

  /* mrs_lib::ScopeTimer timer = mrs_lib::ScopeTimer(node_, "timerRgb()"); */

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
      RCLCPP_WARN(node_->get_logger(), "failed to obtain rgb camera from uav%lu", i + 1);
      continue;
    }

    if (cameraData.empty()) {
      RCLCPP_WARN(node_->get_logger(), "rgb camera from uav%lu is empty!", i + 1);
      continue;
    }

    cv::Mat image = cv::imdecode(cameraData, cv::IMREAD_COLOR);

    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image).toImageMsg();

    msg->header.frame_id = "uav" + std::to_string(i + 1) + "/rgb";

    msg->header.stamp = flightforgeTimeToSimtime(stamp);

    ph_img_rgb_[i].publish(msg);

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
      RCLCPP_WARN(node_->get_logger(), "failed to obtain stereo camera from uav%lu", i + 1);
      continue;
    }

    if (image_left.empty() || image_right.empty()) {
      RCLCPP_WARN(node_->get_logger(), "stereo camera from uav%lu is empty!", i + 1);
      continue;
    }

    cv::Mat cv_left  = cv::imdecode(image_left, cv::IMREAD_COLOR);
    cv::Mat cv_right = cv::imdecode(image_right, cv::IMREAD_COLOR);

    auto msg_left  = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", cv_left).toImageMsg();
    auto msg_right = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", cv_right).toImageMsg();

    msg_left->header.frame_id = "uav" + std::to_string(i + 1) + "/stereo_left";

    msg_right->header.frame_id = "uav" + std::to_string(i + 1) + "/stereo_right";

    msg_left->header.stamp  = flightforgeTimeToSimtime(stamp);
    msg_right->header.stamp = msg_left->header.stamp;

    ph_imp_stereo_left_[i].publish(msg_left);
    ph_imp_stereo_right_[i].publish(msg_right);

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
      RCLCPP_WARN(node_->get_logger(), "failed to obtain segmented camera from uav%lu", i + 1);
      continue;
    }

    if (cameraData.empty()) {
      RCLCPP_WARN(node_->get_logger(), "segmented camera from uav%lu is empty!", i + 1);
      continue;
    }

    cv::Mat image = cv::Mat(rgb_height_, rgb_width_, CV_8UC3, cameraData.data());

    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image).toImageMsg();

    msg->header.frame_id = "uav" + std::to_string(i + 1) + "/rgb";

    msg->header.stamp = flightforgeTimeToSimtime(stamp);

    ph_imp_rgbd_segmented_[i].publish(msg);

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

  for (size_t i = 0; i < uavs_.size(); i++) {

    bool                  res;
    std::vector<uint16_t> cameraData;
    uint32_t              size;
    double                stamp;

    {
      std::scoped_lock lock(mutex_flightforge_);

      std::tie(res, cameraData, stamp, size) = ueds_connectors_[i]->GetDepthCameraData();
    }

    if (abs(stamp - last_rgb_ue_stamp_.at(i)) < 0.001) {
      return;
    }

    last_rgb_ue_stamp_.at(i) = stamp;

    if (!res) {
      RCLCPP_WARN(node_->get_logger(), "failed to obtain depth camera from uav%lu", i + 1);
      continue;
    }

    if (cameraData.empty()) {
      RCLCPP_WARN(node_->get_logger(), "depth camera from uav%lu is empty!", i + 1);
      continue;
    }

    cv::Mat image = cv::Mat(rgb_height_, rgb_width_, CV_16UC1, cameraData.data());

    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono16", image).toImageMsg();

    msg->header.frame_id = "uav" + std::to_string(i + 1) + "/rgb";

    msg->header.stamp = flightforgeTimeToSimtime(stamp);

    ph_imp_depth_[i].publish(msg);

    auto camera_info = rgb_camera_info_;

    camera_info.header = msg->header;

    ph_depth_camera_info_[i].publish(camera_info);
  }
}

//}

// | ------------------------ callbacks ----------------------- |

/* dynamic parameter callbacks //{ */

/* callbackRealtimeFactor() //{ */

void FlightforgeSimulator::callbackRealtimeFactor(const double& param_value) {

  timer_main_->cancel();

  timer_main_ = create_wall_timer(std::chrono::duration<double>(1.0 / (_clock_rate_ * param_value)), std::bind(&FlightforgeSimulator::timerMain, this), cbgrp_main_);

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

    timer_main_ = create_wall_timer(std::chrono::duration<double>(1.0 / (_clock_rate_ * drs_params_.realtime_factor)), std::bind(&FlightforgeSimulator::timerMain, this), cbgrp_main_);

    timer_status_ = create_wall_timer(std::chrono::duration<double>(1.0), std::bind(&FlightforgeSimulator::timerStatus, this), cbgrp_status_);

    RCLCPP_INFO(get_logger(), "unpaused");
  }
}

//}

/* callbackRangefinderEnable() //{ */

void FlightforgeSimulator::callbackRangefinderEnable(const bool& param_value) {

  RCLCPP_INFO(get_logger(), "callbackRangefinderEnable()");

  if (param_value) {
    timer_rangefinder_->start();
  } else {
    timer_rangefinder_->stop();
  }
}

//}

/* callbackRangefinderRate() //{ */

void FlightforgeSimulator::callbackRangefinderRate(const double& param_value) {

  RCLCPP_INFO(get_logger(), "callbackRangefinderRate()");

  timer_rangefinder_->stop();

  timer_rangefinder_->setPeriod(std::chrono::duration<double>(1.0 / param_value));

  if (drs_params_.rangefinder_enabled) {
    timer_rangefinder_->start();
  }

  RCLCPP_INFO(get_logger(), "rangefinder rate updated to %.2f Hz", param_value);
}

//}

/* callbackLidarEnable() //{ */

void FlightforgeSimulator::callbackLidarEnable(const bool& param_value) {

  RCLCPP_INFO(get_logger(), "callbackLidarEnable()");

  if (param_value) {
    timer_lidar_->start();
  } else {
    timer_lidar_->stop();
  }
}

//}

/* callbackLidarRate() //{ */

void FlightforgeSimulator::callbackLidarRate(const double& param_value) {

  RCLCPP_INFO(get_logger(), "callbackLidarRate()");

  timer_lidar_->stop();

  timer_lidar_->setPeriod(std::chrono::duration<double>(1.0 / param_value));

  if (drs_params_.lidar_enabled) {
    timer_lidar_->start();
  }

  RCLCPP_INFO(get_logger(), "lidar rate updated to %.2f Hz", param_value);
}

//}

/* callbackLidarSegEnable() //{ */

void FlightforgeSimulator::callbackLidarSegEnable(const bool& param_value) {

  RCLCPP_INFO(get_logger(), "callbackLidarSegEnable()");

  if (param_value) {
    timer_seg_lidar_->start();
  } else {
    timer_seg_lidar_->stop();
  }
}

//}

/* callbackLidarSegRate() //{ */

void FlightforgeSimulator::callbackLidarSegRate(const double& param_value) {

  RCLCPP_INFO(get_logger(), "callbackLidarSegRate()");

  timer_seg_lidar_->stop();

  timer_seg_lidar_->setPeriod(std::chrono::duration<double>(1.0 / param_value));

  if (drs_params_.lidar_seg_enabled) {
    timer_seg_lidar_->start();
  }

  RCLCPP_INFO(get_logger(), "lidar segmented rate updated to %.2f Hz", param_value);
}

//}

/* callbackLidarIntEnable() //{ */

void FlightforgeSimulator::callbackLidarIntEnable(const bool& param_value) {

  RCLCPP_INFO(get_logger(), "callbackLidarIntEnable()");

  if (param_value) {
    timer_int_lidar_->start();
  } else {
    timer_int_lidar_->stop();
  }
}

//}

/* callbackLidarSegRate() //{ */

void FlightforgeSimulator::callbackLidarIntRate(const double& param_value) {

  RCLCPP_INFO(get_logger(), "callbackLidarIntRate()");

  timer_int_lidar_->stop();

  timer_int_lidar_->setPeriod(std::chrono::duration<double>(1.0 / param_value));

  if (drs_params_.lidar_int_enabled) {
    timer_int_lidar_->start();
  }

  RCLCPP_INFO(get_logger(), "lidar intensity rate updated to %.2f Hz", param_value);
}

//}

/* callbackRgbEnable() //{ */

void FlightforgeSimulator::callbackRgbEnable(const bool& param_value) {

  RCLCPP_INFO(get_logger(), "callbackRgbEnable()");

  if (param_value) {
    timer_rgb_->start();
  } else {
    timer_rgb_->stop();
  }
}

//}

/* callbackRgbRate() //{ */

void FlightforgeSimulator::callbackRgbRate(const double& param_value) {

  RCLCPP_INFO(get_logger(), "callbackRgbRate()");

  timer_rgb_->stop();

  timer_rgb_->setPeriod(std::chrono::duration<double>(1.0 / param_value));

  if (drs_params_.rgb_enabled) {
    timer_rgb_->start();
  }

  RCLCPP_INFO(get_logger(), "rgb rate updated to %.2f Hz", param_value);
}

//}

/* callbackRgbSegEnable() //{ */

void FlightforgeSimulator::callbackRgbSegEnable(const bool& param_value) {

  RCLCPP_INFO(get_logger(), "callbackRgbSegEnable()");

  if (param_value) {
    timer_rgb_segmented_->start();
  } else {
    timer_rgb_segmented_->stop();
  }
}

//}

/* callbackRgbSegRate() //{ */

void FlightforgeSimulator::callbackRgbSegRate(const double& param_value) {

  RCLCPP_INFO(get_logger(), "callbackRgbSegRate()");

  timer_rgb_segmented_->stop();

  timer_rgb_segmented_->setPeriod(std::chrono::duration<double>(1.0 / param_value));

  if (drs_params_.rgb_segmented_enabled) {
    timer_rgb_segmented_->start();
  }

  RCLCPP_INFO(get_logger(), "rgb segmented rate updated to %.2f Hz", param_value);
}

//}

/* callbackDepthEnable() //{ */

void FlightforgeSimulator::callbackDepthEnable(const bool& param_value) {

  RCLCPP_INFO(get_logger(), "callbackDepthEnable()");

  if (param_value) {
    timer_depth_->start();
  } else {
    timer_depth_->stop();
  }
}

//}

/* callbackDepthRate() //{ */

void FlightforgeSimulator::callbackDepthRate(const double& param_value) {

  RCLCPP_INFO(get_logger(), "callbackDepthRate()");

  timer_depth_->stop();

  timer_depth_->setPeriod(std::chrono::duration<double>(1.0 / param_value));

  if (drs_params_.rgb_depth_enabled) {
    timer_depth_->start();
  }

  RCLCPP_INFO(get_logger(), "depth rate updated to %.2f Hz", param_value);
}

//}

/* callbackStereoEnable() //{ */

void FlightforgeSimulator::callbackStereoEnable(const bool& param_value) {

  RCLCPP_INFO(get_logger(), "callbackStereoEnable()");

  if (param_value) {
    timer_stereo_->start();
  } else {
    timer_stereo_->stop();
  }
}

//}

/* callbackStereoRate() //{ */

void FlightforgeSimulator::callbackStereoRate(const double& param_value) {

  RCLCPP_INFO(get_logger(), "callbackStereoRate()");

  timer_stereo_->stop();

  timer_stereo_->setPeriod(std::chrono::duration<double>(1.0 / param_value));

  if (drs_params_.stereo_enabled) {
    timer_stereo_->start();
  }

  RCLCPP_INFO(get_logger(), "stereo rate updated to %.2f Hz", param_value);
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

      Eigen::Matrix3d dynamic_tf = mrs_lib::AttitudeConverter(M_PI * (rgb_rotation_roll_ / 180.0), M_PI * (rgb_rotation_pitch_ / 180.0), M_PI * (rgb_rotation_yaw_ / 180.0));

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
      RCLCPP_INFO(node_->get_logger(), "cam0:\n");
      RCLCPP_INFO(node_->get_logger(), "  T_imu_cam:\n");
      RCLCPP_INFO(node_->get_logger(), "    - [%f, %f, %f, %f]\n", final_tf(0, 0), final_tf(0, 1), final_tf(0, 2), tf.transform.translation.x);
      RCLCPP_INFO(node_->get_logger(), "    - [%f, %f, %f, %f]\n", final_tf(1, 0), final_tf(1, 1), final_tf(1, 2), tf.transform.translation.y);
      RCLCPP_INFO(node_->get_logger(), "    - [%f, %f, %f, %f]\n", final_tf(2, 0), final_tf(2, 1), final_tf(2, 2), tf.transform.translation.z);
      RCLCPP_INFO(node_->get_logger(), "    - [%f, %f, %f, %f]\n", 0.0, 0.0, 0.0, 1.0);
      RCLCPP_INFO(node_->get_logger(), "  cam_overlaps: [0]\n");
      RCLCPP_INFO(node_->get_logger(), "  camera_model: pinhole\n");
      RCLCPP_INFO(node_->get_logger(), "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n");
      RCLCPP_INFO(node_->get_logger(), "  distortion_model: radtan\n");
      RCLCPP_INFO(node_->get_logger(), "  intrinsics: [%f, %f, %f, %f]\n", rgb_camera_info_.k[0], rgb_camera_info_.k[4], rgb_camera_info_.k[2], rgb_camera_info_.k[5]);
      RCLCPP_INFO(node_->get_logger(), "  resolution: [%d, %d]\n", rgb_width_, rgb_height_);
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

      Eigen::Matrix3d dynamic_tf = mrs_lib::AttitudeConverter(M_PI * (stereo_rotation_roll_ / 180.0), M_PI * (stereo_rotation_pitch_ / 180.0), M_PI * (stereo_rotation_yaw_ / 180.0));

      Eigen::Matrix3d final_tf = dynamic_tf * initial_tf;

      tf.transform.rotation = mrs_lib::AttitudeConverter(final_tf);

      try {
        static_broadcaster_->sendTransform(tf);
      }
      catch (...) {
        RCLCPP_ERROR(node_->get_logger(), "Could not publish stereo left tf");
      }

      // | ------------- print the tf matrix for kalibr ------------- |

      RCLCPP_INFO(node_->get_logger(), "[uav%d]: Stereo left camera-imu chain for kalibr config:", int(i + 1));
      RCLCPP_INFO(node_->get_logger(), "cam0:\n");
      RCLCPP_INFO(node_->get_logger(), "  T_imu_cam:\n");
      RCLCPP_INFO(node_->get_logger(), "    - [%f, %f, %f, %f]\n", final_tf(0, 0), final_tf(0, 1), final_tf(0, 2), tf.transform.translation.x);
      RCLCPP_INFO(node_->get_logger(), "    - [%f, %f, %f, %f]\n", final_tf(1, 0), final_tf(1, 1), final_tf(1, 2), tf.transform.translation.y);
      RCLCPP_INFO(node_->get_logger(), "    - [%f, %f, %f, %f]\n", final_tf(2, 0), final_tf(2, 1), final_tf(2, 2), tf.transform.translation.z);
      RCLCPP_INFO(node_->get_logger(), "    - [%f, %f, %f, %f]\n", 0.0, 0.0, 0.0, 1.0);
      RCLCPP_INFO(node_->get_logger(), "  cam_overlaps: [0]\n");
      RCLCPP_INFO(node_->get_logger(), "  camera_model: pinhole\n");
      RCLCPP_INFO(node_->get_logger(), "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n");
      RCLCPP_INFO(node_->get_logger(), "  distortion_model: radtan\n");
      RCLCPP_INFO(node_->get_logger(), "  intrinsics: [%f, %f, %f, %f]\n", stereo_camera_info_.k[0], stereo_camera_info_.k[4], stereo_camera_info_.k[2], stereo_camera_info_.k[5]);
      RCLCPP_INFO(node_->get_logger(), "  resolution: [%d, %d]\n", stereo_width_, stereo_height_);
    }

    {
      tf.header.stamp = clock_->now();

      tf.header.frame_id = "uav" + std::to_string(i + 1) + "/fcu";
      tf.child_frame_id  = "uav" + std::to_string(i + 1) + "/stereo_right";

      tf.transform.translation.x = stereo_offset_x_;
      tf.transform.translation.y = stereo_offset_y_ - stereo_baseline_;
      tf.transform.translation.z = stereo_offset_z_;

      Eigen::Matrix3d initial_tf = mrs_lib::AttitudeConverter(Eigen::Quaterniond(-0.5, 0.5, -0.5, 0.5));

      Eigen::Matrix3d dynamic_tf = mrs_lib::AttitudeConverter(M_PI * (stereo_rotation_roll_ / 180.0), M_PI * (stereo_rotation_pitch_ / 180.0), M_PI * (stereo_rotation_yaw_ / 180.0));

      Eigen::Matrix3d final_tf = dynamic_tf * initial_tf;

      tf.transform.rotation = mrs_lib::AttitudeConverter(final_tf);

      try {
        static_broadcaster_->sendTransform(tf);
      }
      catch (...) {
        RCLCPP_ERROR(node_->get_logger(), "Could not publish stereo right tf");
      }

      // | ------------- print the tf matrix for kalibr ------------- |

      RCLCPP_INFO(node_->get_logger(), "[uav%d]: Stereo right camera-imu chain for kalibr config:", int(i + 1));
      RCLCPP_INFO(node_->get_logger(), "cam0:\n");
      RCLCPP_INFO(node_->get_logger(), "  T_imu_cam:\n");
      RCLCPP_INFO(node_->get_logger(), "    - [%f, %f, %f, %f]\n", final_tf(0, 0), final_tf(0, 1), final_tf(0, 2), tf.transform.translation.x);
      RCLCPP_INFO(node_->get_logger(), "    - [%f, %f, %f, %f]\n", final_tf(1, 0), final_tf(1, 1), final_tf(1, 2), tf.transform.translation.y);
      RCLCPP_INFO(node_->get_logger(), "    - [%f, %f, %f, %f]\n", final_tf(2, 0), final_tf(2, 1), final_tf(2, 2), tf.transform.translation.z);
      RCLCPP_INFO(node_->get_logger(), "    - [%f, %f, %f, %f]\n", 0.0, 0.0, 0.0, 1.0);
      RCLCPP_INFO(node_->get_logger(), "  cam_overlaps: [0]\n");
      RCLCPP_INFO(node_->get_logger(), "  camera_model: pinhole\n");
      RCLCPP_INFO(node_->get_logger(), "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n");
      RCLCPP_INFO(node_->get_logger(), "  distortion_model: radtan\n");
      RCLCPP_INFO(node_->get_logger(), "  intrinsics: [%f, %f, %f, %f]\n", stereo_camera_info_.k[0], stereo_camera_info_.k[4], stereo_camera_info_.k[2], stereo_camera_info_.k[5]);
      RCLCPP_INFO(node_->get_logger(), "  resolution: [%d, %d]\n", stereo_width_, stereo_height_);
    }


    // | ------------------------- lidar tf ------------------------- |
    {
      tf.header.stamp    = clock_->now();
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

/* flightforgeTimeToSimtime() //{ */

rclcpp::Time FlightforgeSimulator::flightforgeTimeToSimtime(const double flightforge_time) {

  auto wall_time_offset = mrs_lib::get_mutexed(mutex_wall_time_offset_, wall_time_offset_);
  auto actual_rtf       = mrs_lib::get_mutexed(mutex_actual_rtf_, actual_rtf_);

  double relative_wall_age = wall_clock_->now().seconds() - (flightforge_time + wall_time_offset);

  double sim_time = clock_->now().seconds() - relative_wall_age * actual_rtf;

  if (sim_time < 0) {
    return rclcpp::Time(0, 0, clock_->get_clock_type());
  }

  const double secs     = floor(sim_time);
  const double nanosecs = (sim_time - secs) * 1e9;

  return rclcpp::Time(secs, nanosecs, clock_->get_clock_type());
}

//}

// | ------------------------ camera routines ------------------------ |

/* fabricateCamInfo() //{ */

void FlightforgeSimulator::fabricateCamInfo(void) {

  // | --------------------------- RGB -------------------------- |

  rgb_camera_info_.height = rgb_height_;
  rgb_camera_info_.width  = rgb_width_;

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

}  // namespace mrs_uav_flightforge_simulator

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(mrs_uav_flightforge_simulator::FlightforgeSimulator)
