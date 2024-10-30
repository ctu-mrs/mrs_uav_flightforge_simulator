/* includes //{ */

#include <ros/ros.h>
#include <nodelet/nodelet.h>

#include <mrs_multirotor_simulator/uav_system_ros.h>

#include <rosgraph_msgs/Clock.h>

#include <geometry_msgs/PoseArray.h>

#include <mrs_lib/param_loader.h>
#include <mrs_lib/publisher_handler.h>
#include <mrs_lib/scope_timer.h>
#include <mrs_lib/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <dynamic_reconfigure/server.h>
#include <mrs_uav_unreal_simulation/unreal_simulatorConfig.h>

#include <Eigen/Dense>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/CameraInfo.h>

#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <ueds_connector/ueds_connector.h>
#include <ueds_connector/game_mode_controller.h>

#include <pcl_ros/transforms.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>

#include <random>

//}

/* defines //{ */

#define API_VERSION 2

//}

using PCLPoint               = pcl::PointXYZ;
using PCLPointCloud          = pcl::PointCloud<PCLPoint>;
using PCLPointCloudColor     = pcl::PointCloud<pcl::PointXYZRGB>;
using PCLPointCloudIntensity = pcl::PointCloud<pcl::PointXYZI>;
namespace mrs_uav_unreal_simulation
{

/* class UnrealSimulator //{ */

class UnrealSimulator : public nodelet::Nodelet {

public:
  virtual void onInit();

private:
  ros::NodeHandle   nh_;
  std::atomic<bool> is_initialized_;

  std::shared_ptr<image_transport::ImageTransport> it_;

  // | ------------------------- params ------------------------- |

  double _simulation_rate_;
  bool   _collisions_;

  ros::Time  sim_time_;
  std::mutex mutex_sim_time_;

  double _clock_min_dt_;

  std::string _world_frame_name_;

  // | ------------------------- timers ------------------------- |

  ros::WallTimer timer_dynamics_;
  void           timerDynamics(const ros::WallTimerEvent& event);

  ros::WallTimer timer_status_;
  void           timerStatus(const ros::WallTimerEvent& event);

  ros::WallTimer timer_time_sync_;
  void           timerTimeSync(const ros::WallTimerEvent& event);

  ros::Timer timer_unreal_sync_;
  void       timerUnrealSync(const ros::TimerEvent& event);

  ros::Timer timer_rangefinder_;
  void       timerRangefinder(const ros::TimerEvent& event);

  ros::Timer timer_lidar_;
  void       timerLidar(const ros::TimerEvent& event);

  ros::Timer timer_seg_lidar_;
  void       timerSegLidar(const ros::TimerEvent& event);

  ros::Timer timer_int_lidar_;
  void       timerIntLidar(const ros::TimerEvent& event);

  ros::Timer timer_rgb_;
  void       timerRgb(const ros::TimerEvent& event);

  ros::Timer timer_rgb_segmented_;
  void       timerRgbSegmented(const ros::TimerEvent& event);

  ros::Timer timer_stereo_;
  void       timerStereo(const ros::TimerEvent& event);

  // | --------------------- service servers -------------------- |

  ros::ServiceServer service_server_realtime_factor_;

  bool callbackSetRealtimeFactor(mrs_msgs::Float64Srv::Request& req, mrs_msgs::Float64Srv::Response& res);

  // | --------------------------- tfs -------------------------- |

  tf2_ros::StaticTransformBroadcaster static_broadcaster_;

  // | ----------------------- camera info ---------------------- |

  sensor_msgs::CameraInfo rgb_camera_info_;
  sensor_msgs::CameraInfo stereo_camera_info_;

  geometry_msgs::TransformStamped rgb_camera_tf_;
  geometry_msgs::TransformStamped stereo_camera_tf_;

  // | --------------------------- rtf -------------------------- |

  double    actual_rtf_ = 1.0;
  ros::Time last_sim_time_status_;

  // | ------------------------- sensors ------------------------ |

  double lidar_horizontal_fov_;
  double lidar_vertical_fov_;
  int    lidar_horizontal_rays_;
  int    lidar_vertical_rays_;
  double lidar_offset_x_;
  double lidar_offset_y_;
  double lidar_offset_z_;
  double lidar_rotation_pitch_;
  double lidar_rotation_roll_;
  double lidar_rotation_yaw_;
  double lidar_beam_length_;
  bool   lidar_show_beams_;

  int    rgb_width_;
  int    rgb_height_;
  double rgb_fov_;
  double rgb_offset_x_;
  double rgb_offset_y_;
  double rgb_offset_z_;
  double rgb_rotation_pitch_;
  double rgb_rotation_yaw_;
  double rgb_rotation_roll_;
  bool   rgb_enable_hdr_;
  bool   rgb_enable_temporal_aa_;
  bool   rgb_enable_raytracing_;

  double stereo_baseline_;
  int    stereo_width_;
  int    stereo_height_;
  double stereo_fov_;
  double stereo_offset_x_;
  double stereo_offset_y_;
  double stereo_offset_z_;
  double stereo_rotation_pitch_;
  double stereo_rotation_yaw_;
  double stereo_rotation_roll_;
  bool   stereo_enable_hdr_;
  bool   stereo_enable_temporal_aa_;
  bool   stereo_enable_raytracing_;

  double lidar_int_grass;
  double lidar_int_road;
  double lidar_int_tree;
  double lidar_int_building;
  double lidar_int_fence;
  double lidar_int_dirt_road;
  double lidar_int_other;
  bool   lidar_int_noise;


  // | ----------------------- publishers ----------------------- |

  mrs_lib::PublisherHandler<rosgraph_msgs::Clock>     ph_clock_;
  mrs_lib::PublisherHandler<geometry_msgs::PoseArray> ph_poses_;

  std::vector<mrs_lib::PublisherHandler<sensor_msgs::Range>>       ph_rangefinders_;

  std::vector<mrs_lib::PublisherHandler<sensor_msgs::PointCloud2>> ph_lidars_;
  std::vector<mrs_lib::PublisherHandler<sensor_msgs::PointCloud2>> ph_seg_lidars_;
  std::vector<mrs_lib::PublisherHandler<sensor_msgs::PointCloud2>> ph_int_lidars_;

  std::vector<image_transport::Publisher> imp_rgb_;
  std::vector<image_transport::Publisher> imp_stereo_left_;
  std::vector<image_transport::Publisher> imp_stereo_right_;
  std::vector<image_transport::Publisher> imp_rgbd_segmented_;

  std::vector<mrs_lib::PublisherHandler<sensor_msgs::CameraInfo>> ph_rgb_camera_info_;
  std::vector<mrs_lib::PublisherHandler<sensor_msgs::CameraInfo>> ph_rgb_seg_camera_info_;
  std::vector<mrs_lib::PublisherHandler<sensor_msgs::CameraInfo>> ph_stereo_left_camera_info_;
  std::vector<mrs_lib::PublisherHandler<sensor_msgs::CameraInfo>> ph_stereo_right_camera_info_;

  // | ------------------------- system ------------------------- |

  std::vector<std::shared_ptr<mrs_multirotor_simulator::UavSystemRos>> uavs_;

  // | -------------------------- time -------------------------- |

  ros::Time last_published_time_;

  // | ------------------------- methods ------------------------ |

  void publishPoses(void);

  // | --------------- dynamic reconfigure server --------------- |

  boost::recursive_mutex                                    mutex_drs_;
  typedef mrs_uav_unreal_simulation::unreal_simulatorConfig DrsConfig_t;
  typedef dynamic_reconfigure::Server<DrsConfig_t>          Drs_t;
  boost::shared_ptr<Drs_t>                                  drs_;
  void                                                      callbackDrs(mrs_uav_unreal_simulation::unreal_simulatorConfig& config, uint32_t level);
  DrsConfig_t                                               drs_params_;
  std::mutex                                                mutex_drs_params_;

  // | ------------------------- Unreal ------------------------- |

  std::unique_ptr<ueds_connector::GameModeController> ueds_game_controller_;

  std::vector<std::shared_ptr<ueds_connector::UedsConnector>> ueds_connectors_;

  std::mutex mutex_ueds_;

  ueds_connector::Coordinates ueds_world_origin_;

  std::vector<ueds_connector::Coordinates> ueds_world_origins_;

  ueds_connector::Coordinates position2ue(const Eigen::Vector3d &pos, const ueds_connector::Coordinates &ueds_world_origin);

  void updateUnrealPoses(const bool teleport_without_collision);

  void checkForCrash(void);

  void fabricateCamInfo(void);

  void publishStaticTfs(void);

  // how much to add to unreal time to get to our wall time
  double        wall_time_offset_             = 0;
  double        wall_time_offset_drift_slope_ = 0;
  ros::WallTime last_sync_time_;
  std::mutex    mutex_wall_time_offset_;

  double ueds_fps_                    = 0;
  int    ueds_world_level_name_enum_  = 2;
  int    ueds_graphics_settings_enum_ = 0;
  int    ueds_forest_density_         = 5;
  int    ueds_forest_hilly_level_     = 3;

  std::vector<double> last_rgb_ue_stamp_;
  std::vector<double> last_rgb_seg_ue_stamp_;
  std::vector<double> last_stereo_ue_stamp_;

  double uedsToWallTime(const double ueds_time);

  std::default_random_engine rng;
};

//}

/* onInit() //{ */

void UnrealSimulator::onInit() {

  is_initialized_ = false;

  nh_ = nodelet::Nodelet::getMTPrivateNodeHandle();

  if (!(nh_.hasParam("/use_sim_time"))) {
    nh_.setParam("/use_sim_time", true);
  }

  srand(time(NULL));

  sim_time_ = ros::Time(ros::WallTime::now().toSec());

  last_published_time_  = sim_time_;
  last_sim_time_status_ = sim_time_;

  it_ = std::make_shared<image_transport::ImageTransport>(nh_);

  mrs_lib::ParamLoader param_loader(nh_, "UnrealSimulator");

  std::string custom_config_path;

  param_loader.loadParam("custom_config", custom_config_path);

  if (custom_config_path != "") {
    param_loader.addYamlFile(custom_config_path);
  }

  param_loader.addYamlFileFromParam("config");
  param_loader.addYamlFileFromParam("config_uavs");

  param_loader.loadParam("ueds_graphics_settings_enum", ueds_graphics_settings_enum_);
  param_loader.loadParam("ueds_world_level_name_enum", ueds_world_level_name_enum_);
  param_loader.loadParam("ueds_forest_density", ueds_forest_density_);
  param_loader.loadParam("ueds_forest_hilly_level", ueds_forest_hilly_level_);

  param_loader.loadParam("simulation_rate", _simulation_rate_);
  param_loader.loadParam("realtime_factor", drs_params_.realtime_factor);
  param_loader.loadParam("dynamic_rtf", drs_params_.dynamic_rtf);
  param_loader.loadParam("frames/world/name", _world_frame_name_);

  param_loader.loadParam("collisions", _collisions_);

  param_loader.loadParam("sensors/rangefinder/enabled", drs_params_.rangefinder_enabled);
  param_loader.loadParam("sensors/rangefinder/rate", drs_params_.rangefinder_rate);

  param_loader.loadParam("sensors/lidar/enabled", drs_params_.lidar_enabled);
  param_loader.loadParam("sensors/lidar/rate", drs_params_.lidar_rate);
  param_loader.loadParam("sensors/lidar/lidar_segmented/enabled", drs_params_.lidar_seg_enabled);
  param_loader.loadParam("sensors/lidar/lidar_segmented/rate", drs_params_.lidar_seg_rate);
  param_loader.loadParam("sensors/lidar/lidar_intensity/enabled", drs_params_.lidar_int_enabled);
  param_loader.loadParam("sensors/lidar/lidar_intensity/rate", drs_params_.lidar_int_rate);
  param_loader.loadParam("sensors/lidar/noise/enabled", drs_params_.lidar_noise_enabled);
  param_loader.loadParam("sensors/lidar/noise/std_at_1m", drs_params_.lidar_std_at_1m);
  param_loader.loadParam("sensors/lidar/noise/std_slope", drs_params_.lidar_std_slope);

  param_loader.loadParam("sensors/lidar/horizontal_fov", lidar_horizontal_fov_);
  param_loader.loadParam("sensors/lidar/vertical_fov", lidar_vertical_fov_);
  param_loader.loadParam("sensors/lidar/horizontal_rays", lidar_horizontal_rays_);
  param_loader.loadParam("sensors/lidar/vertical_rays", lidar_vertical_rays_);
  param_loader.loadParam("sensors/lidar/offset_x", lidar_offset_x_);
  param_loader.loadParam("sensors/lidar/offset_y", lidar_offset_y_);
  param_loader.loadParam("sensors/lidar/offset_z", lidar_offset_z_);
  param_loader.loadParam("sensors/lidar/rotation_pitch", lidar_rotation_pitch_);
  param_loader.loadParam("sensors/lidar/rotation_roll", lidar_rotation_roll_);
  param_loader.loadParam("sensors/lidar/rotation_yaw", lidar_rotation_yaw_);
  param_loader.loadParam("sensors/lidar/beam_length", lidar_beam_length_);
  param_loader.loadParam("sensors/lidar/show_beams", lidar_show_beams_);

  param_loader.loadParam("sensors/rgb/enabled", drs_params_.rgb_enabled);
  param_loader.loadParam("sensors/rgb/rate", drs_params_.rgb_rate);

  param_loader.loadParam("sensors/rgb/rgb_segmented/enabled", drs_params_.rgb_segmented_enabled);
  param_loader.loadParam("sensors/rgb/rgb_segmented/rate", drs_params_.rgb_segmented_rate);

  param_loader.loadParam("sensors/rgb/width", rgb_width_);
  param_loader.loadParam("sensors/rgb/height", rgb_height_);
  param_loader.loadParam("sensors/rgb/fov", rgb_fov_);
  param_loader.loadParam("sensors/rgb/offset_x", rgb_offset_x_);
  param_loader.loadParam("sensors/rgb/offset_y", rgb_offset_y_);
  param_loader.loadParam("sensors/rgb/offset_z", rgb_offset_z_);
  param_loader.loadParam("sensors/rgb/rotation_pitch", rgb_rotation_pitch_);
  param_loader.loadParam("sensors/rgb/rotation_roll", rgb_rotation_roll_);
  param_loader.loadParam("sensors/rgb/rotation_yaw", rgb_rotation_yaw_);
  param_loader.loadParam("sensors/rgb/enable_hdr", rgb_enable_hdr_);
  param_loader.loadParam("sensors/rgb/enable_temporal_aa", rgb_enable_temporal_aa_);
  param_loader.loadParam("sensors/rgb/enable_raytracing", rgb_enable_raytracing_);

  param_loader.loadParam("sensors/stereo/enabled", drs_params_.stereo_enabled);
  param_loader.loadParam("sensors/stereo/rate", drs_params_.stereo_rate);

  param_loader.loadParam("sensors/stereo/width", stereo_width_);
  param_loader.loadParam("sensors/stereo/height", stereo_height_);
  param_loader.loadParam("sensors/stereo/fov", stereo_fov_);
  param_loader.loadParam("sensors/stereo/baseline", stereo_baseline_);
  param_loader.loadParam("sensors/stereo/offset_x", stereo_offset_x_);
  param_loader.loadParam("sensors/stereo/offset_y", stereo_offset_y_);
  param_loader.loadParam("sensors/stereo/offset_z", stereo_offset_z_);
  param_loader.loadParam("sensors/stereo/rotation_pitch", stereo_rotation_pitch_);
  param_loader.loadParam("sensors/stereo/rotation_roll", stereo_rotation_roll_);
  param_loader.loadParam("sensors/stereo/rotation_yaw", stereo_rotation_yaw_);
  param_loader.loadParam("sensors/stereo/enable_hdr", stereo_enable_hdr_);
  param_loader.loadParam("sensors/stereo/enable_temporal_aa", stereo_enable_temporal_aa_);
  param_loader.loadParam("sensors/stereo/enable_raytracing", stereo_enable_raytracing_);

  param_loader.loadParam("sensors/lidar/lidar_intensity/values/grass", drs_params_.lidar_int_value_grass);
  param_loader.loadParam("sensors/lidar/lidar_intensity/values/road", drs_params_.lidar_int_value_road);
  param_loader.loadParam("sensors/lidar/lidar_intensity/values/tree", drs_params_.lidar_int_value_tree);
  param_loader.loadParam("sensors/lidar/lidar_intensity/values/building", drs_params_.lidar_int_value_building);
  param_loader.loadParam("sensors/lidar/lidar_intensity/values/fence", drs_params_.lidar_int_value_fence);
  param_loader.loadParam("sensors/lidar/lidar_intensity/values/dirt_road", drs_params_.lidar_int_value_dirt_road);
  param_loader.loadParam("sensors/lidar/lidar_intensity/values/other", drs_params_.lidar_int_value_other);
  param_loader.loadParam("sensors/lidar/lidar_intensity/noise/enabled", drs_params_.lidar_int_noise_enabled);
  param_loader.loadParam("sensors/lidar/lidar_intensity/noise/std_at_1m", drs_params_.lidar_int_std_at_1m);
  param_loader.loadParam("sensors/lidar/lidar_intensity/noise/std_slope", drs_params_.lidar_int_std_slope);

  double clock_rate;
  param_loader.loadParam("clock_rate", clock_rate);

  drs_params_.paused = false;

  std::vector<std::string> uav_names;

  param_loader.loadParam("uav_names", uav_names);

  for (size_t i = 0; i < uav_names.size(); i++) {

    std::string uav_name = uav_names[i];

    ROS_INFO("[UnrealSimulator]: initializing '%s'", uav_name.c_str());

    uavs_.push_back(std::make_unique<mrs_multirotor_simulator::UavSystemRos>(nh_, uav_name));
  }

  // | ----------- initialize the Unreal Sim connector ---------- |

  ueds_game_controller_ = std::make_unique<ueds_connector::GameModeController>(LOCALHOST, 8000);

  while (true) {

    bool connect_result = ueds_game_controller_->Connect();

    if (connect_result != 1) {
      ROS_ERROR("[UnrealSimulator]: Error connecting to Unreal's game mode controller, connect_result was %d", connect_result);
    } else {
      break;
    }

    //ros::Duration(1.0).sleep();
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  // | ------------------ check the API version ----------------- |

  auto [res, ueds_api_version] = ueds_game_controller_->GetApiVersion();

  if (!res || ueds_api_version != API_VERSION) {

    ROS_ERROR("[UnrealSimulator]: the API versions don't match! (ROS side '%d' != Unreal side '%d')", API_VERSION, ueds_api_version);
    ROS_ERROR("[UnrealSimulator]:");
    ROS_ERROR("[UnrealSimulator]: Solution:");
    ROS_ERROR("[UnrealSimulator]:           1. make sure the mrs_uav_unreal_simulation package is up to date");
    ROS_ERROR("[UnrealSimulator]:              sudo apt update && sudo apt upgrade");
    ROS_ERROR("[UnrealSimulator]:");
    ROS_ERROR("[UnrealSimulator]:           2. make sure you have the right version of the Unreal Simulator 'game'");
    ROS_ERROR("[UnrealSimulator]:              download at: https://github.com/ctu-mrs/mrs_uav_unreal_simulation");
    ROS_ERROR("[UnrealSimulator]:");

    ros::shutdown();
  }

  // | --------------------- Set graphical settings and choose World Level --------------------- |

  Serializable::GameMode::WorldLevelEnum worldLevelEnum = Serializable::GameMode::WorldLevelEnum(ueds_world_level_name_enum_);
  res                                                   = ueds_game_controller_->SwitchWorldLevel(worldLevelEnum);
  if (res) {
    ROS_INFO("[UnrealSimulator] World was switched succesfully.");
  } else {
    ROS_ERROR("[UnrealSimulator] World was not switched succesfully.");
  }

  res = ueds_game_controller_->Disconnect();
  if (res) {
    ROS_INFO("[UnrealSimulator] ueds_game_controller_ was Disconnected succesfully.");
  } else {
    ROS_ERROR("[UnrealSimulator] ueds_game_controller_ was not Disconnected succesfully.");
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));

  while (true) {
    bool connect_result = ueds_game_controller_->Connect();
    if (connect_result != 1) {
      ROS_ERROR("[UnrealSimulator]: Error connecting to Unreal's game mode controller, connect_result was %d", connect_result);
    } else {
      break;
    }
    // ros::Duration(1.0).sleep();
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  Serializable::GameMode::GraphicsSettingsEnum graphicsSettings = Serializable::GameMode::GraphicsSettingsEnum(ueds_graphics_settings_enum_);
  res                                                           = ueds_game_controller_->SetGraphicsSettings(graphicsSettings);

  if (res) {
    ROS_INFO("[UnrealSimulator]: Graphical Settings was set succesfully to '%d'", graphicsSettings);
  } else {
    ROS_ERROR("[UnrealSimulator]: Graphical Settings was not set succesfully to '%d'", graphicsSettings);
  }

  // | --------------------- These graphical settings influence only Forest Game World --------------------- |

  res = ueds_game_controller_->SetForestDensity(ueds_forest_density_);
  if (res) {
    ROS_INFO("[UnrealSimulator]: Forest Density was set succesfully to '%d'", ueds_forest_density_);
  } else {
    ROS_ERROR("[UnrealSimulator]: Forest Density wasn't set succesfully to '%d'", ueds_forest_density_);
  }

  res = ueds_game_controller_->SetForestHillyLevel(ueds_forest_hilly_level_);
  if (res) {
    ROS_INFO("[UnrealSimulator]: Forest Hilly Level was set succesfully to '%d'", ueds_forest_hilly_level_);
  } else {
    ROS_ERROR("[UnrealSimulator]: Forest Hilly Level wasn't set succesfully to '%d'", ueds_forest_hilly_level_);
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));

  // | --------------------- Spawn the UAVs --------------------- |

  const auto [result, world_origin] = ueds_game_controller_->GetWorldOrigin();

  if (!result) {
    ROS_ERROR("[UnrealSimulator]: GameError: getting world origin");
    ros::shutdown();
  } else {
    ueds_world_origin_ = world_origin;
  }

  for (size_t i = 0; i < uav_names.size(); i++) {

    const std::string uav_name = uav_names[i];

    mrs_multirotor_simulator::MultirotorModel::State uav_state = uavs_[i]->getState();

    ueds_connector::Coordinates pos = position2ue(uav_state.x, ueds_world_origin_);

    ROS_INFO("[UnrealSimulator]: %s spawning at [%.2lf, %.2lf, %.2lf] ...", uav_name.c_str(), uav_state.x.x(), uav_state.x.y(), uav_state.x.z());

    std::string uav_type; // = "x500";
    param_loader.loadParam(uav_names[i] + "/type", uav_type);

    ROS_INFO("[UnrealSimulator]: Frame type to spawn is %s", uav_type.c_str());

    auto [resSpawn, port] = ueds_game_controller_->SpawnDroneAtLocation(pos, uav_type);

    if (!resSpawn) {
      ROS_ERROR("[UnrealSimulator]: failed to spawn %s", uav_names[i].c_str());
      ros::shutdown();
    }

    ROS_INFO("[UnrealSimulator]: %s spawned", uav_name.c_str());

    std::shared_ptr<ueds_connector::UedsConnector> ueds_connector = std::make_shared<ueds_connector::UedsConnector>(LOCALHOST, port);

    ueds_connectors_.push_back(ueds_connector);


    auto connect_result = ueds_connector->Connect();

    if (connect_result != 1) {

      ROS_ERROR("[UnrealSimulator]: %s - Error connecting to drone controller, connect_result was %d", uav_name.c_str(), connect_result);
      ros::shutdown();

    }
    else {

      ROS_INFO("[UnrealSimulator]: %s - Connection succeed: %d", uav_name.c_str(), connect_result);

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

    ph_rangefinders_.push_back(mrs_lib::PublisherHandler<sensor_msgs::Range>(nh_, "/" + uav_name + "/rangefinder", 10));

    ph_lidars_.push_back(mrs_lib::PublisherHandler<sensor_msgs::PointCloud2>(nh_, "/" + uav_name + "/lidar/points", 10));
    ph_seg_lidars_.push_back(mrs_lib::PublisherHandler<sensor_msgs::PointCloud2>(nh_, "/" + uav_name + "/lidar_segmented/points", 10));
    ph_int_lidars_.push_back(mrs_lib::PublisherHandler<sensor_msgs::PointCloud2>(nh_, "/" + uav_name + "/lidar_intensity/points", 10));

    imp_rgb_.push_back(it_->advertise("/" + uav_name + "/rgb/image_raw", 10));
    imp_stereo_left_.push_back(it_->advertise("/" + uav_name + "/stereo/left/image_raw", 10));
    imp_stereo_right_.push_back(it_->advertise("/" + uav_name + "/stereo/right/image_raw", 10));
    imp_rgbd_segmented_.push_back(it_->advertise("/" + uav_name + "/rgb_segmented/image_raw", 10));

    ph_rgb_camera_info_.push_back(mrs_lib::PublisherHandler<sensor_msgs::CameraInfo>(nh_, "/" + uav_name + "/rgb/camera_info", 10));
    ph_rgb_seg_camera_info_.push_back(mrs_lib::PublisherHandler<sensor_msgs::CameraInfo>(nh_, "/" + uav_name + "/rgb_segmented/camera_info", 10));

    ph_stereo_left_camera_info_.push_back(mrs_lib::PublisherHandler<sensor_msgs::CameraInfo>(nh_, "/" + uav_name + "/stereo/left/camera_info", 10));
    ph_stereo_right_camera_info_.push_back(mrs_lib::PublisherHandler<sensor_msgs::CameraInfo>(nh_, "/" + uav_name + "/stereo/right/camera_info", 10));

    // | ------------------ set RGB camera config ----------------- |

    {
      ueds_connector::RgbCameraConfig cameraConfig{};

      cameraConfig.width_              = rgb_width_;
      cameraConfig.height_             = rgb_height_;
      cameraConfig.fov_                = rgb_fov_;
      cameraConfig.offset_             = ueds_connector::Coordinates(rgb_offset_x_ * 100.0, rgb_offset_y_ * 100.0, rgb_offset_z_ * 100.0);
      cameraConfig.orientation_        = ueds_connector::Rotation(-rgb_rotation_pitch_, rgb_rotation_yaw_, rgb_rotation_roll_);
      cameraConfig.enable_raytracing_  = rgb_enable_raytracing_;
      cameraConfig.enable_hdr_         = rgb_enable_hdr_;
      cameraConfig.enable_temporal_aa_ = rgb_enable_temporal_aa_;

      const auto res = ueds_connectors_[i]->SetRgbCameraConfig(cameraConfig);

      last_rgb_ue_stamp_.push_back(0.0);
      last_rgb_seg_ue_stamp_.push_back(0.0);

      if (!res) {
        ROS_ERROR("[UnrealSimulator]: failed to set camera config for uav %lu", i + 1);
      } else {
        ROS_INFO("[UnrealSimulator]: camera config set for uav%lu", i + 1);
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
        ROS_ERROR("[UnrealSimulator]: failed to set camera config for uav %lu", i + 1);
      } else {
        ROS_INFO("[UnrealSimulator]: camera config set for uav%lu", i + 1);
      }
    }

    // | -------------------- set LiDAR config -------------------- |

    {
      ueds_connector::LidarConfig lidarConfig{};

      lidarConfig.BeamHorRays  = lidar_horizontal_rays_;
      lidarConfig.BeamVertRays = lidar_vertical_rays_;
      lidarConfig.FOVVert      = lidar_vertical_fov_;
      lidarConfig.FOVHor       = lidar_horizontal_fov_;
      lidarConfig.beamLength   = lidar_beam_length_ * 100.0;
      lidarConfig.offset       = ueds_connector::Coordinates(lidar_offset_x_ * 100.0, lidar_offset_y_ * 100.0, lidar_offset_z_ * 100.0);
      lidarConfig.orientation  = ueds_connector::Rotation(-lidar_rotation_pitch_, lidar_rotation_yaw_, lidar_rotation_roll_);
      lidarConfig.showBeams    = lidar_show_beams_;

      /* LidarConfig. */

      const auto res = ueds_connectors_[i]->SetLidarConfig(lidarConfig);

      if (!res) {
        ROS_ERROR("[UnrealSimulator]: failed to set lidar config for uav %lu", i + 1);
      } else {
        ROS_INFO("[UnrealSimulator]: lidar config set for uav%lu", i + 1);
      }
    }
  }

  // ROS_INFO("[UnrealSimulator]: teleporting the UAVs to their spawn positions");

  // updateUnrealPoses(true);

  ROS_INFO("[UnrealSimulator]: Unreal UAVs are initialized");

  // | --------------- dynamic reconfigure server --------------- |

  drs_.reset(new Drs_t(mutex_drs_, nh_));
  drs_->updateConfig(drs_params_);
  Drs_t::CallbackType f = boost::bind(&UnrealSimulator::callbackDrs, this, _1, _2);
  drs_->setCallback(f);

  if (!param_loader.loadedSuccessfully()) {
    ROS_ERROR("[UnrealSimulator]: could not load all parameters!");
    ros::shutdown();
  }

  _clock_min_dt_ = 1.0 / clock_rate;

  // | ----------------------- publishers ----------------------- |

  ph_clock_ = mrs_lib::PublisherHandler<rosgraph_msgs::Clock>(nh_, "clock_out", 10, false);

  ph_poses_ = mrs_lib::PublisherHandler<geometry_msgs::PoseArray>(nh_, "uav_poses_out", 10, false);

  // | --------------------- service servers -------------------- |

  service_server_realtime_factor_ = nh_.advertiseService("set_realtime_factor_in", &UnrealSimulator::callbackSetRealtimeFactor, this);

  // | ------------------------- timers ------------------------- |

  timer_dynamics_ = nh_.createWallTimer(ros::WallDuration(1.0 / (_simulation_rate_ * drs_params_.realtime_factor)), &UnrealSimulator::timerDynamics, this);

  timer_status_ = nh_.createWallTimer(ros::WallDuration(1.0), &UnrealSimulator::timerStatus, this);

  timer_time_sync_ = nh_.createWallTimer(ros::WallDuration(1.0), &UnrealSimulator::timerTimeSync, this);

  timer_rangefinder_ = nh_.createTimer(ros::Duration(1.0 / drs_params_.rangefinder_rate), &UnrealSimulator::timerRangefinder, this);

  timer_lidar_ = nh_.createTimer(ros::Duration(1.0 / drs_params_.lidar_rate), &UnrealSimulator::timerLidar, this);

  timer_unreal_sync_ = nh_.createTimer(ros::Duration(1.0 / _simulation_rate_), &UnrealSimulator::timerUnrealSync, this);

  timer_seg_lidar_ = nh_.createTimer(ros::Duration(1.0 / drs_params_.lidar_rate), &UnrealSimulator::timerSegLidar, this);

  timer_int_lidar_ = nh_.createTimer(ros::Duration(1.0 / drs_params_.lidar_rate), &UnrealSimulator::timerIntLidar, this);

  timer_rgb_ = nh_.createTimer(ros::Duration(1.0 / drs_params_.rgb_rate), &UnrealSimulator::timerRgb, this);

  timer_stereo_ = nh_.createTimer(ros::Duration(1.0 / drs_params_.stereo_rate), &UnrealSimulator::timerStereo, this);

  timer_rgb_segmented_ = nh_.createTimer(ros::Duration(1.0 / drs_params_.rgb_segmented_rate), &UnrealSimulator::timerRgbSegmented, this);

  // | -------------------- finishing methods ------------------- |

  fabricateCamInfo();

  publishStaticTfs();

  // | ----------------------- finish init ---------------------- |

  is_initialized_ = true;

  ROS_INFO("[UnrealSimulator]: initialized");
}

//}

// | ------------------------- timers ------------------------- |

/* timerDynamics() //{ */

void UnrealSimulator::timerDynamics([[maybe_unused]] const ros::WallTimerEvent& event) {

  if (!is_initialized_) {
    return;
  }

  ROS_INFO_ONCE("[UnrealSimulator]: main timer spinning");

  // | ------------------ make simulation step ------------------ |

  double simulation_step_size = 1.0 / _simulation_rate_;

  sim_time_ = sim_time_ + ros::Duration(simulation_step_size);

  {
    std::scoped_lock lock(mutex_wall_time_offset_);

    const double wall_dt = (event.current_real - event.last_real).toSec();

    if (wall_dt > 0) {

      wall_time_offset_ += wall_time_offset_drift_slope_ * wall_dt;
    }
  }

  for (size_t i = 0; i < uavs_.size(); i++) {
    if (!uavs_[i]->hasCrashed()) {
      uavs_[i]->makeStep(simulation_step_size);
    }
  }

  publishPoses();

  // | ---------------------- publish time ---------------------- |

  if ((sim_time_ - last_published_time_).toSec() >= _clock_min_dt_) {

    rosgraph_msgs::Clock ros_time;

    ros_time.clock.fromSec(sim_time_.toSec());

    ph_clock_.publish(ros_time);

    last_published_time_ = sim_time_;
  }
}

//}

/* timerStatus() //{ */

void UnrealSimulator::timerStatus([[maybe_unused]] const ros::WallTimerEvent& event) {

  if (!is_initialized_) {
    return;
  }

  auto sim_time   = mrs_lib::get_mutexed(mutex_sim_time_, sim_time_);
  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  ros::Duration last_sec_sim_dt = sim_time - last_sim_time_status_;

  last_sim_time_status_ = sim_time;

  double last_sec_rtf = last_sec_sim_dt.toSec() / 1.0;

  actual_rtf_ = 0.9 * actual_rtf_ + 0.1 * last_sec_rtf;

  double fps;

  {
    std::scoped_lock lock(mutex_ueds_);

    bool res;

    std::tie(res, fps) = ueds_game_controller_->GetFps();

    if (!res) {
      ROS_ERROR("[UnrealSimulator]: failed to get FPS from ueds");
      return;
    }
  }

  // filter out the ueds fps
  const double fps_filter_const = 0.9;
  ueds_fps_                     = fps_filter_const * ueds_fps_ + (1.0 - fps_filter_const) * fps;

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

  const double ueds_rtf_ = ueds_fps_ / highest_fps;

  const double desired_rtf = (drs_params.dynamic_rtf && ueds_rtf_ < drs_params.realtime_factor) ? ueds_rtf_ : drs_params.realtime_factor;

  timer_dynamics_.setPeriod(ros::WallDuration(1.0 / (_simulation_rate_ * desired_rtf)), false);

  if (_collisions_) {
    checkForCrash();
  }

  ROS_INFO_THROTTLE(0.1, "[UnrealSimulator]: %s, desired RTF = %.2f, actual RTF = %.2f, ueds FPS = %.2f", drs_params.paused ? "paused" : "running", desired_rtf,
                    actual_rtf_, fps);
}

//}

/* timerUnrealSync() //{ */

void UnrealSimulator::timerUnrealSync([[maybe_unused]] const ros::TimerEvent& event) {

  if (!is_initialized_) {
    return;
  }

  updateUnrealPoses(false);
}

//}

/* timerTimeSync() //{ */

void UnrealSimulator::timerTimeSync([[maybe_unused]] const ros::WallTimerEvent& event) {

  if (!is_initialized_) {
    return;
  }

  auto wall_time_offset = mrs_lib::get_mutexed(mutex_wall_time_offset_, wall_time_offset_);

  const double sync_start = ros::WallTime::now().toSec();

  bool   res;
  double ueds_time;

  {
    std::scoped_lock lock(mutex_ueds_);

    std::tie(res, ueds_time) = ueds_game_controller_->GetTime();
  }

  const double sync_end = ros::WallTime::now().toSec();

  if (!res) {
    ROS_ERROR("[UnrealSimulator]: failed to get ueds's time");
    ros::shutdown();
  }

  const double true_ueds_time = ueds_time - (sync_end - sync_start) / 2.0;

  const double new_wall_time_offset = sync_start - true_ueds_time;

  // | --------------- time drift slope estimation -------------- |

  if (event.current_real.toSec() > 0 && event.last_real.toSec() > 0) {

    const double wall_dt = (event.current_real - event.last_real).toSec();

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

    last_sync_time_ = ros::WallTime::now();
  }

  ROS_DEBUG("[UnrealSimulator]: wall time %f ueds %f time offset: %f, offset slope %f s/s", sync_start, ueds_time, wall_time_offset_,
            wall_time_offset_drift_slope_);
}

//}

void UnrealSimulator::timerRangefinder([[maybe_unused]] const ros::TimerEvent& event){
  if(!is_initialized_){
    return;
  }

  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  if (!drs_params_.rangefinder_enabled) {
    return;
  }

  for(size_t i = 0; i < uavs_.size(); i++){
    bool res;
    double range;
    {
      std::scoped_lock lock(mutex_ueds_);

      std::tie(res, range) = ueds_connectors_[i]->GetRangefinderData();
    }

    if(!res){
      ROS_ERROR_THROTTLE(1.0, "[UnrealSimulator]: [uav%d] - ERROR GetRangefinderData", int(i));
      continue;
    }

    sensor_msgs::Range msg_range;
    msg_range.header.stamp    = ros::Time::now();
    msg_range.header.frame_id = "uav" + std::to_string(i + 1) + "/fcu";
    msg_range.radiation_type  = 1;
    msg_range.min_range       = 0.1;
    msg_range.max_range       = 30;
    msg_range.range           = range / 100;

    ph_rangefinders_[i].publish(msg_range);
  }
}

/* timerLidar() //{ */

void UnrealSimulator::timerLidar([[maybe_unused]] const ros::TimerEvent& event) {

  if (!is_initialized_) {
    return;
  }

  /* mrs_lib::ScopeTimer timer = mrs_lib::ScopeTimer("timerLidar()"); */

  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  if (!drs_params_.lidar_enabled) {
    return;
  }

  for (size_t i = 0; i < uavs_.size(); i++) {

    //mrs_multirotor_simulator::MultirotorModel::State state = uavs_[i]->getState();

    bool                                   res;
    std::vector<ueds_connector::LidarData> lidarData;
    ueds_connector::Coordinates            start;

    {
      std::scoped_lock lock(mutex_ueds_);

      std::tie(res, lidarData, start) = ueds_connectors_[i]->GetLidarData();
    }

    if (!res) {
      ROS_ERROR_THROTTLE(1.0, "[UnrealSimulator]: [uav%d] - ERROR getLidarData", int(i));
      continue;
    }

    sensor_msgs::PointCloud2 pcl_msg;

    // Modifier to describe what the fields are.
    sensor_msgs::PointCloud2Modifier modifier(pcl_msg);
    modifier.setPointCloud2Fields(4, "x", 1, sensor_msgs::PointField::FLOAT32, "y", 1, sensor_msgs::PointField::FLOAT32, "z", 1,
                                  sensor_msgs::PointField::FLOAT32, "intensity", 1, sensor_msgs::PointField::FLOAT32);
    // Msg header
    pcl_msg.header.stamp    = ros::Time::now();
    pcl_msg.header.frame_id = "uav" + std::to_string(i + 1) + "/fcu";

    pcl_msg.height   = lidar_horizontal_rays_;
    pcl_msg.width    = lidar_vertical_rays_;
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

      tf::Vector3 dir = tf::Vector3(ray.directionX, ray.directionY, ray.directionZ);

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

void UnrealSimulator::timerSegLidar([[maybe_unused]] const ros::TimerEvent& event) {

  if (!is_initialized_) {
    return;
  }

  /* mrs_lib::ScopeTimer timer = mrs_lib::ScopeTimer("timerLidar()"); */

  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  if (!drs_params_.lidar_seg_enabled) {
    return;
  }

  for (size_t i = 0; i < uavs_.size(); i++) {

    //mrs_multirotor_simulator::MultirotorModel::State state = uavs_[i]->getState();

    bool                                      res;
    std::vector<ueds_connector::LidarSegData> lidarSegData;
    ueds_connector::Coordinates               start;

    {
      std::scoped_lock lock(mutex_ueds_);

      std::tie(res, lidarSegData, start) = ueds_connectors_[i]->GetLidarSegData();
    }

    if (!res) {
      ROS_ERROR("[UnrealSimulator]: [uav%d] - ERROR getLidarSegData", int(i));
      continue;
    }

    PCLPointCloudColor pcl_cloud;

    for (const ueds_connector::LidarSegData& ray : lidarSegData) {

      pcl::PointXYZRGB point;

      tf::Vector3 dir = tf::Vector3(ray.directionX, ray.directionY, ray.directionZ);

      double ray_distance = ray.distance / 100.0;

      if (drs_params.lidar_noise_enabled && ray_distance > 0) {

        const double std = ray_distance * drs_params.lidar_std_slope * drs_params.lidar_std_at_1m;

        std::normal_distribution<double> distribution(0, std);

        ray_distance += distribution(rng);
      }

      dir = dir.normalized() * ray_distance;

      point.x = dir.x();
      point.y = -dir.y();  // convert left-hand to right-hand coordinates
      point.z = dir.z();

      switch (ray.segmentation) {
        case 0: {
          point.r = 0;
          point.g = 0;
          point.b = 255;
          break;
        }
        case 1: {
          point.r = 0;
          point.g = 0;
          point.b = 255;
          break;
        }
        case 2: {
          point.r = 255;
          point.g = 0;
          point.b = 0;
          break;
        }
        case 3: {
          point.r = 0;
          point.g = 255;
          point.b = 0;
          break;
        }
        case 4: {
          point.r = 120;
          point.g = 255;
          point.b = 0;
          break;
        }
        case 5: {
          point.r = 255;
          point.g = 150;
          point.b = 255;
          break;
        }
        case 6: {
          point.r = 255;
          point.g = 75;
          point.b = 0;
          break;
        }
        case 7: {
          point.r = 0;
          point.g = 20;
          point.b = 255;
          break;
        }
        case 8: {
          point.r = 255;
          point.g = 255;
          point.b = 0;
          break;
        }
        default: {
          point.r = 0;
          point.g = 0;
          point.b = 0;
          break;
        }
      }

      pcl_cloud.push_back(point);
    }

    sensor_msgs::PointCloud2 pcl_msg;
    pcl::toROSMsg(pcl_cloud, pcl_msg);
    pcl_msg.header.stamp    = ros::Time::now();
    pcl_msg.header.frame_id = "uav" + std::to_string(i + 1) + "/fcu";

    ph_seg_lidars_[i].publish(pcl_msg);
  }
}  // namespace mrs_uav_unreal_simulation

//}

/* timerIntLidar() //{ */

void UnrealSimulator::timerIntLidar([[maybe_unused]] const ros::TimerEvent& event) {

  if (!is_initialized_) {
    return;
  }

  /* mrs_lib::ScopeTimer timer = mrs_lib::ScopeTimer("timerLidar()"); */

  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  if (!drs_params_.lidar_int_enabled) {
    return;
  }

  for (size_t i = 0; i < uavs_.size(); i++) {

    mrs_multirotor_simulator::MultirotorModel::State state = uavs_[i]->getState();

    bool                                      res;
    std::vector<ueds_connector::LidarIntData> lidarIntData;
    ueds_connector::Coordinates               start;

    {
      std::scoped_lock lock(mutex_ueds_);

      std::tie(res, lidarIntData, start) = ueds_connectors_[i]->GetLidarIntData();
    }

    if (!res) {
      ROS_ERROR("[UnrealSimulator]: [uav%d] - ERROR getLidarIntData", int(i));
      continue;
    }

    PCLPointCloudIntensity pcl_cloud;

    for (const ueds_connector::LidarIntData& ray : lidarIntData) {

      pcl::PointXYZI point;

      tf::Vector3 dir = tf::Vector3(ray.directionX, ray.directionY, ray.directionZ);

      double ray_distance = ray.distance / 100.0;

      if (drs_params.lidar_noise_enabled && ray_distance > 0) {

        const double std = ray_distance * drs_params.lidar_std_slope * drs_params.lidar_std_at_1m;

        std::normal_distribution<double> distribution(0, std);

        ray_distance += distribution(rng);
      }

      dir = dir.normalized() * ray_distance;

      point.x = dir.x();
      point.y = -dir.y();  // convert left-hand to right-hand coordinates
      point.z = dir.z();
      /* TODO: add the noise to the intensity */
      switch (ray.intensity) {
        case 0: {
          point.intensity = drs_params_.lidar_int_value_other;
          break;
        }
        case 1: {
          point.intensity = drs_params_.lidar_int_value_grass;
          break;
        }
        case 2: {
          point.intensity = drs_params_.lidar_int_value_road;
          break;
        }
        case 3: {
          point.intensity = drs_params_.lidar_int_value_tree;
          break;
        }
        case 4: {
          point.intensity = drs_params_.lidar_int_value_building;
          break;
        }
        case 5: {
          point.intensity = drs_params_.lidar_int_value_fence;
          break;
        }
        case 6: {
          point.intensity = drs_params_.lidar_int_value_dirt_road;
          break;
        }
      }
      if (drs_params.lidar_int_noise_enabled && ray_distance > 0) {
        const double                     intensity_std = ray_distance * drs_params.lidar_int_std_slope * drs_params.lidar_int_std_at_1m;
        std::normal_distribution<double> intensity_distribution(0, intensity_std);
        point.intensity += intensity_distribution(rng);

        // Clamp intensity to a reasonable range (e.g., 0-255)
        point.intensity = std::clamp(point.intensity, 0.0f, 255.0f);
      }
      pcl_cloud.push_back(point);
    }

    sensor_msgs::PointCloud2 pcl_msg;
    pcl::toROSMsg(pcl_cloud, pcl_msg);
    pcl_msg.header.stamp    = ros::Time::now();
    pcl_msg.header.frame_id = "uav" + std::to_string(i + 1) + "/fcu";

    ph_int_lidars_[i].publish(pcl_msg);
  }
}  // namespace mrs_uav_unreal_simulation

//}

/* timerRgb() //{ */

void UnrealSimulator::timerRgb([[maybe_unused]] const ros::TimerEvent& event) {

  if (!is_initialized_) {
    return;
  }

  /* mrs_lib::ScopeTimer timer = mrs_lib::ScopeTimer("timerRgb()"); */

  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  if (!drs_params_.rgb_enabled) {
    return;
  }

  for (size_t i = 0; i < uavs_.size(); i++) {

    bool                       res;
    std::vector<unsigned char> cameraData;
    uint32_t                   size;
    double                     stamp;

    {
      std::scoped_lock lock(mutex_ueds_);

      std::tie(res, cameraData, stamp, size) = ueds_connectors_[i]->GetRgbCameraData();
    }

    if (abs(stamp - last_rgb_ue_stamp_.at(i)) < 0.001) {
      return;
    }

    last_rgb_ue_stamp_.at(i) = stamp;

    if (!res) {
      ROS_WARN("[UnrealSimulator]: failed to obtain rgb camera from uav%lu", i + 1);
      continue;
    }

    if (cameraData.empty()) {
      ROS_WARN("[UnrealSimulator]: rgb camera from uav%lu is empty!", i + 1);
      continue;
    }

    cv::Mat image = cv::imdecode(cameraData, cv::IMREAD_COLOR);

    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();

    msg->header.frame_id = "uav" + std::to_string(i + 1) + "/rgb";

    const double relative_wall_age = ros::WallTime::now().toSec() - uedsToWallTime(stamp);

    if (abs(relative_wall_age) < 1.0) {
      msg->header.stamp = ros::Time(ros::Time::now().toSec() - (relative_wall_age * actual_rtf_));
    }

    imp_rgb_[i].publish(msg);

    auto camera_info = rgb_camera_info_;

    camera_info.header = msg->header;

    ph_rgb_camera_info_[i].publish(camera_info);
  }
}

//}

/* timerStereo() //{ */

void UnrealSimulator::timerStereo([[maybe_unused]] const ros::TimerEvent& event) {

  if (!is_initialized_) {
    return;
  }

  ROS_INFO_ONCE("[UnrealSimulator]: timereStereo() spinning");

  /* mrs_lib::ScopeTimer timer = mrs_lib::ScopeTimer("timerStereo()"); */

  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  if (!drs_params_.stereo_enabled) {
    return;
  }

  for (size_t i = 0; i < uavs_.size(); i++) {

    bool                       res;
    std::vector<unsigned char> image_left;
    std::vector<unsigned char> image_right;
    double                     stamp;

    {
      std::scoped_lock lock(mutex_ueds_);

      std::tie(res, image_left, image_right, stamp) = ueds_connectors_[i]->GetStereoCameraData();
    }

    if (abs(stamp - last_stereo_ue_stamp_.at(i)) < 0.001) {
      return;
    }

    last_stereo_ue_stamp_.at(i) = stamp;

    if (!res) {
      ROS_WARN("[UnrealSimulator]: failed to obtain stereo camera from uav%lu", i + 1);
      continue;
    }

    if (image_left.empty() || image_right.empty()) {
      ROS_WARN("[UnrealSimulator]: stereo camera data from uav%lu is empty!", i + 1);
      continue;
    }

    cv::Mat cv_left  = cv::imdecode(image_left, cv::IMREAD_COLOR);
    cv::Mat cv_right = cv::imdecode(image_right, cv::IMREAD_COLOR);

    sensor_msgs::ImagePtr msg_left  = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_left).toImageMsg();
    sensor_msgs::ImagePtr msg_right = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_right).toImageMsg();

    msg_left->header.frame_id = "uav" + std::to_string(i + 1) + "/stereo_left";

    const double relative_wall_age = ros::WallTime::now().toSec() - uedsToWallTime(stamp);

    if (abs(relative_wall_age) < 1.0) {
      msg_left->header.stamp = ros::Time(ros::Time::now().toSec() - (relative_wall_age * actual_rtf_));
    }

    msg_right->header.frame_id = "uav" + std::to_string(i + 1) + "/stereo_right";
    msg_right->header.stamp    = msg_left->header.stamp;

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

      camera_info.P[3] = -camera_info.P[0] * stereo_baseline_;

      ph_stereo_right_camera_info_[i].publish(camera_info);
    }
  }
}

//}

/* timerRgbSegmented() //{ */

void UnrealSimulator::timerRgbSegmented([[maybe_unused]] const ros::TimerEvent& event) {

  if (!is_initialized_) {
    return;
  }

  /* mrs_lib::ScopeTimer timer = mrs_lib::ScopeTimer("timerRgbSegmented()"); */

  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  if (!drs_params_.rgb_segmented_enabled) {
    return;
  }

  for (size_t i = 0; i < uavs_.size(); i++) {

    bool                       res;
    std::vector<unsigned char> cameraData;
    uint32_t                   size;
    double                     stamp;

    {
      std::scoped_lock lock(mutex_ueds_);

      std::tie(res, cameraData, stamp, size) = ueds_connectors_[i]->GetRgbSegmented();
    }

    if (abs(stamp - last_rgb_seg_ue_stamp_.at(i)) < 0.001) {
      return;
    }

    last_rgb_seg_ue_stamp_.at(i) = stamp;

    if (!res) {
      ROS_WARN("[UnrealSimulator]: failed to obtain rgb camera from uav%lu", i + 1);
      continue;
    }

    if (cameraData.empty()) {
      ROS_WARN("[UnrealSimulator]: rgb segmented camera from uav%lu is empty!", i + 1);
      continue;
    }

    /* cv::Mat               image = cv::imdecode(cameraData, cv::IMREAD_COLOR); */
    cv::Mat image = cv::Mat(rgb_height_, rgb_width_, CV_8UC3, cameraData.data());
    /* cv::cvtColor(image, image, cv::COLOR_BGR2RGB); */
    sensor_msgs::ImagePtr msg   = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();

    msg->header.frame_id = "uav" + std::to_string(i + 1) + "/rgb";

    const double relative_wall_age = ros::WallTime::now().toSec() - uedsToWallTime(stamp);

    if (abs(relative_wall_age) < 1.0) {
      msg->header.stamp = ros::Time(ros::Time::now().toSec() - (relative_wall_age * actual_rtf_));
    }

    imp_rgbd_segmented_[i].publish(msg);

    auto camera_info = rgb_camera_info_;

    camera_info.header = msg->header;

    ph_rgb_seg_camera_info_[i].publish(camera_info);
  }
}

//}

// | ------------------------ callbacks ----------------------- |

/* callbackDrs() //{ */

void UnrealSimulator::callbackDrs(mrs_uav_unreal_simulation::unreal_simulatorConfig& config, [[maybe_unused]] uint32_t level) {

  {
    // | ----------------- pausing the simulation ----------------- |

    auto old_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

    if (!old_params.paused && config.paused) {
      timer_dynamics_.stop();
    } else if (old_params.paused && !config.paused) {
      timer_dynamics_.start();
    }
  }

  // | --------------------- save the params -------------------- |

  {
    std::scoped_lock lock(mutex_drs_params_);

    drs_params_ = config;
  }

  // | ----------------- set the realtime factor ---------------- |

  timer_dynamics_.setPeriod(ros::WallDuration(1.0 / (_simulation_rate_ * config.realtime_factor)), true);

  // | ------------------ set the camera rates ------------------ |

  timer_rgb_.setPeriod(ros::Duration(1.0 / config.rgb_rate));
  timer_stereo_.setPeriod(ros::Duration(1.0 / config.stereo_rate));
  timer_rgb_segmented_.setPeriod(ros::Duration(1.0 / config.rgb_segmented_rate));
  timer_lidar_.setPeriod(ros::Duration(1.0 / config.lidar_rate));
  timer_seg_lidar_.setPeriod(ros::Duration(1.0 / config.lidar_rate));
  timer_int_lidar_.setPeriod(ros::Duration(1.0 / config.lidar_rate));

  ROS_INFO("[UnrealSimulator]: DRS updated params");
}

//}

// | ------------------------ routines ------------------------ |

/* publishPoses() //{ */

void UnrealSimulator::publishPoses(void) {

  auto sim_time = mrs_lib::get_mutexed(mutex_sim_time_, sim_time_);

  geometry_msgs::PoseArray pose_array;

  pose_array.header.stamp    = sim_time;
  pose_array.header.frame_id = _world_frame_name_;

  for (size_t i = 0; i < uavs_.size(); i++) {

    auto state = uavs_[i]->getState();

    geometry_msgs::Pose pose;

    pose.position.x  = state.x[0];
    pose.position.y  = state.x[1];
    pose.position.z  = state.x[2];
    pose.orientation = mrs_lib::AttitudeConverter(state.R);

    pose_array.poses.push_back(pose);
  }

  ph_poses_.publish(pose_array);
}

//}

/* updateUnrealPoses() //{ */

void UnrealSimulator::updateUnrealPoses(const bool teleport_without_collision) {

  // | ------------ set each UAV's position in unreal ----------- |

  {
    std::scoped_lock lock(mutex_ueds_);

    for (size_t i = 0; i < uavs_.size(); i++) {

      mrs_multirotor_simulator::MultirotorModel::State state = uavs_[i]->getState();

      auto [roll, pitch, yaw] = mrs_lib::AttitudeConverter(state.R).getExtrinsicRPY();

      ueds_connector::Coordinates pos;

      pos = position2ue(state.x, ueds_world_origin_);
      // pos.x = ueds_world_origins_[i].x + state.x.x() * 100.0;
      // pos.y = ueds_world_origins_[i].y - state.x.y() * 100.0;
      // pos.z = ueds_world_origins_[i].z + state.x.z() * 100.0;

      ueds_connector::Rotation rot;
      rot.pitch = 180.0 * (-pitch / M_PI);
      rot.roll  = 180.0 * (roll / M_PI);
      rot.yaw   = 180.0 * (-yaw / M_PI);

      ueds_connectors_[i]->SetLocationAndRotationAsync(pos, rot, !teleport_without_collision && _collisions_);
    }
  }
}

ueds_connector::Coordinates UnrealSimulator::position2ue(const Eigen::Vector3d &pos, const ueds_connector::Coordinates &ueds_world_origin){
  ueds_connector::Coordinates pos_ue;

  pos_ue.x = ueds_world_origin.x + pos.x() * 100.0;
  pos_ue.y = ueds_world_origin.y - pos.y() * 100.0;
  pos_ue.z = ueds_world_origin.z + pos.z() * 100.0;

  return pos_ue;
}

//}

/* checkForCrash() //{ */

void UnrealSimulator::checkForCrash(void) {

  // | ------------ set each UAV's position in unreal ----------- |

  {
    std::scoped_lock lock(mutex_ueds_);

    for (size_t i = 0; i < uavs_.size(); i++) {

      auto [res, crashed] = ueds_connectors_[i]->GetCrashState();

      if (!res) {
        ROS_ERROR_THROTTLE(1.0, "[UnrealSimulator]: failed to obtain crash state for uav%lu", i + 1);
      }

      if (res && crashed && !uavs_[i]->hasCrashed()) {

        ROS_WARN_THROTTLE(1.0, "[UnrealSimulator]: uav%lu crashed", i + 1);

        uavs_[i]->crash();
      }
    }
  }
}

//}

/* uedsToWallTime() //{ */

double UnrealSimulator::uedsToWallTime(const double ueds_time) {

  auto wall_time_offset = mrs_lib::get_mutexed(mutex_wall_time_offset_, wall_time_offset_);

  return ueds_time + wall_time_offset;
}

//}

/* fabricateCamInfo() //{ */

void UnrealSimulator::fabricateCamInfo(void) {

  // | --------------------------- RGB -------------------------- |

  rgb_camera_info_.height = rgb_width_;
  rgb_camera_info_.width  = rgb_height_;

  // distortion
  rgb_camera_info_.distortion_model = "plumb_bob";

  rgb_camera_info_.D.resize(5);
  rgb_camera_info_.D[0] = 0;
  rgb_camera_info_.D[1] = 0;
  rgb_camera_info_.D[2] = 0;
  rgb_camera_info_.D[3] = 0;
  rgb_camera_info_.D[4] = 0;

  // original camera matrix
  rgb_camera_info_.K[0] = rgb_width_ / (2.0 * tan(0.5 * M_PI * (rgb_fov_ / 180.0)));
  rgb_camera_info_.K[1] = 0.0;
  rgb_camera_info_.K[2] = rgb_width_ / 2.0;
  rgb_camera_info_.K[3] = 0.0;
  rgb_camera_info_.K[4] = rgb_width_ / (2.0 * tan(0.5 * M_PI * (rgb_fov_ / 180.0)));
  rgb_camera_info_.K[5] = rgb_height_ / 2.0;
  rgb_camera_info_.K[6] = 0.0;
  rgb_camera_info_.K[7] = 0.0;
  rgb_camera_info_.K[8] = 1.0;

  // rectification
  rgb_camera_info_.R[0] = 1.0;
  rgb_camera_info_.R[1] = 0.0;
  rgb_camera_info_.R[2] = 0.0;
  rgb_camera_info_.R[3] = 0.0;
  rgb_camera_info_.R[4] = 1.0;
  rgb_camera_info_.R[5] = 0.0;
  rgb_camera_info_.R[6] = 0.0;
  rgb_camera_info_.R[7] = 0.0;
  rgb_camera_info_.R[8] = 1.0;

  // camera projection matrix (same as camera matrix due to lack of distortion/rectification) (is this generated?)
  rgb_camera_info_.P[0]  = rgb_width_ / (2.0 * tan(0.5 * M_PI * (rgb_fov_ / 180.0)));
  rgb_camera_info_.P[1]  = 0.0;
  rgb_camera_info_.P[2]  = rgb_width_ / 2.0;
  rgb_camera_info_.P[3]  = 0.0;
  rgb_camera_info_.P[4]  = 0.0;
  rgb_camera_info_.P[5]  = rgb_width_ / (2.0 * tan(0.5 * M_PI * (rgb_fov_ / 180.0)));
  rgb_camera_info_.P[6]  = rgb_height_ / 2.0;
  rgb_camera_info_.P[7]  = 0.0;
  rgb_camera_info_.P[8]  = 0.0;
  rgb_camera_info_.P[9]  = 0.0;
  rgb_camera_info_.P[10] = 1.0;
  rgb_camera_info_.P[11] = 0.0;

  // | ------------------------- stereo ------------------------- |

  stereo_camera_info_.width  = stereo_width_;
  stereo_camera_info_.height = stereo_height_;

  // distortion
  stereo_camera_info_.distortion_model = "plumb_bob";

  stereo_camera_info_.D.resize(5);
  stereo_camera_info_.D[0] = 0;
  stereo_camera_info_.D[1] = 0;
  stereo_camera_info_.D[2] = 0;
  stereo_camera_info_.D[3] = 0;
  stereo_camera_info_.D[4] = 0;

  // original camera matrix
  stereo_camera_info_.K[0] = stereo_width_ / (2.0 * tan(0.5 * M_PI * (stereo_fov_ / 180.0)));
  stereo_camera_info_.K[1] = 0.0;
  stereo_camera_info_.K[2] = stereo_width_ / 2.0;
  stereo_camera_info_.K[3] = 0.0;
  stereo_camera_info_.K[4] = stereo_width_ / (2.0 * tan(0.5 * M_PI * (stereo_fov_ / 180.0)));
  stereo_camera_info_.K[5] = stereo_height_ / 2.0;
  stereo_camera_info_.K[6] = 0.0;
  stereo_camera_info_.K[7] = 0.0;
  stereo_camera_info_.K[8] = 1.0;

  // rectification
  stereo_camera_info_.R[0] = 1.0;
  stereo_camera_info_.R[1] = 0.0;
  stereo_camera_info_.R[2] = 0.0;
  stereo_camera_info_.R[3] = 0.0;
  stereo_camera_info_.R[4] = 1.0;
  stereo_camera_info_.R[5] = 0.0;
  stereo_camera_info_.R[6] = 0.0;
  stereo_camera_info_.R[7] = 0.0;
  stereo_camera_info_.R[8] = 1.0;

  // camera projection matrix (same as camera matrix due to lack of distortion/rectification) (is this generated?)
  stereo_camera_info_.P[0]  = stereo_width_ / (2.0 * tan(0.5 * M_PI * (stereo_fov_ / 180.0)));
  stereo_camera_info_.P[1]  = 0.0;
  stereo_camera_info_.P[2]  = stereo_width_ / 2.0;
  stereo_camera_info_.P[3]  = 0.0;
  stereo_camera_info_.P[4]  = 0.0;
  stereo_camera_info_.P[5]  = stereo_width_ / (2.0 * tan(0.5 * M_PI * (stereo_fov_ / 180.0)));
  stereo_camera_info_.P[6]  = stereo_height_ / 2.0;
  stereo_camera_info_.P[7]  = 0.0;
  stereo_camera_info_.P[8]  = 0.0;
  stereo_camera_info_.P[9]  = 0.0;
  stereo_camera_info_.P[10] = 1.0;
  stereo_camera_info_.P[11] = 0.0;
}

//}

/* publishStaticTfs() //{ */

void UnrealSimulator::publishStaticTfs(void) {

  for (size_t i = 0; i < uavs_.size(); i++) {

    geometry_msgs::TransformStamped tf;

    // | ------------------------- rgb tf ------------------------- |

    {
      tf.header.stamp = ros::Time::now();

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
        static_broadcaster_.sendTransform(tf);
      }
      catch (...) {
        ROS_ERROR("[UnrealSimulator]: could not publish rgb tf");
      }

      // | ------------- print the tf matrix for kalibr ------------- |

      ROS_INFO("[UnrealSimulator]: RGB camera-imu chain for kalibr config:");
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
      printf("  intrinsics: [%f, %f, %f, %f]\n", rgb_camera_info_.K[0], rgb_camera_info_.K[4], rgb_camera_info_.K[2], rgb_camera_info_.K[5]);
      printf("  resolution: [%d, %d]\n", rgb_width_, rgb_height_);
    }

    // | ----------------------- stereo left ---------------------- |

    {
      tf.header.stamp = ros::Time::now();

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
        static_broadcaster_.sendTransform(tf);
      }
      catch (...) {
        ROS_ERROR("[UnrealSimulator]: could not publish stereo left tf");
      }

      // | ------------- print the tf matrix for kalibr ------------- |

      ROS_INFO("[UnrealSimulator]: Stereo left camera-imu chain for kalibr config:");
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
      printf("  intrinsics: [%f, %f, %f, %f]\n", stereo_camera_info_.K[0], stereo_camera_info_.K[4], stereo_camera_info_.K[2], stereo_camera_info_.K[5]);
      printf("  resolution: [%d, %d]\n", stereo_width_, stereo_height_);
    }

    {
      tf.header.stamp = ros::Time::now();

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
        static_broadcaster_.sendTransform(tf);
      }
      catch (...) {
        ROS_ERROR("[UnrealSimulator]: could not publish stereo right tf");
      }

      // | ------------- print the tf matrix for kalibr ------------- |

      ROS_INFO("[UnrealSimulator]: Stereo right camera-imu chain for kalibr config:");
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
      printf("  intrinsics: [%f, %f, %f, %f]\n", stereo_camera_info_.K[0], stereo_camera_info_.K[4], stereo_camera_info_.K[2], stereo_camera_info_.K[5]);
      printf("  resolution: [%d, %d]\n", stereo_width_, stereo_height_);
    }
  }
}

//}

/* callbackSetRealtimeFactor() //{ */

bool UnrealSimulator::callbackSetRealtimeFactor(mrs_msgs::Float64Srv::Request& req, mrs_msgs::Float64Srv::Response& res) {

  if (!is_initialized_) {
    return false;
  }

  auto params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  params.realtime_factor = req.value;

  drs_->updateConfig(params);

  mrs_lib::set_mutexed(mutex_drs_params_, params, drs_params_);

  res.message = "new realtime factor set";
  res.success = true;

  return true;
}

//}

}  // namespace mrs_uav_unreal_simulation

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(mrs_uav_unreal_simulation::UnrealSimulator, nodelet::Nodelet)
