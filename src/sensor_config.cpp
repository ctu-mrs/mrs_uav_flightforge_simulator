#include <mrs_uav_flightforge_simulator/sensor_config.h>

#include <algorithm>
#include <map>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace mrs_uav_flightforge_simulator
{

const std::vector<std::string> kSensorTypes = {"rgb", "rgb_segmented", "depth", "instance_seg", "fisheye", "stereo", "event", "lidar", "rangefinder"};

namespace
{

/* class BlockLoader //{ */

class BlockLoader {

public:
  BlockLoader(mrs_lib::ParamLoader& loader, const std::string& type_path, int index)
      : loader_(loader), default_path_(type_path + "default/"), sensor_path_(type_path + "sensor_" + std::to_string(index) + "/") {
  }

  template <class T>
  T get(const std::string& key) {

    T value;

    loader_.loadParamReusable(default_path_ + key, value);

    if (!loader_.loadedSuccessfully()) {
      throw std::runtime_error("missing parameter " + default_path_ + key);
    }

    loader_.loadParam(sensor_path_ + key, value, value);

    return value;
  }

  double positive(const std::string& key) {

    const double value = get<double>(key);

    if (value <= 0) {
      throw std::runtime_error(sensor_path_ + key + " must be positive, got " + std::to_string(value));
    }

    return value;
  }

  int positiveInt(const std::string& key) {

    const int value = get<int>(key);

    if (value <= 0) {
      throw std::runtime_error(sensor_path_ + key + " must be positive, got " + std::to_string(value));
    }

    return value;
  }

  Mount mount() {
    return Mount{get<double>("offset/x"), get<double>("offset/y"),     get<double>("offset/z"),
                 get<double>("rotation/roll"), get<double>("rotation/pitch"), get<double>("rotation/yaw")};
  }

private:
  mrs_lib::ParamLoader& loader_;
  std::string           default_path_;
  std::string           sensor_path_;
};

//}

/* loadCamera() //{ */

CameraParams loadCamera(BlockLoader& b) {

  CameraParams p;

  p.rate                   = b.positive("rate");
  p.width                  = b.positiveInt("width");
  p.height                 = b.positiveInt("height");
  p.fov                    = b.positive("fov");
  p.mount                  = b.mount();
  p.enable_hdr             = b.get<bool>("enable_hdr");
  p.enable_temporal_aa     = b.get<bool>("enable_temporal_aa");
  p.enable_raytracing      = b.get<bool>("enable_raytracing");
  p.enable_motion_blur     = b.get<bool>("enable_motion_blur");
  p.motion_blur_amount     = b.get<double>("motion_blur_amount");
  p.motion_blur_distortion = b.get<double>("motion_blur_distortion");

  return p;
}

//}

/* loadFisheye() //{ */

FisheyeParams loadFisheye(BlockLoader& b) {

  FisheyeParams p;

  p.rate       = b.positive("rate");
  p.width      = b.positiveInt("width");
  p.height     = b.positiveInt("height");
  p.fov        = b.positive("fov");
  p.lens_model = b.get<int>("lens_model");
  p.full_frame = b.get<bool>("full_frame");
  p.mount      = b.mount();

  return p;
}

//}

/* loadStereo() //{ */

StereoParams loadStereo(BlockLoader& b) {

  StereoParams p;

  p.rate               = b.positive("rate");
  p.width              = b.positiveInt("width");
  p.height             = b.positiveInt("height");
  p.fov                = b.positive("fov");
  p.baseline           = b.positive("baseline");
  p.mount              = b.mount();
  p.enable_hdr         = b.get<bool>("enable_hdr");
  p.enable_temporal_aa = b.get<bool>("enable_temporal_aa");
  p.enable_raytracing  = b.get<bool>("enable_raytracing");

  return p;
}

//}

/* loadEvent() //{ */

EventParams loadEvent(BlockLoader& b) {

  EventParams p;

  p.rate                   = b.positive("rate");
  p.width                  = b.positiveInt("width");
  p.height                 = b.positiveInt("height");
  p.fov                    = b.positive("fov");
  p.contrast_threshold_pos = b.positive("contrast_threshold_pos");
  p.contrast_threshold_neg = b.positive("contrast_threshold_neg");
  p.max_events_per_frame   = b.positiveInt("max_events_per_frame");
  p.mount                  = b.mount();

  return p;
}

//}

/* loadLidar() //{ */

LidarParams loadLidar(BlockLoader& b) {

  LidarParams p;

  p.rate                 = b.positive("rate");
  p.livox                = b.get<bool>("livox");
  p.livox_sensor         = b.get<std::string>("livox_sensor");
  p.intensity            = b.get<bool>("intensity");
  p.labels               = b.get<bool>("labels");
  p.publish_free         = b.get<bool>("publish_free");
  p.horizontal_rays      = b.positiveInt("horizontal_rays");
  p.vertical_rays        = b.positiveInt("vertical_rays");
  p.horizontal_fov_left  = b.get<double>("horizontal_fov_left");
  p.horizontal_fov_right = b.get<double>("horizontal_fov_right");
  p.vertical_fov_up      = b.get<double>("vertical_fov_up");
  p.vertical_fov_down    = b.get<double>("vertical_fov_down");
  p.beam_length          = b.positive("beam_length");
  p.mount                = b.mount();
  p.noise.enabled        = b.get<bool>("noise/enabled");
  p.noise.std_at_1m      = b.get<double>("noise/std_at_1m");
  p.noise.std_slope      = b.get<double>("noise/std_slope");

  return p;
}

//}

/* loadRangefinder() //{ */

RangefinderParams loadRangefinder(BlockLoader& b) {
  return RangefinderParams{b.positive("rate"), b.mount()};
}

//}

/* loadType() //{ */

template <class Params, class Fn>
std::vector<Params> loadType(mrs_lib::ParamLoader& loader, const std::string& prefix, const std::string& type, Fn load_one) {

  const std::string type_path = prefix + "sensors/" + type + "/";

  int n_sensors;
  loader.loadParam(type_path + "n_sensors", n_sensors);

  if (!loader.loadedSuccessfully()) {
    throw std::runtime_error("missing parameter " + type_path + "n_sensors");
  }

  if (n_sensors < 0) {
    throw std::runtime_error(type_path + "n_sensors must not be negative, got " + std::to_string(n_sensors));
  }

  std::vector<Params> sensors;

  for (int i = 0; i < n_sensors; i++) {
    BlockLoader block(loader, type_path, i);
    sensors.push_back(load_one(block));
  }

  return sensors;
}

//}

}  // namespace

/* SensorsConfig::highestRate() //{ */

double SensorsConfig::highestRate() const {

  double highest = 0;

  const auto take = [&highest](const auto& sensors) {
    for (const auto& s : sensors) {
      highest = std::max(highest, s.rate);
    }
  };

  take(rgb);
  take(rgb_segmented);
  take(depth);
  take(instance_seg);
  take(fisheye);
  take(stereo);
  take(event);
  take(lidar);
  take(rangefinder);

  return highest;
}

//}

/* loadSensors() //{ */

SensorsConfig loadSensors(mrs_lib::ParamLoader& loader, const std::string& prefix) {

  SensorsConfig cfg;

  cfg.rgb           = loadType<CameraParams>(loader, prefix, "rgb", loadCamera);
  cfg.rgb_segmented = loadType<CameraParams>(loader, prefix, "rgb_segmented", loadCamera);
  cfg.depth         = loadType<CameraParams>(loader, prefix, "depth", loadCamera);
  cfg.instance_seg  = loadType<CameraParams>(loader, prefix, "instance_seg", loadCamera);
  cfg.fisheye       = loadType<FisheyeParams>(loader, prefix, "fisheye", loadFisheye);
  cfg.stereo        = loadType<StereoParams>(loader, prefix, "stereo", loadStereo);
  cfg.event         = loadType<EventParams>(loader, prefix, "event", loadEvent);
  cfg.lidar         = loadType<LidarParams>(loader, prefix, "lidar", loadLidar);
  cfg.rangefinder   = loadType<RangefinderParams>(loader, prefix, "rangefinder", loadRangefinder);

  return cfg;
}

//}

/* validateSensorKeys() //{ */

std::string validateSensorKeys(const std::vector<std::string>& yaml_files, const std::string& root_key) {

  std::vector<YAML::Node> sensors;

  for (const auto& file : yaml_files) {
    const YAML::Node root = YAML::LoadFile(file);
    if (root[root_key] && root[root_key]["sensors"]) {
      sensors.push_back(root[root_key]["sensors"]);
    }
  }

  // the first file that sets a count wins, as in ParamLoader
  std::map<std::string, int> counts;

  for (const auto& s : sensors) {
    for (const auto& type : s) {
      const auto name = type.first.as<std::string>();
      if (type.second["n_sensors"] && !counts.count(name)) {
        counts[name] = type.second["n_sensors"].as<int>();
      }
    }
  }

  const std::string base = root_key + "/sensors/";

  for (const auto& s : sensors) {
    for (const auto& type : s) {

      const auto name = type.first.as<std::string>();

      if (std::find(kSensorTypes.begin(), kSensorTypes.end(), name) == kSensorTypes.end()) {
        return "unknown sensor type " + base + name;
      }

      for (const auto& child : type.second) {

        const auto key = child.first.as<std::string>();

        if (key == "n_sensors" || key == "default") {
          continue;
        }

        if (key.rfind("sensor_", 0) == 0) {
          try {
            size_t    used  = 0;
            const int index = std::stoi(key.substr(7), &used);
            if (used == key.size() - 7 && index >= 0 && index < counts[name]) {
              continue;
            }
          }
          catch (const std::exception&) {
          }
        }

        return "unexpected key " + base + name + "/" + key + " (n_sensors is " + std::to_string(counts[name]) + ")";
      }
    }
  }

  return "";
}

//}

}  // namespace mrs_uav_flightforge_simulator
