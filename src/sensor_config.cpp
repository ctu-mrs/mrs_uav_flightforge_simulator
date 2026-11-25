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
  BlockLoader(mrs_lib::ParamLoader& loader, const std::string& type_path, const std::string& instance, int index)
      : loader_(loader), default_path_(type_path + "default/"), sensor_path_(type_path + instance + std::to_string(index) + "/") {
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

  double nonNegative(const std::string& key) {

    const double value = get<double>(key);

    if (value < 0) {
      throw std::runtime_error(sensor_path_ + key + " must not be negative, got " + std::to_string(value));
    }

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

  const std::string& path() const {
    return sensor_path_;
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
  p.backend              = b.get<std::string>("backend");
  if (p.backend != "auto" && p.backend != "cpu" && p.backend != "gpu") {
    throw std::runtime_error(b.path() + "backend must be auto, cpu or gpu, got " + p.backend);
  }
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

/* loadDevice() //{ */

DeviceParams loadDevice(BlockLoader& b) {

  DeviceParams p;

  p.type = b.get<std::string>("type");

  if (p.type.empty()) {
    throw std::runtime_error(b.path() + "type must name a device");
  }

  p.camera_rate     = b.positive("camera_rate");
  p.lidar_rate      = b.nonNegative("lidar_rate");
  p.show_mesh       = b.get<bool>("show_mesh");
  p.mount           = b.mount();
  p.intensity       = b.get<bool>("intensity");
  p.labels          = b.get<bool>("labels");
  p.publish_free    = b.get<bool>("publish_free");
  p.noise.enabled   = b.get<bool>("noise/enabled");
  p.noise.std_at_1m = b.get<double>("noise/std_at_1m");
  p.noise.std_slope = b.get<double>("noise/std_slope");

  return p;
}

//}

/* loadType() //{ */

template <class Params, class Fn>
std::vector<Params> loadInstances(mrs_lib::ParamLoader& loader, const std::string& path, const std::string& count_key, const std::string& instance, Fn load_one) {

  int count;
  loader.loadParam(path + count_key, count);

  if (!loader.loadedSuccessfully()) {
    throw std::runtime_error("missing parameter " + path + count_key);
  }

  if (count < 0) {
    throw std::runtime_error(path + count_key + " must not be negative, got " + std::to_string(count));
  }

  std::vector<Params> instances;

  for (int i = 0; i < count; i++) {
    BlockLoader block(loader, path, instance, i);
    instances.push_back(load_one(block));
  }

  return instances;
}

template <class Params, class Fn>
std::vector<Params> loadType(mrs_lib::ParamLoader& loader, const std::string& prefix, const std::string& type, Fn load_one) {
  return loadInstances<Params>(loader, prefix + "sensors/" + type + "/", "n_sensors", "sensor_", load_one);
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
  cfg.devices       = loadInstances<DeviceParams>(loader, prefix + "devices/", "n_devices", "device_", loadDevice);

  return cfg;
}

//}

/* deviceNames() //{ */

std::vector<std::string> deviceNames(const std::vector<DeviceParams>& devices) {

  std::map<std::string, int> seen;
  std::vector<std::string>   names;

  for (const auto& d : devices) {
    names.push_back(d.type + "_" + std::to_string(seen[d.type]++));
  }

  return names;
}

//}

/* loadScene() //{ */

std::string loadScene(const std::string& path) {

  YAML::Node scene;

  try {
    scene = YAML::LoadFile(path);
  }
  catch (const YAML::Exception& e) {
    throw std::runtime_error("could not read scene " + path + ": " + e.what());
  }

  if (!scene.IsMap() || !scene["objects"] || !scene["objects"].IsSequence()) {
    throw std::runtime_error("scene " + path + " has no list of objects");
  }

  // z = 0 means the floor here, as it does for a UAV's spawn
  if (!scene["frame"]) {
    scene["frame"] = "ground";
  }

  return YAML::Dump(scene);
}

//}

/* validateSensorKeys() //{ */

namespace
{

// "" when every key under block also exists under reference
std::string unknownKey(const YAML::Node& block, const YAML::Node& reference, const std::string& path) {

  if (!block.IsMap()) {
    return "";
  }

  for (const auto& child : block) {

    const auto key = child.first.as<std::string>();

    if (!reference.IsMap() || !reference[key]) {
      return path + key;
    }

    const auto nested = unknownKey(child.second, reference[key], path + key + "/");

    if (!nested.empty()) {
      return nested;
    }
  }

  return "";
}

// "" when every key of block is the count, the default or an in-range instance holding only default keys
std::string checkInstances(const YAML::Node& block, const YAML::Node& defaults, const std::string& path, const std::string& count_key, const std::string& instance,
                           int count) {

  for (const auto& child : block) {

    const auto key = child.first.as<std::string>();

    if (key == count_key || key == "default") {
      continue;
    }

    bool in_range = false;

    if (key.rfind(instance, 0) == 0) {
      try {
        size_t    used  = 0;
        const int index = std::stoi(key.substr(instance.size()), &used);
        in_range        = used == key.size() - instance.size() && index >= 0 && index < count;
      }
      catch (const std::exception&) {
      }
    }

    if (!in_range) {
      return "unexpected key " + path + key + " (" + count_key + " is " + std::to_string(count) + ")";
    }

    const auto unknown = unknownKey(child.second, defaults, path + key + "/");

    if (!unknown.empty()) {
      return "unknown parameter " + unknown + ", it is not in the default block";
    }
  }

  return "";
}

}  // namespace

std::string validateSensorKeys(const std::vector<std::string>& yaml_files, const std::string& root_key) {

  std::vector<YAML::Node> sensors, devices;

  for (const auto& file : yaml_files) {
    const YAML::Node root = YAML::LoadFile(file);
    if (root[root_key] && root[root_key]["sensors"]) {
      sensors.push_back(root[root_key]["sensors"]);
    }
    if (root[root_key] && root[root_key]["devices"]) {
      devices.push_back(root[root_key]["devices"]);
    }
  }

  // the first file that sets a count or a default wins, as in ParamLoader
  std::map<std::string, int>        counts;
  std::map<std::string, YAML::Node> defaults;

  for (const auto& s : sensors) {
    for (const auto& type : s) {
      const auto name = type.first.as<std::string>();
      if (type.second["n_sensors"] && !counts.count(name)) {
        counts[name] = type.second["n_sensors"].as<int>();
      }
      if (type.second["default"] && !defaults.count(name)) {
        defaults[name] = type.second["default"];
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

      // a file's own count bounds its own blocks
      const int count = type.second["n_sensors"] ? type.second["n_sensors"].as<int>() : counts[name];

      const auto error = checkInstances(type.second, defaults[name], base + name + "/", "n_sensors", "sensor_", count);

      if (!error.empty()) {
        return error;
      }
    }
  }

  int        n_devices = 0;
  YAML::Node device_defaults;

  for (auto it = devices.rbegin(); it != devices.rend(); ++it) {
    if ((*it)["n_devices"]) {
      n_devices = (*it)["n_devices"].as<int>();
    }
    if ((*it)["default"]) {
      device_defaults = (*it)["default"];
    }
  }

  for (const auto& d : devices) {

    const int count = d["n_devices"] ? d["n_devices"].as<int>() : n_devices;

    const auto error = checkInstances(d, device_defaults, root_key + "/devices/", "n_devices", "device_", count);

    if (!error.empty()) {
      return error;
    }
  }

  return "";
}

//}

}  // namespace mrs_uav_flightforge_simulator
