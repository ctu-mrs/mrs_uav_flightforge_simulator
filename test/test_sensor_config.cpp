#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include <rclcpp/rclcpp.hpp>
#include <mrs_lib/param_loader.h>
#include <yaml-cpp/yaml.h>

#include <mrs_uav_flightforge_simulator/sensor_config.h>

using namespace mrs_uav_flightforge_simulator;

namespace
{

const std::string kRoot = "mrs_uav_flightforge_simulator";

std::string fixture(const std::string& name) {
  return std::string(TEST_FIXTURES) + "/" + name;
}

SensorsConfig load(const std::string& path) {
  static int  counter = 0;
  auto        node    = std::make_shared<rclcpp::Node>("test_sensor_config_" + std::to_string(counter++));
  mrs_lib::ParamLoader loader(node, node->get_name());
  loader.addYamlFile(path);
  return loadSensors(loader, kRoot + "/");
}

template <class T>
SensorsConfig loadWith(const std::string& key, const T& value) {
  YAML::Node root = YAML::LoadFile(fixture("sensors_valid.yaml"));
  YAML::Node node = root[kRoot]["sensors"];

  std::stringstream ss(key);
  std::string       part;
  std::vector<std::string> parts;
  while (std::getline(ss, part, '/')) {
    parts.push_back(part);
  }
  for (size_t i = 0; i + 1 < parts.size(); i++) {
    node.reset(node[parts[i]]);
  }
  node[parts.back()] = value;

  const auto path = std::filesystem::temp_directory_path() / "test_sensor_config_override.yaml";
  std::ofstream(path) << root;
  return load(path.string());
}

}  // namespace

TEST(SensorConfig, DefaultsFallBackPerKey) {
  auto cfg = load(fixture("sensors_valid.yaml"));
  ASSERT_EQ(cfg.rgb.size(), 2u);
  EXPECT_DOUBLE_EQ(cfg.rgb[0].rate, 30.0);
  EXPECT_DOUBLE_EQ(cfg.rgb[1].rate, 10.0);
  EXPECT_DOUBLE_EQ(cfg.rgb[1].mount.pitch, 90.0);
  EXPECT_DOUBLE_EQ(cfg.rgb[1].mount.yaw, cfg.rgb[0].mount.yaw);
  EXPECT_DOUBLE_EQ(cfg.rgb[1].mount.x, 0.14);
  EXPECT_EQ(cfg.rgb[1].width, 640);
  EXPECT_EQ(cfg.lidar.size(), 1u);
  EXPECT_TRUE(cfg.lidar[0].noise.enabled);
  EXPECT_TRUE(cfg.depth.empty());
  EXPECT_DOUBLE_EQ(cfg.highestRate(), 30.0);
}

TEST(SensorConfig, NegativeCountIsFatal) {
  EXPECT_THROW(loadWith("rgb/n_sensors", -1), std::runtime_error);
}

TEST(SensorConfig, NonPositiveRateIsFatal) {
  EXPECT_THROW(loadWith("rgb/sensor_1/rate", 0.0), std::runtime_error);
}

TEST(SensorConfig, ErrorNamesThePath) {
  try {
    loadWith("rgb/sensor_1/width", 0);
    FAIL();
  }
  catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("rgb/sensor_1/width"), std::string::npos) << e.what();
  }
}

TEST(SensorConfig, MisspeltTypeIsRejected) {
  EXPECT_NE(validateSensorKeys({fixture("sensors_bad_type.yaml")}, kRoot).find("sensors/rbg"), std::string::npos);
}

TEST(SensorConfig, BlockBeyondCountIsRejected) {
  EXPECT_NE(validateSensorKeys({fixture("sensors_bad_key.yaml")}, kRoot).find("rgb/sensor_2"), std::string::npos);
}

TEST(SensorConfig, ValidFilePasses) {
  EXPECT_EQ(validateSensorKeys({fixture("sensors_valid.yaml")}, kRoot), "");
}

std::string writeTemp(const std::string& name, const std::string& text) {
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream(path) << text;
  return path.string();
}

TEST(SensorConfig, CustomCountOfZeroOverridesDefaultBlocks) {
  const auto custom = writeTemp("test_sensor_config_custom.yaml", "mrs_uav_flightforge_simulator:\n  sensors:\n    lidar:\n      n_sensors: 0\n");
  EXPECT_EQ(validateSensorKeys({custom, fixture("sensors_valid.yaml")}, kRoot), "");
}

TEST(SensorConfig, MisspeltKeyInsideABlockIsRejected) {
  const auto custom = writeTemp("test_sensor_config_typo.yaml", "mrs_uav_flightforge_simulator:\n  sensors:\n    rgb:\n      sensor_1:\n        rat: 5.0\n");
  EXPECT_NE(validateSensorKeys({custom, fixture("sensors_valid.yaml")}, kRoot).find("rgb/sensor_1/rat"), std::string::npos);
}

TEST(SensorConfig, MisspeltNestedKeyIsRejected) {
  const auto custom = writeTemp("test_sensor_config_nested.yaml", "mrs_uav_flightforge_simulator:\n  sensors:\n    rgb:\n      sensor_0:\n        rotation: {pich: 5.0}\n");
  EXPECT_NE(validateSensorKeys({custom, fixture("sensors_valid.yaml")}, kRoot).find("rgb/sensor_0/rotation/pich"), std::string::npos);
}

TEST(Devices, OverridesFallBackPerKey) {
  const auto cfg = load(fixture("sensors_valid.yaml"));
  ASSERT_EQ(cfg.devices.size(), 3u);
  EXPECT_EQ(cfg.devices[1].type, "realsense_d435i");
  EXPECT_DOUBLE_EQ(cfg.devices[0].camera_rate, 30.0);
  EXPECT_DOUBLE_EQ(cfg.devices[1].camera_rate, 15.0);
  EXPECT_DOUBLE_EQ(cfg.devices[1].mount.x, 0.14);
  EXPECT_DOUBLE_EQ(cfg.devices[1].mount.z, 0.1);
  EXPECT_TRUE(cfg.devices[0].show_mesh);
}

TEST(Devices, NamesCountPerType) {
  const auto cfg = load(fixture("sensors_valid.yaml"));
  EXPECT_EQ(deviceNames(cfg.devices), (std::vector<std::string>{"livox_mid360_0", "realsense_d435i_0", "livox_mid360_1"}));
}

TEST(Devices, MissingTypeIsFatal) {
  const auto path = writeTemp("test_devices_no_type.yaml", "mrs_uav_flightforge_simulator:\n  devices:\n    n_devices: 4\n");
  auto       node = std::make_shared<rclcpp::Node>("test_devices_no_type");

  mrs_lib::ParamLoader loader(node, node->get_name());
  loader.addYamlFile(path);
  loader.addYamlFile(fixture("sensors_valid.yaml"));

  try {
    loadSensors(loader, kRoot + "/");
    FAIL();
  }
  catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("device_3/type"), std::string::npos) << e.what();
  }
}

TEST(Devices, BlockBeyondCountIsRejected) {
  const auto custom = writeTemp("test_devices_count.yaml", "mrs_uav_flightforge_simulator:\n  devices:\n    n_devices: 1\n    device_1:\n      type: \"odin1\"\n");
  EXPECT_NE(validateSensorKeys({custom, fixture("sensors_valid.yaml")}, kRoot).find("devices/device_1"), std::string::npos);
}

TEST(Devices, MisspeltKeyIsRejected) {
  const auto custom = writeTemp("test_devices_typo.yaml", "mrs_uav_flightforge_simulator:\n  devices:\n    device_0:\n      camera_rat: 5.0\n");
  EXPECT_NE(validateSensorKeys({custom, fixture("sensors_valid.yaml")}, kRoot).find("device_0/camera_rat"), std::string::npos);
}

TEST(SensorConfig, ShippedConfigsPass) {
  const std::string package = std::string(TEST_FIXTURES) + "/../..";
  EXPECT_EQ(validateSensorKeys({package + "/tmux/one_drone/config/simulator.yaml", package + "/config/flightforge_simulator.yaml"}, kRoot), "");
  EXPECT_EQ(validateSensorKeys({package + "/config/flightforge_simulator.yaml"}, kRoot), "");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  const int ret = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return ret;
}
