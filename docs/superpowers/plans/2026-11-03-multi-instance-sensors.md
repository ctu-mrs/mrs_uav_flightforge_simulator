# Multi-instance Sensors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Any number of sensors per type per UAV, plus depth, instance segmentation, fisheye and event cameras, configured with `n_sensors` / `sensor_<i>` blocks.

**Architecture:** Pure functions (config loading, connector-data → ROS message conversion) move into a small library `MrsUavFlightforgeSimulator_Sensors` that gtest links. The node keeps fetching, scheduling and publishing, holding one vector of plain per-type structs per UAV, with one sim-clock timer per instance.

**Tech Stack:** ROS 2 Jazzy, C++20, mrs_lib, image_transport, cv_bridge/OpenCV, yaml-cpp, event_camera_msgs + event_camera_codecs (built from source in `~/ws/src`), ament_cmake_gtest, FlightForge connector API 0.23.

**Spec:** `docs/superpowers/specs/2026-11-03-multi-instance-sensors-design.md`

## Global Constraints

- Repo `~/ws/src/mrs_uav_flightfore_simulator`, branch `ff_api_0_23`; build with `cd ~/ws && colcon build --packages-select mrs_uav_flightforge_simulator` in bash after `source /opt/ros/jazzy/setup.bash`.
- Type keys, exactly: `rgb rgb_segmented depth instance_seg fisheye stereo event lidar rangefinder`.
- Topics `/<uav>/<type>_<i>/...`; TF frame `<uav>/<type>_<i>`, static from `<uav>/fcu`; camera frames have the optical rotation `Quaterniond(-0.5, 0.5, -0.5, 0.5)` built in, as `publishStaticTfs` does today.
- Mount → UE: offset `(x*100, -y*100, z*100)` cm, rotation `Rotation(-pitch, yaw, roll)` deg, as the node does today.
- Colour images `bgr8` (decoded JPEG); depth `16UC1` mm; instance ids `32SC1` = `r + (g << 8) + (b << 16)`.
- Every conversion checks data length against width × height before writing; on mismatch it returns `std::nullopt`.
- Comments: one clause, gotchas only. Functions touched get `//{` … `//}` fold markers in the file's existing style.
- Do not touch ports 8551 / 4000–4010. The dev sim runs on 18581 / 14300 (`tmux/one_drone/config/simulator.yaml`).
- Commit after each task with a `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` trailer. Never stage the connector symlinks (`include/cereal`, `include/flight_forge_connector`, `include/kissnet`, `src/flight_forge_connector`), the `tmux/one_drone` test edits, or David's local hunks in `config/flightforge_simulator.yaml` (`simulator_ip`, `rangefinder enabled`, `individual_takeoff_platform`).

## Review Focus

1. A camera whose JPEG decodes to a different size than configured (the sim clamps resolutions): the frame is dropped with a throttled error, not published with a stale `camera_info`. Test in Task 2.
2. `GetCameraInfo` returning two entries for a stereo sensor in either order: left/right are chosen by `eye_`, not by position. Test in Task 2.
3. A lidar scan whose `intensity` / `class_id` vectors are shorter than `points` (labels requested but the sim returns none): no out-of-bounds read; the cloud is dropped. Test in Task 3.
4. An event batch with zero events: an empty packet with a valid header is published, not skipped, so consumers see time advance. Test in Task 4.
5. The same frame fetched twice (the sim has not rendered a new one): each instance compares against its own `last_stamp`, and a repeat is skipped with `continue`, not `return`, which today silently skips every later UAV. Covered in Task 5's live check (no duplicate stamps per topic).

---

### Task 1: Config types, loader and key validation

**Files:**
- Create: `include/mrs_uav_flightforge_simulator/sensor_config.h`, `src/sensor_config.cpp`
- Create: `test/test_sensor_config.cpp`, `test/fixtures/sensors_valid.yaml`, `test/fixtures/sensors_bad_type.yaml`, `test/fixtures/sensors_bad_key.yaml`
- Create: `msg/InstanceMap.msg` (`uint32[] ids`, `string[] actors`, `uint16[] class_ids`, after a `std_msgs/Header header`)
- Modify: `CMakeLists.txt` (new library `MrsUavFlightforgeSimulator_Sensors` from `src/sensor_config.cpp` and, from Task 2, `src/sensor_conversions.cpp`; add it to `LIBRARIES`; link it into the simulator; add `msg/InstanceMap.msg` to `rosidl_generate_interfaces` with `std_msgs`; `if(BUILD_TESTING)` + `ament_add_gtest(test_sensors test/test_*.cpp)` linking the library and passing `TEST_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures"`; find yaml-cpp, event_camera_msgs, event_camera_codecs)
- Modify: `package.xml` (`<depend>` event_camera_msgs, event_camera_codecs, yaml-cpp; `<test_depend>` ament_cmake_gtest)

**Interfaces:**
- Produces, in namespace `mrs_uav_flightforge_simulator`:
  - `struct Mount { double x, y, z; double roll, pitch, yaw; };` m and deg.
  - `struct CameraParams { double rate; int width, height; double fov; Mount mount; bool enable_hdr, enable_temporal_aa, enable_raytracing, enable_motion_blur; double motion_blur_amount, motion_blur_distortion; };`
  - `struct FisheyeParams { double rate; int width, height; double fov; int lens_model; bool full_frame; Mount mount; };`
  - `struct StereoParams { double rate; int width, height; double fov; double baseline; Mount mount; bool enable_hdr, enable_temporal_aa, enable_raytracing; };` baseline in m.
  - `struct EventParams { double rate; int width, height; double fov; double contrast_threshold_pos, contrast_threshold_neg; int max_events_per_frame; Mount mount; };`
  - `struct LidarNoise { bool enabled; double std_at_1m, std_slope; };`
  - `struct LidarParams { double rate; bool livox; std::string livox_sensor; bool intensity, labels, publish_free; int horizontal_rays, vertical_rays; double horizontal_fov_left, horizontal_fov_right, vertical_fov_up, vertical_fov_down, beam_length; Mount mount; LidarNoise noise; };`
  - `struct RangefinderParams { double rate; Mount mount; };`
  - `struct SensorsConfig { std::vector<CameraParams> rgb, rgb_segmented, depth, instance_seg; std::vector<FisheyeParams> fisheye; std::vector<StereoParams> stereo; std::vector<EventParams> event; std::vector<LidarParams> lidar; std::vector<RangefinderParams> rangefinder; double highestRate() const; };`
  - `extern const std::vector<std::string> kSensorTypes;`
  - `SensorsConfig loadSensors(mrs_lib::ParamLoader& loader, const std::string& prefix);` where `prefix` is e.g. `"mrs_uav_flightforge_simulator/"`. It throws `std::runtime_error` whose message contains the full parameter path.
  - `std::string validateSensorKeys(const std::vector<std::string>& yaml_files, const std::string& root_key);` returns `""` when valid, else a message containing the offending path (`root_key/sensors/<key>`).

- [ ] **Step 1: Write the fixtures and failing tests**

`sensors_valid.yaml` has `mrs_uav_flightforge_simulator/sensors` with:
- `rgb`: `n_sensors: 2`, a full `default` block (rate 30, 640×480, fov 120), `sensor_0: {}`, `sensor_1: {rate: 10.0, rotation: {pitch: 90.0}}`;
- `lidar`: `n_sensors: 1` with a full default block;
- every other type: `n_sensors: 0` with a full default block.

`sensors_bad_type.yaml` is the same with `rbg:` added. `sensors_bad_key.yaml` has `sensor_2:` under `rgb` with `n_sensors: 2`.

```cpp
TEST(SensorConfig, DefaultsFallBackPerKey) {
  auto cfg = load(fixture("sensors_valid.yaml"));
  ASSERT_EQ(cfg.rgb.size(), 2u);
  EXPECT_DOUBLE_EQ(cfg.rgb[0].rate, 30.0);
  EXPECT_DOUBLE_EQ(cfg.rgb[1].rate, 10.0);
  EXPECT_DOUBLE_EQ(cfg.rgb[1].mount.pitch, 90.0);
  EXPECT_DOUBLE_EQ(cfg.rgb[1].mount.yaw, cfg.rgb[0].mount.yaw);
  EXPECT_EQ(cfg.rgb[1].width, 640);
  EXPECT_TRUE(cfg.depth.empty());
  EXPECT_DOUBLE_EQ(cfg.highestRate(), 30.0);
}
TEST(SensorConfig, NegativeCountIsFatal)   { EXPECT_THROW(loadWith("rgb/n_sensors", -1), std::runtime_error); }
TEST(SensorConfig, NonPositiveRateIsFatal) { EXPECT_THROW(loadWith("rgb/sensor_1/rate", 0.0), std::runtime_error); }
TEST(SensorConfig, MisspeltTypeIsRejected) { EXPECT_NE(validateSensorKeys({fixture("sensors_bad_type.yaml")}, "mrs_uav_flightforge_simulator").find("sensors/rbg"), std::string::npos); }
TEST(SensorConfig, BlockBeyondCountIsRejected) { EXPECT_NE(validateSensorKeys({fixture("sensors_bad_key.yaml")}, "mrs_uav_flightforge_simulator").find("rgb/sensor_2"), std::string::npos); }
TEST(SensorConfig, ValidFilePasses) { EXPECT_EQ(validateSensorKeys({fixture("sensors_valid.yaml")}, "mrs_uav_flightforge_simulator"), ""); }
```

`load(path)` builds an `rclcpp::Node`, a `mrs_lib::ParamLoader` on it, calls `addYamlFile(path)` and `loadSensors(loader, "mrs_uav_flightforge_simulator/")`. `loadWith(key, value)` writes a temp copy of the valid fixture with that one value replaced (yaml-cpp), then calls `load`. `main` calls `rclcpp::init` and `rclcpp::shutdown`.

- [ ] **Step 2: Run, expect a build failure** (no `sensor_config.h`)

Run: `colcon build --packages-select mrs_uav_flightforge_simulator && colcon test --packages-select mrs_uav_flightforge_simulator --ctest-args -R test_sensors; colcon test-result --verbose`

- [ ] **Step 3: Implement `loadSensors` and `validateSensorKeys`**

- For each type, load `sensors/<type>/n_sensors` and the `default` block as compulsory params. Then for each `i`, load every field from `sensor_<i>/<key>` with the default value as the fallback (`loadParam(name, out, default)`). Nested keys are `offset/x|y|z` and `rotation/roll|pitch|yaw`; lidar noise is `noise/enabled|std_at_1m|std_slope`.
- Throw when `loader.loadedSuccessfully()` is false, `n_sensors < 0`, `rate <= 0`, or `width`/`height <= 0`.
- `validateSensorKeys` walks `root_key → sensors` in each file with yaml-cpp. Allowed type keys are `kSensorTypes`. Allowed keys under a type are `n_sensors`, `default` and `sensor_0..sensor_<n-1>`, taking `n` from the same file if present, else from any earlier file.

- [ ] **Step 4: Run the tests, expect 6 PASS**

Same command as Step 2. Expected: `test_sensors` passes, 0 failures.

- [ ] **Step 5: Commit** `Add the multi-instance sensor config loader`

---

### Task 2: Image and camera-info conversions

**Files:**
- Create: `include/mrs_uav_flightforge_simulator/sensor_conversions.h`, `src/sensor_conversions.cpp`, `test/test_image_conversions.cpp`

**Interfaces:**
- Consumes: `Mount`, `CameraParams` (Task 1).
- Produces:
  - `std::optional<cv::Mat> decodeFrame(const std::vector<unsigned char>& data, int width, int height);` uses `cv::imdecode(data, cv::IMREAD_COLOR)`; returns `nullopt` if `data` is empty, does not decode, or the size is not width × height.
  - `sensor_msgs::msg::Image toColorImage(const cv::Mat& bgr, const std_msgs::msg::Header& header);` `bgr8`.
  - `sensor_msgs::msg::Image toInstanceImage(const cv::Mat& bgr, const std_msgs::msg::Header& header);` `32SC1`, `r + (g << 8) + (b << 16)` with r = channel 2.
  - `std::optional<sensor_msgs::msg::Image> toDepthImage(const std::vector<uint16_t>& mm, int width, int height, const std_msgs::msg::Header& header);` `16UC1`.
  - `sensor_msgs::msg::CameraInfo toCameraInfo(const Serializable::Drone::CameraInfo& ci);` K from fx fy cx cy, `D = d_`, `distortion_model = distortion_model_`, R = identity, `P = [fx 0 cx Tx; 0 fy cy 0; 0 0 1 0]` with `Tx = (eye_ == 1) ? -fx * baseline_ / 100.0 : 0` (the baseline is in UE cm).
  - `const Serializable::Drone::CameraInfo* findEye(const std::vector<Serializable::Drone::CameraInfo>& infos, int eye);`
  - `ueds_connector::Coordinates toUeOffset(const Mount& m);` and `ueds_connector::Rotation toUeRotation(const Mount& m);` per Global Constraints.

- [ ] **Step 1: Write failing tests**

```cpp
TEST(Images, DecodesJpegToBgr8) { auto jpg = encode(".jpg", solid(64, 48, {255, 0, 0})); auto m = decodeFrame(jpg, 64, 48); ASSERT_TRUE(m); auto img = toColorImage(*m, {}); EXPECT_EQ(img.encoding, "bgr8"); EXPECT_EQ(img.step, 64u * 3); }
TEST(Images, WrongSizeIsRejected) { EXPECT_FALSE(decodeFrame(encode(".jpg", solid(64, 48, {0, 0, 0})), 640, 480)); }
TEST(Images, EmptyIsRejected) { EXPECT_FALSE(decodeFrame({}, 64, 48)); }
TEST(Images, InstanceIdFromBgr) { auto m = solid(2, 2, {0x03, 0x02, 0x01}); auto img = toInstanceImage(m, {}); EXPECT_EQ(img.encoding, "32SC1"); EXPECT_EQ(reinterpret_cast<const int32_t*>(img.data.data())[0], 0x010203); }
TEST(Images, DepthSizeChecked) { EXPECT_FALSE(toDepthImage(std::vector<uint16_t>(10), 4, 4, {})); auto d = toDepthImage(std::vector<uint16_t>(16, 1500), 4, 4, {}); ASSERT_TRUE(d); EXPECT_EQ(d->encoding, "16UC1"); }
TEST(CameraInfo, RightEyeCarriesBaseline) { auto ci = info(/*eye*/ 1, /*fx*/ 400, /*baseline cm*/ 7.5); EXPECT_DOUBLE_EQ(toCameraInfo(ci).p[3], -400 * 0.075); }
TEST(CameraInfo, FisheyeKeepsEquidistantModel) { auto ci = info(0, 300, 0); ci.distortion_model_ = "equidistant"; ci.d_ = {0.1, 0.2, 0.3, 0.4}; auto out = toCameraInfo(ci); EXPECT_EQ(out.distortion_model, "equidistant"); EXPECT_EQ(out.d.size(), 4u); }
TEST(CameraInfo, EyeChosenByField) { auto r = info(1, 400, 7.5), l = info(0, 400, 7.5); std::vector v{r, l}; EXPECT_EQ(findEye(v, 0)->eye_, 0); EXPECT_EQ(findEye(v, 1)->eye_, 1); EXPECT_EQ(findEye({l}, 1), nullptr); }
TEST(Mount, ToUe) { Mount m{1, 2, 3, 10, 20, 30}; auto o = toUeOffset(m); EXPECT_DOUBLE_EQ(o.y, -200); auto r = toUeRotation(m); EXPECT_DOUBLE_EQ(r.pitch, -20); EXPECT_DOUBLE_EQ(r.yaw, 30); }
```

`solid(w, h, bgr)` makes a `CV_8UC3` Mat; `encode` wraps `cv::imencode`; `info(eye, fx, baseline)` fills a `Serializable::Drone::CameraInfo` with fx = fy and cx/cy at the centre of 640×480.

- [ ] **Step 2: Run, expect a build failure**
- [ ] **Step 3: Implement the functions above in `src/sensor_conversions.cpp`**
- [ ] **Step 4: Run `test_sensors`, expect all PASS**
- [ ] **Step 5: Commit** `Add image and camera info conversions`

---

### Task 3: Lidar conversion

**Files:**
- Modify: `include/mrs_uav_flightforge_simulator/sensor_conversions.h`, `src/sensor_conversions.cpp`
- Create: `test/test_lidar_conversions.cpp`

**Interfaces:**
- Consumes: `LidarParams` (Task 1).
- Produces:
  - `struct LidarClouds { sensor_msgs::msg::PointCloud2 points, free; };`
  - `std::optional<LidarClouds> toLidarClouds(const ueds_connector::LidarScan& scan, const LidarParams& p, const std_msgs::msg::Header& header, std::mt19937& rng);`
    - `points` gets the rays with `distance > 0`, at `normalized(direction) * distance / 100` with y negated, and Gaussian range noise of std `distance_m * std_slope * std_at_1m` when `noise.enabled`.
    - Fields are `x y z` float32, then `intensity` float32 (the material class) if `p.intensity`, then `class_id` uint16 and `instance_id` uint32 if `p.labels`.
    - `free` gets `x y z` unit directions for rays with `distance <= 0`.
    - Returns `nullopt` if a requested channel vector's size differs from `points.size()`.

- [ ] **Step 1: Write failing tests**

```cpp
TEST(Lidar, SplitsHitsAndMisses) { auto c = toLidarClouds(scan({{150, 1, 0, 0}, {-1, 0, 1, 0}}), params(false, false), {}, rng); ASSERT_TRUE(c); EXPECT_EQ(c->points.width, 1u); EXPECT_EQ(c->free.width, 1u); EXPECT_EQ(c->points.height, 1u); }
TEST(Lidar, MetresAndRightHanded) { auto c = toLidarClouds(scan({{200, 0, 1, 0}}), params(false, false), {}, rng); EXPECT_FLOAT_EQ(y(c->points, 0), -2.0f); }
TEST(Lidar, FieldsFollowFlags) { EXPECT_EQ(names(toLidarClouds(withAll(scan({{100, 1, 0, 0}})), params(true, true), {}, rng)->points), (std::vector<std::string>{"x", "y", "z", "intensity", "class_id", "instance_id"})); EXPECT_EQ(names(toLidarClouds(scan({{100, 1, 0, 0}}), params(false, false), {}, rng)->points), (std::vector<std::string>{"x", "y", "z"})); }
TEST(Lidar, LabelsCarried) { auto s = withAll(scan({{100, 1, 0, 0}})); s.class_id = {7}; s.instance_id = {0x010203}; auto c = toLidarClouds(s, params(false, true), {}, rng); EXPECT_EQ(u16(c->points, "class_id", 0), 7); EXPECT_EQ(u32(c->points, "instance_id", 0), 0x010203u); }
TEST(Lidar, ShortChannelIsRejected) { auto s = scan({{100, 1, 0, 0}, {100, 0, 1, 0}}); s.class_id = {1}; s.instance_id = {1}; EXPECT_FALSE(toLidarClouds(s, params(false, true), {}, rng)); }
TEST(Lidar, NoiseOffMeansExact) { auto p = params(false, false); p.noise.enabled = false; auto c = toLidarClouds(scan({{1234, 1, 0, 0}}), p, {}, rng); EXPECT_FLOAT_EQ(x(c->points, 0), 12.34f); }
```

`scan(rays)` builds a `LidarScan` from `{distance_cm, dx, dy, dz}`; `withAll` fills intensity, class_id and instance_id to match; `params(intensity, labels)` returns a `LidarParams` with the noise disabled.

- [ ] **Step 2: Run, expect a build failure**
- [ ] **Step 3: Implement `toLidarClouds`**
- [ ] **Step 4: Run `test_sensors`, expect all PASS**
- [ ] **Step 5: Commit** `Add lidar scan conversion`

---

### Task 4: Event packet and instance map conversions

**Files:**
- Modify: `include/mrs_uav_flightforge_simulator/sensor_conversions.h`, `src/sensor_conversions.cpp`
- Create: `test/test_event_conversions.cpp`

**Interfaces:**
- Produces:
  - `event_camera_msgs::msg::EventPacket toEventPacket(const ueds_connector::CameraEvents& ev, const std_msgs::msg::Header& header);`
    - encoding `"evt3"`, `width`/`height` from `ev`, `time_base = rclcpp::Time(header.stamp).nanoseconds()`;
    - `event_camera_codecs::evt3::Encoder` with `setSensorTime(uint64_t(ev.base_stamp * 1e9))`, then `encodeCD(int32_t(t_us[i] * 1000), x, y, p)` per event, where `p = (xyp[i] >> 31) & 1`, `x = xyp[i] & 0xFFFF` and `y = (xyp[i] >> 16) & 0x7FFF`.
  - `mrs_uav_flightforge_simulator::msg::InstanceMap toInstanceMap(const std::vector<Serializable::Drone::InstanceSegMapEntry>& entries, const std_msgs::msg::Header& header);`

- [ ] **Step 1: Write failing tests**

```cpp
TEST(Events, RoundTripsThroughEvt3) {
  auto ev = events({{10, 20, 1, 0}, {11, 20, 0, 5}, {400, 300, 1, 1500}});  // x, y, polarity, t_us
  auto pkt = toEventPacket(ev, header(100.0));
  EXPECT_EQ(pkt.encoding, "evt3");
  auto got = decodeAll(pkt);  // event_camera_codecs DecoderFactory + collecting EventProcessor
  ASSERT_EQ(got.size(), 3u);
  EXPECT_EQ(got[2].x, 400); EXPECT_EQ(got[2].y, 300); EXPECT_EQ(got[1].p, 0);
  EXPECT_EQ(got[2].t - got[0].t, 1500u * 1000u);
}
TEST(Events, EmptyBatchStillHasHeader) { auto pkt = toEventPacket(events({}), header(5.0)); EXPECT_TRUE(pkt.events.empty()); EXPECT_EQ(pkt.header.stamp.sec, 5); EXPECT_EQ(pkt.width, 640u); }
TEST(InstanceMap, KeepsTwentyFourBitIds) { auto m = toInstanceMap({{0x123456, "/Game/A", 3}}, {}); EXPECT_EQ(m.ids[0], 0x123456u); EXPECT_EQ(m.class_ids[0], 3); }
```

- [ ] **Step 2: Run, expect a build failure**
- [ ] **Step 3: Implement both functions**
- [ ] **Step 4: Run `test_sensors`, expect all PASS**
- [ ] **Step 5: Commit** `Add event packet and instance map conversions`

---

### Task 5: Node, cameras

**Files:**
- Create: `include/mrs_uav_flightforge_simulator/sensors.h`
- Modify: `src/flightforge_simulator.cpp`, `config/flightforge_simulator.yaml` (new `rgb rgb_segmented depth instance_seg fisheye stereo` blocks, all `n_sensors: 0` except `rgb` and `stereo` at their current enabled state; each `default` carries today's values), `tmux/one_drone/config/simulator.yaml` (2 rgb, 1 each of the other camera types), `tmux/one_drone/rviz.rviz`

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces, in `sensors.h` (needs `TimerType`, so move its `typedef` block from `flightforge_simulator.cpp` into the header):
  - `struct CameraSensor { CameraParams params; int id = -1; std::string name; image_transport::Publisher ph_image; mrs_lib::PublisherHandler<sensor_msgs::msg::CameraInfo> ph_info; mrs_lib::PublisherHandler<mrs_uav_flightforge_simulator::msg::InstanceMap> ph_map; sensor_msgs::msg::CameraInfo info; std::shared_ptr<TimerType> timer; double last_stamp = 0.0; };` (`ph_map` is used by instance_seg only).
  - `struct FisheyeSensor` and `struct StereoSensor` with the same shape and their own params; stereo holds `ph_image_left/right`, `ph_info_left/right`, `info_left/right`.
  - Node members `std::vector<std::vector<CameraSensor>> rgb_, rgb_segmented_, depth_, instance_seg_; std::vector<std::vector<FisheyeSensor>> fisheye_; std::vector<std::vector<StereoSensor>> stereo_;` indexed `[uav][i]`, plus `SensorsConfig sensors_config_`.
  - `void timerCamera(std::vector<std::vector<CameraSensor>>* sensors, int type, size_t uav, size_t i);` dispatches on `type` (`SensorTypeEnum`) to the right getter and conversion. `void timerFisheye(size_t uav, size_t i)` and `void timerStereoInstance(size_t uav, size_t i)` likewise.

- [ ] **Step 1: Replace loading.** In `timerInit` call `validateSensorKeys(config_files + custom_config, "mrs_uav_flightforge_simulator")`, then `loadSensors(param_loader, yaml_prefix)`. On a non-empty message or an exception, log it with `RCLCPP_ERROR` and `rclcpp::shutdown()`, as other fatal startup errors do.
- [ ] **Step 2: Replace spawning.** In the per-UAV setup loop, for each camera type and instance:
  - `AddSensor(type)` → `Set*Config(cfg, id)`, with `cfg` built from params via `toUeOffset`/`toUeRotation`;
  - `GetCameraInfo(id)` → `toCameraInfo` (stereo: `findEye` 0 and 1);
  - advertise `/<uav>/<type>_<i>/image_raw` and `camera_info` (stereo: `left/`, `right/`; instance_seg also `labels`, latched with transient-local QoS, published once after the first frame from `GetInstanceSegMap(id)`);
  - send the static TF `<uav>/fcu → <uav>/<type>_<i>` (stereo eyes at ±baseline/2 along y: `<type>_<i>_left` / `_right`);
  - create the timer at `params.rate` on `clock_` and start it.

  Any false or empty result logs the UAV, type and index, then shuts down.
- [ ] **Step 3: Write the timer bodies.** Each does: the getter under `mutex_flightforge_`; `continue` on a repeated stamp or a failure, with a throttled error naming `<uav>/<type>_<i>`; `decodeFrame` / `toDepthImage` with the configured size, dropping on `nullopt`; stamp from `flightforgeTimeToSimtime`; publish image and info.
- [ ] **Step 4: Delete the old camera code.** That's `timerRgb`, `timerStereo`, `timerRgbSegmented`, the commented `timerDepth`, `fabricateCamInfo`, the camera parts of `publishStaticTfs`, the rgb/stereo/rgb_seg enable and rate dynparams with their callbacks, and the `rgb_*`/`stereo_*` members. `dynamic_rtf` takes the highest rate from `sensors_config_.highestRate()` (lidar and rangefinder rates stay in it until Task 6).
- [ ] **Step 5: Build** with zero new warnings in `flightforge_simulator.cpp`, and `test_sensors` still passes.
- [ ] **Step 6: Live check.** Restart `tmux/one_drone` against the dev sim, then run `ros2 topic list | grep -E "rgb_|depth_|instance_seg_|fisheye_|stereo_"`. Expected: every configured topic is listed. `python3 /tmp/ffdev/stamps.py 20` reports each camera at its configured sim rate with `backwards=0`.
- [ ] **Step 7: Commit** `Configure cameras as multi-instance sensors`

---

### Task 6: Node, lidar, rangefinder and event; old sensor code gone

**Files:**
- Modify: `include/mrs_uav_flightforge_simulator/sensors.h`, `src/flightforge_simulator.cpp`, `config/flightforge_simulator.yaml` (`lidar rangefinder event` blocks; the `lidar_segmented`/`lidar_intensity` blocks and the intensity table are removed), `tmux/one_drone/config/simulator.yaml` (1 Livox at 100 Hz with `intensity` and `labels`, 1 rangefinder, 1 event camera), `tmux/one_drone/rviz.rviz`

**Interfaces:**
- Consumes: Tasks 1, 3, 4.
- Produces:
  - `struct LidarSensor { LidarParams params; int id = -1; std::string name; mrs_lib::PublisherHandler<sensor_msgs::msg::PointCloud2> ph_points, ph_free; std::shared_ptr<TimerType> timer; double last_stamp = 0.0; };`
  - `struct RangefinderSensor { RangefinderParams params; int id = -1; std::string name; mrs_lib::PublisherHandler<sensor_msgs::msg::Range> ph_range; std::shared_ptr<TimerType> timer; };`
  - `struct EventSensor { EventParams params; int id = -1; std::string name; mrs_lib::PublisherHandler<event_camera_msgs::msg::EventPacket> ph_events; std::shared_ptr<TimerType> timer; };`
  - `rclcpp::Time ueWorldTimeToSimtime(double ue_world_time);` returns `ue_world_time + ue_world_offset_`, where `timerTimeSync` sets `ue_world_offset_ = ros_sim_now - (sim_seconds + rtt / 2 * dilation)` from `GetSimTime(sim_seconds, dilation)`, called inside the same lock as `GetTime`. Guarded by `mutex_wall_time_offset_`.

- [ ] **Step 1: Lidar spawning.** `AddSensor(livox ? SENSOR_LIDAR_LIVOX : SENSOR_LIDAR)`, then `SetLidarConfig` with `Frequency = rate`, `Livox`, `LivoxSensor`, `beamLength = beam_length * 100` and the FOV/ray fields. Topics are `points` plus `points_free` if `publish_free`, frame `<uav>/lidar_<i>`. The timer calls `GetLidarScan(labels, intensity, id)`, then `toLidarClouds`, stamped with `flightforgeTimeToSimtime(scan.stamp)`.
- [ ] **Step 2: Rangefinder and event spawning.** The rangefinder publishes `range`, frame `<uav>/rangefinder_<i>`. The event camera uses `SetEventCameraConfig` with `use_sim_time_ = true` and publishes `events`. The header stamp is `ueWorldTimeToSimtime(ev.base_stamp)`, and `dropped > 0` logs `RCLCPP_WARN_THROTTLE` with the count.
- [ ] **Step 3: Delete the old sensor code.** That's `timerLidar`, `timerSegLidar`, `timerIntLidar`, `timerRangefinder`, the lidar/rangefinder dynparams and callbacks, `drs_params_` sensor fields, the `lidar_*`/`seg_rgb_` members, the remaining sensor parts of `publishStaticTfs`, and `ph_*` vectors for removed topics. `highestRate()` now covers every type.
- [ ] **Step 4: Build clean; `test_sensors` passes.**
- [ ] **Step 5: Live check.** Restart the session. `python3 /tmp/ffdev/livox.py 30 /uav1/lidar_0/points` and `/uav1/lidar_0/points_free` together give ≈200k points per sim-second, with 0 backwards stamps. `ros2 topic hz /uav1/event_0/events` shows the configured rate, and `ros2 topic echo --once /uav1/lidar_0/points --field fields` lists `intensity`, `class_id` and `instance_id`.
- [ ] **Step 6: Commit** `Configure lidars, rangefinders and event cameras as multi-instance sensors`

---

### Task 7: Connector multi-instance integration test

**Files:**
- Modify: `~/git/flight_forge/.claude/worktrees/refactor/mrs_flight_forge_connector/tests/sim_integration.cpp` (after `AsyncPoseIsAcknowledgedPromptly`)

- [ ] **Step 1: Write the test**

```cpp
FF_TEST(TwoCamerasOfOneTypeKeepTheirOwnResolution)
{
  const auto [OkA, A] = Drone->AddSensor(ueds_connector::SENSOR_RGB_CAMERA);
  const auto [OkB, B] = Drone->AddSensor(ueds_connector::SENSOR_RGB_CAMERA);
  FF_REQUIRE(OkA && OkB && A != B);
  ueds_connector::RgbCameraConfig Cfg{};
  Cfg.width_ = 320; Cfg.height_ = 240; FF_REQUIRE(Drone->SetRgbCameraConfig(Cfg, A));
  Cfg.width_ = 160; Cfg.height_ = 120; FF_REQUIRE(Drone->SetRgbCameraConfig(Cfg, B));
  FF_CHECK(decodedSize(Drone->GetRgbCameraData(A)) == std::make_pair(320, 240));
  FF_CHECK(decodedSize(Drone->GetRgbCameraData(B)) == std::make_pair(160, 120));
  FF_REQUIRE(Drone->RemoveSensor(A));
  FF_CHECK(decodedSize(Drone->GetRgbCameraData(B)) == std::make_pair(160, 120));
  Drone->RemoveSensor(B);
}
```

`decodedSize` reads the JPEG SOF0 header for width and height. If the file has no such helper yet, add it beside the test as a static function; OpenCV is not a connector test dependency.

- [ ] **Step 2: Run** `FLIGHTFORGE_GAMEMODE_PORT=18581 ./build/tests/sim_integration` from the connector dir after `cmake --build build`. Expected: the new test PASS.
- [ ] **Step 3: Commit** in the connector repo, `tests/sim_integration.cpp` only: `Test two cameras of one type keep their own resolution`

---

### Task 8: Live sensor check and regression

**Files:**
- Create: `tmux/one_drone/check_sensors.py`

**Interfaces:**
- Consumes: the topics from Tasks 5–6.

- [ ] **Step 1: Write `check_sensors.py [duration]`.**
  - It reads `config/simulator.yaml` next to it, and for every configured `<type>_<i>` subscribes to its data topics.
  - It reports `rate_sim`, `backwards`, `duplicates`, `info_matches` (camera_info width/height equal the image's) and `tf_ok` (`tf2_ros.Buffer.can_transform('<uav>/fcu', frame, Time())`).
  - It exits non-zero if any rate is below 90% of the configured one, or any of the other checks fails.
- [ ] **Step 2: Run** `python3 tmux/one_drone/check_sensors.py 30` with the full Task 5–6 session. Expected: exit 0, every row OK.
- [ ] **Step 3: Regression.** Start the session with the default config (no custom sensors). The takeoff pane reaches `TAKEOFF -> FINISHED`, and `/uav1/estimation_manager/uav_state` z holds 1.5 ± 0.2 m for 30 s.
- [ ] **Step 4: Commit** `Add a live check of every configured sensor` (`check_sensors.py` only; the tmux config edits stay uncommitted).
