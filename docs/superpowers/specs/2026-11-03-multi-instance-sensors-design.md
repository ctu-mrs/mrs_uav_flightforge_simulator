# Multi-instance sensors for mrs_uav_flightforge_simulator

Date: 2026-11-03
Branch: `ff_api_0_23` (builds on the FlightForge connector API 0.23 port)

## Goal

Let one UAV carry any number of sensors of each type, and add the sensor types the 0.23 API
offers that the node does not use yet: depth, instance segmentation, fisheye and event cameras.
Today the node has exactly one sensor per type, its state kept as loose node members
(`rgb_width_`, `timer_rgb_`, `ph_img_rgb_[uav]`, ...).

Success means:

- a config with two RGB cameras and one of every other type spawns all of them, each publishing
  at its own rate on its own topics with its own TF frame and camera info;
- `n_sensors: 0` turns a type off;
- the full MRS stack still takes off and hovers.

## Non-goals

- Device presets (`AddDevice`: realsense_d435i, livox_mid360, davis346, ...). A later
  `n_devices` group can build on this schema.
- Lockstep and fixed-timestep time models. The current wall-clock model with time dilation
  (synced in `timerStatus`) stays.
- Reconnecting after the simulator restarts. The node still dies in that case.
- Moving the fetch, scheduling and publishing code out of `flightforge_simulator.cpp`. Only the
  pure conversion and config-loading functions move out, because the unit tests must link them
  without the node.

## Connector API used

Sensors are created with `AddSensor(SensorTypeEnum)`, which returns an id; every getter and
setter takes that id as its trailing `sensorId` argument.

| Type key | SensorTypeEnum | Config setter | Getter |
|---|---|---|---|
| `rgb` | `SENSOR_RGB_CAMERA` | `SetRgbCameraConfig` | `GetRgbCameraData` |
| `rgb_segmented` | `SENSOR_RGB_SEG_CAMERA` | `SetRgbCameraConfig` | `GetRgbSegmented` |
| `depth` | `SENSOR_DEPTH_CAMERA` | `SetRgbCameraConfig` | `GetDepthCameraData` |
| `instance_seg` | `SENSOR_INSTANCE_SEG_CAMERA` | `SetRgbCameraConfig` | `GetInstanceSegData`, `GetInstanceSegMap` |
| `fisheye` | `SENSOR_FISHEYE_CAMERA` | `SetFisheyeCameraConfig` | `GetFisheyeCameraData` |
| `stereo` | `SENSOR_STEREO_CAMERA` | `SetStereoCameraConfig` | `GetStereoCameraData` |
| `event` | `SENSOR_EVENT_CAMERA` | `SetEventCameraConfig` | `GetEventCameraData` |
| `lidar` | `SENSOR_LIDAR`, or `SENSOR_LIDAR_LIVOX` when `livox: true` | `SetLidarConfig` | `GetLidarScan(labels, intensity, id)` |
| `rangefinder` | `SENSOR_RANGEFINDER` | none | `GetRangefinderData` |

`GetLidarScan` replaces the three separate lidar getters, which traced separately and, on a
Livox, each advanced their own pattern cursor. Camera intrinsics come from
`GetCameraInfo(id)`, read once after configuration.

## Config schema

Same shape as `mrs_losos_server`: per type, `n_sensors` followed by `sensor_0`, `sensor_1`, ...
Each type also has a `default` block in the package config. A key missing from `sensor_<i>`
is taken from `default`, so a sensor block only lists what differs.

```yaml
sensors:

  rgb:
    n_sensors: 2

    default:
      rate: 30.0 # [Hz]
      width: 640 # [px]
      height: 480 # [px]
      fov: 120.0 # [deg]
      offset: {x: 0.14, y: 0.0, z: 0.0} # [m]
      rotation: {roll: 0.0, pitch: 0.0, yaw: 0.0} # [deg]
      enable_hdr: true
      enable_temporal_aa: true
      enable_raytracing: true
      enable_motion_blur: true
      motion_blur_amount: 0.5
      motion_blur_distortion: 50.0

    sensor_0: {}
    sensor_1:
      rate: 10.0
      rotation: {pitch: 90.0}

  lidar:
    n_sensors: 1

    default:
      rate: 10.0
      livox: false
      livox_sensor: "mid360" # "mid360" or "avia"
      intensity: false
      labels: false
      publish_free: true
      horizontal_rays: 128
      vertical_rays: 64
      horizontal_fov_left: 180.0
      horizontal_fov_right: 180.0
      vertical_fov_up: 45.0
      vertical_fov_down: 45.0
      beam_length: 20.0 # [m]
      offset: {x: 0.0, y: 0.0, z: 0.25}
      rotation: {roll: 0.0, pitch: 0.0, yaw: 0.0}
      noise: {enabled: true, std_at_1m: 0.01, std_slope: 0.2}

    sensor_0: {}
```

Per-type keys beyond `rate`, `offset` and `rotation`:

- `rgb`, `rgb_segmented`, `depth`, `instance_seg`: today's RGB keys (width, height, fov, the
  enable_* flags and motion blur).
- `fisheye`: width, height, fov, `lens_model`, `full_frame`.
- `stereo`: width, height, fov, `baseline` and the enable_* flags. The left and right cameras
  are placed ±baseline/2 along y around `offset` and share `rotation`, replacing today's
  separate left/right offset and rotation blocks.
- `event`: width, height, fov, `contrast_threshold_pos`, `contrast_threshold_neg`,
  `max_events_per_frame`.
- `lidar`: as above. `GetLidarScan` returns a material class per point, which becomes the
  `intensity` field as is. Today's material-to-intensity table and intensity noise are dropped:
  the old intensity lidar loaded them but wrote the ray distance into `intensity`.
- `rangefinder`: only rate, offset and rotation.

The `lidar_segmented` and `lidar_intensity` sub-blocks go away; their behaviour becomes the
`labels` and `intensity` flags of a lidar sensor. `livox` selects the `SENSOR_LIDAR_LIVOX`
type, and the sensor's `rate` is sent as `LidarConfig.Frequency`.

## Code structure

`include/mrs_uav_flightforge_simulator/sensors.h` holds one data-only struct per type: no
methods, no inheritance, no factory. Each struct holds:

- the loaded parameters;
- the connector sensor id;
- its publishers;
- its camera info (cameras only);
- its timer.

The node holds a `std::vector<RgbSensor>` and so on for each type, indexed `[uav][i]`.

`sensor_conversions.{h,cpp}` holds the pure functions that turn connector data into ROS
messages, and `sensor_config.{h,cpp}` the per-type loaders. Both are built into a small library
the node and the tests link.

Everything else stays in `flightforge_simulator.cpp`, written the way `timerRgb` is today:

- spawning loops over `n_sensors`: `AddSensor`, the type's `Set*Config` with the returned
  id, `GetCameraInfo` for cameras, then the static TF;
- each instance gets its own timer on the sim clock, all instances of a type sharing one
  callback that takes the instance index. Two RGB cameras can then run at different rates.

Removed from the node:

- the per-type sensor members;
- the enable and rate dynparams and their callbacks;
- the three separate lidar getters;
- the hand-computed camera intrinsics.

`dynamic_rtf` takes the highest rate across all instances.

Kept: the global `mutex_flightforge_` around every connector call; physics, clock, pose sync,
spawning and world setup; `flightforgeTimeToSimtime`, which maps each sample's capture stamp
to sim time. It is now interpolated between clock ticks and was verified free of backwards
stamps.

## Topics and frames

Every sensor publishes under `/<uav>/<type>_<i>/` and has one TF frame `<uav>/<type>_<i>`,
sent once as a static transform from `<uav>/fcu`. Camera frames keep today's convention of
having the optical rotation built in.

| Type | Topics | Message |
|---|---|---|
| rgb | `image_raw`, `camera_info` | Image `bgr8` via image_transport |
| rgb_segmented | `image_raw`, `camera_info` | Image `bgr8` |
| depth | `image_raw`, `camera_info` | Image `16UC1`, millimetres |
| instance_seg | `image_raw`, `camera_info`, `labels` | Image `32SC1` holding the 24-bit instance id (`r + (g << 8) + (b << 16)` of the decoded PNG); latched `mrs_uav_flightforge_simulator/InstanceMap` (`uint32[] ids`, `string[] actors`, `uint16[] class_ids`) from `GetInstanceSegMap`, because `vision_msgs/LabelInfo` ids are uint16 |
| fisheye | `image_raw`, `camera_info` | Image `bgr8`; CameraInfo `distortion_model: equidistant`, KB coefficients in `D` |
| stereo | `left/image_raw`, `left/camera_info`, `right/image_raw`, `right/camera_info` | Image `bgr8`; the right `P` carries the baseline (`P[3] = -fx * baseline`) |
| event | `events` | `event_camera_msgs/EventPacket`, encoding `evt3` |
| lidar | `points`, `points_free` if `publish_free` | PointCloud2 `x y z`, plus `intensity` (float32, the material class) if `intensity`, plus `class_id` (uint16) and `instance_id` (uint32) if `labels` |
| rangefinder | `range` | sensor_msgs/Range |

Lidar clouds are unorganized (`height = 1`), as today. `points` holds the rays with a return,
at `direction * distance` with distance in m (the sim reports cm) and y negated into the
right-handed frame, and range noise applied when `noise.enabled`. `points_free` holds a unit
direction for every ray without a return.

Camera frames arrive as JPEG (rgb, fisheye, stereo) or PNG (instance_seg) and are decoded with
`cv::imdecode`, which yields BGR channel order; rgb_segmented frames arrive as uncompressed BGR
and are taken as is. Every frame is checked against the size `GetCameraInfo` reports.

Events are encoded to EVT3 with `event_camera_codecs::evt3::Encoder`: `time_base` and
`setSensorTime` take the batch's `base_stamp` in ns, and each event is encoded at
`t_us * 1000` ns after it. The event camera keeps `use_sim_time: true`, so its stamps are UE
world time, which the time-dilation sync keeps at the ROS rate. Its header stamp is the batch's end
(`stamp`) mapped through a UE-world-to-ROS-sim offset that `timerTimeSync` tracks with
`GetSimTime` and low-pass filters, not through `flightforgeTimeToSimtime`, which maps wall-clock
capture stamps. A poll whose batch did not advance publishes nothing, so `rate` is a poll rate. New package dependencies:
`event_camera_msgs` and `event_camera_codecs`. The first two are not installed
system-wide and are built from source in the workspace until they are installed from apt.

Today's topic names (`/uav1/rgb/image_raw`, `/uav1/lidar/points`, ...) go away. The only
consumer in this repo is `tmux/one_drone/rviz.rviz`, which is updated.

## Error handling

- **Config errors are fatal at startup.** These are:
  - a key under `sensors` that is not a known type;
  - a key under a type other than `n_sensors`, `default` and `sensor_0` … `sensor_<N-1>`;
  - a negative `n_sensors`;
  - an invalid value (rate ≤ 0, width or height ≤ 0).

  The error names the full parameter path. `ParamLoader` cannot list keys, so the key checks
  run with yaml-cpp over the same yaml files the loader reads. A missing `sensor_<i>` block
  means all defaults, exactly like `sensor_<i>: {}`, which yields no parameters at all.
- **Setup errors are fatal at startup.** These are: `AddSensor` or `Set*Config` returning
  false, or `GetCameraInfo` returning nothing for a camera. The message names the UAV, type
  and index.
- **Runtime fetch errors skip one sample.** The error is logged throttled, per instance; the
  next scheduled sample proceeds normally.
- **Every conversion checks the data length first.** Data that does not match
  width × height × channels is dropped with a throttled error, never written past a buffer.
  Lidar clouds are sized from the data.
- **`CameraEvents::dropped > 0`** logs a throttled warning with the count.
- **A sample the simulator cannot keep up with is skipped, not queued.**

## Testing

1. **gtest unit tests (new; `ament_cmake_gtest`).** The conversion functions are fed hand-made
   connector data:
   - image encodings and `step`;
   - depth `16UC1`;
   - instance ids in `32SC1`;
   - lidar field layout for each `intensity` / `labels` combination, and the split between
     `points` and `points_free`;
   - fisheye CameraInfo;
   - the EVT3 encoder, round-tripped through `event_camera_codecs`;
   - config parsing from a yaml fixture, including the default fallback and each fatal
     config error, including a misspelt type key.
2. **Connector integration test** (extends `mrs_flight_forge_connector/tests/sim_integration.cpp`).
   Two sensors of the same type with different resolutions return frames of their own sizes,
   and removing one leaves the other working.
3. **Live check.** A tmux config with 2 RGB cameras and one of every other type. A script in
   the style of `stamps.py` checks, for every topic:
   - the rate matches the config, in sim time;
   - stamps never go backwards;
   - `camera_info` matches the image size;
   - the TF frame exists.
4. **Regression.** The full MRS stack takes off and hovers with the default config.
