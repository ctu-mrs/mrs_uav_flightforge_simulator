# Device presets

Date: 2026-11-06
Repo: mrs_uav_flightforge_simulator (`ff_api_0_23`). The plugin and connector need no change.

## Problem

Putting a real sensor on a drone means copying its datasheet into `sensors:` by hand, and the
housing never shows up. The plugin already has a device library (`FSensorDeviceLibrary`) with
datasheet parameters, mount offsets and CAD housings, and the connector has `AddDevice`, but the
node does not use them. Gazebo users expect one line per sensor.

## Config

A `devices:` block beside `sensors:`, with the same shape:

    devices:
      n_devices: 2
      device_0:
        type: "livox_mid360"
        offset: {x: 0.0, y: 0.0, z: 0.1}
      device_1:
        type: "realsense_d435i"
        offset: {x: 0.14, y: 0.0, z: 0.0}

`default` holds `type`, `camera_rate`, `lidar_rate`, `show_mesh`, `offset`, `rotation`, `intensity`,
`labels`, `publish_free` and `noise`; `device_<i>` overrides any of them. `type` has no default.

## Behaviour

- The node calls `AddDevice` with the mount, then reads every created sensor's config back from the
  simulator. The simulator is the single source of the datasheet numbers and the sensor's exact
  pose in the airframe, so TF includes the device's internal offsets.
- Device sensors join the same `cameras_`, `fisheyes_`, `stereos_` and `lidars_` lists as `sensors:`
  instances, so fetching, scheduling and both time models are unchanged.
- Topics are `/<uav>/<device>_<k>/<role>/...`, TF frames `<uav>/<device>_<k>_<role>`, where `k`
  counts devices of that type and `role` is `lidar`, `rgb`, `stereo` or `fisheye`.
- `lidar_rate: 0` keeps the device's own rate. Cameras have no rate in the library, so they take
  `camera_rate`.
- Rgb, stereo, fisheye and lidar are supported. A device with any other sensor (depth, event,
  rangefinder) fails at startup with a named error, as does an unknown `type`, which lists what
  the simulator offers.

## Non-goals

- New devices in the library, or changes to their numbers.
- Per-sensor camera quality overrides on a device; use `sensors:` for that.

## Testing

Unit: config defaults and overrides, a missing `type`, bad keys, the name counter, the UE to ROS
mount round trip and each read-back conversion. Live: each of `livox_mid360`, `livox_avia`,
`ouster_os0_32`, `odin1` and `realsense_d435i` on one drone with `check_sensors.py`, TF present,
and a look at a frame from each camera, since the housing must not block its own view.
