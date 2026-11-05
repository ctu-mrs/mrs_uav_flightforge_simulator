# Camera Throughput and Lockstep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cameras deliver a frame per simulator tick, render only at their configured sim-time rate, and the ROS node can run the simulator in lockstep so every sensor rate is exact in simulation time.

**Architecture:** The plugin's `UCamera` gets an in-order ring of readback slots and a per-camera due time; the connector mirrors a `capture_rate_` field (API 0.24). The node replaces per-sensor timers with a due-time scheduler run from `timerMain`, one connector socket and worker thread per sensor, and a lockstep mode that calls `Step(1)` per physics step with a barrier on the fetches.

**Tech Stack:** UE 5.7 plugin (FRHIGPUTextureReadback), connector (kissnet + cereal), ROS 2 Jazzy node (C++20, mrs_lib), gtest.

**Spec:** `docs/superpowers/specs/2026-11-05-camera-throughput-lockstep-design.md`

## Global Constraints

- Plugin: `~/git/flight_forge/.claude/worktrees/refactor/FlightForgePlugin`; rebuild the host with `~/Data/UnrealEngine/Engine/Build/BatchFiles/RunUBT.sh mrs_flight_forgeEditor Linux Development -Project="/home/david/Data/mrs_flight_forge 5.7/mrs_flight_forge.uproject"` (~1 min incremental). Dev sim: `run-editor.sh ... -FlightForgeGameModePort=18581 -FlightForgeDroneStartPort=14300` launched with `setsid nohup`. Never touch 8551 / 4000–4010 / 18551 / 14000.
- Connector: `~/git/flight_forge/.claude/worktrees/refactor/mrs_flight_forge_connector`, branch `V0.23`; build `cmake --build build`; tests `FF_FILTER=<name> FLIGHTFORGE_GAMEMODE_PORT=18581 ./build/tests/sim_integration`.
- Node: `~/git/mrs_uav_unreal_simulation`, branch `ff_api_0_23`; build/tests via `source /opt/ros/jazzy/setup.bash && cd ~/ws && colcon build --packages-select mrs_uav_flightforge_simulator --cmake-args -DBUILD_TESTING=ON && colcon test --packages-select mrs_uav_flightforge_simulator --ctest-args -R test_sensors && colcon test-result --verbose`.
- Comments: one clause, gotchas only; `//{ //}` folds on touched functions. Commit per task with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; never stage the connector symlinks, the `tmux/one_drone` test edits or David's local yaml hunks (`simulator_ip`, `individual_takeoff_platform`, `rangefinder n_sensors: 0`).
- Units: UE cm/deg; capture rates in Hz; world time in seconds.

## Review Focus

1. A camera whose ring is full in free-running mode skips the capture rather than blocking the game thread: the frame counter still advances only on published frames. Test in Task 1 (probe: no stall, fold counter grows under load).
2. A `Step(1)` that returns false or times out: the node exits with a named error instead of spinning with a stale clock. Test in Task 5 (kill the sim mid-run, node exits within the connector timeout).
3. A sensor whose worker is still busy when it is due again: the slot is skipped, not queued, and the barrier never waits on a dispatch that was not made. Test in Task 4 (scheduler unit test with a `busy` flag).
4. A due time more than one period behind (node paused, sim stalled): one dispatch, not a burst. Test in Task 4.
5. Stereo and fisheye frames whose slots complete out of order between eyes/faces: a frame is published only when all its slots are copied. Test in Task 1 (stereo stamps equal between eyes across 100 frames, probe).

---

### Task 1: Plugin, readback ring in `UCamera`

**Files:**
- Modify: `FlightForgePlugin/Source/FlightForgeCore/Sensors/Camera.h` (replace `PendingReadback` with `TArray<TSharedPtr<FCameraReadback>> InflightReadbacks`, `int32 MaxInflight = 3`, `uint32 FramesFolded`), `Camera.cpp` (`PumpReadback` publishes the oldest copied slot and issues resolves for ready ones; `EnqueueReadback` appends; `HasPendingReadback` → `IsRingFull()`; `DrainReadback` waits for the ring to empty; `RequestFrameFromServerThread` shares the newest slot), `RgbCamera.cpp`, `RgbSegCamera.cpp`, `DepthCamera.cpp`, `SegIdCamera.cpp`, `StereoCamera.cpp` (both eyes' slots go into one frame record), `FisheyeCamera.cpp` (`PendingFaces` becomes a ring of face sets).

**Interfaces:**
- Produces: `bool UCamera::IsRingFull() const`, `uint32 UCamera::FramesFolded` (logged on stream off like the event camera), unchanged `GetRgbCameraDataFromServerThread` semantics (returns the newest published frame).

- [ ] **Step 1: Baseline probe.** Launch the dev sim; run `/tmp/ffdev/probe/camrate2` (rebuild from `/tmp/ffdev/probe/camrate2.cpp` if `/tmp` was wiped). Record fresh frames/s and sim FPS. Expected: ~20 frames/s at ~120 FPS.
- [ ] **Step 2: Implement the ring** per the spec, free-running fold when full, lockstep drain unchanged.
- [ ] **Step 3: Rebuild the host project.** Expected: `Result: Succeeded`.
- [ ] **Step 4: Probe again.** Expected: fresh frames/s ≥ 0.8 × sim FPS for 640×480; stereo probe (add to camrate2: 100 `GetStereoCameraData`, left/right stamps equal) passes.
- [ ] **Step 5: Run the existing connector suite subset** `FF_FILTER=Camera` and `FF_FILTER=Stereo`. Expected: all OK.
- [ ] **Step 6: Commit** (plugin): `Pipeline camera readbacks three deep`

---

### Task 2: Connector + plugin, `capture_rate_`

**Files:**
- Modify: connector `include/flight_forge_connector/data_types.h` (`RgbCameraConfig::capture_rate_ = 0.0`, `FisheyeCameraConfig::capture_rate_ = 0.0`), `serialization/serializable_shared.h` (serialize the field in the three configs incl. `StereoCameraConfig`; API minor → 24), `tests/unit_tests.cpp` (round trip), `python/flightforge/client.py` (+ mirror), `tests/sim_integration.cpp`.
- Modify: plugin `Sensors/Camera.h/.cpp` (`double CaptureRate`, `double NextCaptureDue`, `bool CaptureDue(double WorldNow)`), the five camera `SetConfig` paths and `TickComponent`s to call `CaptureDue` before `UpdateCamera`, `Server/FlightForgeGameModeServer.cpp` API version constant.

**Interfaces:**
- Produces: `capture_rate_` on the three configs; `bool UCamera::CaptureDue(double WorldNow)` (advances the due time; resets it to now when more than one period behind).

- [ ] **Step 1: Write the failing unit test** in `tests/unit_tests.cpp`: a `RgbCameraConfig` with `capture_rate_ = 12.5` survives the request round trip.
- [ ] **Step 2: Write the failing integration tests**:
```cpp
FF_TEST(CaptureRateHoldsInLockstep)
{
  double applied; FF_REQUIRE(GameMode->SetFixedTimestep(0.01, 0.0, applied));
  FF_REQUIRE(GameMode->SetLockstep(true));
  const auto [OkA, A] = Drone->AddSensor(ueds_connector::SENSOR_RGB_CAMERA);
  ueds_connector::RgbCameraConfig Cfg{}; Cfg.width_ = 320; Cfg.height_ = 240; Cfg.capture_rate_ = 20.0;
  FF_REQUIRE(Drone->SetRgbCameraConfig(Cfg, A));
  double T; uint64_t Ticks; FF_REQUIRE(GameMode->Step(5, T, Ticks));  // warm up
  std::set<double> Stamps;
  for (int i = 0; i < 100; i++) { FF_REQUIRE(GameMode->Step(1, T, Ticks)); Stamps.insert(std::get<2>(Drone->GetRgbCameraData(A))); }
  FF_CHECK(Stamps.size() >= 19 && Stamps.size() <= 21);
  GameMode->SetLockstep(false); GameMode->SetFixedTimestep(0.0, 0.0, applied); Drone->RemoveSensor(A);
}
FF_TEST(CaptureRateZeroMeansEveryTick) { /* same with capture_rate_ = 0 over Step(10): 10 stamps */ }
```
- [ ] **Step 3: Run** `FF_FILTER=CaptureRate ...`. Expected: FAIL (field absent / stamps ≈ 100).
- [ ] **Step 4: Implement** the field, the serialization, `CaptureDue`, the version bump.
- [ ] **Step 5: Rebuild plugin host and connector; run** unit tests and `FF_FILTER=CaptureRate`. Expected: PASS.
- [ ] **Step 6: Commit** connector `Add capture_rate to the camera configs (API 0.24)` and plugin `Capture cameras at their configured rate`.

---

### Task 3: Node, per-sensor connections and workers

**Files:**
- Modify: `include/mrs_uav_flightforge_simulator/sensors.h` (every sensor struct gains `std::shared_ptr<ueds_connector::UedsConnector> connector; double next_due = 0; std::atomic<bool> busy{false};` and loses `timer`), `src/flightforge_simulator.cpp` (`setupSensors` opens one connector per sensor; `startSensorTimers` → `startSensorWorkers`; a `SensorWorker` per sensor: thread + condvar + `pending` flag; the timer bodies become `fetchX(uav, i)` run on the worker; the per-sensor connector replaces `ueds_connectors_[uav]` in fetches, and no `mutex_flightforge_` around them), `CMakeLists.txt` (nothing new; threads via std).
- Create: `include/mrs_uav_flightforge_simulator/sensor_scheduler.h`, `src/sensor_scheduler.cpp`, `test/test_sensor_scheduler.cpp`.

**Interfaces:**
- Produces: `struct ScheduledSensor { double rate; double next_due; bool busy; };` and `std::vector<size_t> dueSensors(double sim_time, std::vector<ScheduledSensor>& sensors);` — returns the indices to dispatch, advances their `next_due` by `1/rate`, snaps a `next_due` more than one period behind to `sim_time + 1/rate`, and skips (but still advances) busy sensors.
- `void FlightforgeSimulator::dispatchDueSensors(double sim_time)` (collects every type's instances into one scheduled list, kicks workers) and `void waitForSensors()` (barrier, used by Task 4).

- [ ] **Step 1: Write failing scheduler tests**:
```cpp
TEST(Scheduler, ExactCountsOverTenSeconds) { ... 5, 10, 30, 100 Hz stepped at 10 ms for 10 s → 50, 100, 300, 1000 dispatches ... }
TEST(Scheduler, StallYieldsOneDispatch) { ... rate 10, sim_time jumps 1 s → dispatched once, next_due == t + 0.1 ... }
TEST(Scheduler, BusySensorSkipsSlot) { ... busy → not returned, next_due advanced ... }
```
- [ ] **Step 2: Run, expect build failure. Step 3: Implement `dueSensors`. Step 4: Tests pass.**
- [ ] **Step 5: Wire the workers** into the node; keep free-running behaviour (dispatch from `timerMain` after each physics step; no barrier).
- [ ] **Step 6: Build; live check in free-running** (`check_sensors.py 40`, the 2-RGB config). Expected: no topic worse than before Task 3; camera rates now ≥ what Task 1 made possible (rgb_0 close to 30 Hz when the sim holds ≥ 60 FPS).
- [ ] **Step 7: Commit** `Fetch each sensor on its own connection from a due-time scheduler`

---

### Task 4: Node, lockstep

**Files:**
- Modify: `src/flightforge_simulator.cpp` (`time_model` param; startup `SetFixedTimestep`/`SetLockstep`; in `timerMain` the physics-step branch calls `Step(1)` under `mutex_flightforge_` after `publishPoses`/`updateUnrealPoses`, then `dispatchDueSensors`, then `waitForSensors`; `timerStatus` skips dilation and RTF retiming in lockstep; `timerTimeSync` sets `ue_world_offset_` exactly from the last step; shutdown restores free-running), `config/flightforge_simulator.yaml` (`time_model: "free_running"` with the lockstep option documented), `tmux/one_drone/config/simulator.yaml` (`time_model: "lockstep"`, uncommitted), `test/test_sensor_scheduler.cpp` (barrier test with fake workers).

**Interfaces:**
- Consumes: `GameModeController::SetFixedTimestep/SetLockstep/Step`, Task 3's scheduler and barrier.

- [ ] **Step 1: Failing barrier test**: three fake workers signalling done out of order → `waitForSensors` returns only after all three; a worker that reports failure also counts as done.
- [ ] **Step 2: Implement** the barrier and lockstep branch.
- [ ] **Step 3: Build; live in lockstep** with the 2-RGB config: `check_sensors.py 40` → every topic within 10% of its rate in sim time, 0 backwards, 0 duplicates; takeoff finishes and hover holds 1.5 ± 0.2 m; `livox.py` on `/uav1/lidar_0/points` + `points_free` → 200k ± 2% pts per sim-second; log shows the wall RTF.
- [ ] **Step 4: Kill-the-sim test**: `kill <sim pid>` during lockstep → the node logs a named error and exits within 90 s.
- [ ] **Step 5: Commit** `Run the simulator in lockstep from the node`

---

### Task 5: Defaults, docs and regression

**Files:**
- Modify: `config/flightforge_simulator.yaml` (flip `time_model` default to `"lockstep"` if Task 4's live run holds; keep `"free_running"` documented), `README.md` (a "Time models" section: what each mode guarantees, how `simulation_rate` quantizes camera rates, `capture_rate` semantics), the node's `.gitman.yaml` pin to the published 0.24 connector once David pushes it (leave a note if not yet).

- [ ] **Step 1: Free-running regression**: the 2-RGB config with `time_model: free_running` → `check_sensors.py` no worse than Task 3's numbers; default package config takes off and hovers.
- [ ] **Step 2: Write the README section.**
- [ ] **Step 3: Commit** `Document the time models and default to lockstep`
