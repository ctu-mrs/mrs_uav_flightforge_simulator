# Camera throughput and lockstep time model

Date: 2026-11-05
Repos: FlightForgePlugin (worktree `refactor`), mrs_flight_forge_connector (`V0.23` → `0.24`),
mrs_uav_flightforge_simulator (`ff_api_0_23`).

## Problem

Measured on an idle simulator at ~120 FPS: one 640×480 RGB camera yields a fresh frame every
~50 ms (20 Hz) while each request returns in 4 ms. A camera keeps one GPU readback in flight,
so every frame waits out the full capture → copy → resolve → publish round trip (5–6 ticks).
Under the full sensor load the simulator falls to 35–60 FPS and the same path gives 7–10 Hz,
whatever rate the node asks for. Cameras also render every tick while anyone polls them, so
seven cameras at 5–30 Hz burn 7 renders per tick for frames nobody reads. The node's sensor
rates are wall-clock timers on a dilated ROS clock, so no rate is exact in simulation time,
and a Livox's point count per request depends on how the polls land between rendered frames.

## Goal

- A camera delivers one frame per simulator tick when asked to, at any resolution the GPU
  can render that fast.
- A camera renders only at its configured rate, in simulation time.
- In lockstep the node owns time: every sensor's rate is exact in simulation time, each
  frame belongs to a known step, and wall-clock speed is simply what the GPU allows.
- The free-running mode keeps working as it does today for users who want wall-clock pace.

Success: `check_sensors.py` with 2 RGB cameras (30 Hz and 10 Hz) and one of every other type
reports every topic within 10% of its configured rate in sim time, no backwards or duplicate
stamps, the full MRS stack takes off and hovers in lockstep, and a Livox delivers
200k ± 2% points per simulated second.

## Non-goals

- Nanite instance ids, engine-side changes of any kind.
- Device presets (`n_devices`) and scene configs; they follow in their own specs on top of
  the scheduler this one introduces.
- Reconnecting after a simulator restart.

## Part A: plugin, pipelined readbacks

`UCamera` keeps a small in-order ring of readback slots (`MaxInflight = 3`) instead of one
`PendingReadback`. A capture enqueues a slot; `PumpReadback` publishes the oldest completed
slot into `CameraBuffer`. The ring is full when three copies are on the GPU:
- free-running: the capture is skipped and counted (`FramesFolded`), as the event camera does;
- lockstep: `DrainReadback` at step end waits for every slot, as it already does.

`RgbCamera`, `RgbSegCamera`, `DepthCamera`, `SegIdCamera` (instance ids) and `StereoCamera`
(two slots per frame) move onto the ring. `FisheyeCamera` renders six faces per frame and
keeps its own `PendingFaces` set, extended to the same depth. `EventCamera` already has its
own pipeline and is untouched.

Expected effect: a camera's fresh-frame rate goes from 1 / round-trip to the simulator's
frame rate (measured before and after with a probe that counts distinct stamps per second).

## Part B: plugin + connector, capture at a configured rate

`RgbCameraConfig`, `FisheyeCameraConfig` and `StereoCameraConfig` gain `capture_rate_` (Hz,
`0` = every tick, the current behaviour). The camera captures on the first tick whose world
time reaches its next due time, then advances the due time by `1 / capture_rate_`; a due time
that falls more than one period behind is reset to now, so a stall produces one frame, not a
burst. The frame stamp is the world time of the tick that captured it. The lidar already has
`Frequency`; the event camera keeps `use_sim_time` and its own model.

The connector mirrors the fields (schema `serializable_shared.h`, the Python client), and the
API version becomes 0.24.

## Part C: node, per-sensor connections and a due-time scheduler

The per-sensor timers go. `timerMain` already advances `sim_time_` by one clock tick
(`1 / clock_rate`) and steps the physics every `1 / simulation_rate`. After each physics
step it runs the scheduler: every sensor whose `next_due <= sim_time` is dispatched, and its
`next_due` advances by `1 / rate` (snapping forward if it is more than one period behind).

Each sensor instance owns its own `UedsConnector` to the drone port and a worker thread that
performs the fetch, the conversion and the publish. The drone server accepts several
clients per drone (verified: two sockets run their requests in parallel), so sensors no
longer wait for each other and `mutex_flightforge_` only guards the pose stream and the
game-mode controller. A sensor whose previous fetch has not returned when it is due again
skips that slot and logs it throttled.

Dispatch is a `std::condition_variable` kick per worker; the worker fetches with the sensor's
own connector, stamps the message with `flightforgeTimeToSimtime` (free-running) or the
step's sim time (lockstep, see D), and publishes. The conversion functions are unchanged.

## Part D: node, lockstep time model

A new parameter `time_model: "free_running" | "lockstep"`. In lockstep, at startup the node
calls `SetFixedTimestep(1 / simulation_rate, 0)` and `SetLockstep(true)`; on shutdown or a
switch to free-running it restores `SetLockstep(false)` and `SetFixedTimestep(0, 0)`.

Per physics step in `timerMain`:
1. push the UAV poses as today (`SetLocationAndRotationAsync`), so the pawn has the pose
   before the tick;
2. `Step(1)`: blocks until the tick rendered and every camera drained; UE world time has
   advanced exactly `1 / simulation_rate`;
3. run the scheduler; every dispatched worker fetches the frame that step produced;
4. wait for those workers to finish before the next step (a counting barrier), so each
   frame is attributed to the step that produced it and no fetch overlaps a render.

The `/clock` keeps ticking at `clock_rate` between physics steps; the wall time of a step
is spent on its fourth tick. The dynamic RTF loop and time dilation are bypassed in
lockstep, since the step itself paces the clock; `realtime_factor` acts as a cap. Sensor
stamps in lockstep are the step's sim time; the UE world offset used by the event camera is
exact (`ue_world_offset_ = sim_time - step.sim_time` after each step), with no filtering.

Rates quantize to the step: at 100 Hz physics a 30 Hz camera captures every 3–4 ticks and
averages 30 Hz with stamps on tick boundaries. A camera needing a finer grid should raise
`simulation_rate`; the Livox's point count per step is exactly `PointRate / simulation_rate`.

Free-running keeps today's behaviour (wall timers, dilation, `flightforgeTimeToSimtime`),
driven by the same scheduler so the two modes share one code path for fetching.

### As built

These things differ from the design above:

- Lockstep cameras run in `CAPTURE_ON_DEMAND`, not on `capture_rate_`: each due fetch renders that
  step's frame. The plugin's due times and the node's could drift a tick apart, which would drop
  frames as duplicates, and on demand nothing renders unless it is due. `capture_rate_` stays
  available to connector clients but the node does not send it.
- Sensor sockets use a 10 s receive timeout in lockstep, since a frame takes as long as the GPU
  needs; the connector's default 1 s turned a slow frame into a failed fetch.
- The node calls `Step` every N physics steps, not every one: N follows the fastest sensor with a
  30 Hz floor, and UE's fixed timestep is N physics steps. Each `Step` costs about three engine
  frames, so stepping every physics step ran a sensorless setup at RTF 0.19.
- The drone server is one thread looping over its clients, so per-sensor sockets give parallel
  decoding and conversion in the node but sensors on one drone are still served one at a time. The
  "run their requests in parallel" line in Part C holds only for replies that need no render.
- Between `Step`s the clock runs up to 4x faster than `realtime_factor`, gated on a wall deadline,
  so a `Step`'s wall time is not paid twice.
- `clock_rate` must be a whole multiple of `simulation_rate` in lockstep; anything else is fatal at
  startup, because UE steps exactly the physics dt.
- Free-running sensors are dispatched by the same scheduler, and a camera frame whose converted
  stamp does not move forward is dropped.

## Error handling

- `Step` returning false, or timing out (the connector allows 90 s for `step`), is fatal:
  the world went away under the node.
- A worker whose fetch fails logs throttled and leaves the sensor's due time advanced; the
  lockstep barrier still releases (failure counts as done).
- `capture_rate_` ≤ 0 means every tick; a negative value is rejected by the plugin's config
  setter with status false, which the node treats as fatal at startup like any config
  failure.
- Switching `time_model` at runtime is not supported; it is a startup parameter.

## Testing

1. **Plugin probe, before/after Part A** (`/tmp/ffdev/probe/camrate2`): fresh frames per
   second for one 640×480 camera on an idle sim. Before: ~20. After: ≥ 0.8 × sim FPS.
2. **Connector unit test**: schema round trip of the new `capture_rate_` fields.
3. **Connector integration tests** (`sim_integration`, run with `FF_FILTER`):
   - `CaptureRateHoldsInLockstep`: fixed dt 10 ms, lockstep, RGB `capture_rate_ = 20`,
     `Step(100)` → exactly 20 distinct stamps 50 ms apart (±1 at the edges).
   - `CaptureRateIsOptional`: `capture_rate_ = 0` over `Step(10)` → 10 distinct stamps.
4. **Node gtest**: the scheduler as a pure function over `(sim_time, rates, due)`: exact
   counts over 10 s for 5, 10, 30, 100 Hz; snapping after a 1 s stall yields one dispatch.
5. **Live** (`check_sensors.py`, lockstep, the 2-RGB config): every topic within 10% of its
   rate in sim time, 0 backwards, 0 duplicates; `rgb_0` at 30 Hz. Takeoff and hover in
   lockstep. Livox 200k ± 2% points per sim-second. Then the same config in free-running:
   no regression against today's numbers.
