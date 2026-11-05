#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <vector>

namespace mrs_uav_flightforge_simulator
{

struct ScheduledSensor
{
  double rate;
  double next_due;
  bool   busy;
};

// a stall gives one dispatch and a busy sensor loses its slot
std::vector<size_t> dueSensors(double sim_time, std::vector<ScheduledSensor>& sensors);

// physics steps per Step, with a 30 Hz floor for the view
int lockstepStride(double simulation_rate, double highest_sensor_rate);

// UE steps exactly the physics dt, so a physics step must be a whole number of clock ticks
bool lockstepClockValid(double clock_rate, double simulation_rate);

// counts fetches in flight, so a lockstep step can wait for every frame it produced
class FetchBarrier {

public:
  void add() {
    std::scoped_lock lock(mutex_);
    in_flight_++;
  }

  void done() {
    {
      std::scoped_lock lock(mutex_);
      in_flight_--;
    }
    cv_.notify_all();
  }

  // false when fetches are still out after timeout_s
  bool wait(double timeout_s) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, std::chrono::duration<double>(timeout_s), [this]() { return in_flight_ == 0; });
  }

private:
  std::mutex              mutex_;
  std::condition_variable cv_;
  int                     in_flight_ = 0;
};

}  // namespace mrs_uav_flightforge_simulator
