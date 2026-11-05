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

// indices due at sim_time; a due time more than a period behind snaps forward, a busy sensor loses its slot
std::vector<size_t> dueSensors(double sim_time, std::vector<ScheduledSensor>& sensors);

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
