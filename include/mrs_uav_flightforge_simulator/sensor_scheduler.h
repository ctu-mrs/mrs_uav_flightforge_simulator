#pragma once

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

}  // namespace mrs_uav_flightforge_simulator
