#include <mrs_uav_flightforge_simulator/sensor_scheduler.h>

#include <algorithm>
#include <cmath>

namespace mrs_uav_flightforge_simulator
{

/* lockstepStride() //{ */

int lockstepStride(double simulation_rate, double highest_sensor_rate) {
  return std::max(1, int(std::floor(simulation_rate / std::max(highest_sensor_rate, 30.0))));
}

//}

/* dueSensors() //{ */

std::vector<size_t> dueSensors(double sim_time, std::vector<ScheduledSensor>& sensors) {

  std::vector<size_t> due;

  for (size_t i = 0; i < sensors.size(); i++) {

    ScheduledSensor& s = sensors[i];

    if (s.rate <= 0) {
      continue;
    }

    const double period = 1.0 / s.rate;

    if (sim_time - s.next_due > period) {
      s.next_due = sim_time;
    }

    if (sim_time + 1e-9 < s.next_due) {
      continue;
    }

    s.next_due += period;

    if (!s.busy) {
      due.push_back(i);
    }
  }

  return due;
}

//}

}  // namespace mrs_uav_flightforge_simulator
