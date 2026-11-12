#include <mrs_uav_flightforge_simulator/ground.h>

#include <algorithm>
#include <cmath>

namespace mrs_uav_flightforge_simulator
{

/* footprintPoints() //{ */

std::array<std::array<double, 3>, kFootprintPoints> footprintPoints(const double x, const double y, const double gear_bottom_z, const double yaw,
                                                                     const double radius) {

  const double c = std::cos(yaw), s = std::sin(yaw);
  const double z = gear_bottom_z + kGroundTraceLift;

  std::array<std::array<double, 3>, kFootprintPoints> points;
  points[0] = {x, y, z};

  const double corners[4][2] = {{radius, radius}, {radius, -radius}, {-radius, -radius}, {-radius, radius}};
  for (size_t i = 0; i < 4; i++) {
    points[i + 1] = {x + c * corners[i][0] - s * corners[i][1], y + s * corners[i][0] + c * corners[i][1], z};
  }

  return points;
}

//}

/* highestSurface() //{ */

double highestSurface(const double* heights, const size_t count) {

  double highest = std::nan("");

  for (size_t i = 0; i < count; i++) {
    if (std::isfinite(heights[i]) && !(heights[i] <= highest)) {
      highest = heights[i];
    }
  }

  return highest;
}

//}

/* touchedDown() //{ */

bool touchedDown(const double z_after, const double vx_after, const double vy_after, const double vz_after, const double vz_before, const double ground_z) {
  return vz_before < 0.0 && z_after == ground_z && vx_after == 0.0 && vy_after == 0.0 && vz_after == 0.0;
}

//}

/* isLanding() //{ */

bool isLanding(const double impact_z, const double surface, const double speed, const double max_touchdown_speed) {
  return std::isfinite(surface) && impact_z <= surface + kGroundTouchTolerance && speed <= max_touchdown_speed;
}

//}

}  // namespace mrs_uav_flightforge_simulator
