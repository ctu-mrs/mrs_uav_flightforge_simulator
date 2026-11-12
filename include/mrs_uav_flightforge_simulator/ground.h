#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace mrs_uav_flightforge_simulator
{

// starts the trace above the gear's bottom so a surface it is about to touch is still below
constexpr double kGroundTraceLift = 0.05;  // [m]

// how far above the surface under the UAV a reported impact still counts as the ground
constexpr double kGroundTouchTolerance = 0.05;  // [m]

constexpr size_t kFootprintPoints = 5;

// centre and corners of a square of half-side radius, yawed with the body
std::array<std::array<double, 3>, kFootprintPoints> footprintPoints(double x, double y, double gear_bottom_z, double yaw, double radius);

// NaN when no point of the footprint has a surface below it
double highestSurface(const double* heights, size_t count);

// the model clamped a descending UAV to ground_z this step
bool touchedDown(double z_after, double vx_after, double vy_after, double vz_after, double vz_before, double ground_z);

bool isLanding(double impact_z, double surface, double speed, double max_touchdown_speed);

}  // namespace mrs_uav_flightforge_simulator
