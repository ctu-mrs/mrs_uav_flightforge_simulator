#include <gtest/gtest.h>

#include <cmath>

#include <mrs_uav_flightforge_simulator/ground.h>

using namespace mrs_uav_flightforge_simulator;

TEST(Ground, FootprintCoversTheCornersAtTheGear) {
  const auto points = footprintPoints(1.0, 2.0, 0.5, 0.0, 0.25);
  EXPECT_DOUBLE_EQ(points[0][0], 1.0);
  EXPECT_DOUBLE_EQ(points[0][1], 2.0);
  for (const auto& p : points) {
    EXPECT_DOUBLE_EQ(p[2], 0.5 + kGroundTraceLift);
  }
  EXPECT_DOUBLE_EQ(points[1][0], 1.25);
  EXPECT_DOUBLE_EQ(points[1][1], 2.25);
  EXPECT_DOUBLE_EQ(points[3][0], 0.75);
  EXPECT_DOUBLE_EQ(points[3][1], 1.75);
}

TEST(Ground, FootprintTurnsWithTheBody) {
  const auto points = footprintPoints(0.0, 0.0, 0.0, M_PI / 4.0, 1.0);
  EXPECT_NEAR(points[1][0], 0.0, 1e-12);
  EXPECT_NEAR(points[1][1], std::sqrt(2.0), 1e-12);
}

TEST(Ground, HighestSurfaceSkipsMisses) {
  const double nan = std::nan("");

  const double some[] = {nan, 0.2, 1.5, nan, 0.9};
  EXPECT_DOUBLE_EQ(highestSurface(some, 5), 1.5);

  const double none[] = {nan, nan};
  EXPECT_TRUE(std::isnan(highestSurface(none, 2)));

  const double first[] = {-3.0, nan};
  EXPECT_DOUBLE_EQ(highestSurface(first, 2), -3.0);
}

TEST(Ground, TouchdownIsTheModelsClamp) {
  EXPECT_TRUE(touchedDown(0.16, 0.0, 0.0, 0.0, -1.2, 0.16));
  EXPECT_FALSE(touchedDown(0.16, 0.0, 0.0, 0.0, 0.0, 0.16));
  EXPECT_FALSE(touchedDown(0.20, 0.0, 0.0, -0.1, -1.2, 0.16));
  EXPECT_FALSE(touchedDown(0.16, 0.1, 0.0, 0.0, -1.2, 0.16));
}

TEST(Ground, LandingIsAGentleContactWithTheGroundBelow) {
  EXPECT_TRUE(isLanding(0.01, 0.0, 0.5, 2.0));
  EXPECT_TRUE(isLanding(0.04, 0.0, 2.0, 2.0));
  EXPECT_FALSE(isLanding(0.30, 0.0, 0.5, 2.0));
  EXPECT_FALSE(isLanding(0.01, 0.0, 3.0, 2.0));
  EXPECT_FALSE(isLanding(0.01, std::nan(""), 0.5, 2.0));
}
