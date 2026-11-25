#include <gtest/gtest.h>

#include <mrs_uav_flightforge_simulator/world.h>

using namespace mrs_uav_flightforge_simulator;

TEST(World, EmptyNameKeepsTheLoadedWorld) {
  EXPECT_FALSE(worldToSwitchTo("").has_value());
  EXPECT_FALSE(worldToSwitchTo("  \t").has_value());
  EXPECT_FALSE(worldToSwitchTo("current").has_value());
  EXPECT_EQ(worldToSwitchTo(" Valley "), "Valley");
  EXPECT_EQ(worldToSwitchTo("/Game/Worlds/Warehouse/Maps/Warehouse"), "/Game/Worlds/Warehouse/Maps/Warehouse");
}

TEST(World, ReadyOnceLoadedAndNotSwitching) {
  flightforge::WorldList list;
  EXPECT_FALSE(worldReady(list));
  list.current = "/Game/Worlds/Warehouse/Maps/Warehouse";
  EXPECT_TRUE(worldReady(list));
  list.loading = "/Game/Worlds/Valley/Maps/Valley";
  EXPECT_FALSE(worldReady(list));
}

TEST(World, NamesListEverySwitchableWorld) {
  flightforge::WorldList list;
  EXPECT_EQ(worldNames(list), "");
  list.worlds = {{"Warehouse", "/Game/Worlds/Warehouse/Maps/Warehouse", "Warehouse"}, {"Valley", "/Game/Worlds/Valley/Maps/Valley", "Valley"}};
  EXPECT_EQ(worldNames(list), "Warehouse, Valley");
}

TEST(World, MatchesTheMapNameIgnoringCase) {
  EXPECT_TRUE(isWorld("/Game/Worlds/Forest/Maps/infinite_forest", "infinite_forest"));
  EXPECT_TRUE(isWorld("/Game/Worlds/Forest/Maps/Infinite_Forest", "infinite_forest"));
  EXPECT_FALSE(isWorld("/Game/Worlds/Forest/Maps/InfinityForest", "infinite_forest"));
  EXPECT_FALSE(isWorld("", "infinite_forest"));
}
