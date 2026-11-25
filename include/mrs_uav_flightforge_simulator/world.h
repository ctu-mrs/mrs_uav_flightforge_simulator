#pragma once

#include <optional>
#include <string>

#include <flight_forge_connector/data_types.h>

namespace mrs_uav_flightforge_simulator
{

// an empty world_name keeps whatever world the simulator has loaded
std::optional<std::string> worldToSwitchTo(const std::string& configured);

// a world has begun play and no switch is loading another
bool worldReady(const flightforge::WorldList& list);

// the names switch_world takes, comma-separated
std::string worldNames(const flightforge::WorldList& list);

// the package's map is named short_name, ignoring case
bool isWorld(const std::string& package, const std::string& short_name);

}  // namespace mrs_uav_flightforge_simulator
