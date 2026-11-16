#include <mrs_uav_flightforge_simulator/world.h>

#include <algorithm>
#include <cctype>

namespace mrs_uav_flightforge_simulator
{

/* worldToSwitchTo() //{ */

std::optional<std::string> worldToSwitchTo(const std::string& configured) {

  const auto first = configured.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return std::nullopt;
  }
  const auto last = configured.find_last_not_of(" \t\r\n");
  return configured.substr(first, last - first + 1);
}

//}

/* worldReady() //{ */

bool worldReady(const flightforge::WorldList& list) {
  return !list.current.empty() && list.loading.empty();
}

//}

/* worldNames() //{ */

std::string worldNames(const flightforge::WorldList& list) {

  std::string names;
  for (const auto& world : list.worlds) {
    names += (names.empty() ? "" : ", ") + world.name;
  }
  return names;
}

//}

/* isWorld() //{ */

bool isWorld(const std::string& package, const std::string& short_name) {

  const std::string map = package.substr(package.find_last_of('/') + 1);
  return std::equal(map.begin(), map.end(), short_name.begin(), short_name.end(),
                    [](const char a, const char b) { return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b)); });
}

//}

}  // namespace mrs_uav_flightforge_simulator
