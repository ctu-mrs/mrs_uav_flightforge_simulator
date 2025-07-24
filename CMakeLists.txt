cmake_minimum_required(VERSION 3.5)
project(mrs_uav_flightforge_simulator)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set the compile options to show code warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

add_compile_options(-Wno-missing-braces)

# 1 == use rclcpp::Timer implementation
# 0 == use MRS's thread timer implementation
set(USE_ROS_TIMER 0)

if(${USE_ROS_TIMER})
  MESSAGE(WARNING "[mrs_uav_flightforge_simulator]: Compiling with ROS Timers. This can cause high CPU load in runtime.")
endif()

if(ENABLE_COVERAGE)
  message(WARNING "building for coverage, the performance might be limited")

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage -fprofile-arcs")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-arcs")

  add_definitions("-O1")
  add_definitions("-g")
endif()

set(DEPENDENCIES
  rclcpp
  rclcpp_components
  Eigen3
  mrs_lib
  mrs_msgs
  mrs_uav_hw_api
  mrs_multirotor_simulator
  nav_msgs
  rosgraph_msgs
  sensor_msgs
  std_msgs
  std_srvs
  image_transport
  cv_bridge
  pcl_ros
  pcl_conversions
  geometry_msgs
  backward_ros
  rosidl_default_generators
)

set(LIBRARIES
  MrsUavFlightforgeSimulator_FlightforgeConnector
  MrsUavFlightforgeSimulator_FlightforgeSimulator
  MrsUavFlightforgeSimulator_HwApiPlugin
)

foreach(DEP IN LISTS DEPENDENCIES)
  find_package(${DEP} REQUIRED)
endforeach()

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SetOrientation.srv"
  DEPENDENCIES geometry_msgs
)

include_directories(
  include
  include/cereal
  include/flight_forge_connector
  include/kissnet
  ${rclcpp_INCLUDE_DIRS}
  ${mrs_lib_INCLUDE_DIRS}
  ${mrs_multirotor_simulator_INCLUDE_DIRS}
  ${mrs_uav_hw_api_INCLUDE_DIRS}
)

## | --------------------- FlightForge connector --------------------- |

add_library(MrsUavFlightforgeSimulator_FlightforgeConnector SHARED
  src/flight_forge_connector/socket_client.cpp
  src/flight_forge_connector/flight_forge_connector.cpp
  src/flight_forge_connector/game_mode_controller.cpp
)

## | ------------------------ FlightForge Simulator ----------------------- |

add_library(MrsUavFlightforgeSimulator_FlightforgeSimulator SHARED
  src/flightforge_simulator.cpp
)

target_link_libraries(MrsUavFlightforgeSimulator_FlightforgeSimulator
  MrsUavFlightforgeSimulator_FlightforgeConnector
  ${PROJECT_NAME}__rosidl_generator_cpp
)

ament_target_dependencies(MrsUavFlightforgeSimulator_FlightforgeSimulator
  ${DEPENDENCIES}
)

rclcpp_components_register_nodes(MrsUavFlightforgeSimulator_FlightforgeSimulator PLUGIN "mrs_uav_flightforge_simulator::FlightforgeSimulator" EXECUTABLE MrsUavFlightforgeSimulator_FlightforgeSimulator)

target_compile_definitions(MrsUavFlightforgeSimulator_FlightforgeSimulator PRIVATE USE_ROS_TIMER=${USE_ROS_TIMER})

## | ---------------------- HW Api Plugin --------------------- |

add_library(MrsUavFlightforgeSimulator_HwApiPlugin SHARED
  src/hw_api_plugin.cpp
)

ament_target_dependencies(MrsUavFlightforgeSimulator_HwApiPlugin
  ${DEPENDENCIES}
)

target_compile_definitions(MrsUavFlightforgeSimulator_HwApiPlugin PRIVATE USE_ROS_TIMER=${USE_ROS_TIMER})

## --------------------------------------------------------------
## |                       Export plugins                       |
## --------------------------------------------------------------

pluginlib_export_plugin_description_file(mrs_uav_hw_api plugins.xml)

## --------------------------------------------------------------
## |                           Install                          |
## --------------------------------------------------------------

ament_export_libraries(
  ${LIBRARIES}
)

install(
  TARGETS ${LIBRARIES}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include
  DESTINATION .
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY tmux
  USE_SOURCE_PERMISSIONS
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY srv
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(
  include
)

ament_export_targets(
  export_${PROJECT_NAME} HAS_LIBRARY_TARGET
)

ament_export_dependencies(
  ${DEPENDENCIES}
)

ament_package()
