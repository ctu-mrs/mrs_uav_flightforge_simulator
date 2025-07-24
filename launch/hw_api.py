#!/usr/bin/env python3

import launch
import os
import sys

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
    IfElseSubstitution,
    PythonExpression,
    PathJoinSubstitution,
    EnvironmentVariable,
)

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    ld = launch.LaunchDescription()

    pkg_name = "mrs_uav_flightforge_simulator"

    this_pkg_path = get_package_share_directory(pkg_name)
    namespace='hw_api'

    # #{ uav_name

    uav_name = LaunchConfiguration('uav_name')

    ld.add_action(DeclareLaunchArgument(
        'uav_name',
        default_value=os.getenv('UAV_NAME', "uav1"),
        description="The uav name used for namespacing.",
    ))

    # #} end of custom_config

    # #{ custom_config

    custom_config = LaunchConfiguration('custom_config')

    # this adds the args to the list of args available for this launch files
    # these args can be listed at runtime using -s flag
    # default_value is required to if the arg is supposed to be optional at launch time
    ld.add_action(DeclareLaunchArgument(
        'custom_config',
        default_value="",
        description="Path to the custom configuration file. The path can be absolute, starting with '/' or relative to the current working directory",
    ))

    # behaviour:
    #     custom_config == "" => custom_config: ""
    #     custom_config == "/<path>" => custom_config: "/<path>"
    #     custom_config == "<path>" => custom_config: "$(pwd)/<path>"
    custom_config = IfElseSubstitution(
            condition=PythonExpression(['"', custom_config, '" != "" and ', 'not "', custom_config, '".startswith("/")']),
            if_value=PathJoinSubstitution([EnvironmentVariable('PWD'), custom_config]),
            else_value=custom_config
    )

    # #} end of custom_config

    # #{ use_sim_time

    use_sim_time = LaunchConfiguration('use_sim_time')

    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value=os.getenv('USE_SIM_TIME', "true"),
        description="Should the node subscribe to sim time?",
    ))

    # #} end of custom_config

    # the first one has the priority
    configs = [
        this_pkg_path + '/config/hw_api.yaml',
        get_package_share_directory("mrs_uav_hw_api") + "/config/hw_api.yaml",
    ]

    ld.add_action(ComposableNodeContainer(

        namespace=uav_name,
        name=namespace+'_container',
        package='rclcpp_components',
        executable='component_container_mt',
        output="screen",

        # prefix=['debug_roslaunch ' + os.ttyname(sys.stdout.fileno())],

        composable_node_descriptions=[

            ComposableNode(

                package="mrs_uav_hw_api",
                plugin='mrs_uav_hw_api::HwApiManager',
                namespace=uav_name,
                name='hw_api',
                parameters=[
                    {"uav_name": uav_name},
                    {"topic_prefix": ["/", uav_name]},
                    {"use_sim_time": use_sim_time},
                    {"configs": configs},
                    {'custom_config': custom_config},
                ],

                remappings=[
                  ("~/simulator_imu_in", ["/flightforge_simulator/", uav_name, "/imu"]),
                  ("~/simulator_odom_in", ["/flightforge_simulator/", uav_name, "/odom"]),
                  ("~/simulator_rangefinder_in", ["/flightforge_simulator/", uav_name, "/rangefinder"]),
                  ("~/simulator_actuators_cmd_out", ["/flightforge_simulator/", uav_name, "/actuators_cmd"]),
                  ("~/simulator_control_group_cmd_out", ["/flightforge_simulator/", uav_name, "/control_group_cmd"]),
                  ("~/simulator_attitude_rate_cmd_out", ["/flightforge_simulator/", uav_name, "/attitude_rate_cmd"]),
                  ("~/simulator_attitude_cmd_out", ["/flightforge_simulator/", uav_name, "/attitude_cmd"]),
                  ("~/simulator_acceleration_hdg_rate_cmd_out", ["/flightforge_simulator/", uav_name, "/acceleration_hdg_rate_cmd"]),
                  ("~/simulator_acceleration_hdg_cmd_out", ["/flightforge_simulator/", uav_name, "/acceleration_hdg_cmd"]),
                  ("~/simulator_velocity_hdg_rate_cmd_out", ["/flightforge_simulator/", uav_name, "/velocity_hdg_rate_cmd"]),
                  ("~/simulator_velocity_hdg_cmd_out", ["/flightforge_simulator/", uav_name, "/velocity_hdg_cmd"]),
                  ("~/simulator_position_cmd_out", ["/flightforge_simulator/", uav_name, "/position_cmd"]),
                  ("~/simulator_tracker_cmd_out", ["/flightforge_simulator/", uav_name, "/tracker_cmd"]),
                ],
            )

        ],

    ))

    return ld
