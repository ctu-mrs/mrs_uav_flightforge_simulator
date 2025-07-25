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

    pkg_share_path = get_package_share_directory(pkg_name)

    namespace='flightforge_simulator'

    mrs_simulator_path = get_package_share_directory("mrs_multirotor_simulator")

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

    # #{ configs

    config_files = [
        # general configs
        pkg_share_path + '/config/flightforge_simulator.yaml',
        # uavs to load
        pkg_share_path + '/config/uavs.yaml',
    ]

    mrs_multirotor_simulator_uav_configs = [
        # general configs
        pkg_share_path + '/config/flightforge_simulator.yaml',
        # uavs to load
        pkg_share_path + '/config/uavs.yaml',
        # general configs
        mrs_simulator_path + '/config/controllers/attitude_controller.yaml',
        mrs_simulator_path + '/config/controllers/rate_controller.yaml',
        mrs_simulator_path + '/config/controllers/position_controller.yaml',
        mrs_simulator_path + '/config/controllers/velocity_controller.yaml',
        mrs_simulator_path + '/config/controllers/mixer.yaml',
        # the UAV configs
        mrs_simulator_path + '/config/uavs/a300.yaml',
        mrs_simulator_path + '/config/uavs/f330.yaml',
        mrs_simulator_path + '/config/uavs/f450.yaml',
        mrs_simulator_path + '/config/uavs/f550.yaml',
        mrs_simulator_path + '/config/uavs/naki.yaml',
        mrs_simulator_path + '/config/uavs/robofly.yaml',
        mrs_simulator_path + '/config/uavs/t650.yaml',
        mrs_simulator_path + '/config/uavs/x500.yaml',
    ]

    # #} end of configs

    ld.add_action(ComposableNodeContainer(

        namespace='',
        name=namespace+'_container',
        package='rclcpp_components',
        executable='component_container_mt',
        output="screen",

        # prefix=['debug_roslaunch ' + os.ttyname(sys.stdout.fileno())],

        parameters=[
            {'thread_num': os.cpu_count()},
        ],

        composable_node_descriptions=[

            ComposableNode(

                package=pkg_name,
                plugin='mrs_uav_flightforge_simulator::FlightforgeSimulator',
                namespace='',
                name='flightforge_simulator',
                parameters=[
                    {'config_files': config_files},
                    {'custom_config': custom_config},
                    {'uav_configs': mrs_multirotor_simulator_uav_configs},
                    {'use_sim_time': True},
                ],

                remappings=[
                    ("~/clock_out", "/clock"),
                    ("~/uav_poses_out", "~/uav_poses"),
                ],
            )

        ],

    ))

    return ld
