import os
import sys
import time
import shutil
import unittest

import launch
import launch_ros
import launch_testing.actions
import launch_testing.asserts
from launch.actions import ExecuteProcess, IncludeLaunchDescription, GroupAction
import rclpy
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Bool

def generate_test_description():

    ld = launch.LaunchDescription()

    uav_type = "x500"
    uav_name = "uav1"
    platform_config = get_package_share_directory("mrs_multirotor_simulator") + "/config/mrs_uav_system/" + uav_type + ".yaml"

    launch_file_path = os.path.abspath(__file__)
    launch_dir = os.path.dirname(launch_file_path)

    test_name = os.path.basename(launch_dir)

    current_rmw = os.environ.get('RMW_IMPLEMENTATION', '')

    if current_rmw == 'rmw_zenoh_cpp':
        ld.add_action(
            launch_ros.actions.Node(
                package='rmw_zenoh_cpp',
                namespace='',
                executable='rmw_zenohd',
                name='zenoh_router',
                output='screen'
            )
        )

    # Detect FlightForge executable
    ff_exec = os.environ.get('FLIGHTFORGE_PATH') or shutil.which('mrs_flight_forge') or shutil.which('flightforge') or '/opt/flightforge/mrs_flight_forge.sh'

    if not os.path.exists(ff_exec):
        raise RuntimeError(f"FlightForge executable not found at '{ff_exec}'. Please set FLIGHTFORGE_PATH or install mrs-flightforge-lite.")

    # Launch the UE5 FlightForge simulator in headless mode without requiring a GPU
    cmd = [ff_exec, '-nullrhi']
    if os.geteuid() == 0:
        # Unreal Engine refuses to run with root privileges ("Refusing to run with the root privileges.").
        # When running in a container as root (e.g. CI), launch the simulator process as an unprivileged user.
        user = 'ubuntu'
        import pwd
        try:
            pwd.getpwnam(user)
        except KeyError:
            import subprocess
            subprocess.run(['useradd', '-m', '-s', '/bin/bash', user], check=False)

        os.makedirs(f'/home/{user}', exist_ok=True)
        if shutil.which('runuser'):
            cmd = ['runuser', '-u', user, '--'] + cmd
        elif shutil.which('su'):
            cmd = ['su', user, '-c', f"{ff_exec} -nullrhi"]

    ld.add_action(
        ExecuteProcess(
            cmd=cmd,
            output='screen',
            name='mrs_flight_forge',
        )
    )

    # Launch MRS UAV System (core + autostart)
    ld.add_action(
        GroupAction([
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    PathJoinSubstitution([
                        FindPackageShare('mrs_uav_testing'),
                        'launch',
                        'mrs_uav_system.launch.py'
                    ])
                ]),
                launch_arguments={
                    'run_automatic_start': "true",
                    'uav_name': uav_name,
                    'platform_config': platform_config,
                    'world_config': os.path.join(launch_dir, 'config', 'world_config.yaml'),
                }.items()
            )
        ])
    )

    # Launch FlightForge Simulator ROS node
    ld.add_action(
        GroupAction([
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    PathJoinSubstitution([
                        FindPackageShare('mrs_uav_flightforge_simulator'),
                        'launch',
                        'flightforge_simulator.launch.py'
                    ])
                ]),
                launch_arguments={
                    'custom_config': os.path.join(launch_dir, 'config', 'simulator.yaml'),
                }.items()
            )
        ])
    )

    # Launch FlightForge HW API plugin node
    ld.add_action(
        GroupAction([
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    PathJoinSubstitution([
                        FindPackageShare('mrs_uav_flightforge_simulator'),
                        'launch',
                        'hw_api.launch.py'
                    ])
                ]),
                launch_arguments={
                    'uav_name': uav_name,
                }.items()
            )
        ])
    )

    # Nodes under test
    ld.add_action(
        launch_ros.actions.Node(
            package='mrs_uav_flightforge_simulator',
            namespace='',
            executable='test_' + test_name,
            name='test_' + test_name,
            output="screen",
            parameters=[
                {'test_name': test_name},
            ],
        )
    )

    # Starts the python test part down below
    ld.add_action(
        launch.actions.TimerAction(
            period=1.0, actions=[launch_testing.actions.ReadyToTest()]
        ),
    )

    return ld

class PublisherHandlerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('integration_test_handler')

    def tearDown(self):
        self.node.destroy_node()

    def test_interactor(self, proc_output, proc_info, timeout=240):

        """Check whether test reported success"""

        test_result = []

        sub = self.node.create_subscription(
            Bool, '/test_result',
            lambda msg: test_result.append(msg), 100)
        try:
            end_time = time.time() + timeout

            while time.time() < end_time:
                if len(test_result) > 0:
                    break

                # fail fast if any part of the system (simulator, FlightForge, MRS UAV System) died,
                # e.g., because of an API version mismatch with the FlightForge binary
                for info in proc_info:
                    returncode = getattr(info, 'returncode', None)
                    name = info.action.process_details['name']
                    if returncode is not None and not name.startswith('test_'):
                        self.fail(f"Process '{name}' exited with code {returncode} before the test finished, see its output above")

                rclpy.spin_once(self.node, timeout_sec=1)

            time.sleep(2.0)

            # check if we have the result
            self.assertTrue(len(test_result) > 0, "Timed out waiting for test result on /test_result")

            # check if the result is true
            self.assertTrue(test_result[0].data, "Test reported failure on /test_result")

        finally:
            self.node.destroy_subscription(sub)

