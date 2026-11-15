#!/usr/bin/python3

# Continuously wobbles the extrinsics of the simulated stereo cameras.
#
# Publishes to /<uav_name>/stereo/{left,right}/extrinsics_adjustment_in (geometry_msgs/Pose).
# The pose is a correction relative to the nominal camera pose, expressed in the camera's
# nominal optical frame (x right, y down, z forward):
#
#   T_fcu_cam = T_fcu_cam_nominal * T_adjustment
#
# Each camera moves up/down by +-amplitude_z and pitches by +-amplitude_pitch.
# The left and right cameras move out of phase (phase shift = camera_phase, pi by default),
# so their relative extrinsics change over time.
#
# On exit (Ctrl+C), identity adjustments are sent to restore the nominal extrinsics.
#
# Parameters (--ros-args -p name:=value):
#   uav_name           [string] default "uav1"
#   amplitude_z        [m]      default 0.025
#   amplitude_pitch    [deg]    default 3.0     (positive = camera tilts up)
#   frequency          [Hz]     default 0.5     (wobble frequency)
#   rate               [Hz]     default 20.0    (publishing rate)
#   camera_phase       [rad]    default pi      (phase between the left and right camera)
#   pitch_phase        [rad]    default 0.0     (phase between pitch and the up/down motion)

import math

import rclpy
from rclpy.signals import SignalHandlerOptions
from rclpy.node import Node
from geometry_msgs.msg import Pose


class StereoWobble(Node):

    def __init__(self):

        super().__init__('stereo_wobble')

        self.uav_name        = self.declare_parameter('uav_name', 'uav1').value
        self.amplitude_z     = self.declare_parameter('amplitude_z', 0.02).value
        self.amplitude_pitch = math.radians(self.declare_parameter('amplitude_pitch', 1.0).value)
        self.frequency       = self.declare_parameter('frequency', 0.5).value
        self.rate            = self.declare_parameter('rate', 20.0).value
        self.camera_phase    = self.declare_parameter('camera_phase', math.pi).value
        self.pitch_phase     = self.declare_parameter('pitch_phase', 0.0).value

        self.pub_left  = self.create_publisher(Pose, '/' + self.uav_name + '/stereo/left/extrinsics_adjustment_in', 10)
        self.pub_right = self.create_publisher(Pose, '/' + self.uav_name + '/stereo/right/extrinsics_adjustment_in', 10)

        self.t0 = self.get_clock().now()

        self.timer = self.create_timer(1.0 / self.rate, self.timerMain)

        self.get_logger().info('wobbling the stereo cameras of %s: +-%.1f cm, +-%.1f deg, %.2f Hz' %
                               (self.uav_name, 100.0 * self.amplitude_z, math.degrees(self.amplitude_pitch), self.frequency))

    def adjustment(self, phase):

        # displacement up (in the fcu frame) and pitch up of the camera
        dz    = self.amplitude_z * math.sin(phase)
        pitch = self.amplitude_pitch * math.sin(phase + self.pitch_phase)

        pose = Pose()

        # optical frame: y points down
        pose.position.x = 0.0
        pose.position.y = -dz
        pose.position.z = 0.0

        # rotation about the optical x axis (right); positive angle tilts the optical axis (z) up
        pose.orientation.w = math.cos(pitch / 2.0)
        pose.orientation.x = math.sin(pitch / 2.0)
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0

        return pose

    def timerMain(self):

        t     = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        phase = 2.0 * math.pi * self.frequency * t

        self.pub_left.publish(self.adjustment(phase))
        self.pub_right.publish(self.adjustment(phase + self.camera_phase))

    def reset(self):

        identity = Pose()
        identity.orientation.w = 1.0

        self.pub_left.publish(identity)
        self.pub_right.publish(identity)


def main(args=None):

    # handle SIGINT ourselves so that the context is still valid for sending the reset
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)

    node = StereoWobble()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.get_logger().info('restoring the nominal extrinsics')
            node.reset()
            rclpy.spin_once(node, timeout_sec=0.2)
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
