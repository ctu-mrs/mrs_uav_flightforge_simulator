#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from mrs_msgs.srv import ReferenceStampedSrv, Vec4

class Goto(Node):

    def __init__(self):

        super().__init__('goto')

        self.get_logger().info('ROS2 node initialized')

        self.get_logger().info('Setting up the client')

        self.client = self.create_client(ReferenceStampedSrv, "/uav1/control_manager/reference")

        while not self.client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info('service not available, waiting again...')

        request = ReferenceStampedSrv.Request()
        request.header.frame_id = "fcu_untilted"
        request.reference.position.x = 10.0
        request.reference.position.y = 10.0
        request.reference.position.z = 0.0
        request.reference.heading = 1.57

        self.future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, self.future)

        self.get_logger().info('Service called')

        rclpy.shutdown()

def main(args=None):

    rclpy.init(args=args)

    node = Goto()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
