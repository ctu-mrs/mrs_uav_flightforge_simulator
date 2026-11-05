#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from mrs_msgs.srv import PathSrv
from mrs_msgs.msg import Path
from mrs_msgs.msg import Reference

class NodeObj(Node):

    def __init__(self):

        super().__init__('goto')

        self.get_logger().info('ROS2 node initialized')

        self.get_logger().info('Setting up the client')

        self.client = self.create_client(PathSrv, "/uav1/trajectory_generation/path")

        while not self.client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info('service not available, waiting again...')

        request = PathSrv.Request()
        request.path.fly_now = True
        request.path.use_heading = True
        request.path.max_execution_time = 5.0
        request.path.header.frame_id = "world_origin"

        p1 = Reference()
        p1.position.x = -26.0
        p1.position.y = 17.0
        p1.position.z = 7.6
        p1.heading = 5.79

        p2 = Reference()
        p2.position.x = -65.91
        p2.position.y = 36.12
        p2.position.z = 10.32
        p2.heading = 3.93

        p3 = Reference()
        p3.position.x = -39.03
        p3.position.y = 67.18
        p3.position.z = 5.89
        p3.heading = 4.35

        p4 = Reference()
        p4.position.x = -32.12
        p4.position.y = 77.53
        p4.position.z = 5.89
        p4.heading = 2.64

        p5 = Reference()
        p5.position.x = 10.41
        p5.position.y = 48.83
        p5.position.z = 2.16
        p5.heading = 2.75

        p6 = Reference()
        p6.position.x = 22.98
        p6.position.y = 44.89
        p6.position.z = 2.66
        p6.heading = 2.6

        p7 = Reference()
        p7.position.x = 58.06
        p7.position.y = 44.75
        p7.position.z = 7.85
        p7.heading = 3.65

        p8 = Reference()
        p8.position.x = 84.51
        p8.position.y = 28.21
        p8.position.z = 7.85
        p8.heading = 3.55

        p9 = Reference()
        p9.position.x = 84.45
        p9.position.y = 28.21
        p9.position.z = 14.75
        p9.heading = 3.14

        p10 = Reference()
        p10.position.x = 89.61
        p10.position.y = -45.77
        p10.position.z = 9.29
        p10.heading = 0.86

        request.path.points = [
                p10,
                p9,
                p7,
                p6,
                p5,
                p4,
                p3,
                p2,
                p10,
                ];

        self.future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, self.future)

        self.get_logger().info('Service called')

        rclpy.shutdown()

def main(args=None):

    rclpy.init(args=args)

    node = NodeObj()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
