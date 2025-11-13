#!/usr/bin/env python3
import rospy
import numpy as np
from stereo_msgs.msg import DisparityImage
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

class DisparityToDepth:
    def __init__(self):
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("depth/image_rect", Image, queue_size=1)

        rospy.Subscriber("disparity", DisparityImage, self.cb, queue_size=1)
        rospy.Subscriber("left/camera_info", CameraInfo, self.caminfo_cb, queue_size=1)
        rospy.Subscriber("right/camera_info", CameraInfo, self.caminfo_cb, queue_size=1)

        self.f = None   # focal length in pixels
        self.B = None   # stereo baseline in meters

    def caminfo_cb(self, msg):
        # Focal length fx
        self.f = msg.K[0]

        # Baseline B (extracted from projection matrix Tx of right camera)
        # P[3] = -fx * Tx, so Tx = -P[3] / fx
        if msg.P[3] != 0:
            Tx = -msg.P[3] / msg.K[0]
            self.B = abs(Tx)

    def cb(self, msg):
        if self.f is None or self.B is None:
            rospy.logwarn_throttle(5.0, "Waiting for camera_info to get f and B...")
            return

        # Convert disparity to depth
        disp = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding="32FC1")
        depth = np.zeros_like(disp, dtype=np.float32)

        valid = disp > 0.0
        depth[valid] = self.f * self.B / disp[valid]  # in meters

        # Scale to millimeters and convert to uint16 such that the output is same as from the gazebo depth camera
        depth_mm = (depth * 1000.0).astype(np.uint16)

        # Publish depth image (16UC1, millimeters)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_mm, encoding="16UC1")
        depth_msg.header = msg.header
        self.pub.publish(depth_msg)

if __name__ == "__main__":
    rospy.init_node("disparity_to_depth")
    DisparityToDepth()
    rospy.spin()
