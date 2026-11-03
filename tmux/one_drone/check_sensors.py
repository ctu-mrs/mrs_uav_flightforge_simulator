#!/usr/bin/env python3
"""usage: check_sensors.py [duration_s]"""

import os
import sys
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from event_camera_msgs.msg import EventPacket
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, Range
from tf2_ros import Buffer, TransformListener

ROOT = "mrs_uav_flightforge_simulator"
CAMERAS = ["rgb", "rgb_segmented", "depth", "instance_seg", "fisheye"]


def merged_sensors():
    here = os.path.dirname(os.path.abspath(__file__))
    custom = yaml.safe_load(open(os.path.join(here, "config", "simulator.yaml")))
    default = yaml.safe_load(open(os.path.join(get_package_share_directory(ROOT), "config", "flightforge_simulator.yaml")))
    uav = custom["uav_names"][0]
    out = []
    for kind, base in default[ROOT]["sensors"].items():
        over = custom.get(ROOT, {}).get("sensors", {}).get(kind, {})
        n = over.get("n_sensors", base["n_sensors"])
        for i in range(n):
            params = dict(base["default"])
            params.update(over.get(f"sensor_{i}", {}) or {})
            out.append((uav, kind, i, params))
    return out


def topics_for(uav, kind, i, params):
    ns = f"/{uav}/{kind}_{i}"
    if kind in CAMERAS:
        return [(ns + "/image_raw", Image, f"{uav}/{kind}_{i}", ns + "/camera_info")]
    if kind == "stereo":
        return [(ns + f"/{eye}/image_raw", Image, f"{uav}/{kind}_{i}_{eye}", ns + f"/{eye}/camera_info") for eye in ("left", "right")]
    if kind == "event":
        return [(ns + "/events", EventPacket, f"{uav}/{kind}_{i}", None)]
    if kind == "lidar":
        out = [(ns + "/points", PointCloud2, f"{uav}/{kind}_{i}", None)]
        if params.get("publish_free", True):
            out.append((ns + "/points_free", PointCloud2, f"{uav}/{kind}_{i}", None))
        return out
    return [(ns + "/range", Range, f"{uav}/{kind}_{i}", None)]


class Checker(Node):

    def __init__(self, sensors):
        super().__init__("check_sensors")
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.rows = []
        for uav, kind, i, params in sensors:
            for topic, msg_type, frame, info_topic in topics_for(uav, kind, i, params):
                row = {"topic": topic, "rate": params["rate"], "frame": frame, "uav": uav, "stamps": [], "size": None, "info": None}
                self.rows.append(row)
                self.create_subscription(msg_type, topic, self.on_msg(row), qos_profile_sensor_data)
                if info_topic:
                    self.create_subscription(CameraInfo, info_topic, self.on_info(row), qos_profile_sensor_data)

    def on_msg(self, row):
        def cb(msg):
            row["stamps"].append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            if isinstance(msg, Image):
                row["size"] = (msg.width, msg.height)
        return cb

    def on_info(self, row):
        def cb(msg):
            row["info"] = (msg.width, msg.height)
        return cb


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    rclpy.init()
    node = Checker(merged_sensors())
    t0 = time.time()
    while time.time() - t0 < duration:
        rclpy.spin_once(node, timeout_sec=0.05)

    failed = False
    print(f"{'topic':42} {'conf':>6} {'sim Hz':>7} {'back':>4} {'dup':>4} {'info':>5} {'tf':>3}")
    for r in node.rows:
        s = r["stamps"]
        gaps = [b - a for a, b in zip(s, s[1:])]
        rate = (len(s) - 1) / (s[-1] - s[0]) if len(s) > 2 and s[-1] > s[0] else 0.0
        back = sum(g < 0 for g in gaps)
        dup = sum(g == 0 for g in gaps)
        info_ok = r["info"] is None or r["size"] is None or r["info"] == r["size"]
        tf_ok = node.tf.can_transform(f"{r['uav']}/fcu", r["frame"], Time())
        # an event camera's rate is only its poll rate
        rate_ok = rate > 0 if r["topic"].endswith("/events") else rate >= 0.9 * r["rate"]
        ok = rate_ok and back == 0 and dup == 0 and info_ok and tf_ok
        failed |= not ok
        print(f"{r['topic']:42} {r['rate']:6.1f} {rate:7.1f} {back:4d} {dup:4d} {str(info_ok):>5} {str(tf_ok):>3} {'' if ok else 'FAIL'}")

    rclpy.shutdown()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
