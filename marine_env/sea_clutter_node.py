#!/usr/bin/env python3
"""
Sea Clutter + False Alarm Injector for X-band Marine Radar.
- Subscribes to raw radar PointCloud2
- Injects Weibull-distributed sea clutter
- Injects random false alarm detections
- Publishes processed PointCloud2 with clutter
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import struct
import time


class SeaClutterNode(Node):
    def __init__(self):
        super().__init__('sea_clutter')

        self.declare_parameter('input_topic', '/wamv/sensors/lidars/xband_radar_sensor/points')
        self.declare_parameter('output_topic', '/wamv/sensors/radars/xband/points_cluttered')
        self.declare_parameter('clutter_shape', 1.5)      # Weibull shape (1=exponential, <1=heavy tail)
        self.declare_parameter('clutter_scale', 0.03)      # Weibull scale (~clutter intensity)
        self.declare_parameter('false_alarm_rate', 0.02)   # probability of false alarm per point
        self.declare_parameter('false_alarm_range', 500.0) # max range for false alarms
        self.declare_parameter('enable_clutter', True)
        self.declare_parameter('enable_false_alarms', True)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.sub = self.create_subscription(PointCloud2, input_topic, self.callback, 10)
        self.pub = self.create_publisher(PointCloud2, output_topic, 10)

        # Also publish LaserScan with clutter
        self.pub_scan = self.create_publisher(
            PointCloud2,
            '/wamv/sensors/radars/xband/scan_cluttered',
            10)

        self.get_logger().info(f'Sea Clutter Node started: {input_topic} -> {output_topic}')

    def callback(self, msg):
        enable_clutter = self.get_parameter('enable_clutter').value
        enable_false = self.get_parameter('enable_false_alarms').value

        if not enable_clutter and not enable_false:
            self.pub.publish(msg)
            return

        # Decode original point cloud
        points = list(pointcloud2_to_xyz(msg))
        if len(points) == 0:
            self.pub.publish(msg)
            return

        pts = np.array(points, dtype=np.float32)

        # --- Weibull sea clutter ---
        if enable_clutter:
            shape = self.get_parameter('clutter_shape').value
            scale = self.get_parameter('clutter_scale').value

            # Add Weibull noise to each point's intensity (last field)
            # For points at longer range, clutter increases
            ranges = np.linalg.norm(pts[:, :3], axis=1)
            range_factor = np.clip(ranges / 200.0, 0.5, 3.0)
            clutter_intensity = scale * range_factor * np.random.weibull(shape, len(pts))
            pts[:, 3] += clutter_intensity

        # --- False alarms ---
        if enable_false:
            false_rate = self.get_parameter('false_alarm_rate').value
            max_range = self.get_parameter('false_alarm_range').value
            n_false = np.random.poisson(false_rate * 360)  # ~7 false alarms/scan at 2%

            if n_false > 0:
                false_pts = np.zeros((n_false, 4), dtype=np.float32)
                for i in range(n_false):
                    az = np.random.uniform(-np.pi, np.pi)
                    r = np.random.uniform(10, max_range)
                    false_pts[i, 0] = r * np.cos(az)
                    false_pts[i, 1] = r * np.sin(az)
                    false_pts[i, 2] = np.random.uniform(-0.5, 0.5)
                    false_pts[i, 3] = np.random.uniform(0.01, 0.2)  # random intensity
                pts = np.vstack([pts, false_pts])

        # Re-encode
        out_msg = xyz_to_pointcloud2(pts, msg.header)
        self.pub.publish(out_msg)
        self.pub_scan.publish(out_msg)


def pointcloud2_to_xyz(cloud):
    """Extract x,y,z,intensity from PointCloud2."""
    assert isinstance(cloud, PointCloud2)
    fmt = '<fff'  # x,y,z = float32
    # find intensity offset
    intensity_offset = None
    for field in cloud.fields:
        if field.name == 'intensity':
            intensity_offset = field.offset
            break
    point_step = cloud.point_step
    for i in range(cloud.width * cloud.height):
        offset = i * point_step
        x, y, z = struct.unpack_from(fmt, cloud.data, offset)
        if intensity_offset is not None:
            intensity = struct.unpack_from('<f', cloud.data, offset + intensity_offset)[0]
        else:
            intensity = 0.0
        yield (x, y, z, intensity)


def xyz_to_pointcloud2(points, header):
    """Encode numpy array to PointCloud2."""
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(points)
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.is_bigendian = False
    msg.is_dense = True
    msg.data = points.astype(np.float32).tobytes()
    return msg


def main():
    rclpy.init()
    node = SeaClutterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
