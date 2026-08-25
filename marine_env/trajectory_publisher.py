#!/usr/bin/env python3
"""
Own-ship trajectory publisher for RViz2 visualization.

Publishes two nav_msgs/Path topics:
  - /wamv/trajectory_gt   — OS ground truth (from /model/wamv/odometry)
  - /wamv/trajectory_ekf  — OS EKF estimate (from /wamv/state/estimated)

Target-ship trajectories are published by target_ship_spawner on /{name}/trajectory.
"""
import rclpy
import math
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped


class TrajectoryPublisher(Node):
    """Accumulates own-ship odometry and publishes Path messages for RViz2."""

    MAX_PTS = 500

    def __init__(self):
        super().__init__('trajectory_publisher')

        # ── OS Ground Truth ──
        self._gt_path: list = []
        self._gt_frame: str = ''
        self._gt_count: int = 0
        self._gt_pub = self.create_publisher(Path, '/wamv/trajectory_gt', 10)
        self.create_subscription(
            Odometry, '/model/wamv/odometry', self._cb_gt, 10)

        # ── OS EKF Estimate ──
        self._ekf_path: list = []
        self._ekf_frame: str = ''
        self._ekf_count: int = 0
        self._ekf_pub = self.create_publisher(Path, '/wamv/trajectory_ekf', 10)
        self.create_subscription(
            Odometry, '/wamv/state/estimated', self._cb_ekf, 10)

        # Publish at 5 Hz
        self._timer = self.create_timer(0.2, self._publish_all)
        # Diagnostic at 0.1 Hz
        self._diag = self.create_timer(5.0, self._diag_print)

        self.get_logger().info(
            'Trajectory Publisher ready — '
            'GT: /model/wamv/odometry → /wamv/trajectory_gt, '
            'EKF: /wamv/state/estimated → /wamv/trajectory_ekf')

    # ── Callbacks ──

    def _cb_gt(self, msg: Odometry):
        self._gt_count += 1
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        self._gt_path.append((x, y))
        if len(self._gt_path) > self.MAX_PTS:
            self._gt_path.pop(0)
        if not self._gt_frame:
            self._gt_frame = msg.header.frame_id or 'map'
            q = msg.pose.pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.get_logger().info(
                f'  ✓ GT start=({x:.1f},{y:.1f}) '
                f'hdg={math.degrees(math.atan2(siny, cosy)):.0f}° '
                f'frame={self._gt_frame}')

    def _cb_ekf(self, msg: Odometry):
        self._ekf_count += 1
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        self._ekf_path.append((x, y))
        if len(self._ekf_path) > self.MAX_PTS:
            self._ekf_path.pop(0)
        if not self._ekf_frame:
            self._ekf_frame = msg.header.frame_id or 'map'
            q = msg.pose.pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.get_logger().info(
                f'  ✓ EKF start=({x:.1f},{y:.1f}) '
                f'hdg={math.degrees(math.atan2(siny, cosy)):.0f}° '
                f'frame={self._ekf_frame}')

    # ── Path builder ──

    def _build_path_msg(self, path_list, frame):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame
        for (x, y) in path_list:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        return msg

    # ── Timers ──

    def _publish_all(self):
        if self._gt_frame and self._gt_path:
            self._gt_pub.publish(self._build_path_msg(self._gt_path, self._gt_frame))
        if self._ekf_frame and self._ekf_path:
            self._ekf_pub.publish(self._build_path_msg(self._ekf_path, self._ekf_frame))

    def _diag_print(self):
        for label, count, path, frame in [
            ('GT', self._gt_count, self._gt_path, self._gt_frame),
            ('EKF', self._ekf_count, self._ekf_path, self._ekf_frame),
        ]:
            pts = len(path)
            last = f'({path[-1][0]:.1f},{path[-1][1]:.1f})' if pts else '--'
            self.get_logger().info(
                f'  OS {label}: msgs={count} pts={pts} last={last} frame={frame}')


def main():
    rclpy.init()
    rclpy.spin(TrajectoryPublisher())


if __name__ == '__main__':
    main()
