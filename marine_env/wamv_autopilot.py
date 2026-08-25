#!/usr/bin/env python3
"""
WAM-V Autopilot with Waypoint Following (Phase 1-2)
=====================================================

Line-of-Sight (LOS) guidance law + PI heading control + speed profile.

Guidance (Fossen, 2011):
  - Desired heading:  χ_d = α_k + atan2(-y_e, Δ)  (LOS with look-ahead)
    where α_k is path tangent angle, y_e is cross-track error, Δ is look-ahead.
  - Cross-track error: y_e = -(x - x_k)·sin(α_k) + (y - y_k)·cos(α_k)
  - Waypoint acceptance: switch when distance to waypoint < R_accept
    OR when along-track progress exceeds segment length.

Control:
  - Heading: PI controller → differential thrust
  - Speed: P controller → common thrust
  - Turn rate limiting for smooth trajectories

Parameters:
  - waypoints: list of [x, y, speed] waypoints
  - look_ahead_distance: Δ for LOS guidance (m)
  - acceptance_radius: R_accept for waypoint switching (m)
  - heading_kp, heading_ki: PI gains for heading
  - speed_kp: P gain for speed
  - max_thrust, max_diff_ratio: thrust limits
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool, String
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import math
import numpy as np
import json
from typing import List, Optional, Tuple


class WamvAutopilot(Node):
    """Waypoint-following autopilot with LOS guidance + PI control."""

    def __init__(self):
        super().__init__('wamv_autopilot')

        # ── Parameters ──
        # Waypoints stored as flat list [x0, y0, s0, x1, y1, s1, ...]
        # (ROS2 Jazzy 不支持 list-of-lists 参数类型)
        self.declare_parameter('waypoints',
            [0.0, 0.0, 1.5,          # x, y, target_speed (m/s)
             0.0, 200.0, 1.5])       # 直行向北
        self.declare_parameter('look_ahead_distance', 15.0)      # m
        self.declare_parameter('acceptance_radius', 5.0)         # m
        self.declare_parameter('heading_kp', 2.0)
        self.declare_parameter('heading_ki', 0.05)
        self.declare_parameter('heading_kd', 0.3)
        self.declare_parameter('speed_kp', 400.0)                # N per m/s error
        self.declare_parameter('speed_ki', 50.0)
        self.declare_parameter('max_thrust', 2000.0)             # N
        self.declare_parameter('max_diff_ratio', 0.6)            # fraction of base thrust
        self.declare_parameter('odom_topic', '/wamv/state/estimated')
        self.declare_parameter('publish_rate', 20.0)             # Hz
        self.declare_parameter('publish_path', True)

        # ── Load waypoints (flat list → grouped in triples) ──
        raw = self.get_parameter('waypoints').value
        self.waypoints: List[Tuple[float, float, float]] = [
            (raw[i], raw[i + 1], raw[i + 2]) for i in range(0, len(raw), 3)
        ]
        self._wp_idx = 0
        self._mission_complete = False

        # ── State ──
        self.current_pos = np.zeros(2)
        self.current_yaw = 0.0
        self.current_speed = 0.0
        self.current_yaw_rate = 0.0
        self._last_yaw = 0.0

        # ── Control integrators ──
        self._heading_error_integral = 0.0
        self._speed_error_integral = 0.0
        self._last_time = self.get_clock().now().nanoseconds * 1e-9

        # ── Publishers ──
        # WAM-V 的 ros_gz_bridge 在 /wamv 命名空间内，ROS2 topic 相对路径 thrusters/<side>/thrust
        # 解析为 /wamv/thrusters/<side>/thrust → Gazebo wamv/thrusters/<side>/thrust
        self.left_pub = self.create_publisher(
            Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_pub = self.create_publisher(
            Float64, '/wamv/thrusters/right/thrust', 10)
        self.status_pub = self.create_publisher(
            String, '/wamv/autopilot/status', 10)

        if self.get_parameter('publish_path').value:
            self.path_pub = self.create_publisher(
                Path, '/wamv/autopilot/path', 10)
        else:
            self.path_pub = None

        # ── Subscriber ──
        self.odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self._odom_cb, 10)

        # ── Timer ──
        rate = max(self.get_parameter('publish_rate').value, 1.0)
        self.timer = self.create_timer(1.0 / rate, self._control_cycle)

        # ── Publish initial path ──
        if self.path_pub:
            self._publish_path()

        self.get_logger().info(
            f'🚢 WAM-V Autopilot ready: {len(self.waypoints)} waypoints, '
            f'LOS Δ={self.get_parameter("look_ahead_distance").value:.0f}m')

    # =====================================================================
    # Callbacks
    # =====================================================================

    def _odom_cb(self, msg: Odometry):
        """Update current state from EKF estimate."""
        self.current_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        ])
        # Full quaternion → yaw
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.current_yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz))
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = math.sqrt(vx**2 + vy**2)
        self.current_yaw_rate = msg.twist.twist.angular.z

    # =====================================================================
    # LOS Guidance
    # =====================================================================

    def _compute_los_guidance(self) -> Tuple[float, float, float]:
        """Compute LOS desired heading and guidance errors.

        Returns:
            chi_d: desired heading (rad)
            cross_track_error: signed cross-track error (m)
            along_track_dist: distance to next waypoint along path (m)
        """
        if self._wp_idx >= len(self.waypoints) - 1:
            # Final waypoint — point directly at it
            wp = self.waypoints[-1]
            dx = wp[0] - self.current_pos[0]
            dy = wp[1] - self.current_pos[1]
            dist = math.sqrt(dx**2 + dy**2)
            if dist < self.get_parameter('acceptance_radius').value:
                self._mission_complete = True
            return math.atan2(dy, dx), 0.0, dist

        # Current path segment: wp_k → wp_{k+1}
        wp_k = self.waypoints[self._wp_idx]
        wp_next = self.waypoints[self._wp_idx + 1]

        # Path tangent angle α_k
        dx_seg = wp_next[0] - wp_k[0]
        dy_seg = wp_next[1] - wp_k[1]
        alpha_k = math.atan2(dy_seg, dx_seg)
        seg_length = math.sqrt(dx_seg**2 + dy_seg**2)

        if seg_length < 0.01:
            return alpha_k, 0.0, 0.0

        # Cross-track error: y_e = -(x-x_k)·sin(α_k) + (y-y_k)·cos(α_k)
        dx_os = self.current_pos[0] - wp_k[0]
        dy_os = self.current_pos[1] - wp_k[1]
        cross_track = -dx_os * math.sin(alpha_k) + dy_os * math.cos(alpha_k)

        # Along-track progress
        along_track = dx_os * math.cos(alpha_k) + dy_os * math.sin(alpha_k)

        # LOS desired heading
        delta = self.get_parameter('look_ahead_distance').value
        chi_d = alpha_k + math.atan2(-cross_track, delta)

        return chi_d, cross_track, along_track

    def _check_waypoint_advance(self, along_track: float) -> bool:
        """Check if we should advance to the next waypoint.

        Advances when:
          - Distance to waypoint < acceptance_radius, OR
          - Along-track progress exceeds segment length (passed waypoint)
        """
        if self._wp_idx >= len(self.waypoints) - 1:
            # At final waypoint — check distance
            wp = self.waypoints[-1]
            dx = self.current_pos[0] - wp[0]
            dy = self.current_pos[1] - wp[1]
            dist = math.sqrt(dx**2 + dy**2)
            return dist < self.get_parameter('acceptance_radius').value

        wp_k = self.waypoints[self._wp_idx]
        wp_next = self.waypoints[self._wp_idx + 1]
        dx_seg = wp_next[0] - wp_k[0]
        dy_seg = wp_next[1] - wp_k[1]
        seg_length = math.sqrt(dx_seg**2 + dy_seg**2)

        # Distance to next waypoint
        dx = self.current_pos[0] - wp_next[0]
        dy = self.current_pos[1] - wp_next[1]
        dist_to_next = math.sqrt(dx**2 + dy**2)

        acceptance = self.get_parameter('acceptance_radius').value

        return dist_to_next < acceptance or along_track >= seg_length

    # =====================================================================
    # Control
    # =====================================================================

    def _control_cycle(self):
        """Main control loop: LOS guidance → PI heading → thrust mapping."""
        if self._mission_complete:
            # Stop thrusters
            self.left_pub.publish(Float64(data=0.0))
            self.right_pub.publish(Float64(data=0.0))
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        dt = now - self._last_time
        self._last_time = now
        if dt <= 0 or dt > 1.0:
            dt = 0.05

        # ── LOS guidance ──
        chi_d, cross_track, along_track = self._compute_los_guidance()

        # ── Waypoint advance ──
        if self._check_waypoint_advance(along_track):
            old_idx = self._wp_idx
            self._wp_idx = min(self._wp_idx + 1, len(self.waypoints) - 1)
            self._heading_error_integral = 0.0  # reset integrator
            if self._wp_idx != old_idx:
                self.get_logger().info(
                    f'📍 Waypoint {old_idx + 1} reached → advancing to '
                    f'WP{self._wp_idx + 1}: '
                    f'({self.waypoints[self._wp_idx][0]:.0f}, '
                    f'{self.waypoints[self._wp_idx][1]:.0f})')

        # ── Target speed from current segment ──
        if self._wp_idx < len(self.waypoints):
            target_speed = self.waypoints[self._wp_idx][2]
        else:
            target_speed = self.waypoints[-1][2]

        # Reduce speed when cross-track error is large
        if abs(cross_track) > 10.0:
            speed_scale = max(0.4, 1.0 - abs(cross_track) / 50.0)
            target_speed *= speed_scale

        # ── Heading control (PI) ──
        heading_kp = self.get_parameter('heading_kp').value
        heading_ki = self.get_parameter('heading_ki').value
        heading_kd = self.get_parameter('heading_kd').value

        # Heading error (shortest rotation)
        yaw_error = (chi_d - self.current_yaw + math.pi) % (2 * math.pi) - math.pi

        # Anti-windup: only integrate when not saturated
        self._heading_error_integral += yaw_error * dt
        max_integral = 1.0  # rad·s — clamp to prevent windup
        self._heading_error_integral = max(-max_integral,
                                           min(max_integral,
                                               self._heading_error_integral))

        # PI + D (yaw rate damping)
        heading_correction = (heading_kp * yaw_error +
                             heading_ki * self._heading_error_integral -
                             heading_kd * self.current_yaw_rate)

        # ── Speed control (PI) ──
        speed_kp = self.get_parameter('speed_kp').value
        speed_ki = self.get_parameter('speed_ki').value

        speed_error = target_speed - self.current_speed
        self._speed_error_integral += speed_error * dt
        self._speed_error_integral = max(-5.0, min(5.0,
                                          self._speed_error_integral))

        # Feedforward + PI (max thrust to overcome WAM-V hull drag)
        feedforward = target_speed * 1200.0  # ~1200N per m/s
        base_thrust = feedforward + speed_kp * speed_error + \
                      speed_ki * self._speed_error_integral

        max_thrust = self.get_parameter('max_thrust').value
        base_thrust = max(0.0, min(max_thrust, base_thrust))

        # ── Differential thrust for turning ──
        # heading_correction > 0: need CCW turn (yaw increase)
        # CCW turn = right thrust > left thrust
        # Left engine @ body Y=+1.03m → left>right = CW turn
        # So: diff = -heading_correction * gain
        diff_gain = 800.0  # N per rad of heading error
        diff = -heading_correction * diff_gain

        max_diff_ratio = self.get_parameter('max_diff_ratio').value
        diff_max = base_thrust * max_diff_ratio
        diff = max(-diff_max, min(diff_max, diff))

        left_cmd = base_thrust + diff
        right_cmd = base_thrust - diff

        # Clamp to valid range (use max_thrust not hardcoded 1800)
        left_cmd = max(0.0, min(max_thrust, left_cmd))
        right_cmd = max(0.0, min(max_thrust, right_cmd))

        # Emergency stop: if ship is going too fast toward waypoint, reduce
        if self.current_speed > 5.0:
            left_cmd *= 0.3
            right_cmd *= 0.3

        # ── Publish ──
        self.left_pub.publish(Float64(data=float(left_cmd)))
        self.right_pub.publish(Float64(data=float(right_cmd)))

        # ── Status (1 Hz) ──
        if int(now) != int(now - dt):
            status = {
                'wp_idx': self._wp_idx,
                'wp_total': len(self.waypoints),
                'mission_complete': self._mission_complete,
                'cross_track_error': round(float(cross_track), 2),
                'heading_error_deg': round(math.degrees(yaw_error), 1),
                'desired_heading_deg': round(math.degrees(chi_d), 1),
                'current_heading_deg': round(math.degrees(self.current_yaw), 1),
                'target_speed': round(target_speed, 2),
                'current_speed': round(self.current_speed, 2),
                'thrust_left': round(float(left_cmd), 0),
                'thrust_right': round(float(right_cmd), 0),
            }
            self.status_pub.publish(String(data=json.dumps(status)))

    def _publish_path(self):
        """Publish waypoint path for RViz visualization."""
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for wp in self.waypoints:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position = Point(x=float(wp[0]), y=float(wp[1]), z=0.0)
            pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)


def main():
    rclpy.init()
    node = WamvAutopilot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
