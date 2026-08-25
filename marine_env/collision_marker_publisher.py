#!/usr/bin/env python3
"""
Collision detection + emergency stop for COLREGS scenarios.

Detects when OS ↔ target ship distance < STOP_DIST,
publishes zero-thrust at 50 Hz to stop both ships,
logs the collision point coordinates.

Key features:
  - Speed estimation from position deltas (for diagnostic accuracy)
  - Per-ship speed logging every 3s
  - STOP_DIST configurable via ROS param
  - Collision prediction: estimates time-to-meeting based on current speeds

No RViz2 markers — collision is visible where trajectory lines end.
"""
import rclpy
import math
import yaml
import os
import json
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64


def _load_ship_names():
    for p in ['/home/xxy/vrx_ws/src/marine_env/config/target_ships.yaml',
              os.path.expanduser('~/vrx_ws/src/marine_env/config/target_ships.yaml')]:
        if os.path.exists(p):
            try:
                cfg = yaml.safe_load(open(p))
                sj = cfg['target_ship_spawner']['ros__parameters']['ships_json']
                ships = json.loads(sj) if isinstance(sj, str) else sj
                return [s['name'] for s in ships]
            except Exception:
                pass
    return []


class CollisionMarkerPublisher(Node):

    def __init__(self):
        super().__init__('collision_marker_publisher')

        self.declare_parameter('stop_distance', 5.0)
        self.STOP_DIST = self.get_parameter('stop_distance').value

        self._os_pos = (0.0, 0.0)
        self._os_prev_pos = (0.0, 0.0)
        self._os_speed = 0.0
        self._os_count = 0
        self._ts_positions = {}
        self._ts_prev_positions = {}
        self._ts_speeds = {}
        self._ts_counts = {}
        self._collided_ts = set()   # target ships that have collided
        self._os_stopped = False
        self._stop_active = False
        self._last_diag_time = 0.0

        # Subscriptions
        self.create_subscription(Odometry, '/model/wamv/odometry', self._cb_os, 10)
        ship_names = _load_ship_names()
        for name in ship_names:
            self.create_subscription(
                Odometry, f'/model/{name}/odometry',
                lambda msg, n=name: self._cb_ts(n, msg), 10)
            self._ts_positions[name] = (0.0, 0.0)
            self._ts_prev_positions[name] = (0.0, 0.0)
            self._ts_speeds[name] = 0.0
            self._ts_counts[name] = 0

        self._total_ts = len(ship_names)
        self.get_logger().error(
            f'INIT: OS + {ship_names}, stop at {self.STOP_DIST:.1f}m, '
            f'OS stops only after all {self._total_ts} TS collide')

        # Zero-thrust publishers (absolute topics — explicit ros_gz_bridge handles OS)
        self._thrust_pubs = {}
        for name in ['wamv'] + ship_names:
            self._thrust_pubs[name] = (
                self.create_publisher(Float64, f'/{name}/thrusters/left/thrust', 10),
                self.create_publisher(Float64, f'/{name}/thrusters/right/thrust', 10),
            )

        # Timers
        self.create_timer(0.2, self._tick)       # 5 Hz detect
        self.create_timer(0.02, self._stop_tick) # 50 Hz override
        self.create_timer(3.0, self._diag)       # status

    def _cb_os(self, msg):
        self._os_prev_pos = self._os_pos
        self._os_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self._os_speed = math.hypot(
            msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self._os_count += 1
        if self._os_count == 1:
            self.get_logger().error(
                f'OS online: ({self._os_pos[0]:.1f}, {self._os_pos[1]:.1f})')

    def _cb_ts(self, name, msg):
        if name in self._ts_positions:
            self._ts_prev_positions[name] = self._ts_positions[name]
        self._ts_positions[name] = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self._ts_speeds[name] = math.hypot(
            msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self._ts_counts[name] += 1
        if self._ts_counts[name] == 1:
            self.get_logger().error(
                f'{name} online: ({self._ts_positions[name][0]:.1f}, '
                f'{self._ts_positions[name][1]:.1f})')

    def _tick(self):
        if self._os_count == 0:
            return
        os_x, os_y = self._os_pos

        for ts_name, (ts_x, ts_y) in self._ts_positions.items():
            if self._ts_counts.get(ts_name, 0) == 0:
                continue
            dist = math.hypot(ts_x - os_x, ts_y - os_y)
            if dist < self.STOP_DIST and ts_name not in self._collided_ts:
                self._collided_ts.add(ts_name)
                self._stop_active = True
                mx, my = (os_x + ts_x) / 2.0, (os_y + ts_y) / 2.0
                self.get_logger().error(
                    f'\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
                    f'!! COLLISION !! OS ↔ {ts_name}\n'
                    f'!! 碰撞点: ({mx:.1f}, {my:.1f})  CPA={dist:.2f}m\n'
                    f'!! OS 速度: {self._os_speed:.2f} m/s  '
                    f'TS 速度: {self._ts_speeds.get(ts_name, 0):.2f} m/s\n'
                    f'!! 已碰撞: {len(self._collided_ts)}/{self._total_ts} 艘船\n'
                    f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        # OS only stops after all target ships have collided
        if len(self._collided_ts) >= self._total_ts and not self._os_stopped:
            self._os_stopped = True
            self.get_logger().error(
                f'!! ALL {self._total_ts} TARGET SHIPS COLLIDED — STOPPING OS !!')

    def _stop_tick(self):
        if not self._stop_active:
            return
        zero = Float64(data=0.0)
        # Always stop collided target ships
        for name in list(self._collided_ts):
            p = self._thrust_pubs.get(name)
            if p:
                p[0].publish(zero)
                p[1].publish(zero)
        # Only stop OS after all TS have collided
        if self._os_stopped:
            p = self._thrust_pubs.get('wamv')
            if p:
                p[0].publish(zero)
                p[1].publish(zero)

    def _diag(self):
        if self._os_count == 0:
            return
        ox, oy = self._os_pos
        parts = [f'OS({ox:.1f},{oy:.1f})@{self._os_speed:.2f}m/s{"⛔" if self._os_stopped else ""}']
        for name, (tx, ty) in self._ts_positions.items():
            if self._ts_counts.get(name, 0) > 0:
                d = math.hypot(tx - ox, ty - oy)
                ts_spd = self._ts_speeds.get(name, 0.0)
                flag = '⛔' if name in self._collided_ts else ''

                # Predict time-to-meeting based on closing speed
                if d > 0.1:
                    dx, dy = tx - ox, ty - oy
                    rel_vx = (self._os_speed * (dx / d) if d > 0 else 0)
                    # Approximate: use OS heading toward TS
                    os_vx = self._os_speed * (dx / d) if abs(dx) > 0.01 else 0.0
                    os_vy = self._os_speed * (dy / d) if abs(dy) > 0.01 else 0.0
                    ts_v = ts_spd
                    # Simple closing speed estimate
                    closing = abs(os_vx) + abs(os_vy) + ts_v
                    ttc = d / max(closing, 0.01)
                    parts.append(f'{flag}{name}:{d:.1f}m spd={ts_spd:.2f} ttc~{ttc:.0f}s')
                else:
                    parts.append(f'{flag}{name}:{d:.1f}m spd={ts_spd:.2f}')

        if self._stop_active:
            parts.append(f'HIT:{len(self._collided_ts)}/{self._total_ts}')
        else:
            parts.append(f'NO HIT')

        self.get_logger().info(' | '.join(parts))


def main():
    rclpy.init()
    rclpy.spin(CollisionMarkerPublisher())


if __name__ == '__main__':
    main()
