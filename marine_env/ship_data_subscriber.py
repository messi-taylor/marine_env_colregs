#!/usr/bin/env python3
"""
COLREGS Scenario Ship Data Subscriber — 多船数据订阅与监控节点
=============================================================================
功能:
  1. 订阅本船 (wamv) 所有传感器与状态话题
  2. 动态发现并订阅所有目标船 (tsXX) 的位姿话题
  3. 实时计算各船的 CPA/TCPA/相对距离/相对方位
  4. 格式化输出各船完整状态数据
  5. 支持任意场景 (1-20) 的动态目标船检测

用法:
  ros2 run marine_env ship_data_subscriber
  ros2 run marine_env ship_data_subscriber --ros-args -p target_names:=["ts01","ts02a","ts02b"]

订阅的话题 (自动建立):
  ┌─ 本船 (Own Ship) ─────────────────────────────────────────────────┐
  │ /wamv/state/estimated          Odometry    EKF估计位姿/速度          │
  │ /wamv/sensors/gps/gps/fix      NavSatFix   GPS定位                   │
  │ /wamv/sensors/imu/imu/data     Imu         IMU数据                   │
  │ /wamv/sensors/ais/nmea         String      AIS报文                   │
  │ /wamv/sensors/radars/xband/    PointCloud2 雷达点云(含杂波)           │
  │   points_cluttered                                                  │
  │ /wamv/tracking/targets         MarkerArray JPDA跟踪目标               │
  │ /wamv/thrusters/left/thrust    Float64     左桨推力                   │
  │ /wamv/thrusters/right/thrust   Float64     右桨推力                   │
  ├─ 目标船 (Target Ships) ────────────────────────────────────────────┤
  │ /{name}/pose                   PoseStamped 目标船位姿                │
  │ /model/{name}/odometry         Odometry    真值里程计                │
  └────────────────────────────────────────────────────────────────────┘

输出示例:
  ================ 本船状态 (OS) ================
  Position:  (12.34, 56.78) m   Yaw: 15.2°
  Velocity:  surge=1.48 sway=0.03 m/s   yaw_rate=0.012 rad/s
  GPS:       lat=-33.7237 lon=150.6799
  IMU:       acc=(0.01, 0.02, -9.80)  gyro=(0.001, 0.001, 0.012)
  Thrust:    L=312.5 N  R=308.2 N
  AIS:       收到 2 条目标船报文
  Radar:     47 个点云目标
  Tracking:  3 个JPDA跟踪目标

  ================ 目标船数据 ================
  ts01:  pos=(3.12, 28.45)m  yaw=184.8°  speed=1.18m/s
         rel_dist=28.5m  rel_bearing=5.3°(右舷)  CPA=2.8m  TCPA=11.2s
         COLREGS判定: 对遇 (Rule 14) — 两船各右转
  ts02a: pos=(9.87, 15.23)m  yaw=269.2°  speed=0.82m/s
         rel_dist=18.1m  rel_bearing=32.1°(右舷)  CPA=0.5m  TCPA=10.8s
         COLREGS判定: 交叉相遇 (Rule 15) — OS为让路船
  =============================================
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, NavSatFix, PointCloud2
from std_msgs.msg import Float64, String
from visualization_msgs.msg import MarkerArray
import numpy as np
import math
import json
from collections import defaultdict


class ShipDataSubscriber(Node):
    """多船数据订阅与态势感知节点."""

    def __init__(self):
        super().__init__('ship_data_subscriber')

        # ── 参数 ──
        self.declare_parameter('target_names', ['ts01', 'ts02a', 'ts02b'])
        self.declare_parameter('output_rate', 1.0)   # Hz — 打印频率
        self.declare_parameter('cpa_warning_dist', 50.0)  # CPA预警距离
        self.declare_parameter('tcpa_warning_time', 30.0)  # TCPA预警时间

        # ── 本船状态存储 ──
        self.os = {
            'position': np.zeros(2),       # [x, y] (ENU)
            'yaw': 0.0,                    # rad
            'velocity_body': np.zeros(2),  # [surge, sway]
            'yaw_rate': 0.0,
            'gps_lat': None,
            'gps_lon': None,
            'imu_acc': np.zeros(3),
            'imu_gyro': np.zeros(3),
            'left_thrust': 0.0,
            'right_thrust': 0.0,
            'ais_messages': [],
            'radar_points': 0,
            'tracked_targets': 0,
        }

        # ── 目标船状态存储 ──
        self.targets = defaultdict(lambda: {
            'position': np.zeros(2),
            'yaw': 0.0,
            'velocity': np.zeros(2),
            'last_update': 0.0,
            'active': False,
        })

        # ── 本船话题订阅 ──
        self._subscribe_own_ship()

        # ── 目标船话题订阅 ──
        target_names = self.get_parameter('target_names').value
        for name in target_names:
            self._subscribe_target_ship(name)

        # ── 定时输出 ──
        rate = self.get_parameter('output_rate').value
        self._output_timer = self.create_timer(
            1.0 / max(rate, 0.1), self._output_cycle)

        # ── 动态目标船发现定时器 ──
        self._discovery_timer = self.create_timer(
            3.0, self._discover_target_ships)

        self.get_logger().info(
            f'🚢 ShipDataSubscriber 启动: 监控 {len(target_names)} 艘目标船')

    # =====================================================================
    # 话题订阅 — 本船
    # =====================================================================

    def _subscribe_own_ship(self):
        """订阅本船所有传感器和状态话题."""
        # EKF 状态估计
        self.create_subscription(
            Odometry, '/wamv/state/estimated', self._os_odom_cb, 10)
        # GPS
        self.create_subscription(
            NavSatFix, '/wamv/sensors/gps/gps/fix', self._os_gps_cb, 10)
        # IMU
        self.create_subscription(
            Imu, '/wamv/sensors/imu/imu/data', self._os_imu_cb, 10)
        # AIS NMEA
        self.create_subscription(
            String, '/wamv/sensors/ais/nmea', self._os_ais_cb, 10)
        # Radar (X-band cluttered)
        self.create_subscription(
            PointCloud2, '/wamv/sensors/radars/xband/points_cluttered',
            self._os_radar_cb, 10)
        # JPDA tracking markers
        self.create_subscription(
            MarkerArray, '/wamv/tracking/targets', self._os_tracking_cb, 10)
        # Thrusters
        self.create_subscription(
            Float64, '/wamv/thrusters/left/thrust', self._os_left_thrust_cb, 10)
        self.create_subscription(
            Float64, '/wamv/thrusters/right/thrust', self._os_right_thrust_cb, 10)

    def _subscribe_target_ship(self, name: str):
        """订阅一艘目标船的话题."""
        if name in self.targets and self.targets[name]['active']:
            return
        self.create_subscription(
            PoseStamped, f'/{name}/pose',
            lambda msg, n=name: self._ts_pose_cb(n, msg), 10)
        self.create_subscription(
            Odometry, f'/model/{name}/odometry',
            lambda msg, n=name: self._ts_odom_cb(n, msg), 10)
        self.targets[name]['active'] = True
        self.get_logger().info(f'  → 订阅目标船: {name}')

    def _discover_target_ships(self):
        """动态发现新的目标船话题 (通过检查已知命名模式)."""
        # 尝试发现 ts01-ts20 以及 tsXXa-tsXXf
        candidates = []
        for i in range(1, 21):
            candidates.append(f'ts{i:02d}')
            for suffix in ['a', 'b', 'c', 'd', 'e', 'f']:
                candidates.append(f'ts{i:02d}{suffix}')

        for name in candidates:
            if name not in self.targets or not self.targets[name]['active']:
                # 尝试订阅 — 如果话题不存在, ROS2会静默忽略
                self._subscribe_target_ship(name)

    # =====================================================================
    # 本船回调
    # =====================================================================

    def _os_odom_cb(self, msg: Odometry):
        self.os['position'][0] = msg.pose.pose.position.x
        self.os['position'][1] = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.os['yaw'] = 2 * math.atan2(qz, qw)
        self.os['velocity_body'][0] = msg.twist.twist.linear.x
        self.os['velocity_body'][1] = msg.twist.twist.linear.y
        self.os['yaw_rate'] = msg.twist.twist.angular.z

    def _os_gps_cb(self, msg: NavSatFix):
        self.os['gps_lat'] = msg.latitude
        self.os['gps_lon'] = msg.longitude

    def _os_imu_cb(self, msg: Imu):
        self.os['imu_acc'] = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z])
        self.os['imu_gyro'] = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z])

    def _os_ais_cb(self, msg: String):
        # 解析 AIS NMEA 报文
        self.os['ais_messages'].append(msg.data)
        if len(self.os['ais_messages']) > 20:
            self.os['ais_messages'] = self.os['ais_messages'][-10:]

    def _os_radar_cb(self, msg: PointCloud2):
        self.os['radar_points'] = msg.width * msg.height

    def _os_tracking_cb(self, msg: MarkerArray):
        self.os['tracked_targets'] = len(msg.markers)

    def _os_left_thrust_cb(self, msg: Float64):
        self.os['left_thrust'] = msg.data

    def _os_right_thrust_cb(self, msg: Float64):
        self.os['right_thrust'] = msg.data

    # =====================================================================
    # 目标船回调
    # =====================================================================

    def _ts_pose_cb(self, name: str, msg: PoseStamped):
        t = self.targets[name]
        t['position'][0] = msg.pose.position.x
        t['position'][1] = msg.pose.position.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        t['yaw'] = 2 * math.atan2(qz, qw)
        t['last_update'] = self.get_clock().now().nanoseconds * 1e-9

    def _ts_odom_cb(self, name: str, msg: Odometry):
        t = self.targets[name]
        # 从 Odometry 中同时获取位置和速度 (以防 Pose bridge 未工作时仍能检测目标)
        t['position'][0] = msg.pose.pose.position.x
        t['position'][1] = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        t['yaw'] = 2 * math.atan2(qz, qw)
        t['velocity'][0] = msg.twist.twist.linear.x
        t['velocity'][1] = msg.twist.twist.linear.y
        t['last_update'] = self.get_clock().now().nanoseconds * 1e-9

    # =====================================================================
    # 态势计算
    # =====================================================================

    def _compute_relative(self, ts_pos, ts_vel):
        """计算相对距离、方位、CPA、TCPA."""
        os_pos = self.os['position']
        os_vel_body = self.os['velocity_body']
        os_yaw = self.os['yaw']

        # 本船速度 (世界坐标系)
        c, s = math.cos(os_yaw), math.sin(os_yaw)
        os_vel_world = np.array([
            c * os_vel_body[0] - s * os_vel_body[1],
            s * os_vel_body[0] + c * os_vel_body[1],
        ])

        # 相对位置
        rel_pos = ts_pos - os_pos
        dist = np.linalg.norm(rel_pos)

        # 相对方位角 (从本船船首方向)
        bearing = math.atan2(rel_pos[0], rel_pos[1]) - os_yaw
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        bearing_deg = math.degrees(bearing)

        # 相对速度
        rel_vel = ts_vel - os_vel_world
        rel_speed = np.linalg.norm(rel_vel)

        # CPA (最近会遇距离)
        if rel_speed < 1e-6:
            cpa = dist
            tcpa = float('inf')
        else:
            # CPA = |rel_pos × rel_vel| / |rel_vel|
            cpa = abs(rel_pos[0] * rel_vel[1] - rel_pos[1] * rel_vel[0]) / rel_speed
            # TCPA = -(rel_pos · rel_vel) / |rel_vel|²
            tcpa = -np.dot(rel_pos, rel_vel) / (rel_speed ** 2)

        return dist, bearing_deg, cpa, tcpa

    def _classify_encounter(self, bearing_deg, ts_yaw, os_yaw):
        """基于相对方位和航向差判定COLREGS会遇类型."""
        # 航向差 (TS相对于OS)
        heading_diff = (ts_yaw - os_yaw + math.pi) % (2 * math.pi) - math.pi

        # 对遇: 相对方位 < 5° 且航向差 ≈ 180°
        if abs(bearing_deg) < 5.0 and abs(abs(heading_diff) - math.pi) < 0.1:
            return '对遇 (Rule 14) — 两船各右转'

        # 右交叉: 相对方位在右舷 (0°~112.5°)
        if 0.0 <= bearing_deg <= 112.5:
            return '交叉相遇 (Rule 15) — OS为让路船, 应右转从TS船尾通过'

        # 左交叉: 相对方位在左舷 (0°~-112.5°)
        if -112.5 <= bearing_deg < 0.0:
            return '交叉相遇 (Rule 15) — OS为直航船, 保向保速'

        # 追越: 相对方位 > 112.5° 或 < -112.5° (后方)
        if abs(heading_diff) < 0.4 and abs(bearing_deg) < 20.0:
            if self.os['velocity_body'][0] > 1.1 * np.linalg.norm(self.targets.get('_tmp_vel', [0, 0])):
                return '追越 (Rule 13) — OS为追越船, 让路'
            else:
                return '追越 (Rule 13) — OS为被追越船, 直航'

        # 小角度交叉: 介于对遇和交叉之间
        if abs(bearing_deg) < 22.5:
            return '小角度交叉 (Rule 14/15边界) — 谨慎判定'

        return '常规交叉 — 按Rule 15处理'

    # =====================================================================
    # 定时输出
    # =====================================================================

    def _output_cycle(self):
        """定时打印所有船只的完整状态数据."""
        now = self.get_clock().now().nanoseconds * 1e-9
        print("\n" + "=" * 70)
        print(f"🚢 COLREGS 场景态势感知 — t={now:.1f}s")
        print("=" * 70)

        # ── 本船状态 ──
        os = self.os
        print(f"\n┌─ 本船 (Own Ship: wamv) ──────────────────────────────────")
        print(f"│ 位置:     ({os['position'][0]:.2f}, {os['position'][1]:.2f}) m  "
              f"Yaw: {math.degrees(os['yaw']):.1f}°")
        print(f"│ 速度(体): surge={os['velocity_body'][0]:.3f} m/s  "
              f"sway={os['velocity_body'][1]:.3f} m/s  "
              f"yaw_rate={os['yaw_rate']:.4f} rad/s")
        if os['gps_lat'] is not None:
            print(f"│ GPS:      lat={os['gps_lat']:.6f}  lon={os['gps_lon']:.6f}")
        print(f"│ IMU:      acc=({os['imu_acc'][0]:.2f}, {os['imu_acc'][1]:.2f}, "
              f"{os['imu_acc'][2]:.2f}) m/s²")
        print(f"│            gyro=({os['imu_gyro'][0]:.4f}, {os['imu_gyro'][1]:.4f}, "
              f"{os['imu_gyro'][2]:.4f}) rad/s")
        print(f"│ 推进器:   L={os['left_thrust']:.1f} N  R={os['right_thrust']:.1f} N")
        print(f"│ 雷达:     {os['radar_points']} 个点云目标")
        print(f"│ AIS:      {len(os['ais_messages'])} 条报文")
        print(f"│ 跟踪:     {os['tracked_targets']} 个JPDA目标")
        print(f"└──────────────────────────────────────────────────────────")

        # ── 目标船状态 ──
        active_targets = {k: v for k, v in self.targets.items()
                          if v['active'] and v['last_update'] > now - 10.0}
        if active_targets:
            print(f"\n┌─ 目标船 ({len(active_targets)} 艘) ─────────────────────────────────")
            for name, t in sorted(active_targets.items()):
                dist, bearing, cpa, tcpa = self._compute_relative(
                    t['position'], t['velocity'])
                speed = np.linalg.norm(t['velocity'])
                encounter = self._classify_encounter(bearing, t['yaw'], os['yaw'])

                # 告警标记
                warn = "⚠️ " if cpa < 20.0 and 0 < tcpa < 120.0 else "  "
                critical = "🚨" if cpa < 5.0 and 0 < tcpa < 60.0 else "  "

                side = "右舷" if bearing >= 0 else "左舷"
                print(f"│ {warn}{critical} {name}: "
                      f"pos=({t['position'][0]:.1f}, {t['position'][1]:.1f})m  "
                      f"yaw={math.degrees(t['yaw']):.0f}°  "
                      f"speed={speed:.2f}m/s")
                print(f"│      rel_dist={dist:.1f}m  "
                      f"rel_bearing={bearing:.1f}°({side})  "
                      f"CPA={cpa:.1f}m  TCPA={tcpa:.1f}s")
                print(f"│      → {encounter}")
            print(f"└──────────────────────────────────────────────────────────")
        else:
            print(f"\n  (无活跃目标船)")

        # ── CPA 告警 ──
        warnings = []
        for name, t in active_targets.items():
            dist, bearing, cpa, tcpa = self._compute_relative(
                t['position'], t['velocity'])
            if cpa < self.get_parameter('cpa_warning_dist').value and \
               0 < tcpa < self.get_parameter('tcpa_warning_time').value:
                warnings.append(
                    f"⚠️  {name}: CPA={cpa:.1f}m TCPA={tcpa:.0f}s — 碰撞危险!")

        if warnings:
            print("\n┌─ ⚠️  碰撞危险告警 ─────────────────────────────────────")
            for w in warnings:
                print(f"│ {w}")
            print("└──────────────────────────────────────────────────────────")

        print("\n" + "-" * 70)


def main():
    rclpy.init()
    node = ShipDataSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
