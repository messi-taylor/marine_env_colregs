#!/usr/bin/env python3
"""
Target Ship Spawner for COLREGS Scenarios — Xacro动态URDF + Gazebo物理驱动
============================================================================
每条目标船:
  1. xacro 生成专用 URDF: namespace=<船名>, locked=false, 传感器=全开
  2. 后处理: 删除 detachable joint (避免 boat→platform 死锁)
  3. ros_gz_sim create 生成模型
  4. 桥接 thruster topics (ROS2 → Gazebo)
  5. 桥接 odometry (Gazebo → ROS2, 由launch文件处理)
  6. 发布轨迹 (Path) 供 RViz2 可视化

为什么之前船不动:
  - detachable joint 尝试连接 {name}/base_link → platform/dummy_upper
  - platform 模型不存在 → joint 回退到 world 原点 → 船被粘住
  - 推力、浮力都在工作，但被机械锁定覆盖
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory
import subprocess
import json
import math
import os
import numpy as np
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict


def _resolve_wamv_xacro():
    """Resolve wamv_gazebo.urdf.xacro path."""
    try:
        share = get_package_share_directory('wamv_gazebo')
        path = os.path.join(share, 'urdf', 'wamv_gazebo.urdf.xacro')
        if os.path.exists(path):
            return path
    except Exception:
        pass
    path = '/home/xxy/vrx_ws/src/vrx/vrx_urdf/wamv_gazebo/urdf/wamv_gazebo.urdf.xacro'
    if os.path.exists(path):
        return path
    raise FileNotFoundError('Cannot locate wamv_gazebo.urdf.xacro')

WAMV_XACRO = _resolve_wamv_xacro()
URDF_OUTPUT_DIR = '/tmp/vrx_target_ships'

MODEL_MAP = {
    'roboboat01': 'vrx_gazebo',
    'roboboat02': 'vrx_gazebo',
    'wamv': 'vrx_gazebo',
}


class TargetShipSpawner(Node):
    def __init__(self):
        super().__init__('target_ship_spawner')

        default_ships = [
            {'name': 'target_ship_1', 'model': 'wamv',
             'x': -500.0, 'y': 200.0, 'z': 0.0, 'yaw': 0.0,
             'speed': 0.5, 'waypoints': []},
        ]

        self.declare_parameter('ships_json', json.dumps(default_ships))
        self.declare_parameter('world_name', 'sydney_regatta')
        self.declare_parameter('control_rate', 5.0)
        self.declare_parameter('thrust_rate', 20.0)
        self.declare_parameter('trajectory_rate', 2.0)
        self.declare_parameter('trajectory_length', 5000)

        ships_json = self.get_parameter('ships_json').value
        if isinstance(ships_json, str):
            ships = json.loads(ships_json)
        else:
            ships = ships_json

        self._ship_configs = {}
        self._odom = {}
        self._trajectory = defaultdict(list)
        self._waypoint_idx = {}

        # ── Target-target COLREGS avoidance state ──
        self._tgt_avoiding: dict = {}       # name → bool
        self._tgt_avoidance_target: dict = {}  # name → other_name
        self._tgt_intended_heading: dict = {}  # name → rad (original heading)
        self._tgt_avoidance_heading: dict = {} # name → rad (starboard deviation)
        self.TGT_CPA_THRESHOLD = 15.0          # m
        self.TGT_TCPA_LOOKAHEAD = 25.0         # s
        self.TGT_AVOIDANCE_STARBOARD = math.radians(35)  # rad
        self.TGT_RECOVERY_RATE = math.radians(3)         # rad/s
        self.TGT_RECOVERY_DISTANCE = 30.0      # m

        self._left_thrust_pubs = {}
        self._right_thrust_pubs = {}
        self._trajectory_pubs = {}
        self._gt_pubs = {}
        self._odom_subs = {}

        os.makedirs(URDF_OUTPUT_DIR, exist_ok=True)

        for ship_cfg in ships:
            self._spawn_ship(ship_cfg)

        # ── 延迟释放所有目标船 (detach from platform) ──
        # 多次发送 release 信号 (每3s一次, 共5次), 确保所有船插件加载完毕都能收到
        self._release_pub = self.create_publisher(Bool, '/vrx/release', 10)
        self._release_count = 0
        self._release_max = 0   # disabled — launch file handles unified release
        self._release_timer = self.create_timer(99.0, self._release_ships)

        # 定时器
        self._ctrl_timer = self.create_timer(
            1.0 / max(self.get_parameter('control_rate').value, 0.5),
            self._waypoint_control)
        self._thrust_timer = self.create_timer(
            1.0 / max(self.get_parameter('thrust_rate').value, 1.0),
            self._publish_thrust)
        self._traj_timer = self.create_timer(
            1.0 / max(self.get_parameter('trajectory_rate').value, 0.5),
            self._publish_trajectories)

        self.get_logger().info(
            f'TargetShipSpawner: {len(self._ship_configs)} WAM-V ships ready')

    # =====================================================================
    # URDF 生成 + 后处理
    # =====================================================================

    def _generate_urdf(self, name: str) -> str:
        """
        xacro → URDF → 删除 detachable joint → 写入磁盘.

        xacro 参数:
          namespace=<name>  → thruster topic 正确命名
          locked:=false     → 不锁定 (但 detachable joint 仍然在URDF中, 需手动删除)
          vrx_sensors_enabled:=true → 与 OS 相同的传感器套件
        """
        urdf_path = os.path.join(URDF_OUTPUT_DIR, f'{name}.urdf')

        # vrx_sensors_enabled 已包含 GPS/IMU/Lidar/Camera/Radar/Xband — 不再单独设置, 避免重复 sensor link
        xacro_cmd = [
            'xacro', WAMV_XACRO,
            f'namespace:={name}',
            'locked:=false',
            'ground_truth_enabled:=true',
            'vrx_sensors_enabled:=true',
            'thruster_config:=H',
        ]
        try:
            result = subprocess.run(
                xacro_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                self.get_logger().error(
                    f'xacro failed for {name}: {result.stderr[:300]}')
                return ''
            urdf = result.stdout
        except Exception as e:
            self.get_logger().error(f'xacro exception for {name}: {e}')
            return ''

        # ── 保留 detachable joint (与OS行为一致) ──
        # 船初始连接到 platform，稍后通过 /vrx/release 统一释放

        with open(urdf_path, 'w') as f:
            f.write(urdf)

        self.get_logger().info(
            f'  URDF: {urdf_path} ({len(urdf)} bytes, detachable joint removed)')
        return urdf_path

    def _strip_detachable_joint(self, urdf: str) -> str:
        """删除 detachable-joint-system plugin 块 (含父标签)."""
        # 匹配整个 <gazebo> 块，其中包含 DetachableJoint
        # 使用非贪婪匹配删除从 <gazebo> 到 </gazebo> 的完整块
        pattern = r'\s*<gazebo>\s*<plugin\s+filename="gz-sim-detachable-joint-system"[^>]*>.*?</plugin>\s*</gazebo>'
        cleaned = re.sub(pattern, '', urdf, flags=re.DOTALL)
        if len(cleaned) < len(urdf):
            self.get_logger().info(f'    ✓ detachable joint stripped')
        return cleaned

    # =====================================================================
    # SDF 后处理: 移除碰撞几何 (防止 CPA→0 时物理引擎推开偏离航线)
    # =====================================================================

    def _strip_collisions(self, sdf_path: str) -> bool:
        """从 SDF 中移除所有 <collision> 元素, 彻底杜绝船舶间碰撞接触力.

        Gazebo Garden + DART 对 <collision_filter> 支持不完整,
        直接移除碰撞几何是最可靠的方式.
        浮力/水动力由 VRX usv_gazebo_plugins 通过 link 属性独立计算,
        不依赖碰撞网格.

        Returns True on success.
        """
        try:
            tree = ET.parse(sdf_path)
            root = tree.getroot()

            # SDF namespace 可能影响 tag 匹配
            # ElementTree 在无 namespace 的 SDF 中直接用 tag 名匹配
            removed = 0
            for parent in root.iter():
                collisions_to_remove = []
                for child in parent:
                    tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    if tag == 'collision':
                        collisions_to_remove.append(child)
                for c in collisions_to_remove:
                    parent.remove(c)
                    removed += 1

            if removed > 0:
                tree.write(sdf_path, xml_declaration=True, encoding='UTF-8')
                self.get_logger().info(
                    f'    ✓ stripped {removed} collision elements '
                    f'(ships pass through each other, buoyancy preserved)')
            return True
        except Exception as e:
            self.get_logger().warn(f'    ⚠ collision strip failed: {e}')
            return False

    # =====================================================================
    # 生成
    # =====================================================================

    def _spawn_ship(self, cfg):
        name = cfg['name']
        model_type = cfg.get('model', 'wamv')
        x, y = cfg['x'], cfg['y']
        z = cfg.get('z', 0.0)
        yaw = cfg.get('yaw', 0.0)
        speed = cfg.get('speed', 0.5)

        # 1. URDF 生成 (xacro)
        if model_type == 'wamv':
            urdf_path = self._generate_urdf(name)
            if not urdf_path:
                return
        else:
            try:
                pkg_share = get_package_share_directory(MODEL_MAP.get(model_type, 'vrx_gazebo'))
            except Exception:
                self.get_logger().error(f'Package not found for {name}')
                return
            urdf_path = os.path.join(pkg_share, 'models', model_type, 'model.sdf')
        if not os.path.exists(urdf_path):
            self.get_logger().error(f'File not found: {urdf_path}')
            return

        # 2. URDF → SDF 转换 (与 VRX Model.generate() 一致, 确保 hydrodynamics 等插件正确加载)
        sdf_path = urdf_path.replace('.urdf', '.sdf')
        try:
            conv = subprocess.run(
                ['gz', 'sdf', '-p', urdf_path],
                capture_output=True, text=True, timeout=30)
            if conv.returncode == 0 and conv.stdout.strip():
                with open(sdf_path, 'w') as f:
                    f.write(conv.stdout)
                self.get_logger().info(f'  ✓ URDF→SDF: {sdf_path} ({len(conv.stdout)} bytes)')
                # ★ 移除碰撞几何: 杜绝船舶间物理碰撞推开 → 轨迹线在 CPA=0 处交汇
                self._strip_collisions(sdf_path)
                spawn_path = sdf_path
            else:
                self.get_logger().warn(
                    f'  ⚠ URDF→SDF failed ({conv.returncode}), falling back to URDF: {conv.stderr[:200]}')
                spawn_path = urdf_path
        except Exception as e:
            self.get_logger().warn(f'  ⚠ URDF→SDF exception: {e}, falling back to URDF')
            spawn_path = urdf_path

        world = self.get_parameter('world_name').value

        # 3. 先删除旧模型, 再 Spawn (防止旧位置残留)
        subprocess.run(
            ['ros2', 'run', 'ros_gz_sim', 'create',
             '-world', world, '-name', name, '-allow_renaming', 'false',
             '-x', str(x), '-y', str(y), '-z', str(z),
             '-R', '0', '-P', '0', '-Y', str(yaw),
             '--remove-old'],    # 如果存在则先删除
            capture_output=True, text=True, timeout=15)
        # Fallback: 如果 --remove-old 不支持, 手动删
        import os as _os
        _os.system(f'gz service -s /world/{world}/remove '
                   f'--reqtype gz.msgs.Entity --reptype gz.msgs.Boolean '
                   f'--req \'name: "{name}", type: MODEL\' 2>/dev/null')
        try:
            result = subprocess.run(
                ['ros2', 'run', 'ros_gz_sim', 'create',
                 '-world', world,
                 '-file', spawn_path,
                 '-name', name,
                 '-allow_renaming', 'false',
                 '-x', str(x), '-y', str(y), '-z', str(z),
                 '-R', '0', '-P', '0', '-Y', str(yaw)],
                capture_output=True, text=True, timeout=30)
            ok = result.returncode == 0 or 'already exists' in (result.stderr or '')
            self.get_logger().info(
                f'  {"✓" if ok else "!"} Spawned {name} at '
                f'({x:.1f},{y:.1f}) yaw={math.degrees(yaw):.0f}° speed={speed:.1f}m/s')
            if not ok:
                self.get_logger().error(f'  Spawn stderr: {result.stderr[:300]}')
        except subprocess.TimeoutExpired:
            self.get_logger().error(f'  ✗ {name} spawn timeout')
            return

        # 3. 初始化
        self._ship_configs[name] = cfg
        self._waypoint_idx[name] = 0
        self._odom[name] = {
            'x': x, 'y': y, 'yaw': yaw,
            'vx': 0.0, 'vy': 0.0, 'v_yaw': 0.0,
            'received': False,
        }
        self._trajectory[name] = [(x, y)]

        # 4. 话题
        self._setup_publishers(name)
        self._setup_odom_subscriber(name)

        # 5. 推力桥接已由 launch file 管理 (不再在此启动 subprocess)

    # =====================================================================
    # 话题
    # =====================================================================

    def _setup_publishers(self, name):
        """
        推力话题 (与 VRX payload_bridges.thrust() 格式一致 — unscoped).
        Gazebo thruster plugin listens on: {name}/thrusters/{side}/thrust
        """
        self._left_thrust_pubs[name] = self.create_publisher(
            Float64, f'/{name}/thrusters/left/thrust', 10)
        self._right_thrust_pubs[name] = self.create_publisher(
            Float64, f'/{name}/thrusters/right/thrust', 10)
        self._trajectory_pubs[name] = self.create_publisher(
            Path, f'/{name}/trajectory', 10)
        self._gt_pubs[name] = self.create_publisher(
            Odometry, f'/{name}/ground_truth', 10)

    def _setup_odom_subscriber(self, name):
        self._odom_subs[name] = self.create_subscription(
            Odometry, f'/model/{name}/odometry',
            lambda msg, n=name: self._odom_callback(n, msg), 10)

    # =====================================================================
    # 释放
    # =====================================================================

    def _release_ships(self):
        """多次发布 /vrx/release 以释放所有锁定的目标船 (应对插件延迟加载)."""
        msg = Bool(data=True)
        self._release_pub.publish(msg)
        self._release_count += 1
        self.get_logger().info(
            f'🔓 /vrx/release ({self._release_count}/{self._release_max}) → 释放所有目标船!')
        if self._release_count >= self._release_max:
            self._release_timer.cancel()
            self.get_logger().info('🔓 释放完成, 目标船应已解锁并开始移动')

    # =====================================================================
    # Odometry
    # =====================================================================

    def _odom_callback(self, name, msg: Odometry):
        o = self._odom[name]
        o['x'] = msg.pose.pose.position.x
        o['y'] = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        o['yaw'] = 2 * math.atan2(qz, qw)
        o['vx'] = msg.twist.twist.linear.x
        o['vy'] = msg.twist.twist.linear.y
        o['v_yaw'] = msg.twist.twist.angular.z
        o['received'] = True

        self._trajectory[name].append((o['x'], o['y']))
        if len(self._trajectory[name]) > self.get_parameter('trajectory_length').value:
            self._trajectory[name] = self._trajectory[name][-5000:]

        gt_msg = Odometry()
        gt_msg.header.stamp = self.get_clock().now().to_msg()
        gt_msg.header.frame_id = 'world'
        gt_msg.child_frame_id = f'{name}/base_link'
        gt_msg.pose.pose.position.x = o['x']
        gt_msg.pose.pose.position.y = o['y']
        gt_msg.pose.pose.orientation.z = qz
        gt_msg.pose.pose.orientation.w = qw
        gt_msg.twist.twist.linear.x = o['vx']
        gt_msg.twist.twist.linear.y = o['vy']
        gt_msg.twist.twist.angular.z = o['v_yaw']
        self._gt_pubs[name].publish(gt_msg)

    # =====================================================================
    # Target-Target COLREGS Avoidance
    # =====================================================================

    def _resolve_target_avoidance(self, dt: float):
        """Check all target-target pairs for collision risk, apply COLREGS starboard avoidance.

        Same algorithm as batch_runner._resolve_target_avoidance():
          - Head-on (Rule 14): both ships turn starboard 35°
          - Crossing (Rule 15): give-way ship turns starboard 35°
          - Overtaking (Rule 13): overtaking ship passes astern
          - One-pair-per-ship constraint (most urgent by min CPA)
          - Recovery: gradual return to intended heading when safe
        """
        names = [n for n in self._ship_configs if n in self._odom and self._odom[n]['received']]
        num_targets = len(names)
        if num_targets < 2:
            self._recover_targets(dt)
            return

        # ── Phase 1: Check all pairs ──
        risks = []  # (cpa, name_a, name_b, give_way_ship, stand_on_ship)
        for i in range(num_targets):
            for j in range(i + 1, num_targets):
                na, nb = names[i], names[j]
                oa, ob = self._odom[na], self._odom[nb]
                p_a = np.array([oa['x'], oa['y']])
                p_b = np.array([ob['x'], ob['y']])
                h_a, h_b = oa['yaw'], ob['yaw']
                sp_a = self._ship_configs[na].get('speed', 1.0)
                sp_b = self._ship_configs[nb].get('speed', 1.0)
                v_a = sp_a * np.array([math.cos(h_a), math.sin(h_a)])
                v_b = sp_b * np.array([math.cos(h_b), math.sin(h_b)])

                # Relative geometry
                dp_ab = p_b - p_a
                rel_vel = v_b - v_a
                dist = float(np.linalg.norm(dp_ab))
                rel_speed = float(np.linalg.norm(rel_vel))
                if rel_speed < 1e-6 or dist < 1e-6:
                    continue

                cross = dp_ab[0] * rel_vel[1] - dp_ab[1] * rel_vel[0]
                cpa = abs(cross) / rel_speed
                tcpa = -np.dot(dp_ab, rel_vel) / (rel_speed ** 2)
                if cpa >= self.TGT_CPA_THRESHOLD or tcpa <= 0 or tcpa > self.TGT_TCPA_LOOKAHEAD:
                    continue

                # Vector-based encounter classification
                d_a = v_a / (sp_a + 1e-10)
                d_b = v_b / (sp_b + 1e-10)
                course_dot = float(np.dot(d_a, d_b))
                dp_ab_unit = dp_ab / (dist + 1e-10)
                ahead_a = float(np.dot(d_a, dp_ab_unit))
                cross_a = float(d_a[0] * dp_ab_unit[1] - d_a[1] * dp_ab_unit[0])

                # Head-on
                is_headon = (course_dot < -0.866 and ahead_a > 0.966 and dist < 100.0)
                # Overtaking
                is_overtaking_ab = (ahead_a < -0.3827 and course_dot > 0.342
                                    and sp_a > sp_b * 1.05)
                is_overtaking_ba = (float(np.dot(d_b, -dp_ab_unit)) < -0.3827
                                    and course_dot > 0.342 and sp_b > sp_a * 1.05)

                if is_headon:
                    give_way = 'both'
                    stand_on = ''
                elif is_overtaking_ab:
                    give_way, stand_on = na, nb
                elif is_overtaking_ba:
                    give_way, stand_on = nb, na
                else:
                    # Crossing
                    cross_b = float(d_b[0] * (-dp_ab_unit)[1] - d_b[1] * (-dp_ab_unit)[0])
                    a_to_starboard_of_b = (cross_b < 0)
                    b_to_starboard_of_a = (cross_a < 0)
                    if b_to_starboard_of_a:
                        give_way, stand_on = na, nb
                    elif a_to_starboard_of_b:
                        give_way, stand_on = nb, na
                    else:
                        continue

                risks.append((cpa, na, nb, give_way, stand_on))

        if not risks:
            self._recover_targets(dt)
            return

        # ── Phase 2: Sort by urgency ──
        risks.sort(key=lambda r: r[0])

        # ── Phase 3: Apply avoidance (one-pair-per-ship) ──
        committed = set()
        for cpa, na, nb, give_way, stand_on in risks:
            if na in committed or nb in committed:
                continue
            ta, tb = self._tgt_avoiding.get(na, False), self._tgt_avoiding.get(nb, False)
            if ta and self._tgt_avoidance_target.get(na, '') not in (nb, ''):
                continue
            if tb and self._tgt_avoidance_target.get(nb, '') not in (na, ''):
                continue
            committed.update([na, nb])

            if give_way == 'both':
                for name in (na, nb):
                    o = self._odom[name]
                    if not self._tgt_avoiding.get(name, False):
                        self._tgt_intended_heading[name] = o['yaw']
                        self._tgt_avoidance_heading[name] = self._norm_angle(
                            o['yaw'] - self.TGT_AVOIDANCE_STARBOARD)
                        self._tgt_avoiding[name] = True
                        self._tgt_avoidance_target[name] = (nb if name == na else na)
            else:
                gw = give_way
                if not self._tgt_avoiding.get(gw, False):
                    o = self._odom[gw]
                    self._tgt_intended_heading[gw] = o['yaw']
                    self._tgt_avoidance_heading[gw] = self._norm_angle(
                        o['yaw'] - self.TGT_AVOIDANCE_STARBOARD)
                    self._tgt_avoiding[gw] = True
                    self._tgt_avoidance_target[gw] = stand_on

        self._recover_targets(dt)

    @staticmethod
    def _norm_angle(a): return (a + math.pi) % (2 * math.pi) - math.pi

    def _recover_targets(self, dt: float):
        """Gradually return avoiding targets to their intended heading."""
        for name in list(self._tgt_avoiding.keys()):
            if not self._tgt_avoiding.get(name, False):
                continue
            other = self._tgt_avoidance_target.get(name, '')
            if other and other in self._odom and self._odom[other]['received']:
                p_self = np.array([self._odom[name]['x'], self._odom[name]['y']])
                p_other = np.array([self._odom[other]['x'], self._odom[other]['y']])
                if float(np.linalg.norm(p_other - p_self)) < self.TGT_RECOVERY_DISTANCE:
                    continue  # still too close

            intended = self._tgt_intended_heading[name]
            current = self._odom[name]['yaw']
            dh = self._norm_angle(intended - current)
            rate = min(abs(dh) / max(dt, 0.01), self.TGT_RECOVERY_RATE)
            if abs(dh) < math.radians(1):
                self._tgt_avoiding[name] = False
                self._odom[name]['yaw'] = intended  # snap
            else:
                self._odom[name]['yaw'] = self._norm_angle(
                    current + math.copysign(rate * dt, dh))

    # =====================================================================
    # 推力
    # =====================================================================

    def _publish_thrust(self):
        """Constant-speed mode (no waypoints): PI speed + COLREGS avoidance turning.

        Applies differential thrust for ships actively avoiding another target ship.
        """
        self._resolve_target_avoidance(0.05)  # dt ≈ 0.05s at 20Hz

        for name, cfg in self._ship_configs.items():
            if cfg.get('waypoints', []):
                continue
            if name not in self._left_thrust_pubs:
                continue
            target_speed = cfg.get('speed', 0.5)
            odom = self._odom.get(name)
            if odom is None or not odom.get('received'):
                thrust = float(target_speed * 1200.0)
                self._left_thrust_pubs[name].publish(Float64(data=thrust))
                self._right_thrust_pubs[name].publish(Float64(data=thrust))
                continue

            # ── Speed PI control ──
            current_speed = math.hypot(odom.get('vx', 0.0), odom.get('vy', 0.0))
            speed_error = target_speed - current_speed
            if not hasattr(self, '_ts_speed_integral'):
                self._ts_speed_integral = {}
            if name not in self._ts_speed_integral:
                self._ts_speed_integral[name] = 0.0
            self._ts_speed_integral[name] += speed_error * 0.05
            self._ts_speed_integral[name] = max(-5.0, min(5.0, self._ts_speed_integral[name]))
            feedforward = target_speed * 1200.0
            base = feedforward + 800.0 * speed_error + 100.0 * self._ts_speed_integral[name]
            base = max(0.0, min(5000.0, float(base)))

            # ── Heading: differential thrust for avoidance or course-keeping ──
            if self._tgt_avoiding.get(name, False):
                desired = self._tgt_avoidance_heading[name]
            else:
                desired = odom['yaw']  # maintain current heading
            yaw_err = self._norm_angle(desired - odom['yaw'])
            # diff: positive → more left thrust → starboard turn (CW in ENU)
            diff = 600.0 * yaw_err - 120.0 * odom['v_yaw']
            diff = max(-base * 0.7, min(base * 0.7, diff))
            self._left_thrust_pubs[name].publish(Float64(data=float(max(0.0, base + diff))))
            self._right_thrust_pubs[name].publish(Float64(data=float(max(0.0, base - diff))))

    # =====================================================================
    # Waypoint
    # =====================================================================

    def _waypoint_control(self):
        self._resolve_target_avoidance(0.05)  # dt ≈ 0.05s at 20Hz
        for name, cfg in self._ship_configs.items():
            waypoints = cfg.get('waypoints', [])
            if not waypoints:
                continue
            odom = self._odom.get(name)
            if odom is None or not odom.get('received'):
                continue
            idx = self._waypoint_idx.get(name, 0)
            if idx >= len(waypoints):
                continue

            wp = waypoints[idx]
            dx, dy = wp[0] - odom['x'], wp[1] - odom['y']
            dist = math.hypot(dx, dy)
            if dist < 3.0:
                self._waypoint_idx[name] = (idx + 1) % len(waypoints)
                self.get_logger().info(f'{name}: WP{idx}✓→{self._waypoint_idx[name]}')
                continue

            desired = math.atan2(dy, dx)
            yaw_err = (desired - odom['yaw'] + math.pi) % (2*math.pi) - math.pi
            target_speed = cfg.get('speed', 0.5)

            # ── Speed PI control (matching OS autopilot feedforward model) ──
            current_speed = math.hypot(odom.get('vx', 0.0), odom.get('vy', 0.0))
            speed_error = target_speed - current_speed

            if not hasattr(self, '_ts_speed_integral'):
                self._ts_speed_integral = {}
            if name not in self._ts_speed_integral:
                self._ts_speed_integral[name] = 0.0
            self._ts_speed_integral[name] += speed_error * 0.05
            self._ts_speed_integral[name] = max(-5.0, min(5.0, self._ts_speed_integral[name]))

            feedforward = target_speed * 1200.0  # match OS feedforward coefficient
            base = feedforward + 800.0 * speed_error + 100.0 * self._ts_speed_integral[name]
            base = max(0.0, min(5000.0, float(base)))

            # left engine @ Y=+1.03 body → 左桨推力大=右转(yaw↓). 需 right>left 才能CCW.
            diff = -600.0 * yaw_err - 120.0 * odom['v_yaw']
            diff = max(-base*0.7, min(base*0.7, diff))

            if name in self._left_thrust_pubs:
                self._left_thrust_pubs[name].publish(
                    Float64(data=float(max(0.0, base + diff))))
                self._right_thrust_pubs[name].publish(
                    Float64(data=float(max(0.0, base - diff))))

    # =====================================================================
    # 轨迹
    # =====================================================================

    def _publish_trajectories(self):
        now = self.get_clock().now().to_msg()
        for name, pts in self._trajectory.items():
            if name not in self._trajectory_pubs or len(pts) < 2:
                continue
            msg = Path()
            msg.header.stamp = now
            msg.header.frame_id = 'world'
            for (px, py) in pts:
                p = PoseStamped()
                p.header.stamp = now
                p.header.frame_id = 'world'
                p.pose.position.x = px
                p.pose.position.y = py
                p.pose.orientation.w = 1.0
                msg.poses.append(p)
            self._trajectory_pubs[name].publish(msg)

    def destroy_node(self):
        super().destroy_node()


def main():
    rclpy.init()
    node = TargetShipSpawner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
