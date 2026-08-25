#!/usr/bin/env python3
"""
Full Mission Launch: VRX Competition + Marine Environment Simulation.
========================================================================
- 启动 VRX 仿真 (WAM-V OS 在 sydney_regatta)
- 为每艘目标船桥接 Gazebo odometry → ROS2
  (推力桥接由 target_ship_spawner 内部处理)
- 启动所有 marine_env 节点
- 不启动 RViz2 (用户自行启动观察轨迹)

用法:
  python3 load_scenario.py 8 --new      # 加载场景
  ros2 launch marine_env full_mission.launch.py  # 启动仿真

  # 另一个终端:
  rviz2   # 手动添加话题: /{name}/trajectory (Path), /{name}/ground_truth (Odometry)
"""

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                             LogInfo, OpaqueFunction, ExecuteProcess, TimerAction)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import yaml
import os
import json
import math


def _load_ship_names(context):
    """从 target_ships.yaml 提取目标船名称."""
    pkg_share = FindPackageShare('marine_env').perform(context)
    config_path = os.path.join(pkg_share, 'config', 'target_ships.yaml')
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        ships_json = cfg['target_ship_spawner']['ros__parameters']['ships_json']
        if isinstance(ships_json, str):
            ships = json.loads(ships_json)
        else:
            ships = ships_json
        return [s['name'] for s in ships]
    except Exception as e:
        print(f'[WARN] Cannot load ship names: {e}')
        return []


def _load_os_waypoints(context):
    """从 own_ship.yaml 读取 OS 初始位姿, 生成 autopilot waypoints.

    own_ship.yaml 格式: [{model_name, model_type, position: {xyz, rpy}}]
    rpy[2] = yaw in ENU (0=east, π/2=north).
    生成: OS 从起点沿初始方向直行 400m.
    """
    pkg_share = FindPackageShare('marine_env').perform(context)
    own_path = os.path.join(pkg_share, 'config', 'own_ship.yaml')
    try:
        with open(own_path) as f:
            cfg = yaml.safe_load(f)
        ship = cfg[0]
        x, y, _ = ship['position']['xyz']
        _, _, yaw = ship['position']['rpy']
        # 从注释中取 speed (load_scenario.py 写入)
        with open(own_path) as f2:
            raw = f2.read()
        import re
        sp_match = re.search(r'target_speed:\s*([\d.]+)', raw)
        speed = float(sp_match.group(1)) if sp_match else 1.5
    except Exception as e:
        print(f'[WARN] Cannot load OS pose: {e}, using defaults')
        x, y, yaw, speed = 0.0, 0.0, math.pi / 2, 1.5

    # 前方 400m 的 waypoint (沿 yaw 方向)
    # 使用设计速度 (不放大): PI 控制器会自动补偿船体阻力达到目标速度
    # TS 现已统一使用相同推力模型 (target_ship_spawner.py feedforward=1200×speed)
    wx = x + 400.0 * math.cos(yaw)
    wy = y + 400.0 * math.sin(yaw)
    waypoints = [x, y, speed, wx, wy, speed]
    print(f'[INFO] OS waypoints: ({x:.1f},{y:.1f}) hdg={math.degrees(yaw):.0f}° → ({wx:.1f},{wy:.1f}) @ {speed:.1f} m/s (design)')
    return waypoints


def launch_nodes(context, *args, **kwargs):
    pkg_share = FindPackageShare('marine_env').perform(context)
    world = LaunchConfiguration('world').perform(context)
    ship_names = _load_ship_names(context)
    os_waypoints = _load_os_waypoints(context)

    LogInfo(msg=f'场景目标船 ({len(ship_names)}): {", ".join(ship_names) if ship_names else "(none)"}')

    # ═══════════════════════════════════════════════════════
    # /vrx/release 桥接 — 释放所有锁定的船
    # ═══════════════════════════════════════════════════════
    release_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        name='vrx_release_bridge', output='screen',
        arguments=['/vrx/release@std_msgs/msg/Bool]gz.msgs.Boolean'],
    )

    # ═══════════════════════════════════════════════════════
    # Gazebo ↔ ROS2 桥接
    # ═══════════════════════════════════════════════════════

    # 环境力: ROS2 → Gazebo
    wrench_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        name='wrench_bridge', output='screen',
        arguments=[f'/world/{world}/wrench@ros_gz_interfaces/msg/EntityWrench]gz.msgs.EntityWrench'],
    )

    # OS (wamv) odometry → 3路视觉里程计 (复用同一个Gazebo topic)
    vo_bridges = []
    for ros_topic, bridge_name in [
        ('/wamv/sensors/cameras/front_left/vo', 'vo_br_front_left'),
        ('/wamv/sensors/cameras/front_right/vo', 'vo_br_front_right'),
        ('/wamv/sensors/cameras/middle_right/vo', 'vo_br_mid_right'),
    ]:
        vo_bridges.append(Node(
            package='ros_gz_bridge', executable='parameter_bridge',
            name=bridge_name, output='screen',
            arguments=['/model/wamv/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry'],
            remappings=[('/model/wamv/odometry', ros_topic)],
        ))

    # OS odometry 真值
    gt_odom = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        name='gt_odom_bridge', output='screen',
        arguments=['/model/wamv/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry'],
    )

    # OS 推力桥 (ROS2 → Gazebo)
    # VRX 内部的 wamv.ros_gz_bridge 使用 namespaced topic，与 NMPC 不匹配。
    # 这里显式创建桥，确保 NMPC 推力能到达 Gazebo Thruster 插件。
    os_thrust_left = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        name='os_thrust_left_bridge', output='screen',
        arguments=['/wamv/thrusters/left/thrust@std_msgs/msg/Float64]gz.msgs.Double'],
    )
    os_thrust_right = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        name='os_thrust_right_bridge', output='screen',
        arguments=['/wamv/thrusters/right/thrust@std_msgs/msg/Float64]gz.msgs.Double'],
    )

    # ═══════════════════════════════════════════════════════
    # 目标船 Gazebo ↔ ROS2 桥接
    # ═══════════════════════════════════════════════════════
    ts_bridges = []
    for name in ship_names:
        # ── Gazebo → ROS2 ──
        # odometry (位置/速度)
        ts_bridges.append(Node(
            package='ros_gz_bridge', executable='parameter_bridge',
            name=f'ts_br_odom_{name}', output='screen',
            arguments=[f'/model/{name}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry'],
        ))
        # pose
        ts_bridges.append(Node(
            package='ros_gz_bridge', executable='parameter_bridge',
            name=f'ts_br_pose_{name}', output='screen',
            arguments=[f'/model/{name}/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose'],
        ))
        # GPS (NavSat)
        ts_bridges.append(Node(
            package='ros_gz_bridge', executable='parameter_bridge',
            name=f'ts_br_gps_{name}', output='screen',
            arguments=[f'/model/{name}/navsat@sensor_msgs/msg/NavSatFix[gz.msgs.NavSat'],
        ))
        # IMU
        ts_bridges.append(Node(
            package='ros_gz_bridge', executable='parameter_bridge',
            name=f'ts_br_imu_{name}', output='screen',
            arguments=[f'/model/{name}/imu@sensor_msgs/msg/Imu[gz.msgs.IMU'],
        ))
        # ── ROS2 → Gazebo (推力) ──
        # 使用 unscoped topic (与 VRX payload_bridges.thrust() 一致): {name}/thrusters/{side}/thrust
        ts_bridges.append(Node(
            package='ros_gz_bridge', executable='parameter_bridge',
            name=f'ts_br_thrust_l_{name}', output='screen',
            arguments=[f'/{name}/thrusters/left/thrust@std_msgs/msg/Float64]gz.msgs.Double'],
        ))
        ts_bridges.append(Node(
            package='ros_gz_bridge', executable='parameter_bridge',
            name=f'ts_br_thrust_r_{name}', output='screen',
            arguments=[f'/{name}/thrusters/right/thrust@std_msgs/msg/Float64]gz.msgs.Double'],
        ))

    # ═══════════════════════════════════════════════════════
    # TF: map → world
    # ═══════════════════════════════════════════════════════
    static_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='map_to_world_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'world'],
    )

    # ═══════════════════════════════════════════════════════
    # Marine Environment 节点
    # ═══════════════════════════════════════════════════════
    nodes = [
        static_tf,
        wrench_bridge,
        *vo_bridges,
        gt_odom,
        os_thrust_left,            # OS 推力 (左)
        os_thrust_right,           # OS 推力 (右)
        release_bridge,            # /vrx/release 桥接

        # ★ 发布 /vrx/release 解锁全部船（重试循环，确保 Gazebo 收到）
        # Gazebo VRX 加载 sydney_regatta + 生成 WAM-V 可能需要 30-60s。
        # 单次发布可能在 Gazebo 就绪前到达，导致 WAM-V 永久锁定。
        # 改为：35s 后每 5s 重试，共 12 次 (60s 窗口)。
        ExecuteProcess(
            cmd=['bash', '-c',
                 'sleep 35; '
                 'for i in $(seq 1 12); do '
                 '  echo "[vrx_release] Attempt $i/12: publishing /vrx/release"; '
                 '  ros2 topic pub --once /vrx/release std_msgs/msg/Bool "{data: true}" 2>&1; '
                 '  sleep 5; '
                 'done; '
                 'echo "[vrx_release] Release sequence complete"'],
            name='vrx_release_retry', output='screen',
        ),

        *ts_bridges,               # 目标船 odom/pose/GPS/IMU/thrust 桥接

        # 感知
        Node(package='marine_env', executable='sea_clutter', name='sea_clutter',
             output='screen',
             parameters=[PathJoinSubstitution([pkg_share, 'config', 'full_mission.yaml'])]),

        # AIS
        Node(package='marine_env', executable='ais_publisher', name='ais_publisher',
             output='screen',
             parameters=[PathJoinSubstitution([pkg_share, 'config', 'full_mission.yaml'])]),

        # 环境力 (风/浪/流)
        Node(package='marine_env', executable='environment_forces', name='environment_forces',
             output='screen',
             parameters=[PathJoinSubstitution([pkg_share, 'config', 'full_mission.yaml'])]),

        # ★ 目标船生成器 (xacro → URDF → spawn → thrust bridge → 轨迹发布)
        Node(package='marine_env', executable='target_ship_spawner',
             name='target_ship_spawner', output='screen',
             parameters=[PathJoinSubstitution([pkg_share, 'config', 'target_ships.yaml'])]),

        # ES-EKF（四元数 SO(3) 误差状态）
        Node(package='marine_env', executable='eskf_estimator', name='eskf_estimator',
             output='screen',
             parameters=[PathJoinSubstitution([pkg_share, 'config', 'full_mission.yaml'])]),

        # JPDA 多目标跟踪
        Node(package='marine_env', executable='jpda_tracker', name='jpda_tracker',
             output='screen',
             parameters=[PathJoinSubstitution([pkg_share, 'config', 'full_mission.yaml'])]),

        # EKF 数据记录
        Node(package='marine_env', executable='ekf_visualizer', name='ekf_visualizer',
             output='screen',
             parameters=[{'output_dir': '/home/xxy/vrx_ws/ekf_plots'}]),

        # ── NMPC 控制器 (COLREGS避碰) ──
        Node(package='marine_env', executable='nmpc_controller', name='nmpc_controller',
             output='screen',
             parameters=[PathJoinSubstitution([pkg_share, 'config', 'full_mission.yaml']),
                         {
                 'odom_topic': '/model/wamv/odometry',
                 'waypoints': os_waypoints,
                 'target_names': ship_names if ship_names else ['ts01'],
             }]),

        # ── COLREGS 裁判节点 (确定性引擎, 提供NMPC约束) ──
        # Uses GT odometry by default for Gazebo debugging (matches NMPC).
        # Switch os_odom_topic to /wamv/state/estimated for realistic EKF-based operation.
        Node(package='marine_env', executable='referee_node', name='referee_node',
             output='screen',
             parameters=[PathJoinSubstitution([pkg_share, 'config', 'full_mission.yaml']),
                         {
                 'backend': 'deterministic',
                 'target_names': ship_names if ship_names else ['ts01'],
                 'trigger_threshold': 0.3,
                 'min_trigger_interval': 2.0,
                 'cpa_safe_distance': 50.0,
                 'os_odom_topic': '/model/wamv/odometry',
             }]),

        # 多船数据订阅 (态势监控)
        Node(package='marine_env', executable='ship_data_subscriber',
             name='ship_data_subscriber', output='screen',
             parameters=[{
                 'target_names': ship_names if ship_names else ['ts01'],
                 'output_rate': 1.0,
                 'cpa_warning_dist': 50.0,
                 'tcpa_warning_time': 30.0,
             }]),

        # ★ Own-ship 轨迹发布器 (OS GT + EKF Path for RViz2)
        Node(package='marine_env', executable='trajectory_publisher',
             name='trajectory_publisher', output='screen'),

        # ★ 碰撞检测 + 紧急停船 + 标记发布 (50Hz 零推力覆盖)
        ExecuteProcess(
            cmd=['ros2', 'run', 'marine_env', 'collision_marker_publisher'],
            name='collision_marker_publisher', output='screen',
        ),
    ]

    # ── RViz2 (默认启动, 由 rviz:=False 禁用) ──
    rviz_arg = LaunchConfiguration('rviz').perform(context)
    if rviz_arg.lower() != 'false':
        rviz_config = os.path.join(pkg_share, 'config', 'colregs_trajectory.rviz')
        nodes.append(ExecuteProcess(
            cmd=['rviz2', '-d', rviz_config],
            name='rviz2', output='screen',
        ))

    rviz_config_path = os.path.join(pkg_share, 'config', 'colregs_trajectory.rviz')
    LogInfo(msg='=' * 60)
    LogInfo(msg='🌊 Full Mission Environment — COLREGS Collision Avoidance')
    LogInfo(msg=f'   场景目标船: {len(ship_names)} 艘 — {", ".join(ship_names) if ship_names else "(none)"}')
    LogInfo(msg='')
    LogInfo(msg='👁  查看避让效果:')
    LogInfo(msg='   Gazebo GUI: 按 T 键 → 俯瞰视角; 鼠标滚轮缩放')
    LogInfo(msg='   RViz2:     加载了轨迹可视化配置, 查看船只轨迹弧线')
    LogInfo(msg=f'   RViz2 手动: rviz2 -d {rviz_config_path}')
    LogInfo(msg='')
    LogInfo(msg='📊 实时监控 (新终端):')
    LogInfo(msg='   ros2 topic echo /colregs/decision        # 裁判分类输出')
    LogInfo(msg='   ros2 topic echo /colregs/nmpc_constraints  # NMPC 约束')
    LogInfo(msg='   ros2 topic echo /wamv/nmpc/status         # NMPC 求解状态')
    LogInfo(msg='')
    LogInfo(msg='🖥  RViz2 轨迹显示:')
    LogInfo(msg='   /wamv/trajectory_gt  — OS 真值 (亮蓝粗线)')
    LogInfo(msg='   /wamv/trajectory_ekf — OS EKF 估计 (黄色细线)')
    for name in (ship_names or []):
        LogInfo(msg=f'   /{name}/trajectory     — 目标船轨迹 (彩色粗线)')
    LogInfo(msg='')
    LogInfo(msg='⏳ 等待 Gazebo 加载 (~40s) + WAM-V 解锁 + EKF 收敛...')
    LogInfo(msg='=' * 60)
    return nodes


def _launch_vrx_and_nodes(context, *args, **kwargs):
    """读取 own_ship.yaml 并启动 VRX 竞赛 + marine_env 节点."""
    pkg_share = FindPackageShare('marine_env').perform(context)

    # 读取 own_ship.yaml 获取 OS 初始位姿
    own_ship_path = os.path.join(pkg_share, 'config', 'own_ship.yaml')
    config_file_arg = ''
    if os.path.exists(own_ship_path):
        config_file_arg = own_ship_path
        LogInfo(msg=f'OS 初始位姿从: {own_ship_path}')

    vrx_competition = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('vrx_gz'), '/launch/competition.launch.py'
        ]),
        launch_arguments={
            'world': LaunchConfiguration('world').perform(context),
            'headless': LaunchConfiguration('headless').perform(context),
            'ground_truth_enabled': 'true',
            'config_file': config_file_arg,
        }.items(),
    )

    # launch_nodes 返回 marine_env 节点列表
    nodes = launch_nodes(context)

    return [vrx_competition, *nodes]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='sydney_regatta'),
        DeclareLaunchArgument('headless', default_value='False'),
        DeclareLaunchArgument('rviz', default_value='True',
                              description='自动启动 RViz2 并加载轨迹可视化配置 (rviz:=False 禁用)'),
        OpaqueFunction(function=_launch_vrx_and_nodes),
    ])
