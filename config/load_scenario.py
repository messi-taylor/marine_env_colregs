#!/usr/bin/env python3
"""
COLREGS 20 Scenarios Loader — 从 colregs_20_scenarios.yaml 加载场景,
生成 target_ships.yaml 和 full_mission.yaml 配置文件供仿真直接使用。

用法:
  python3 load_scenario.py 6              # 加载场景 6, 写入 target_ships.yaml
  python3 load_scenario.py 6 --dry-run    # 仅打印, 不写入
  python3 load_scenario.py list           # 列出所有场景
  python3 load_scenario.py 20 --output /tmp/test_scenario/  # 输出到指定目录
"""

import yaml
import json
import sys
import os
import math
import argparse
from pathlib import Path

SCENARIOS_FILE = Path(__file__).parent / 'colregs_20_scenarios.yaml'
SCENARIOS_FILE_NEW = Path(__file__).parent / 'colregs_20_new_scenarios.yaml'
CONFIG_DIR = Path(__file__).parent


# =============================================================================
# 坐标系转换: 罗经方位 → ENU
# YAML 使用罗经方位: 0=北(N), π/2=东(E), 顺时针
# ROS2/Gazebo ENU:   0=东(E), π/2=北(N), 逆时针
# 转换: ENU_yaw = π/2 - compass_yaw, 归一化到 [-π, π]
# =============================================================================
def compass_to_enu(yaw_compass: float) -> float:
    """罗经方位角 → ENU yaw (归一化到 [-π, π])."""
    ros_yaw = math.pi / 2.0 - yaw_compass
    return math.atan2(math.sin(ros_yaw), math.cos(ros_yaw))


def load_scenarios(new_scenarios: bool = False):
    filepath = SCENARIOS_FILE_NEW if new_scenarios else SCENARIOS_FILE
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def list_scenarios(scenarios):
    print(f"{'ID':<6} {'名称':<45} {'类型':<14} {'船数':<6} {'能见度':<12} {'难度':<10}")
    print("=" * 100)
    for key, s in scenarios.items():
        if not key.startswith('scenario_'):
            continue
        sid = key.split('_')[1]
        ts_count = len(s.get('target_ships', []))
        print(f"S{sid:<5} {s['description'].strip().split(chr(10))[0]:<45} "
              f"{s['encounter_type']:<14} {ts_count:<6} "
              f"{s.get('visibility','clear'):<12} {s['difficulty']:<10}")


def generate_target_ships_yaml(scenario):
    """将场景的 target_ships 转换为 target_ships.yaml 内容。

    YAML 中 yaw 使用罗经方位角 (0=北, π/2=东, 顺时针).
    ROS2/Gazebo 使用 ENU 坐标系 (0=东, π/2=北, 逆时针).
    转换公式: ros_yaw = π/2 - compass_yaw, 归一化到 [-π, π].
    """
    ships = []
    for ts in scenario['target_ships']:
        # 罗经方位角 → ENU yaw
        yaml_yaw = ts['yaw']
        ros_yaw = (math.pi / 2.0) - yaml_yaw
        ros_yaw = math.atan2(math.sin(ros_yaw), math.cos(ros_yaw))  # 归一化 [-π, π]

        ship = {
            'name': ts['name'],
            'model': ts.get('model', 'wamv'),
            'x': ts['x'],
            'y': ts['y'],
            'z': ts.get('z', 0.0),
            'yaw': ros_yaw,
            'speed': ts['speed'],
            'waypoints': ts.get('waypoints', []),
        }
        ships.append(ship)

    ships_json = json.dumps(ships)

    yaml_content = f"""# Target ship spawner configuration — 场景自动生成
target_ship_spawner:
  ros__parameters:
    ships_json: '{ships_json}'
    world_name: "sydney_regatta"
    pose_rate: 5.0
    control_rate: 2.0
"""
    return yaml_content


def generate_own_ship_yaml(scenario):
    """根据 scenario['own_ship'] 生成 OS 配置文件 (供 VRX competition launch 使用).

    YAML 中 own_ship.yaw 是罗经角 (0=北, 顺时针). 转换为 ENU.
    """
    os_cfg = scenario.get('own_ship', {})
    os_x = os_cfg.get('x', 0.0)
    os_y = os_cfg.get('y', 0.0)
    os_yaw_compass = os_cfg.get('yaw', 0.0)
    os_yaw_enu = compass_to_enu(os_yaw_compass)
    os_speed = os_cfg.get('speed', 1.5)

    config = [{
        'model_name': 'wamv',
        'model_type': 'wam-v',
        'position': {
            'xyz': [os_x, os_y, 0.0],
            'rpy': [0.0, 0.0, os_yaw_enu],
        }
    }]

    ships_yaml = yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)

    content = f"""# Own Ship (本船) 初始状态 — 场景自动生成
# 由 VRX competition.launch.py 读取, 设置 OS 初始位姿
{ships_yaml}
# autopilot 参数 (由 full_mission.yaml 读取)
# target_speed: {os_speed}
# target_heading: {os_yaw_compass} (罗经) / {os_yaw_enu:.4f} (ENU)
"""
    return content


def generate_full_mission_yaml(scenario):
    """根据场景环境参数生成 full_mission.yaml。"""
    env = scenario.get('environment', {})
    vis = scenario.get('visibility', 'clear')

    # 能见度不良时调整雷达参数
    sea_clutter = {
        'input_topic': '/wamv/sensors/lidars/xband_radar_sensor/points',
        'output_topic': '/wamv/sensors/radars/xband/points_cluttered',
        'clutter_shape': 1.5,
        'clutter_scale': 0.03 if vis == 'clear' else 0.06,
        'false_alarm_rate': 0.02 if vis == 'clear' else 0.05,
        'false_alarm_range': 500.0,
        'enable_clutter': True,
        'enable_false_alarms': True,
    }

    target_ships_names = [ts['name'] for ts in scenario['target_ships']]

    # ── own_ship 参数 ──
    os_cfg = scenario.get('own_ship', {})
    os_speed = os_cfg.get('speed', 1.5)
    os_yaw_compass = os_cfg.get('yaw', 0.0)
    os_yaw_enu = compass_to_enu(os_yaw_compass)

    content = f"""# Full mission environment parameters — 场景自动生成
# ── 本船 (Own Ship) 自动舵参数 ──
own_ship_autopilot:
  ros__parameters:
    target_speed: {os_speed}
    target_heading: {os_yaw_enu}
    enable_heading_hold: true
sea_clutter:
  ros__parameters:
    input_topic: "{sea_clutter['input_topic']}"
    output_topic: "{sea_clutter['output_topic']}"
    clutter_shape: {sea_clutter['clutter_shape']}
    clutter_scale: {sea_clutter['clutter_scale']}
    false_alarm_rate: {sea_clutter['false_alarm_rate']}
    false_alarm_range: {sea_clutter['false_alarm_range']}
    enable_clutter: {str(sea_clutter['enable_clutter']).lower()}
    enable_false_alarms: {str(sea_clutter['enable_false_alarms']).lower()}

ais_publisher:
  ros__parameters:
    publish_rate: 0.1
    packet_loss_rate: 0.05
    max_delay_ms: 500
    target_ships: {json.dumps(target_ships_names)}

environment_forces:
  ros__parameters:
    wind_speed: {env.get('wind_speed', 5.0)}
    wind_direction: {env.get('wind_direction', 240.0)}
    gust_amplitude: 3.0
    significant_wave_height: {env.get('significant_wave_height', 0.5)}
    peak_period: {env.get('peak_period', 4.0)}
    current_speed: {env.get('current_speed', 0.3)}
    current_direction: {env.get('current_direction', 90.0)}
    world_name: "sydney_regatta"
    model_name: "wamv"
    publish_rate: 10.0

eskf_estimator:
  ros__parameters:
    imu_topic: "/wamv/sensors/imu/imu/data"
    gps_topic: "/wamv/sensors/gps/gps/fix"
    odom_topic: "/wamv/state/estimated"
    world_frame: "map"
    body_frame: "wamv/eskf_base_link"
    gps_origin_lat: -33.724223
    gps_origin_lon: 150.679736
    gps_origin_alt: 0.0
    init_from_gps: false
    gt_odom_topic: "/model/wamv/odometry"
    enable_vo: true
    vo_topics_json: '["/wamv/sensors/cameras/front_left/vo","/wamv/sensors/cameras/front_right/vo","/wamv/sensors/cameras/middle_right/vo"]'
    vo_noise_json: '[[1.5,3.0,0.05],[1.5,3.0,0.05],[2.0,4.0,0.05]]'
    vo_rate: 15.0
    sigma_accel: 0.1
    sigma_gyro: 0.01
    sigma_accel_bias: 0.001
    sigma_gyro_bias: 0.0001
    r_gps_pos: 2.0
    r_gps_cog: 0.15
    cog_min_speed: 0.5
    p_init_pos: 10.0
    p_init_vel: 1.0
    p_init_att: 0.5
    p_init_ab: 0.1
    p_init_wb: 0.05

jpda_tracker:
  ros__parameters:
    radar_topic: "/wamv/sensors/radars/xband/points_cluttered"
    radar_x_body: 0.85
    radar_y_body: 0.0
    enable_mmwave: true
    mmwave_topic: "/wamv/sensors/lidars/radar_wamv_sensor/points"
    mmwave_x_body: 0.3
    mmwave_y_body: 0.0
    ais_topic: "/wamv/sensors/ais/nmea"
    marker_topic: "/wamv/tracking/targets"
    world_frame: "map"
    radar_frame: "wamv/xband_radar_sensor"
    cluster_eps: 15.0
    cluster_min_samples: 3
    gate_threshold: 4.0
    max_coast: 5
    min_hits_confirm: 3
    init_cov_pos: 25.0
    init_cov_vel: 4.0
    q_pos: 0.1
    q_vel: 0.5
    r_meas: 9.0
    ais_assoc_dist: 50.0

nmpc_controller:
  ros__parameters:
    target_names: {json.dumps(target_ships_names)}
    rci_disturbance_bound: 1.0
    odom_topic: "/wamv/state/estimated"
    control_rate: 5.0
    prediction_horizon: 20
    time_step: 0.5

colregs_referee:
  ros__parameters:
    backend: "deterministic"
    target_names: {json.dumps(target_ships_names)}
    os_odom_topic: "/wamv/state/estimated"
    trigger_threshold: 0.3
    min_trigger_interval: 2.0
    cpa_safe_distance: 50.0
"""
    return content


# ═════════════════════════════════════════════════════════════════════════════
# RViz2 轨迹可视化配置文件生成
# ═════════════════════════════════════════════════════════════════════════════

# 目标船颜色调色板 (R,G,B) — 高对比度, 避免与 OS 蓝/绿混淆
_TS_COLORS = [
    (255, 40, 40),     # 红 — 最醒目的碰撞警告色
    (255, 140, 0),     # 橙 — 与红/黄均有区分
    (200, 60, 255),    # 紫 — 与蓝/绿/红完全不同色系
    (255, 200, 0),     # 金 — 明亮但非红/橙
    (0, 230, 230),     # 青 — 冷色系
    (255, 60, 160),    # 粉 — 暖色系, 区别于红/紫
]


def _format_color(rgb):
    return f'{rgb[0]}; {rgb[1]}; {rgb[2]}'


def _make_path_display(name, topic, color, width, alpha=1.0):
    """生成 RViz Path 显示块."""
    return f"""    - Alpha: {alpha}
      Buffer Length: 1
      Class: rviz_default_plugins/Path
      Color: {_format_color(color)}
      Enabled: true
      Line Style: {{Line Width: {width}, Value: Lines}}
      Name: {name}
      Topic: {{Depth: 5, Durability Policy: Volatile, History Policy: Keep Last, Reliability Policy: Reliable, Value: {topic}}}
      Value: true"""


def _make_marker_display(name, topic):
    """生成 RViz MarkerArray 显示块."""
    return f"""    - Class: rviz_default_plugins/MarkerArray
      Enabled: true
      Name: {name}
      Topic: {{Depth: 5, Durability Policy: Volatile, History Policy: Keep Last, Reliability Policy: Reliable, Value: {topic}}}
      Value: true"""


def generate_rviz_config(scenario, scenario_id):
    """为指定场景生成 colregs_trajectory.rviz 内容。

    包含:
      - OS GT 轨迹:      /wamv/trajectory_gt     (青蓝特粗线)
      - OS EKF 轨迹:     /wamv/trajectory_ekf    (鲜绿细线)
      - 目标船轨迹:      /{name}/trajectory      (高对比色粗线)
      - 碰撞点标记:      /colregs/collision_markers  (实心球)
      - CPA 最小点标记:  /colregs/cpa_markers        (持久标记)
    """
    ts_names = [ts['name'] for ts in scenario['target_ships']]

    # ── 动态构建 Expanded 列表 ──
    expanded_items = [
        '/Global Options1',
        '/OS GT1',
        '/OS EKF1',
    ]
    expanded_items += [f'/{name}1' for name in ts_names]
    expanded_str = ', '.join(expanded_items)

    # ── 构建显示块 ──
    displays = []

    # OS GT — 青蓝特粗线 (最粗, 最醒目)
    displays.append(_make_path_display(
        'OS GT', '/wamv/trajectory_gt',
        (0, 220, 255), 0.08, 1.0))

    # OS EKF — 鲜绿细线 (与蓝色完全不同的色系)
    displays.append(_make_path_display(
        'OS EKF', '/wamv/trajectory_ekf',
        (80, 255, 100), 0.04, 0.7))

    # 目标船 — 高对比色粗线
    for i, name in enumerate(ts_names):
        c = _TS_COLORS[i % len(_TS_COLORS)]
        displays.append(_make_path_display(
            name, f'/{name}/trajectory',
            c, 0.06, 1.0))

    displays_block = '\n'.join(displays)

    rviz_content = f"""Panels:
  - Class: rviz_common/Displays
    Name: Displays
    Property Tree Widget:
      Expanded: [{expanded_str}]
      Splitter Ratio: 0.5
    Tree Height: 450
  - Class: rviz_common/Selection
    Name: Selection
  - Class: rviz_common/Views
    Name: Views
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.12
      Cell Size: 5
      Class: rviz_default_plugins/Grid
      Color: 80; 80; 80
      Enabled: true
      Line Style: {{Line Width: 0.02, Value: Lines}}
      Name: Grid
      Plane: XY
      Plane Cell Count: 30
      Reference Frame: map
      Value: true
{displays_block}
  Enabled: true
  Global Options:
    Background Color: 20; 20; 22
    Fixed Frame: map
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
  Transformation:
    Current: {{Class: rviz_default_plugins/TF}}
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 20
      Focal Point: {{X: -3, Y: 0, Z: 0}}
      Name: Current View
      Pitch: 1.2
      Target Frame: map
      Yaw: 3.0
    Saved: ~
Window Geometry:
  Displays: {{collapsed: false}}
  Height: 900
  Hide Left Dock: false
  Hide Right Dock: true
  Width: 1400
"""
    return rviz_content


def load_scenario_by_id(scenario_id, new_scenarios: bool = False):
    scenarios = load_scenarios(new_scenarios)
    key = f'scenario_{scenario_id:02d}_'
    for k, v in scenarios.items():
        if k.startswith(key):
            return k, v
    return None, None


def main():
    parser = argparse.ArgumentParser(description='COLREGS 20 Scenarios Loader')
    parser.add_argument('scenario', help='场景编号 (1-20) 或 "list"')
    parser.add_argument('--new', action='store_true',
                        help='使用 colregs_20_new_scenarios.yaml (新场景)')
    parser.add_argument('--dry-run', action='store_true', help='仅打印, 不写入文件')
    parser.add_argument('--output', '-o', default=None,
                        help='输出目录 (默认: config/)')
    args = parser.parse_args()

    if args.scenario == 'list':
        scenarios = load_scenarios(args.new)
        print(f"\n使用场景文件: {'colregs_20_new_scenarios.yaml' if args.new else 'colregs_20_scenarios.yaml'}")
        list_scenarios(scenarios)
        return

    try:
        sid = int(args.scenario)
    except ValueError:
        print(f"错误: 无效场景编号 '{args.scenario}', 请输入 1-20 或 'list'")
        sys.exit(1)

    if sid < 1 or sid > 20:
        print(f"错误: 场景编号 {sid} 超出范围 (1-20)")
        sys.exit(1)

    key, scenario = load_scenario_by_id(sid, args.new)
    scenario_file = 'colregs_20_new_scenarios.yaml' if args.new else 'colregs_20_scenarios.yaml'
    if scenario is None:
        print(f"错误: 未找到场景 {sid}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else CONFIG_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打印场景信息
    print(f"\n{'='*70}")
    print(f"场景 S{sid:02d}: {key}")
    print(f"{'='*70}")
    print(f"类型: {scenario['encounter_type']}")
    print(f"难度: {scenario['difficulty']}")
    print(f"能见度: {scenario.get('visibility', 'clear')}")
    print(f"COLREGS规则: {', '.join(scenario['colregs_rules'])}")
    print(f"目标船数: {len(scenario['target_ships'])}")
    for ts in scenario['target_ships']:
        print(f"  - {ts['name']}: ({ts['x']}, {ts['y']}), "
              f"yaw={ts['yaw']:.2f}rad ({ts['yaw']*57.3:.0f}°), "
              f"speed={ts['speed']} m/s")
    print(f"\n描述: {scenario['description'].strip()}")
    print(f"{'='*70}\n")

    target_ships_yaml = generate_target_ships_yaml(scenario)
    full_mission_yaml = generate_full_mission_yaml(scenario)
    own_ship_yaml = generate_own_ship_yaml(scenario)

    rviz_config = generate_rviz_config(scenario, sid)

    if args.dry_run:
        print("--- target_ships.yaml (preview) ---")
        print(target_ships_yaml)
        print("--- full_mission.yaml (preview) ---")
        print(full_mission_yaml[:2000])
        print("--- own_ship.yaml (preview) ---")
        print(own_ship_yaml)
        print("--- colregs_trajectory.rviz (preview) ---")
        print(rviz_config[:2000])
        print("...")
        return

    # 写入文件
    ts_path = output_dir / 'target_ships.yaml'
    fm_path = output_dir / 'full_mission.yaml'
    os_path = output_dir / 'own_ship.yaml'
    rviz_path = output_dir / 'colregs_trajectory.rviz'

    with open(ts_path, 'w') as f:
        f.write(target_ships_yaml)
    print(f"✓ target_ships.yaml → {ts_path}")

    with open(fm_path, 'w') as f:
        f.write(full_mission_yaml)
    print(f"✓ full_mission.yaml → {fm_path}")

    with open(os_path, 'w') as f:
        f.write(own_ship_yaml)
    print(f"✓ own_ship.yaml → {os_path}")

    with open(rviz_path, 'w') as f:
        f.write(rviz_config)
    print(f"✓ colregs_trajectory.rviz → {rviz_path}")

    print(f"\n启动仿真: ros2 launch marine_env full_mission.launch.py")
    print(f"查看轨迹: rviz2 -d {rviz_path}")


if __name__ == '__main__':
    main()
