#!/bin/bash
# =============================================================================
# 碰撞调试脚本 — 加载场景 + 启动仿真 + 监控关键话题
#
# 用法:
#   ./debug_collision.sh 4          # 场景 4 (双侧交叉, CPA=0)
#   ./debug_collision.sh 4 --rviz   # 同上 + 自动打开 RViz2
# =============================================================================
set -e
SCENE=${1:-4}
RViz=${2:-}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WS="/home/xxy/vrx_ws"

source "${WS}/install/setup.bash"

echo "╔══════════════════════════════════════════════════╗"
echo "║  加载场景 S${SCENE}                                ║"
echo "╚══════════════════════════════════════════════════╝"
python3 "${SCRIPT_DIR}/config/load_scenario.py" "${SCENE}" --new

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  启动仿真                                        ║"
echo "╚══════════════════════════════════════════════════╝"

if [ "$RViz" = "--rviz" ]; then
    ros2 launch marine_env full_mission.launch.py rviz:=True &
else
    ros2 launch marine_env full_mission.launch.py &
fi
SIM_PID=$!

echo "仿真 PID: $SIM_PID"
echo ""
echo "┌──────────────────────────────────────────────────┐"
echo "│  等待 15 秒让 Gazebo + 节点全部初始化...          │"
echo "└──────────────────────────────────────────────────┘"
sleep 15

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  验证: 检查所有关键话题                            ║"
echo "╚══════════════════════════════════════════════════╝"

check_topic() {
    local topic="$1"
    local label="$2"
    if ros2 topic info "$topic" &>/dev/null; then
        local count=$(ros2 topic echo "$topic" --once --timeout 3 2>/dev/null | wc -l)
        echo "  ✓ $label: $topic ($count 行)"
    else
        echo "  ✗ $label: $topic — 无数据!"
    fi
}

echo "── 本船轨迹 ──"
check_topic "/wamv/trajectory_gt" "OS GT 轨迹"
check_topic "/wamv/trajectory_ekf" "OS EKF 轨迹"

echo "── 碰撞标记 ──"
check_topic "/colregs/collision_markers" "碰撞接近标记"
check_topic "/colregs/collision_points" "碰撞点持久标记"

echo "── 目标船轨迹 ──"
for name in $(python3 -c "
import yaml,json,os
for p in ['${SCRIPT_DIR}/config/target_ships.yaml']:
    if os.path.exists(p):
        c=yaml.safe_load(open(p))
        s=c['target_ship_spawner']['ros__parameters']['ships_json']
        ships=json.loads(s) if isinstance(s,str) else s
        print(' '.join([s['name'] for s in ships]))
" 2>/dev/null); do
    check_topic "/${name}/trajectory" "${name} 轨迹"
done

echo "── 本船里程计 ──"
check_topic "/model/wamv/odometry" "OS GT Odometry"

echo "── 碰撞检测节点 ──"
if ros2 node list 2>/dev/null | grep -q collision_marker; then
    echo "  ✓ collision_marker_publisher 运行中"
else
    echo "  ✗ collision_marker_publisher 未运行!"
fi
if ros2 node list 2>/dev/null | grep -q trajectory_pub; then
    echo "  ✓ trajectory_publisher 运行中"
else
    echo "  ✗ trajectory_publisher 未运行!"
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  实时监控 (Ctrl+C 退出)                           ║"
echo "║  观察碰撞检测输出: 接近 → COLLISION → 停船        ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "监控命令:"
echo "  # 碰撞检测日志"
echo "  ros2 topic echo /colregs/collision_markers --once"
echo "  # 碰撞点持久标记"
echo "  ros2 topic echo /colregs/collision_points --once"
echo "  # OS 本船位置"
echo "  ros2 topic echo /model/wamv/odometry --once"
echo ""

# 持续监控碰撞检测日志
ros2 topic echo /colregs/collision_markers 2>/dev/null | while read line; do
    echo "[COLLISION] $line"
done
