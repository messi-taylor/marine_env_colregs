#!/bin/bash
# =============================================================================
# COLREGS 场景一键启动脚本
#
# 用法:
#   ./run_scenario.sh <1-20>           加载场景 + 启动仿真 (RViz2 手动打开)
#   ./run_scenario.sh <1-20> --rviz    加载场景 + 启动仿真 + 自动打开 RViz2
#   ./run_scenario.sh list             列出所有场景
#
# 示例:
#   ./run_scenario.sh 8 --rviz          # 场景 8, 自动 RViz2
#   ./run_scenario.sh 2                 # 场景 2, RViz2 手动
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/config"
WS_DIR="/home/xxy/vrx_ws"

# ── 参数解析 ──
if [ $# -lt 1 ]; then
    echo "用法: $0 <1-20|list> [--rviz]"
    echo "示例: $0 8 --rviz"
    exit 1
fi

SCENARIO_ID="$1"
RViz_FLAG="${2:-}"

# ── 环境 ──
if [ -f "${WS_DIR}/install/setup.bash" ]; then
    source "${WS_DIR}/install/setup.bash"
else
    echo "[WARN] 未找到 ${WS_DIR}/install/setup.bash, 请先 colcon build"
fi

# ── 列出场景 ──
if [ "$SCENARIO_ID" = "list" ]; then
    python3 "${CONFIG_DIR}/load_scenario.py" list --new
    exit 0
fi

# ── 验证场景编号 ──
if ! [[ "$SCENARIO_ID" =~ ^[0-9]+$ ]] || [ "$SCENARIO_ID" -lt 1 ] || [ "$SCENARIO_ID" -gt 20 ]; then
    echo "错误: 场景编号必须是 1-20, 收到: $SCENARIO_ID"
    exit 1
fi

# ── 加载场景配置 ──
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  COLREGS Scenario S${SCENARIO_ID} — 加载配置                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
python3 "${CONFIG_DIR}/load_scenario.py" "${SCENARIO_ID}" --new

# ── 启动仿真 ──
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  启动 VRX 仿真 + Marine Environment                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"

RViz_CONFIG="${CONFIG_DIR}/colregs_trajectory.rviz"

if [ "$RViz_FLAG" = "--rviz" ]; then
    echo "[INFO] 自动启动 RViz2: ${RViz_CONFIG}"
    ros2 launch marine_env full_mission.launch.py rviz:=True
else
    ros2 launch marine_env full_mission.launch.py
    echo ""
    echo "┌──────────────────────────────────────────────────────────────┐"
    echo "│  RViz2 轨迹可视化 (在另一个终端):                              │"
    echo "│    rviz2 -d ${RViz_CONFIG}                                   │"
    echo "│                                                              │"
    echo "│  或重新运行本脚本加 --rviz 自动打开:                           │"
    echo "│    $0 ${SCENARIO_ID} --rviz                                  │"
    echo "└──────────────────────────────────────────────────────────────┘"
fi
