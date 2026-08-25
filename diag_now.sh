#!/bin/bash
# 在线诊断 collision_marker_publisher — 仿真运行时执行
echo "=== 1. 节点在不在？ ==="
ros2 node list 2>/dev/null | grep collision || echo "❌ 节点不在!"

echo ""
echo "=== 2. 节点收到 OS 数据没？ ==="
# 看节点自己的诊断输出 (最近 10 行)
if ros2 node list 2>/dev/null | grep -q collision; then
    echo "节点存在，检查它收到多少 OS 消息..."
else
    echo "❌ 节点不存在! 检查 launch 文件是否正确启动"
fi

echo ""
echo "=== 3. 手动算距离 (OS ↔ ts04a) ==="
# 取 OS 位置
OS_POS=$(ros2 topic echo /model/wamv/odometry --once 2>/dev/null | grep -A2 'position:' | tail -1 | awk '{print $2, $3}')
echo "OS 位置: $OS_POS"

# 取 ts04a 位置
TS_POS=$(ros2 topic echo /model/ts04a/odometry --once 2>/dev/null | grep -A2 'position:' | tail -1 | awk '{print $2, $3}')
echo "ts04a 位置: $TS_POS"

# 算距离
if [ -n "$OS_POS" ] && [ -n "$TS_POS" ]; then
    OS_X=$(echo $OS_POS | cut -d' ' -f1 | tr -d ',')
    OS_Y=$(echo $OS_POS | cut -d' ' -f2)
    TS_X=$(echo $TS_POS | cut -d' ' -f1 | tr -d ',')
    TS_Y=$(echo $TS_POS | cut -d' ' -f2)
    DIST=$(python3 -c "import math; print(f'{math.hypot($OS_X - $TS_X, $OS_Y - $TS_Y):.1f}')")
    echo "距离: ${DIST}m"
    if python3 -c "exit(0 if float($DIST) < 5 else 1)"; then
        echo "✅ 距离 < 5m — 应该看到红色球!"
    elif python3 -c "exit(0 if float($DIST) < 50 else 1)"; then
        echo "⚠️  距离 < 50m — 应该看到黄色/橙色球"
    else
        echo "距离 > 50m — 还没靠近"
    fi
fi

echo ""
echo "=== 4. 碰撞标记话题有数据吗？ ==="
MARKER_DATA=$(ros2 topic echo /colregs/collision_markers --once 2>/dev/null | head -5)
if [ -n "$MARKER_DATA" ]; then
    echo "✅ /colregs/collision_markers 有数据"
else
    echo "❌ /colregs/collision_markers 无数据"
fi

echo ""
echo "=== 5. 碰撞点话题有数据吗？ ==="
SITE_DATA=$(ros2 topic echo /colregs/collision_points --once 2>/dev/null | head -5)
if [ -n "$SITE_DATA" ]; then
    echo "✅ /colregs/collision_points 有数据"
else
    echo "⚠️  /colregs/collision_points 无数据 (还没碰撞或节点未检测到)"
fi

echo ""
echo "=== 6. 推力话题 ==="
THRUST=$(ros2 topic echo /wamv/thrusters/left/thrust --once 2>/dev/null)
echo "wamv/left: $THRUST"
THRUST=$(ros2 topic echo /ts04a/thrusters/left/thrust --once 2>/dev/null)
echo "ts04a/left: $THRUST"
