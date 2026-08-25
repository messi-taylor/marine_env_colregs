#!/usr/bin/env python3
"""
Φ Operator ROS 2 Node — Scene Descriptor (Phase 3-4)
======================================================
Standalone node that implements the Φ operator pipeline in real-time:
  1. Subscribe to own-ship odometry
  2. **Dynamically discover** target-ship odometry topics (/model/*/odometry)
  3. Call Φ: build_scene_description() — numerical state → natural language
  4. Publish the factual long-text stream to /colregs/scene_description
  5. Log to console for direct observation

Architecture role (per 文档1.doc Section 3.1):
  Environment Perception → State Fusion → **Scene Semanticization (Φ)** → Symbolic Referee

Key design choice: target ship names vary per scenario (ts01, ts08a, target_0, etc.).
Instead of requiring a parameter that must match the scenario, this node dynamically
discovers /model/*/odometry topics via the ROS 2 graph. No manual configuration needed.

Usage:
  ros2 run marine_env scene_descriptor_node
  ros2 topic echo /colregs/scene_description   # watch the text stream

Parameters:
  publish_rate        — Hz at which to generate scene descriptions (default: 1.0)
  discovery_period    — seconds between topic re-discovery scans (default: 2.0)
  visibility          — visibility condition string (default: "clear")
  sea_state           — Douglas sea state 0-9 (default: 2)
  wind_speed          — m/s (default: 5.0)
  wave_height         — m (default: 0.5)
  current_speed       — m/s (default: 0.3)
"""

import json
import math
import re
import time

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from .scene_descriptor import build_scene_description
from .deterministic_referee import ShipObservation


# Regex for VRX / Gazebo odometry topics: /model/{name}/odometry
_MODEL_ODOM_RE = re.compile(r'^/model/(.+)/odometry$')


class SceneDescriptorNode(Node):
    """ROS 2 node for the Φ operator — numerical state → natural language.

    Dynamically discovers target ship odometry topics so it works with any
    scenario without manual target_names configuration.

    Subscribes:
      /wamv/state/estimated      (Odometry) — own ship ES-EKF estimate
      /model/{name}/odometry     (Odometry) — auto-discovered target ships

    Publishes:
      /colregs/scene_description  (std_msgs/String) — full scene text
      /colregs/scene_summary      (std_msgs/String) — compact JSON summary
    """

    def __init__(self):
        super().__init__('scene_descriptor_node')

        # Parameters
        self.declare_parameter('publish_rate', 1.0)
        self.declare_parameter('discovery_period', 2.0)
        self.declare_parameter('visibility', 'clear')
        self.declare_parameter('sea_state', 2)
        self.declare_parameter('wind_speed', 5.0)
        self.declare_parameter('wave_height', 0.5)
        self.declare_parameter('current_speed', 0.3)

        self.publish_rate = self.get_parameter('publish_rate').value
        self.discovery_period = self.get_parameter('discovery_period').value

        # State storage
        self._own_ship: ShipObservation | None = None
        self._target_states: dict[str, ShipObservation] = {}
        self._target_last_seen: dict[str, float] = {}
        self._known_topics: set[str] = set()      # set of /model/{name}/odometry we already sub'd
        self._generation_count = 0

        # Publishers
        self.scene_pub = self.create_publisher(String, '/colregs/scene_description', 10)
        self.summary_pub = self.create_publisher(String, '/colregs/scene_summary', 10)

        # Own ship subscriber — /model/wamv/odometry is the Gazebo ground truth
        # (ES-EKF publishes to /wamv/state/estimated separately; both are valid.
        #  We use /model/wamv/odometry because it's always available during simulation.)
        self.create_subscription(
            Odometry, '/model/wamv/odometry', self._os_odom_cb, 10)

        # Periodic topic discovery + scene generation
        self._discovery_timer = self.create_timer(self.discovery_period, self._discover_targets)
        self._publish_timer = self.create_timer(1.0 / self.publish_rate, self._generate_scene)

        # Run discovery immediately on startup
        self._discover_targets()

        self.get_logger().info(
            f'Φ Scene Descriptor Node started '
            f'(rate={self.publish_rate} Hz, discovery every {self.discovery_period}s)')
        self.get_logger().info(
            'Publishing to: /colregs/scene_description, /colregs/scene_summary')
        self.get_logger().info('Target ships: auto-discovered from /model/*/odometry')

    # =========================================================================
    # Dynamic topic discovery
    # =========================================================================

    def _discover_targets(self):
        """Discover /model/{name}/odometry topics on the ROS graph.

        Called periodically so newly-spawned target ships are picked up
        without restarting the node. Skips own-ship topics.
        """
        topic_names = self.get_topic_names_and_types()
        own_ship_patterns = ('/wamv/', '/wamv_', '/model/wamv/')

        for topic_name, _ in topic_names:
            m = _MODEL_ODOM_RE.match(topic_name)
            if not m:
                continue
            model_name = m.group(1)
            # Skip own ship
            if any(pattern in topic_name for pattern in own_ship_patterns):
                continue
            # Skip if already subscribed
            if topic_name in self._known_topics:
                continue

            # Subscribe to this target ship
            self.create_subscription(
                Odometry, topic_name,
                lambda msg, n=model_name: self._ts_odom_cb(n, msg), 10)
            self._known_topics.add(topic_name)
            self.get_logger().info(f'Discovered target: /model/{model_name}/odometry')

    # =========================================================================
    # Callbacks — same pattern as referee_node._os_odom_cb / _ts_odom_cb
    # =========================================================================

    def _os_odom_cb(self, msg: Odometry):
        """Own ship ES-EKF state update."""
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        qx, qy, qz, qw = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                           msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                         1.0 - 2.0 * (qy * qy + qz * qz))
        vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
        self._own_ship = ShipObservation(
            name='wamv', position=pos, heading=yaw, speed=vel,
            length=5.0, is_own_ship=True,
        )

    def _ts_odom_cb(self, name: str, msg: Odometry):
        """Target ship odometry callback."""
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        qx, qy, qz, qw = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                           msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                         1.0 - 2.0 * (qy * qy + qz * qz))
        vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
        self._target_states[name] = ShipObservation(
            name=name, position=pos, heading=yaw, speed=vel, length=5.0,
        )
        self._target_last_seen[name] = time.time()

    # =========================================================================
    # Φ operator invocation
    # =========================================================================

    def _generate_scene(self):
        """Φ operator — numerical state → natural language text stream."""
        if self._own_ship is None:
            return

        # Cleanup stale targets (10s threshold, same as referee_node)
        now = time.time()
        stale = [n for n, t in self._target_last_seen.items() if now - t > 10.0]
        for name in stale:
            self._target_states.pop(name, None)
            self._target_last_seen.pop(name, None)

        visibility = self.get_parameter('visibility').value
        sea_state = self.get_parameter('sea_state').value
        wind_speed = self.get_parameter('wind_speed').value
        wave_height = self.get_parameter('wave_height').value
        current_speed = self.get_parameter('current_speed').value

        targets = list(self._target_states.values())
        self._generation_count += 1

        # ── Φ: numerical → natural language ──
        t0 = time.perf_counter()
        scene = build_scene_description(
            self._own_ship, targets,
            visibility=visibility, sea_state=sea_state,
            wind_speed=wind_speed, wave_height=wave_height,
            current_speed=current_speed,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # ── Publish scene text ──
        scene_msg = String()
        scene_msg.data = scene.scene_text
        self.scene_pub.publish(scene_msg)

        # ── Publish JSON summary ──
        summary = {
            'generation': self._generation_count,
            'timestamp': time.time(),
            'own_ship': scene.own_ship_state,
            'num_targets': len(targets),
            'targets': scene.target_ships,
            'rules_applicable': scene.colregs_rules_applicable,
            'environment': scene.environment_context,
            'text_length': len(scene.scene_text),
            'generation_ms': round(elapsed_ms, 2),
        }
        summary_msg = String()
        summary_msg.data = json.dumps(summary, indent=2, ensure_ascii=False)
        self.summary_pub.publish(summary_msg)

        # ── Console log ──
        os_pos = self._own_ship.position
        self.get_logger().info(
            f'Φ #{self._generation_count} | '
            f'OS @ ({os_pos[0]:.0f}, {os_pos[1]:.0f}) | '
            f'{len(targets)} targets | '
            f'rules={scene.colregs_rules_applicable} | '
            f'{len(scene.scene_text)} chars | '
            f'{elapsed_ms:.1f}ms'
        )


def main(args=None):
    rclpy.init(args=args)
    node = SceneDescriptorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
