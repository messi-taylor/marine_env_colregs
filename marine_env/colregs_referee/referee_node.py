#!/usr/bin/env python3
"""
COLREGS Symbolic Referee ROS 2 Node
=====================================

Event-triggered symbolic referee node for the dual-loop neuro-symbolic
control architecture.

Architecture (Section 3.1-3.2):
  ┌─────────────────────────────────────────────────────┐
  │  referee_node (低频异步, event-triggered)             │
  │  ┌─────────────────┐    ┌──────────────────────┐    │
  │  │  Referee Engine  │ →  │  Constraint Mapper   │    │
  │  │  (LLM/Det)       │    │  (symbolic→numeric)  │    │
  │  └─────────────────┘    └──────────────────────┘    │
  │           ↓                          ↓               │
  │  /colregs/decision      /colregs/nmpc_constraints   │
  └─────────────────────────────────────────────────────┘

Event Trigger Condition (Section 3.2):
  φ(cpa, tcpa) · κ(visibility, traffic) > threshold
  → trigger LLM referee refresh

Usage:
  ros2 run marine_env referee_node
  ros2 run marine_env referee_node --ros-args -p backend:=deterministic
  ros2 run marine_env referee_node --ros-args -p backend:=simulated_llm
  ros2 run marine_env referee_node --ros-args -p backend:=anthropic
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# ROS 2 message types
from std_msgs.msg import String, Float64, Bool
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import builtin_interfaces.msg

import numpy as np
import math
import json
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future

from .deterministic_referee import (
    DeterministicReferee,
    ShipObservation,
    _compute_relative_state,
    _compute_cpa_risk_field,
)
from .output_schema import (
    COLREGSConstraintOutput,
    EncounterType,
    ShipRole,
    ManeuverType,
    ForbiddenManeuver,
    validate_output,
)
from .constraint_mapper import ConstraintMapper, NMPCConstraints


class RefereeNode(Node):
    """Event-triggered COLREGS symbolic referee ROS 2 node.

    Subscribes: own ship state, target ship poses
    Publishes: COLREGS decision (structured JSON), NMPC constraints

    Parameters:
      backend: "deterministic" | "simulated_llm" | "anthropic"
      visibility: "clear" | "restricted"
      trigger_threshold: float — CPA risk field threshold for event trigger
      min_trigger_interval: float — minimum seconds between triggers
      cpa_safe_distance: float — base CPA safe distance (m)
    """

    def __init__(self):
        super().__init__('colregs_referee')

        # ── Parameters ──
        self.declare_parameter('backend', 'deterministic')
        self.declare_parameter('visibility', 'clear')
        self.declare_parameter('sea_state', 2)
        self.declare_parameter('trigger_threshold', 0.3)
        self.declare_parameter('min_trigger_interval', 2.0)  # seconds
        self.declare_parameter('cpa_safe_distance', 50.0)
        self.declare_parameter('prediction_horizon', 20)
        self.declare_parameter('target_names', ['ts01', 'ts02a', 'ts02b'])
        self.declare_parameter('os_odom_topic', '/wamv/state/estimated')

        # ── Initialize referee engine ──
        backend = self.get_parameter('backend').value
        self.referee = self._init_referee(backend)
        self.mapper = ConstraintMapper(
            prediction_horizon=self.get_parameter('prediction_horizon').value)

        # ── State storage ──
        self.os_state: Optional[ShipObservation] = None
        self.target_states: Dict[str, ShipObservation] = {}
        self.target_last_seen: Dict[str, float] = {}    # unix timestamp per target
        self.last_trigger_time = 0.0
        self.last_output: Optional[COLREGSConstraintOutput] = None
        self.inference_count = 0
        self.trigger_was_active: bool = False            # for transition detection
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_future: Optional[Future] = None
        self._pending_now: float = 0.0

        # ── Degradation FSM (Section 4.6) ──
        self.degradation_level: int = 0
        self._cfg_validation_failures: int = 0           # consecutive CFG failures
        self.MAX_CFG_FAILURES_BEFORE_DEGRADE = 3
        self.LLM_TIMEOUT_S = 15.0                        # LLM response timeout
        self._degradation_history: list = []              # transition log

        # ── Subscribers ──
        self._setup_subscribers()

        # ── Publishers ──
        # Structured COLREGS decision (JSON string)
        self.decision_pub = self.create_publisher(
            String, '/colregs/decision', 10)
        # NMPC constraints as JSON (for control layer)
        self.nmpc_pub = self.create_publisher(
            String, '/colregs/nmpc_constraints', 10)
        # Degradation level
        self.degradation_pub = self.create_publisher(
            Float64, '/colregs/degradation_level', 10)
        # Event trigger signal
        self.trigger_pub = self.create_publisher(
            Bool, '/colregs/trigger_active', 10)

        # ── Main loop timer ──
        self._loop_timer = self.create_timer(1.0, self._main_loop)

        # ── Status timer ──
        self._status_timer = self.create_timer(10.0, self._print_status)

        self.get_logger().info(
            f'⚖️  COLREGS Referee Node started (backend={backend}, '
            f'visibility={self.get_parameter("visibility").value})')

    @staticmethod
    def _quaternion_to_yaw(orientation) -> float:
        """Convert quaternion to yaw angle using the full 3-rotation formula.

        This is more robust than the simplified 2*atan2(qz,qw) formula
        (which only works for pure Z-axis rotations with qx=qy=0).

        Formula: yaw = atan2(2(qw*qz + qx*qy), 1 - 2(qy² + qz²))

        Returns yaw in ENU convention (0 = East, CCW positive).
        """
        qx = orientation.x
        qy = orientation.y
        qz = orientation.z
        qw = orientation.w
        return math.atan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz))

    @staticmethod
    def _enu_to_maritime_heading(enu_yaw: float) -> float:
        """Convert ENU yaw (0=East, CCW+) to maritime heading (0=North, CW+).

        Conversion: maritime = π/2 - enu, normalized to [-π, π].

        The deterministic_referee bearing formula expects maritime heading.
        Gazebo/ROS2 odometry provides ENU yaw, so we must convert before
        constructing ShipObservation.
        """
        maritime = math.pi / 2.0 - enu_yaw
        return (maritime + math.pi) % (2.0 * math.pi) - math.pi

    def _init_referee(self, backend: str):
        """Initialize the appropriate referee backend."""
        visibility = self.get_parameter('visibility').value
        sea_state = self.get_parameter('sea_state').value

        if backend == 'deterministic':
            return DeterministicReferee(
                visibility=visibility, sea_state=sea_state)

        elif backend == 'simulated_llm':
            from .llm_referee import SimulatedLLMReferee
            return SimulatedLLMReferee(
                visibility=visibility, sea_state=sea_state)

        elif backend == 'anthropic':
            from .llm_referee import AnthropicReferee
            return AnthropicReferee(
                visibility=visibility, sea_state=sea_state)

        elif backend == 'ollama':
            from .llm_referee import OllamaReferee
            return OllamaReferee(
                visibility=visibility, sea_state=sea_state)

        else:
            self.get_logger().warn(
                f"Unknown backend '{backend}', falling back to deterministic")
            return DeterministicReferee(
                visibility=visibility, sea_state=sea_state)

    def _setup_subscribers(self):
        """Set up subscribers for own ship and target ship states."""
        # Own ship: configurable odometry source (ES-EKF by default,
        # can switch to /model/wamv/odometry for GT debugging)
        os_odom = self.get_parameter('os_odom_topic').value
        self.create_subscription(
            Odometry, os_odom, self._os_odom_cb, 10)

        # Target ships: we use dynamic discovery + topic pattern
        target_names = self.get_parameter('target_names').value
        for name in target_names:
            self.create_subscription(
                Odometry, f'/model/{name}/odometry',
                lambda msg, n=name: self._ts_odom_cb(n, msg), 10)

    # =====================================================================
    # Callbacks
    # =====================================================================

    def _os_odom_cb(self, msg: Odometry):
        """Own ship state update.

        Converts ENU yaw from odometry to maritime heading expected by the
        deterministic referee bearing formula (0=North, CW+).
        """
        pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        ])
        enu_yaw = self._quaternion_to_yaw(msg.pose.pose.orientation)
        maritime_heading = self._enu_to_maritime_heading(enu_yaw)
        vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
        ])
        self.os_state = ShipObservation(
            name='wamv',
            position=pos,
            heading=maritime_heading,
            speed=vel,
            is_own_ship=True,
        )

    def _ts_odom_cb(self, name: str, msg: Odometry):
        """Target ship state update.

        Converts ENU yaw from odometry to maritime heading expected by the
        deterministic referee bearing formula (0=North, CW+).
        """
        pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        ])
        enu_yaw = self._quaternion_to_yaw(msg.pose.pose.orientation)
        maritime_heading = self._enu_to_maritime_heading(enu_yaw)
        vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
        ])
        self.target_states[name] = ShipObservation(
            name=name,
            position=pos,
            heading=maritime_heading,
            speed=vel,
            length=5.0,
        )
        self.target_last_seen[name] = time.time()

    # =====================================================================
    # Main Loop & Event Trigger
    # =====================================================================

    def _main_loop(self):
        """Main processing loop — checks trigger condition and runs referee."""
        if self.os_state is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        # ── Stale target cleanup (Issue 3) ──
        # Remove targets that haven't been seen for > 10 seconds
        stale_timeout = 10.0
        stale_names = [
            name for name, last_seen in self.target_last_seen.items()
            if now - last_seen > stale_timeout
        ]
        for name in stale_names:
            self.get_logger().warn(
                f'Target {name} lost (no updates for {now - self.target_last_seen[name]:.1f}s), removing')
            self.target_states.pop(name, None)
            self.target_last_seen.pop(name, None)

        # ── Handle pending async evaluation (Issue 5) ──
        # MUST check before trigger to avoid dropped results
        if self._pending_future is not None:
            if self._pending_future.done():
                # Computation finished — publish results in main thread
                try:
                    result = self._pending_future.result()
                    if result:
                        output, nmpc_constraints = result
                        self.last_output = output
                        self._publish_decision(output)
                        self._publish_nmpc_constraints(nmpc_constraints)
                        self._log_referee_result(output)
                        # Reset CFG failure counter on success
                        self._cfg_validation_failures = 0
                except Exception as e:
                    self.get_logger().error(f'Async referee evaluation failed: {e}')
                    self._handle_evaluation_failure(str(e))
                self._pending_future = None
            else:
                # ── LLM Timeout Check (Section 4.6) ──
                elapsed = now - self._pending_now
                if elapsed > self.LLM_TIMEOUT_S:
                    self.get_logger().error(
                        f'⏰ LLM timeout: {elapsed:.1f}s without response → '
                        f'cancelling + triggering degradation')
                    # Cancel the stuck future (best effort)
                    self._pending_future = None
                    self._escalate_referee_degradation('llm_timeout')
                    # Fall back to deterministic
                    self._fallback_to_deterministic()
                else:
                    # Still waiting for LLM — skip trigger check this cycle
                    return

        # ── Event trigger check ──
        should_trigger = self._should_trigger(now)

        # Publish trigger state on transitions (Issue 4)
        if should_trigger != self.trigger_was_active:
            trig_msg = Bool()
            trig_msg.data = should_trigger
            self.trigger_pub.publish(trig_msg)
            self.trigger_was_active = should_trigger

        if not should_trigger:
            return

        # Submit new referee evaluation to background thread
        active_targets = list(self.target_states.values())
        if not active_targets:
            return

        self.last_trigger_time = now
        self.inference_count += 1

        self._pending_now = now
        self._pending_future = self._executor.submit(
            self._evaluate_referee_blocking,
            now,
            active_targets,
            f"referee_{self.inference_count:04d}",
        )

    def _should_trigger(self, now: float) -> bool:
        """Check event-trigger condition (Section 3.2).

        φ(cpa, tcpa) · κ(visibility, traffic) > threshold

        Also requires minimum interval between triggers to prevent
        excessive LLM API calls.
        """
        if not self.target_states:
            # No targets → only trigger on first run
            return self.last_output is None

        # Minimum interval check
        interval = self.get_parameter('min_trigger_interval').value
        if now - self.last_trigger_time < interval:
            return False

        # Always trigger on first evaluation
        if self.last_output is None:
            return True

        # Compute CPA risk field for all targets
        max_risk = 0.0
        for name, ts in self.target_states.items():
            if self.os_state:
                geo = _compute_relative_state(self.os_state, ts)
                risk = _compute_cpa_risk_field(
                    geo['cpa'], geo['tcpa'], geo['rel_distance'])
                max_risk = max(max_risk, risk)

        # Environment context factor
        visibility = self.get_parameter('visibility').value
        kappa = 0.5 if visibility == 'restricted' else 1.0

        threshold = self.get_parameter('trigger_threshold').value

        return max_risk * kappa > threshold

    def _evaluate_referee_blocking(
        self,
        now: float,
        active_targets: List[ShipObservation],
        scenario_id: str,
    ) -> Optional[Tuple[COLREGSConstraintOutput, NMPCConstraints]]:
        """Run referee evaluation in a background thread (Issue 5).

        This method is designed to be called via ThreadPoolExecutor.submit().
        It does NO ROS publishing — results are returned to the main thread.

        Returns:
            (output, nmpc_constraints) on success, None on failure.
        """
        if self.os_state is None:
            return None

        try:
            # Step 1: Run referee (potentially blocking LLM call)
            output = self.referee.evaluate(
                self.os_state,
                active_targets,
                scenario_id=scenario_id,
            )

            # Step 2: Validate
            validate_output(output)

            # Step 3: Map to NMPC constraints
            nmpc_constraints = self.mapper.map(output)

            return (output, nmpc_constraints)

        except Exception as e:
            self.get_logger().error(f'Background referee evaluation failed: {e}')
            return None

    def _log_referee_result(self, output: COLREGSConstraintOutput):
        """Log referee evaluation result (called from main thread)."""
        self.get_logger().info(
            f'⚖️  Referee #{self.inference_count}: '
            f'encounter={output.encounter_classification.primary_encounter.value}, '
            f'risk={output.encounter_classification.risk_level}, '
            f'maneuver={output.required_maneuver.value}, '
            f'min_cpa={output.global_min_cpa:.0f}m, '
            f'conf={output.confidence_score:.2f}, '
            f'degrad={output.degradation_level}, '
            f'time={output.inference_time_ms:.1f}ms'
        )

    def _run_referee(self, now: float):
        """Execute the referee evaluation pipeline synchronously.

        Pipeline:
          1. Referee Engine → COLREGSConstraintOutput
          2. Constraint Mapper → NMPCConstraints
          3. Publish results

        Note: When backend is LLM-based (anthropic), prefer the async path
        via _evaluate_referee_blocking in the main loop to avoid blocking.
        """
        if self.os_state is None:
            return

        active_targets = list(self.target_states.values())

        if not active_targets:
            self.get_logger().debug('No active target ships, skipping')
            return

        self.inference_count += 1

        result = self._evaluate_referee_blocking(
            self.get_clock().now().nanoseconds * 1e-9,
            active_targets,
            f"referee_{self.inference_count:04d}",
        )

        if not result:
            return

        output, nmpc_constraints = result
        self.last_output = output
        self._publish_decision(output)
        self._publish_nmpc_constraints(nmpc_constraints)
        self._log_referee_result(output)

    def _publish_decision(self, output: COLREGSConstraintOutput):
        """Publish the structured COLREGS decision as JSON."""
        msg = String()
        msg.data = output.to_json(indent=2)
        self.decision_pub.publish(msg)

        # Also publish degradation level
        deg_msg = Float64()
        deg_msg.data = float(output.degradation_level)
        self.degradation_pub.publish(deg_msg)

    def _publish_nmpc_constraints(self, nmpc: NMPCConstraints):
        """Publish NMPC constraints as JSON."""
        msg = String()
        msg.data = json.dumps(nmpc.to_dict(), indent=2)
        self.nmpc_pub.publish(msg)

    # =====================================================================
    # Degradation FSM helpers (Section 4.6)
    # =====================================================================

    def _handle_evaluation_failure(self, error_msg: str):
        """Handle a referee evaluation failure.

        Distinguishes CFG/GBNF validation failures from other errors.
        After MAX_CFG_FAILURES_BEFORE_DEGRADE consecutive CFG failures,
        escalates degradation and falls back to deterministic.
        """
        is_cfg_failure = any(kw in error_msg.lower()
                             for kw in ['validation', 'schema', 'grammar',
                                        'cfg', 'gbnf', 'json', 'parse'])

        if is_cfg_failure:
            self._cfg_validation_failures += 1
            self.get_logger().warn(
                f'CFG validation failure #{self._cfg_validation_failures}/'
                f'{self.MAX_CFG_FAILURES_BEFORE_DEGRADE}: {error_msg[:120]}')
            if self._cfg_validation_failures >= self.MAX_CFG_FAILURES_BEFORE_DEGRADE:
                self._escalate_referee_degradation('cfg_validation_failure')
                self._fallback_to_deterministic()
        else:
            self.get_logger().error(
                f'Referee evaluation error: {error_msg[:200]}')

    def _escalate_referee_degradation(self, trigger_reason: str):
        """Escalate degradation and publish to NMPC controller.

        Args:
            trigger_reason: 'llm_timeout' | 'cfg_validation_failure'
        """
        if self.degradation_level >= 3:
            return
        old_level = self.degradation_level
        self.degradation_level += 1

        import datetime
        event = {
            'timestamp': datetime.datetime.now().isoformat(),
            'unix_time': time.time(),
            'level_before': old_level,
            'level_after': self.degradation_level,
            'trigger_reason': trigger_reason,
            'inference_count': self.inference_count,
            'cfg_failures': self._cfg_validation_failures,
        }
        self._degradation_history.append(event)
        if len(self._degradation_history) > 100:
            self._degradation_history = self._degradation_history[-100:]

        self.get_logger().error(
            f'⚠️ Referee Degradation L{old_level}→L{self.degradation_level}: '
            f'{trigger_reason} (event #{len(self._degradation_history)})')

        # Publish degradation to NMPC controller
        deg_msg = Float64()
        deg_msg.data = float(self.degradation_level)
        self.degradation_pub.publish(deg_msg)

    def _fallback_to_deterministic(self):
        """Fall back to deterministic referee after LLM/CFG failures.

        Runs a synchronous deterministic evaluation to maintain control continuity.
        """
        if self.os_state is None or not self.target_states:
            return

        try:
            det = DeterministicReferee(
                visibility=self.get_parameter('visibility').value,
                sea_state=self.get_parameter('sea_state').value)

            active_targets = list(self.target_states.values())
            output = det.evaluate(
                self.os_state, active_targets,
                scenario_id=f'fallback_{self.inference_count:04d}')

            from .output_schema import validate_output
            validate_output(output)

            nmpc_constraints = self.mapper.map(output)
            self.last_output = output
            self._publish_decision(output)
            self._publish_nmpc_constraints(nmpc_constraints)
            self.get_logger().info(
                f'🔄 Fallback deterministic: '
                f'encounter={output.encounter_classification.primary_encounter.value}, '
                f'maneuver={output.required_maneuver.value}')
        except Exception as e:
            self.get_logger().error(
                f'❌ Even deterministic fallback failed: {e}', throttle_duration_sec=5.0)

    def _print_status(self):
        """Periodic status report."""
        if self.last_output:
            self.get_logger().info(
                f'📊 Status: {self.inference_count} evaluations, '
                f'last maneuver={self.last_output.required_maneuver.value}, '
                f'degradation={self.degradation_level}, '
                f'CFG fails={self._cfg_validation_failures}, '
                f'avg inference={self.referee.avg_inference_time_ms:.1f}ms'
                if hasattr(self.referee, 'avg_inference_time_ms')
                else f'📊 Status: {self.inference_count} evaluations, '
                      f'degrad={self.degradation_level}'
            )


def main():
    rclpy.init()
    node = RefereeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
