#!/usr/bin/env python3
"""
Resilient NMPC Controller Node (Phase 7-8)
============================================

High-frequency numerical control layer. Uses async solver execution (ThreadPoolExecutor)
to avoid blocking the ROS executor — same pattern as referee_node.

Key design:
  - Timer at 5 Hz (200ms) checks for completed solves
  - Solver runs in background thread
  - While waiting: publish last valid thrust (or simple fallback on first cycle)
  - Warm-start from previous solution for speed

Usage:
  ros2 run marine_env nmpc_controller
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String
from nav_msgs.msg import Odometry
import numpy as np
import math
import json
import time
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future


def _json_safe(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj

try:
    from .nmpc_solver import NMPCSolver, NMPCParams
    from .robustness import (
        RobustnessVerifier, RobustnessConfig,
        compute_robust_constraints,
    )
except ImportError:
    from nmpc_solver import NMPCSolver, NMPCParams
    from robustness import (
        RobustnessVerifier, RobustnessConfig,
        compute_robust_constraints,
    )


class NMPCControllerNode(Node):
    """ROS 2 NMPC controller — async solver, 5 Hz publish rate."""

    def __init__(self):
        super().__init__('nmpc_controller')

        # ── Parameters ──
        self.declare_parameter('waypoints',
            [0.0, 0.0, 2.0, 0.0, 400.0, 2.0])  # 直行向北 2m/s
        self.declare_parameter('control_rate', 5.0)            # Hz — match solver speed
        self.declare_parameter('odom_topic', '/wamv/state/estimated')
        self.declare_parameter('prediction_horizon', 20)
        self.declare_parameter('time_step', 0.5)
        self.declare_parameter('max_thrust', 1500.0)
        self.declare_parameter('max_yaw_moment', 800.0)
        self.declare_parameter('target_names', ['ts01', 'ts02a', 'ts02b'])

        # ── Waypoints ──
        raw = self.get_parameter('waypoints').value
        self.waypoints: List[Tuple[float, float, float]] = [
            (raw[i], raw[i + 1], raw[i + 2]) for i in range(0, len(raw), 3)
        ]
        self._wp_idx = 0
        self._mission_complete = False

        # ── NMPC solver (built once at init) ──
        solver_params = NMPCParams(
            N=self.get_parameter('prediction_horizon').value,
            dt=self.get_parameter('time_step').value,
            max_yaw_moment=self.get_parameter('max_yaw_moment').value,
            max_thrust_per_engine=self.get_parameter('max_thrust').value,
        )
        self.solver = NMPCSolver(params=solver_params)
        self.get_logger().info('Building NMPC symbolic problem (CasADi Opti + IPOPT)...')
        t0 = time.perf_counter()
        self.solver.setup()
        self.get_logger().info(
            f'NMPC solver ready in {(time.perf_counter() - t0)*1000:.0f}ms '
            f'(N={solver_params.N}, dt={solver_params.dt}s)')

        # ── Robustness Verifier (Section 4.5) ──
        self.declare_parameter('rci_disturbance_bound', 5.0)
        rci_bound = self.get_parameter('rci_disturbance_bound').value
        robustness_cfg = RobustnessConfig(
            tau_nominal=0.05,
            tau_max_default=0.20,
            tau_safety_factor=2.0,
            ts_pos_sigma_default=1.0,
            rci_disturbance_bound=rci_bound,
        )
        self._robustness = RobustnessVerifier(config=robustness_cfg)
        self._last_robustness_report: dict = {}
        self._rci_update_counter: int = 0
        self._RCI_UPDATE_INTERVAL = 10  # re-linearize every 10 solves

        # ── State ──
        self.own_state: Optional[np.ndarray] = None
        self.target_states: Dict[str, np.ndarray] = {}
        self.target_last_seen: Dict[str, float] = {}

        # ── COLREGS ──
        self.colregs_constraints: Optional[dict] = None
        self.last_constraints_time: float = 0.0

        # ── Last valid output (published while solver runs) ──
        self._last_tau_u: float = 0.0
        self._last_tau_r: float = 0.0
        self._has_valid_output: bool = False

        # ── Spin-up: accelerate OS to cruise speed before NMPC (mirrors offline evaluator) ──
        self._spinup_complete: bool = False
        self._spinup_target_speed: float = 0.5  # m/s surge — lowered from 1.0 for VRX
        self._spinup_start_time: float = 0.0    # unix time when odometry first arrived
        self._spinup_timeout: float = 60.0      # force spin-up complete after this many seconds

        # ── Starboard reference bias (mirrors batch_runner.py) ──
        self._need_starboard_turn: bool = False
        self._starboard_bias_applied: bool = False

        # ── Degradation FSM (Section 4.6) ──
        self.degradation_level: int = 0
        self.consecutive_infeasible: int = 0
        self.consecutive_solved: int = 0          # for auto-recovery
        self.MAX_INFEASIBLE_BEFORE_DEGRADE = 5
        self.MIN_SOLVED_BEFORE_RECOVERY = 10       # consecutive SOLVED to recover
        self.SOLVER_TIMEOUT_S = 15.0               # async solver timeout → degrade
        self.RECOVERY_RISK_THRESHOLD = 0.15        # max risk field for recovery

        # Degradation history log
        self._degradation_history: list = []
        # Communication timeout tracking
        self._last_solver_submit_time: float = 0.0
        self._solver_timeout_degraded: bool = False

        # ── Async solver ──
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_future: Optional[Future] = None
        self._pending_solve_start: float = 0.0

        # ── Stats ──
        self.solve_count: int = 0
        self.solve_times_ms: List[float] = []
        self.infeasible_count: int = 0

        # ── Publishers ──
        # Absolute topics — our launch file provides explicit ros_gz_bridge
        # for these (not relying on VRX's internal namespaced bridge).
        self.left_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        self.status_pub = self.create_publisher(String, '/wamv/nmpc/status', 10)

        # ── Subscribers ──
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value,
                                 self._own_odom_cb, 10)
        self.create_subscription(String, '/colregs/nmpc_constraints',
                                 self._constraints_cb, 10)
        self.create_subscription(Float64, '/colregs/degradation_level',
                                 self._referee_degradation_cb, 10)
        for name in self.get_parameter('target_names').value:
            self.create_subscription(
                Odometry, f'/model/{name}/odometry',
                lambda msg, n=name: self._ts_odom_cb(n, msg), 10)

        # ── Timers ──
        rate = max(self.get_parameter('control_rate').value, 1.0)
        self._timer = self.create_timer(1.0 / rate, self._timer_callback)
        self._status_timer = self.create_timer(5.0, self._publish_status)
        self._startup_time = time.time()  # for startup thrust timeout
        self._startup_thrust_logged = False

        self.get_logger().info(
            f'🚢 NMPC Controller ready: {len(self.waypoints)} waypoints, '
            f'{rate:.0f} Hz, horizon={solver_params.N}')

    # =====================================================================
    # Callbacks
    # =====================================================================

    def _own_odom_cb(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        qx, qy, qz, qw = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                           msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        psi = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        vx, vy, r = msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z
        c, s = math.cos(psi), math.sin(psi)
        u_body = c * vx + s * vy
        v_body = -s * vx + c * vy
        self.own_state = np.array([px, py, psi, u_body, v_body, r])

    def _ts_odom_cb(self, name: str, msg: Odometry):
        px, py = msg.pose.pose.position.x, msg.pose.pose.position.y
        qx, qy, qz, qw = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                           msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        psi = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        vx, vy = msg.twist.twist.linear.x, msg.twist.twist.linear.y
        c, s = math.cos(psi), math.sin(psi)
        self.target_states[name] = np.array([
            px, py, psi, c * vx + s * vy, -s * vx + c * vy])
        self.target_last_seen[name] = time.time()

    def _constraints_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.colregs_constraints = self._parse_colregs_json(data)
            self.last_constraints_time = time.time()
            # Store starboard-turn signal for reference trajectory bias
            self._need_starboard_turn = (data.get('turn_direction_sign', 0) > 0 or
                                         data.get('forbidden_maneuver', '') == 'alter_to_port')
        except Exception as e:
            self.get_logger().warn(f'Failed to parse constraints: {e}')

    def _referee_degradation_cb(self, msg: Float64):
        # Ignore external degradation during spin-up (OS needs time to accelerate)
        if not self._spinup_complete:
            return
        ref_deg = int(msg.data)
        # Cap external degradation at L1 — never let referee kill speed (L3)
        ref_deg = min(ref_deg, 1)
        if ref_deg > self.degradation_level:
            old = self.degradation_level
            self.degradation_level = ref_deg
            self._log_degradation_event(old, self.degradation_level, 'referee_signal')
            self.get_logger().warn(
                f'⚠️ Degradation L{old}→L{ref_deg} by referee signal (external)')

    # =====================================================================
    # Timer callback — async solver pattern
    # =====================================================================

    def _timer_callback(self):
        """Main timer: publishes thrust every cycle. Starts async solver if idle.

        CRITICAL: always publishes thrust — even before odometry arrives.
        On startup (before EKF converges), publishes a default forward thrust
        to get the WAM-V moving. After odometry arrives, switches to NMPC.

        Degradation Level 3: emergency brake — zero thrust unconditionally.
        """
        now = time.time()

        # ── Level 3 Emergency: simple PD starboard avoidance (never zero thrust) ──
        # Zero thrust = dead in water = guaranteed collision. Instead, use a
        # minimal starboard PD controller that at least attempts avoidance.
        # Periodically retry NMPC (every 30s) to allow auto-recovery.
        if self.degradation_level >= 3:
            # Cancel any stuck pending solve
            if self._pending_future is not None and not self._pending_future.done():
                if now - self._last_solver_submit_time > self.SOLVER_TIMEOUT_S:
                    self._pending_future = None
                    self._solver_timeout_degraded = False

            # Periodic NMPC retry for auto-recovery
            retry_interval = 30.0  # try NMPC every 30s
            if (self._pending_future is None and self.own_state is not None
                    and now - self._last_solver_submit_time > retry_interval):
                self.get_logger().info(
                    f'🔄 L3 retry: attempting NMPC solve for auto-recovery...')
                self._last_solver_submit_time = now
                self._start_background_solve()

            # Check for completed retry solve
            if self._pending_future is not None and self._pending_future.done():
                try:
                    new_result = self._pending_future.result()
                except Exception as e:
                    self.get_logger().error(f'L3 retry solve crashed: {e}')
                    new_result = None
                self._pending_future = None
                if new_result is not None:
                    self._process_solve_result(new_result)

            # PD starboard avoidance (same as offline batch_runner Phase 1)
            if self.own_state is not None:
                left, right = self._l3_pd_avoidance()
                self._publish_thrust(left, right)
            else:
                self._publish_thrust(300.0, 300.0)  # gentle forward if no odom
            return

        # ── Check solver timeout (independent degradation path) ──
        if self._pending_future is not None and self._pending_future.done():
            # ... (existing done-check logic)
            pass  # handled below

        if (self._pending_future is not None
                and now - self._last_solver_submit_time > self.SOLVER_TIMEOUT_S
                and not self._solver_timeout_degraded):
            self.get_logger().error(
                f'⏰ Solver timeout: {now - self._last_solver_submit_time:.1f}s '
                f'without result → triggering degradation')
            self._solver_timeout_degraded = True
            self._escalate_degradation('solver_timeout')
            # Cancel the stuck future
            self._pending_future = None

        # ── Check if a background solve just finished ──
        new_result = None
        if self._pending_future is not None and self._pending_future.done():
            try:
                new_result = self._pending_future.result()
            except Exception as e:
                self.get_logger().error(f'Background solve crashed: {e}')
            self._pending_future = None
            self._solver_timeout_degraded = False

        if new_result is not None:
            try:
                self._process_solve_result(new_result)
            except Exception as e:
                self.get_logger().error(
                    f'Process solve result crashed: {e}', throttle_duration_sec=5.0)
                self.consecutive_infeasible += 1

        # ── Stale target cleanup ──
        stale = [n for n, t in self.target_last_seen.items() if now - t > 10.0]
        for n in stale:
            self.target_states.pop(n, None)
            self.target_last_seen.pop(n, None)

        # ── Spin-up: accelerate to cruise speed before NMPC (mirrors offline evaluator) ──
        if not self._spinup_complete and self.own_state is not None:
            # Record when odometry first arrived
            if self._spinup_start_time == 0.0:
                self._spinup_start_time = now

            surge = float(self.own_state[3])
            spinup_elapsed = now - self._spinup_start_time

            if surge >= self._spinup_target_speed:
                self._spinup_complete = True
                # Reset degradation + NMPC state for clean start
                self.degradation_level = 0
                self.consecutive_infeasible = 0
                self._has_valid_output = False
                self.get_logger().info(
                    f'🚀 Spin-up complete (surge={surge:.1f} m/s, '
                    f'elapsed={spinup_elapsed:.1f}s). Starting NMPC.')
            elif spinup_elapsed > self._spinup_timeout:
                # Timeout: force spin-up complete even if target speed not reached.
                # In VRX, hydrodynamic drag + late release may prevent reaching
                # target speed. Better to start NMPC at low speed than never.
                self._spinup_complete = True
                self.degradation_level = 0
                self.consecutive_infeasible = 0
                self._has_valid_output = False
                self.get_logger().warn(
                    f'⏰ Spin-up timeout ({spinup_elapsed:.0f}s > {self._spinup_timeout:.0f}s), '
                    f'surge={surge:.1f} m/s. Starting NMPC anyway.')

        # ── Start background solve if idle AND have state AND spin-up done ──
        if self._pending_future is None and self.own_state is not None and self._spinup_complete:
            self._start_background_solve()

        # ── Determine thrust to publish ──
        if not self._spinup_complete and self.own_state is not None:
            # Spin-up phase: pure LOS thrust to reach cruise speed
            left, right = self._initial_fallback_thrust()
        elif self._has_valid_output and self.own_state is not None:
            # NMPC solution available
            left, right = self._thrust_mapping(self._last_tau_u, self._last_tau_r)
        elif self.own_state is not None:
            # Have odometry but no NMPC solution yet → simple LOS fallback
            left, right = self._initial_fallback_thrust()
        elif now - self._startup_time > 2.0:
            # No odometry after 2s → publish default thrust to spin up
            # (EKF may need time to converge after Gazebo release)
            if not self._startup_thrust_logged:
                self.get_logger().info(
                    '⏳ No odometry yet, publishing startup thrust (500N each)')
                self._startup_thrust_logged = True
            left, right = 500.0, 500.0
        else:
            # First 2 seconds — wait for EKF to start publishing
            left, right = 0.0, 0.0

        self._publish_thrust(left, right)

    def _start_background_solve(self):
        """Submit a new NMPC solve to the background thread."""
        # Build inputs (copy state to avoid race conditions)
        x0 = self.own_state.copy()
        wp_idx = self._wp_idx

        # ── Starboard reference bias (mirrors batch_runner.py) ──
        # When COLREGS requires starboard turn, bias reference waypoints to guide NMPC
        if self._need_starboard_turn and not self._starboard_bias_applied:
            os_heading = float(x0[2])
            starboard_hdg = os_heading - math.radians(40)  # CW in ENU
            wp_mid = (float(x0[0]) + 50.0 * math.cos(starboard_hdg),
                      float(x0[1]) + 50.0 * math.sin(starboard_hdg),
                      self._spinup_target_speed)
            wp_end = self.waypoints[-1] if self.waypoints else (
                float(x0[0]) + 200.0 * math.cos(starboard_hdg),
                float(x0[1]) + 200.0 * math.sin(starboard_hdg),
                self._spinup_target_speed)
            waypoints = [wp_mid, wp_end]
            self._starboard_bias_applied = True
            self.get_logger().info(
                f'⭐ Starboard bias applied: mid=({wp_mid[0]:.0f},{wp_mid[1]:.0f}) '
                f'end=({wp_end[0]:.0f},{wp_end[1]:.0f})')
        else:
            waypoints = list(self.waypoints)

        constraints = self._build_active_constraints()
        constraints = self._apply_degradation(constraints)
        target_trajs = self._predict_target_trajectories()

        self._pending_solve_start = time.perf_counter()
        self._last_solver_submit_time = time.time()
        self._pending_future = self._executor.submit(
            self._solve_blocking,
            x0, waypoints, wp_idx, constraints, target_trajs,
        )

    def _solve_blocking(self, x0, waypoints, wp_idx, constraints, target_trajs):
        """Run the solver in a background thread with progressive relaxation.

        Mirrors batch_runner.py Plan D: when primary solve fails, retry with
        up to 3 levels of progressively relaxed constraints.

        Returns (result, tau_u, tau_r).
        """
        try:
            # ── Input validation ──
            x0 = np.asarray(x0, dtype=float).ravel()
            if len(x0) != 6 or not np.all(np.isfinite(x0)):
                self.get_logger().error(
                    f'Invalid state for NMPC solve: {x0}', throttle_duration_sec=5.0)
                return {
                    'status': 'INFEASIBLE', 'u_opt': np.zeros((2, 1)),
                    'x_pred': np.zeros((6, self.solver.p.N + 1)),
                    'cost': float('inf'), 'solve_time_ms': 0.0,
                    'epsilon_legal': 0.0, 'epsilon_smooth': 0.0,
                    'epsilon_speed': 0.0, 'wp_idx': wp_idx,
                    'retry_level': 0,
                }
            # Clamp velocity to reasonable range before solve
            x0 = x0.copy()
            x0[3] = max(-10.0, min(10.0, x0[3]))   # surge
            x0[4] = max(-5.0, min(5.0, x0[4]))     # sway
            x0[5] = max(-2.0, min(2.0, x0[5]))     # yaw rate

            x_ref = self.solver.generate_reference(x0, waypoints, wp_idx)
            t_start = time.perf_counter()
            result = self.solver.solve(
                x0=x0, x_ref=x_ref, target_trajs=target_trajs,
                constraints=constraints, tau_env=np.zeros(3), warm_start=True,
            )
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            result['solve_time_ms'] = elapsed_ms
            result['wp_idx'] = wp_idx
            result['retry_level'] = 0

            # ── Progressive constraint relaxation (Plan D, mirrors batch_runner.py) ──
            # When primary solve fails, retry with up to 3 levels of relaxation.
            # Relaxation order (safest first):
            #   L1: reduce alteration by 15°, allow -0.1 rudder
            #   L2: reduce CPA by 40%, remove alteration
            #   L3: free rudder, CPA floor at 12m
            MAX_RETRY = 3
            max_yaw_moment = 800.0  # matches solver tau_r bounds
            retry_level = 0
            while result['status'] != 'SOLVED' and retry_level < MAX_RETRY:
                retry_level += 1
                rc = dict(constraints)  # shallow copy
                rc['cpa_radius_per_target'] = dict(constraints.get('cpa_radius_per_target', {}))
                if retry_level == 1:
                    rc['alteration_min_angle'] = max(
                        0.0, rc.get('alteration_min_angle', 0.0) - math.radians(15))
                    rc['alteration_active'] = rc['alteration_min_angle'] > 0.01
                    rc['tau_r_min'] = max(
                        rc.get('tau_r_min', -max_yaw_moment), -0.1 * max_yaw_moment)
                elif retry_level == 2:
                    for name in rc['cpa_radius_per_target']:
                        rc['cpa_radius_per_target'][name] *= 0.6
                    rc['alteration_min_angle'] = 0.0
                    rc['alteration_active'] = False
                elif retry_level == 3:
                    rc['tau_r_min'] = -max_yaw_moment
                    rc['tau_r_max'] = max_yaw_moment
                    rc['alteration_min_angle'] = 0.0
                    rc['alteration_active'] = False
                    for name in rc['cpa_radius_per_target']:
                        rc['cpa_radius_per_target'][name] = max(
                            12.0, rc['cpa_radius_per_target'][name] * 0.5)

                t_retry = time.perf_counter()
                rr = self.solver.solve(
                    x0=x0, x_ref=x_ref, target_trajs=target_trajs,
                    constraints=rc, tau_env=np.zeros(3), warm_start=False,
                )
                rr_elapsed = (time.perf_counter() - t_retry) * 1000.0
                rr['solve_time_ms'] = rr_elapsed
                rr['wp_idx'] = wp_idx
                rr['retry_level'] = retry_level
                if rr['status'] == 'SOLVED':
                    result = rr
                    self.get_logger().info(
                        f'NMPC retry L{retry_level} solved ({rr_elapsed:.0f}ms)',
                        throttle_duration_sec=5.0)

            if result['retry_level'] > 0 and result['status'] != 'SOLVED':
                result['solve_time_ms'] = (time.perf_counter() - t_start) * 1000.0

            return result
        except Exception as e:
            self.get_logger().error(
                f'NMPC solve thread crashed: {e}', throttle_duration_sec=5.0)
            return {
                'status': 'INFEASIBLE', 'u_opt': np.zeros((2, 1)),
                'x_pred': np.tile(x0.reshape(-1, 1),
                                  (1, self.solver.p.N + 1)) if 'x0' in dir() else np.zeros((6, self.solver.p.N + 1)),
                'cost': float('inf'), 'solve_time_ms': 0.0,
                'epsilon_legal': 0.0, 'epsilon_smooth': 0.0,
                'epsilon_speed': 0.0, 'wp_idx': wp_idx,
                'retry_level': 0,
            }

    def _process_solve_result(self, result: dict):
        """Handle a completed solve result (called from main thread).

        Manages degradation FSM transitions:
          - SOLVED: increment consecutive_solved, check auto-recovery
          - INFEASIBLE: increment consecutive_infeasible, check escalation
        """
        self.solve_count += 1
        self.solve_times_ms.append(result['solve_time_ms'])
        if len(self.solve_times_ms) > 50:
            self.solve_times_ms.pop(0)

        if result['status'] == 'SOLVED':
            self.consecutive_infeasible = 0
            self.consecutive_solved += 1
            self._last_tau_u = float(result['u_opt'][0, 0])
            self._last_tau_r = float(result['u_opt'][1, 0])
            self._has_valid_output = True
            # Waypoint advance check
            self._check_waypoint_advance()

            # ── Section 4.5 Robustness Verification ──
            self._run_robustness_verification(result)

            # ── Auto-recovery check ──
            self._check_recovery()
        else:
            self.consecutive_solved = 0
            self.consecutive_infeasible += 1
            self.infeasible_count += 1
            self.get_logger().warn(
                f'NMPC infeasible (#{self.infeasible_count}, '
                f'consec={self.consecutive_infeasible}) — '
                f'{result["solve_time_ms"]:.0f}ms',
                throttle_duration_sec=2.0)
            if self.consecutive_infeasible >= self.MAX_INFEASIBLE_BEFORE_DEGRADE:
                self._escalate_degradation('consecutive_infeasible')
            # Keep last valid output as fallback

    def _run_robustness_verification(self, result: dict):
        """Execute Section 4.5 robustness verification pass (Theorem 1 + Lemma 2).

        Called after each successful NMPC solve. Updates:
          - RCI linearization (periodic)
          - Delay estimator
          - Safety tube radius
          - Constraint tightening recommendations
        """
        if self.own_state is None:
            return

        try:
            p = self.solver.p
            x_current = self.own_state.copy()
            u_prev = np.array([self._last_tau_u, self._last_tau_r])
            solve_time_s = result['solve_time_ms'] / 1000.0

            # Periodic RCI linearization update
            self._rci_update_counter += 1
            if self._rci_update_counter >= self._RCI_UPDATE_INTERVAL:
                self._rci_update_counter = 0
                # Use the state-space linearization from the ship model
                A_disc, B_disc = self._compute_nmpc_linearization()
                self._robustness.update_linearization(A_disc, B_disc)

            # Full robustness verification
            M_inv = np.linalg.inv(p.M_matrix)
            self._last_robustness_report = self._robustness.verify(
                x_current=x_current,
                u_prev=u_prev,
                x_pred=result['x_pred'],
                solve_time_s=solve_time_s,
                M_inv=M_inv,
                p=p,
                ts_states=self.target_states,
                base_cpa=self.colregs_constraints.get('cpa_radius_per_target', {})
                if self.colregs_constraints else {},
            )

            # Log warnings if safety is compromised
            if not self._last_robustness_report['is_safe']:
                rci_v = self._last_robustness_report['rci_violations']
                tube_s = self._last_robustness_report['tube_safe']
                self.get_logger().warn(
                    f'⚠️ Robustness: RCI violations={rci_v}, tube_safe={tube_s}, '
                    f"τ̂_max={self._last_robustness_report['tau_hat_ms']:.0f}ms",
                    throttle_duration_sec=3.0)
        except Exception as e:
            self.get_logger().error(
                f'Robustness verification failed: {e}', throttle_duration_sec=5.0)

    def _compute_nmpc_linearization(self):
        """Compute linearized discrete-time dynamics around current state.

        Uses Fossen 3DOF state-space linearization (ship_dynamics.py) with
        NMPC's discretization timestep.

        Returns:
          A_disc: (6,6) discrete-time system matrix at dt=0.5s
          B_disc: (6,2) discrete-time input matrix (τ_u, τ_r)
        """
        if self.own_state is None:
            return np.eye(6), np.zeros((6, 2))

        try:
            from .ship_dynamics import FossenShip, ShipParams
        except ImportError:
            from ship_dynamics import FossenShip, ShipParams
        sp = ShipParams()
        ship = FossenShip(params=sp, dt=self.solver.p.dt)
        eta = self.own_state[:3].copy()
        nu = self.own_state[3:].copy()
        ship.set_state(eta, nu)

        A_disc, B_disc_full = ship.state_space_matrices(nu)
        # B_disc_full is (6,2) for [τ_u, τ_r]
        return A_disc, B_disc_full

    # =====================================================================
    # Simple fallback control (used before first solve completes)
    # =====================================================================

    def _initial_fallback_thrust(self) -> Tuple[float, float]:
        """Simple LOS heading toward first waypoint — used before first NMPC solve.

        Provides aggressive thrust at low speeds to overcome VRX hydrodynamic drag
        and get the WAM-V moving. When COLREGS requires starboard turn, applies
        starboard bias to the reference heading.

        Thrust mapping: positive diff → more left thrust → starboard turn (CW in ENU).
        """
        if self.own_state is None or len(self.waypoints) == 0:
            return 0.0, 0.0

        surge = float(self.own_state[3])
        os_heading = float(self.own_state[2])

        # ── Reference heading: waypoint (with starboard bias if COLREGS requires) ──
        wp = self.waypoints[min(self._wp_idx, len(self.waypoints) - 1)]
        dx = wp[0] - self.own_state[0]
        dy = wp[1] - self.own_state[1]
        target_heading = math.atan2(dy, dx)

        # Starboard bias: when referee says alter_to_starboard, bias heading starboard
        if self._need_starboard_turn:
            starboard_bias = math.radians(30)
            target_heading = self._norm_angle(target_heading - starboard_bias)

        heading_error = self._norm_angle(target_heading - os_heading)

        # ── Speed-dependent base thrust ──
        if surge < 0.3:
            base = 1200.0
        elif surge < 0.8:
            base = 1000.0
        else:
            base = 800.0

        # ── Heading correction: positive diff → more left thrust → starboard turn ──
        diff = 600.0 * heading_error - 100.0 * float(self.own_state[5])
        diff = max(-base * 0.7, min(base * 0.7, diff))

        left = max(0.0, min(1500.0, base + diff))
        right = max(0.0, min(1500.0, base - diff))

        if surge < 0.3:
            left = max(left, 800.0)
            right = max(right, 800.0)

        return left, right

    @staticmethod
    def _norm_angle(a): return (a + math.pi) % (2 * math.pi) - math.pi

    def _l3_pd_avoidance(self) -> Tuple[float, float]:
        """Level 3 emergency PD controller: starboard-only turn + speed maintain.

        Mirrors batch_runner.py Phase 1 (starboard PD) — NOT zero thrust.
        Zero thrust in a collision scenario is dangerous: the boat drifts into
        the target. Instead, always turn starboard (COLREGS-safe default) and
        maintain forward speed.
        """
        if self.own_state is None:
            return 300.0, 300.0

        surge = float(self.own_state[3])
        r_yaw = float(self.own_state[5])

        # ── Speed control: maintain ~1.0 m/s ──
        target_speed = 1.0
        speed_error = target_speed - surge
        base_thrust = 800.0 + 500.0 * speed_error
        base_thrust = max(300.0, min(1500.0, base_thrust))

        # ── Starboard-only rudder: proportional + damping ──
        # Target: -40° cumulative starboard turn
        # In ENU/Fossen: starboard = negative τ_r = CW rotation
        Kp_turn = 25.0   # P gain for starboard turn
        Kd_yaw = 40.0    # D gain for yaw rate damping
        target_turn_rate = -0.15  # rad/s starboard (CW)

        turn_error = target_turn_rate - r_yaw
        tau_r_raw = Kp_turn * turn_error - Kd_yaw * r_yaw
        # Clamp to starboard-only [-60, 0] Nm
        tau_r = max(-60.0, min(0.0, tau_r_raw))

        # ── Thrust mapping ──
        d = 2.06
        max_t = self.get_parameter('max_thrust').value
        T_left = max(0.0, min(max_t, (base_thrust - 2.0 * tau_r / d) / 2.0))
        T_right = max(0.0, min(max_t, (base_thrust + 2.0 * tau_r / d) / 2.0))

        return T_left, T_right

    # =====================================================================
    # COLREGS parsing, trajectory prediction, constraints, etc.
    # =====================================================================

    def _parse_colregs_json(self, data: dict) -> dict:
        constraints = {
            'tau_r_min': -800.0, 'tau_r_max': 800.0,
            'alteration_min_angle': 0.0, 'alteration_active': False,
            'v_min': 0.5, 'v_max': 5.0, 'cpa_radius_per_target': {},
            'pass_astern_per_target': {},  # for half-plane normal computation
        }
        rudder_bounds = data.get('rudder_bounds', [-0.5, 0.5])
        # SIGN FLIP: mapper +rudder=starboard → Fossen -τ_r=starboard
        # (mirrors batch_runner.py B3 fix)
        constraints['tau_r_min'] = -(rudder_bounds[1] * 800.0)
        constraints['tau_r_max'] = -(rudder_bounds[0] * 800.0)
        constraints['alteration_min_angle'] = math.radians(
            data.get('alteration_min_angle_deg', 0))
        constraints['alteration_active'] = abs(data.get('turn_direction_sign', 0)) > 0
        speed_bounds = data.get('speed_bounds', [0.5, 5.0])
        constraints['v_min'] = speed_bounds[0]
        constraints['v_max'] = speed_bounds[1]
        cpa_dists = data.get('min_cpa_distances', [])
        ts_names = list(self.target_states.keys())
        for i, cpa in enumerate(cpa_dists):
            if i < len(ts_names):
                constraints['cpa_radius_per_target'][ts_names[i]] = cpa
        # Parse per-target spatial constraints (pass_astern for half-plane normals)
        for st in data.get('spatial_targets', []):
            name = st.get('target_name', '')
            if name:
                constraints['pass_astern_per_target'][name] = st.get('pass_astern', None)
        return constraints

    def _predict_target_trajectories(self) -> Dict[str, np.ndarray]:
        """Predict TS trajectories with uncertainty propagation (Section 4.5.3).

        Uses the UncertaintyPropagator to produce mean trajectories.
        Also updates uncertainty state for constraint tightening.
        """
        N, dt = self.solver._N, self.solver.p.dt
        trajectories = {}
        now = time.time()
        for name, state in self.target_states.items():
            if now - self.target_last_seen.get(name, 0) > 10.0:
                continue
            px, py, psi, u_body, v_body = state
            c, s = math.cos(psi), math.sin(psi)
            vx_w, vy_w = c * u_body - s * v_body, s * u_body + c * v_body

            # Update uncertainty state
            self._robustness.uncertainty.update_target(name, state)

            # Predict mean trajectory (uses uncertainty-augmented state internally)
            pred = self._robustness.uncertainty.predict_with_uncertainty(
                name, state, N, dt, confidence=2.0
            )
            trajectories[name] = pred['mu']

            # Store enlarged CPA recommendation for constraint tightening
            if not hasattr(self, '_uncertainty_cpa'):
                self._uncertainty_cpa = {}
            self._uncertainty_cpa[name] = float(np.max(pred['enlarged_radius']))

        return trajectories

    def _build_active_constraints(self) -> dict:
        constraints = {
            'tau_r_min': -800.0, 'tau_r_max': 800.0,
            'alteration_min_angle': 0.0, 'alteration_active': False,
            'v_min': 0.5, 'v_max': 5.0, 'cpa_radius_per_target': {},
            'hp_normals_per_target': {},
        }
        if self.colregs_constraints and time.time() - self.last_constraints_time < 30.0:
            constraints.update(self.colregs_constraints)
        for name in self.target_states:
            if name not in constraints['cpa_radius_per_target']:
                constraints['cpa_radius_per_target'][name] = 50.0

        # ── Section 4.5 Constraint Tightening ──
        # Apply Lemma 2: enlarge CPA by tube_radius + uncertainty_margin.
        # CAP the tightening at +50% of base CPA to avoid pushing CPA beyond
        # current ship-to-ship distance (which guarantees infeasibility).
        tube_radius = self._robustness.safety_tube.tube_radius
        uncertainty_cpa = getattr(self, '_uncertainty_cpa', {})

        for name in constraints['cpa_radius_per_target']:
            base_cpa = constraints['cpa_radius_per_target'][name]
            uncertainty_margin = uncertainty_cpa.get(name, 0.0)
            robust_cpa = compute_robust_constraints(
                base_cpa, tube_radius, uncertainty_margin, safety_factor=1.2
            )
            # Cap: never exceed base_cpa * 1.5 (avoid infeasibility from over-tightening)
            constraints['cpa_radius_per_target'][name] = min(robust_cpa, base_cpa * 1.5)

        # ── Adaptive CPA: scale with current distance (B4 fix) ──
        # Referee's full CPA (e.g. 61m) is infeasible when targets are closer.
        # Scale down to max(10, min(full, dist * 0.82)) so NMPC can converge.
        # Floor lowered from 20m→10m for close-quarters Gazebo scenarios.
        if self.own_state is not None:
            os_pos = self.own_state[:2]
            for name in list(constraints['cpa_radius_per_target'].keys()):
                if name in self.target_states:
                    ts_pos = self.target_states[name][:2]
                    current_dist = float(np.linalg.norm(ts_pos - os_pos))
                    full_cpa = constraints['cpa_radius_per_target'][name]
                    constraints['cpa_radius_per_target'][name] = max(
                        10.0, min(full_cpa, current_dist * 0.82))

        # ── Half-plane normals (Plan C, mirrors batch_runner.py) ──
        # COLREGS-aware half-plane normals replace non-convex exclusion circles
        # with convex linear constraints n̂·(p_OS - p_TS) >= r_hp.
        # Uses pass_astern from referee to orient the half-plane.
        if self.own_state is not None:
            os_pos = self.own_state[:2]
            pass_astern_map = constraints.get('pass_astern_per_target', {})
            for name, ts_state in self.target_states.items():
                ts_pos = ts_state[:2]
                ts_heading = float(ts_state[2])
                rel_vec = os_pos - ts_pos  # TS → OS direction
                dist_ts = float(np.linalg.norm(rel_vec))
                n_rel = rel_vec / max(dist_ts, 1e-6) if dist_ts > 1e-6 else np.array([1.0, 0.0])

                pass_astern = pass_astern_map.get(name, None)
                if pass_astern is True:
                    ts_h_vec = np.array([math.cos(ts_heading), math.sin(ts_heading)])
                    n_stern = -ts_h_vec  # normal pointing to TS stern
                    proj_stern = float(np.dot(n_stern, -rel_vec))
                    r_cpa = constraints['cpa_radius_per_target'].get(name, 20.0)
                    if proj_stern > r_cpa * 0.3:
                        blend = min(0.6, proj_stern / (r_cpa * 2 + 1.0))
                        n_blended = (1 - blend) * n_stern + blend * n_rel
                        n_blended /= max(np.linalg.norm(n_blended), 1e-6)
                        constraints['hp_normals_per_target'][name] = n_blended
                    elif proj_stern > -r_cpa * 0.5:
                        constraints['hp_normals_per_target'][name] = n_stern
                    else:
                        constraints['hp_normals_per_target'][name] = n_rel
                else:
                    constraints['hp_normals_per_target'][name] = n_rel

        return constraints

    def _apply_degradation(self, constraints: dict) -> dict:
        if self.degradation_level == 0:
            return constraints
        elif self.degradation_level == 1:
            constraints['tau_r_min'] = -800.0
            constraints['tau_r_max'] = 800.0
            constraints['alteration_active'] = False
        elif self.degradation_level >= 2:
            constraints['tau_r_min'] = -800.0
            constraints['tau_r_max'] = 800.0
            constraints['alteration_active'] = False
            constraints['v_max'] = min(constraints['v_max'], 2.5)
            for name in constraints['cpa_radius_per_target']:
                constraints['cpa_radius_per_target'][name] = max(
                    constraints['cpa_radius_per_target'].get(name, 50.0), 40.0)
        if self.degradation_level >= 3:
            constraints['v_max'] = 0.5
        return constraints

    # =====================================================================
    # Degradation FSM (Section 4.6) — escalation, recovery, logging
    # =====================================================================

    def _escalate_degradation(self, trigger_reason: str = 'unknown'):
        """Escalate degradation by one level. Logs the transition.

        Args:
            trigger_reason: human-readable reason for escalation
              ('consecutive_infeasible', 'solver_timeout', 'referee_signal',
               'communication_timeout', 'cfg_validation_failure')
        """
        if self.degradation_level >= 3:
            return
        old_level = self.degradation_level
        self.degradation_level += 1
        self._log_degradation_event(old_level, self.degradation_level, trigger_reason)
        self.get_logger().error(
            f'⚠️ Degradation L{old_level}→L{self.degradation_level}: {trigger_reason} '
            f'(infeasible={self.consecutive_infeasible})')

    def _check_recovery(self):
        """Check if conditions allow recovery to a lower degradation level.

        Conditions for recovery (all must be met):
          1. degradation_level > 0 (not already normal)
          2. consecutive_solved >= MIN_SOLVED_BEFORE_RECOVERY (stable solving)
          3. Risk field is low (CPA large enough or referee reports low risk)

        Recovery is one level at a time (conservative).
        """
        if self.degradation_level <= 0:
            return
        if self.consecutive_solved < self.MIN_SOLVED_BEFORE_RECOVERY:
            return

        # Check risk field: if we have target ships, check CPA distances
        risk_low = self._is_risk_low()

        if risk_low:
            old_level = self.degradation_level
            self.degradation_level -= 1
            self.consecutive_solved = 0  # reset counter after recovery
            self._log_degradation_event(old_level, self.degradation_level,
                                        'auto_recovery')
            self.get_logger().info(
                f'✅ Auto-recovery L{old_level}→L{self.degradation_level}: '
                f'{self.MIN_SOLVED_BEFORE_RECOVERY} consecutive SOLVED + low risk')

    def _is_risk_low(self) -> bool:
        """Check if the current collision risk is low enough for recovery.

        Uses:
          - Minimum CPA distance to any target ship
          - Referee degradation level (if available)
        Returns True if risk is low enough for recovery.
        """
        # If no target ships, always low risk
        if not self.target_states or self.own_state is None:
            return True

        # Check minimum distance to any target ship
        min_dist = float('inf')
        ox, oy = self.own_state[0], self.own_state[1]
        for name, state in self.target_states.items():
            d = math.hypot(state[0] - ox, state[1] - oy)
            if d < min_dist:
                min_dist = d

        # Recovery threshold: all target ships > 50m away
        # OR CPA-based risk is demonstrably low
        if min_dist > 50.0:
            return True

        # Check against CPA constraints if available
        if self.colregs_constraints:
            cpa_map = self.colregs_constraints.get('cpa_radius_per_target', {})
            for name in self.target_states:
                cpa_req = cpa_map.get(name, 50.0)
                state = self.target_states[name]
                d = math.hypot(state[0] - ox, state[1] - oy)
                if d < cpa_req * 0.5:  # well within danger zone
                    return False

        # Conservative: if unsure, don't recover
        return min_dist > 100.0

    def _log_degradation_event(self, from_level: int, to_level: int,
                                trigger_reason: str):
        """Record a degradation state transition with timestamp.

        Args:
            from_level: previous degradation level (0-3)
            to_level: new degradation level (0-3)
            trigger_reason: human-readable cause of transition
        """
        import datetime
        event = {
            'timestamp': datetime.datetime.now().isoformat(),
            'unix_time': time.time(),
            'level_before': from_level,
            'level_after': to_level,
            'trigger_reason': trigger_reason,
            'consecutive_infeasible': self.consecutive_infeasible,
            'consecutive_solved': self.consecutive_solved,
            'solve_count': self.solve_count,
        }
        self._degradation_history.append(event)
        # Keep last 100 events max
        if len(self._degradation_history) > 100:
            self._degradation_history = self._degradation_history[-100:]
        # Also log to ROS logger for immediate visibility
        direction = '⬆' if to_level > from_level else '⬇'
        self.get_logger().warn(
            f'📋 Degradation Event {direction} L{from_level}→L{to_level}: '
            f'{trigger_reason} (event #{len(self._degradation_history)})')

    def _thrust_mapping(self, tau_u: float, tau_r: float) -> Tuple[float, float]:
        d = 2.06
        max_t = self.get_parameter('max_thrust').value
        T_left = max(0.0, min(max_t, (tau_u - 2.0 * tau_r / d) / 2.0))
        T_right = max(0.0, min(max_t, (tau_u + 2.0 * tau_r / d) / 2.0))
        return T_left, T_right

    def _check_waypoint_advance(self):
        if self._wp_idx >= len(self.waypoints) - 1 or self.own_state is None:
            return
        wp_next = self.waypoints[self._wp_idx + 1]
        dist = math.sqrt((self.own_state[0] - wp_next[0])**2 +
                         (self.own_state[1] - wp_next[1])**2)
        if dist < 5.0:
            old = self._wp_idx
            self._wp_idx = min(self._wp_idx + 1, len(self.waypoints) - 1)
            if self._wp_idx != old:
                self.get_logger().info(
                    f'📍 WP{old+1}→WP{self._wp_idx+1}: '
                    f'({self.waypoints[self._wp_idx][0]:.0f},'
                    f'{self.waypoints[self._wp_idx][1]:.0f})')
        if self._wp_idx >= len(self.waypoints) - 1:
            wp_f = self.waypoints[-1]
            if math.sqrt((self.own_state[0]-wp_f[0])**2 + (self.own_state[1]-wp_f[1])**2) < 3.0:
                if not self._mission_complete:
                    self._mission_complete = True
                    self.get_logger().info('✅ Mission complete!')

    def _publish_thrust(self, left: float, right: float):
        self.left_pub.publish(Float64(data=float(left)))
        self.right_pub.publish(Float64(data=float(right)))

    def _publish_status(self):
        try:
            avg = (sum(self.solve_times_ms) / len(self.solve_times_ms)
                   if self.solve_times_ms else 0)
            status = {
                'solve_count': self.solve_count,
                'infeasible_count': self.infeasible_count,
                'degradation_level': self.degradation_level,
                'consecutive_infeasible': self.consecutive_infeasible,
                'consecutive_solved': self.consecutive_solved,
                'degradation_events': len(self._degradation_history),
                'last_degradation': (self._degradation_history[-1]
                                     if self._degradation_history else None),
                'wp_idx': self._wp_idx, 'wp_total': len(self.waypoints),
                'mission_complete': self._mission_complete,
                'avg_solve_ms': round(avg, 1),
                'has_odometry': self.own_state is not None,
                'has_valid_output': self._has_valid_output,
                'pending_solve': self._pending_future is not None,
            }
            if self.own_state is not None:
                status['state'] = {
                    'x': round(float(self.own_state[0]), 1),
                    'y': round(float(self.own_state[1]), 1),
                    'psi_deg': round(math.degrees(self.own_state[2]), 1),
                    'u_mps': round(float(self.own_state[3]), 2),
                }
                status['n_targets'] = len(self.target_states)

            # ── Section 4.5 Robustness telemetry ──
            if self._last_robustness_report:
                status['robustness'] = {
                    'is_safe': self._last_robustness_report['is_safe'],
                    'tau_max_hat_ms': self._last_robustness_report['tau_hat_ms'],
                    'delay_stats': self._last_robustness_report['delay_stats'],
                    'tube_radius_m': self._last_robustness_report['tube_radius_m'],
                    'tube_safe': self._last_robustness_report['tube_safe'],
                    'rci_diameter_m': self._last_robustness_report.get('rci_diameter_m', 0),
                    'rci_violations': self._last_robustness_report['rci_violations'],
                    'enlarged_cpa': self._last_robustness_report['enlarged_cpa'],
                }

            self.status_pub.publish(String(data=json.dumps(_json_safe(status))))
            self.get_logger().info(
                f'📊 NMPC #{self.solve_count}: avg={avg:.0f}ms, '
                f'infeas={self.infeasible_count}, degrad={self.degradation_level}, '
                f'odom={"✓" if self.own_state is not None else "✗"}, '
                f'valid={self._has_valid_output}')
        except Exception as e:
            self.get_logger().error(
                f'_publish_status crashed: {e}', throttle_duration_sec=5.0)


def main():
    rclpy.init()
    node = NMPCControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
