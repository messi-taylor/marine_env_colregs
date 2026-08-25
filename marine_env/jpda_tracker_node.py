#!/usr/bin/env python3
"""
JPDA Multi-Target Tracking Node (Phase 1-2)
=============================================

Joint Probabilistic Data Association tracker for maritime target ships.

Algorithm (Bar-Shalom & Fortmann, 1988):
  1. Predict all tracks to current time via constant-velocity KF
  2. Gate incoming measurements with Mahalanobis distance (chi² test)
  3. Form validation matrix Ω(j,t) — which measurements could belong to which tracks
  4. Enumerate feasible joint association events (Murty's heuristic for >4 targets)
  5. Compute marginal association probabilities β(j,t)
  6. Kalman update each track with probability-weighted innovation
  7. Track lifecycle: tentative → confirmed → coasting → deleted

Subscribes:
  - /model/{name}/odometry  (target ship ground truth, with simulated noise added)

Publishes:
  - /tracked_targets          (TrackedTargetArray — custom message as JSON)
  - /tracked_targets/poses    (PoseStamped per confirmed track — for RViz)

Parameters:
  - target_names: list of model names to track
  - measurement_noise_std: simulated radar position noise (m)
  - detection_probability: P_D — probability of detecting a target
  - gate_probability: P_G — gating probability for chi² threshold
  - track_confirmation_threshold: hits needed to confirm a track
  - track_deletion_threshold: consecutive misses before deletion
  - publish_rate: Hz
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import String, Header
import numpy as np
import math
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.stats import chi2


# =============================================================================
# Kalman Filter (Constant Velocity, 4-state: x, y, vx, vy)
# =============================================================================

class ConstantVelocityKF:
    """Linear Kalman filter with constant-velocity motion model.

    State:  [x, y, vx, vy]ᵀ
    Measurement: [x, y]ᵀ  (position only)
    """

    def __init__(self, dt: float = 1.0,
                 process_noise_q: float = 0.01,
                 measurement_noise_r: float = 1.0):
        self.dt = dt

        # State transition: F = [I₂  dt·I₂; 0  I₂]
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        # Measurement matrix: H = [I₂  0]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        # Process noise covariance (discrete white noise acceleration)
        q = process_noise_q
        self.Q = q * np.array([
            [dt**3/3, 0,       dt**2/2, 0],
            [0,       dt**3/3, 0,       dt**2/2],
            [dt**2/2, 0,       dt,      0],
            [0,       dt**2/2, 0,       dt],
        ])

        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise_r

        self.x = np.zeros(4)       # state estimate
        self.P = np.eye(4) * 100.  # covariance (large initial uncertainty)

    def predict(self):
        """Predict state to current time."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def innovation(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute innovation ν = z - Hx and innovation covariance S.

        Returns:
            nu: innovation vector (2,)
            S: innovation covariance (2,2)
            nis: normalized innovation squared (Mahalanobis distance²)
        """
        nu = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        nis = float(nu.T @ np.linalg.inv(S) @ nu)
        return nu, S, nis

    def update(self, z: np.ndarray):
        """Standard Kalman update with measurement z."""
        nu, S, _ = self.innovation(z)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.P = self.P - K @ self.H @ self.P

    def update_weighted(self, z: np.ndarray, beta: float):
        """JPDA weighted update: weight the innovation by association probability β.

        Uses the PDAF covariance inflation formula (Bar-Shalom):
          P(k|k) = P(k|k-1) - (1-β)·K·S·Kᵀ + β_correction
        """
        nu, S, _ = self.innovation(z)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Weighted state update
        self.x = self.x + beta * (K @ nu)

        # Covariance update with JPDA inflation
        # P̃ = K·S·Kᵀ (standard Kalman reduction)
        P_tilde = K @ S @ K.T
        # Inflation term: β·(1-β)·(K·ν)·(K·ν)ᵀ  (spread of innovations)
        innovation_spread = beta * (1.0 - beta) * np.outer(K @ nu, K @ nu)

        self.P = self.P - (1.0 - beta) * P_tilde + innovation_spread

    def get_state(self) -> np.ndarray:
        return self.x.copy()

    def get_position(self) -> np.ndarray:
        return self.x[:2]

    def get_velocity(self) -> np.ndarray:
        return self.x[2:4]

    def clone(self) -> 'ConstantVelocityKF':
        """Deep copy for track initialization."""
        kf = ConstantVelocityKF(dt=self.dt)
        kf.F = self.F.copy()
        kf.H = self.H.copy()
        kf.Q = self.Q.copy()
        kf.R = self.R.copy()
        kf.x = self.x.copy()
        kf.P = self.P.copy()
        return kf


# =============================================================================
# Track data structure
# =============================================================================

@dataclass
class Track:
    """Single target track with Kalman filter and lifecycle state."""
    track_id: int
    kf: ConstantVelocityKF
    status: str = "tentative"      # tentative → confirmed → coasting → deleted
    hits: int = 0                  # consecutive detections (for confirmation)
    misses: int = 0                # consecutive misses (for deletion)
    age: int = 0                   # total prediction steps
    name: str = ""                 # associated ship name (if known)

    # Association result for current step
    beta_sum: float = 0.0          # sum of association probabilities (1.0 = well-tracked)
    associated_measurement: Optional[np.ndarray] = None

    HISTORY_LENGTH = 30

    def predict(self):
        self.kf.predict()
        self.age += 1

    def update_weighted(self, z: np.ndarray, beta: float):
        self.kf.update_weighted(z, beta)
        self.beta_sum += beta

    def update_single(self, z: np.ndarray):
        """Standard Kalman update (single unambiguous measurement)."""
        self.kf.update(z)
        self.beta_sum = 1.0


# =============================================================================
# JPDA Engine
# =============================================================================

class JPDAEngine:
    """Joint Probabilistic Data Association for multi-target tracking.

    Given M measurements and N tracks, JPDA computes the probability β(j,t)
    that measurement j originated from track t, considering all feasible
    joint association events.

    For maritime use with typically ≤5 targets, we use exact enumeration
    of feasible joint events. Above ~8 tracks, Murty's k-best approximation
    should be used instead.
    """

    def __init__(self,
                 detection_prob: float = 0.95,
                 gate_prob: float = 0.99,
                 clutter_density: float = 1e-6):
        """
        Args:
            detection_prob: P_D — probability of detecting a target
            gate_prob: P_G — probability that true measurement falls in gate
            clutter_density: λ — Poisson clutter spatial density (per m²)
        """
        self.P_D = detection_prob
        self.P_G = gate_prob

        # Chi² threshold for 2D position measurements
        # gate_prob=0.99 → chi²_2(0.99) ≈ 9.21
        self.gate_threshold = chi2.ppf(gate_prob, df=2)

        # Clutter density per validation region area
        self.clutter_density = clutter_density

    def gate(self, track: Track, measurements: List[np.ndarray]) -> List[int]:
        """Return indices of measurements that fall within track's validation gate."""
        gated = []
        for j, z in enumerate(measurements):
            _, _, nis = track.kf.innovation(z)
            if nis <= self.gate_threshold:
                gated.append(j)
        return gated

    def associate(self, tracks: List[Track],
                  measurements: List[np.ndarray]) -> Dict[int, Dict[int, float]]:
        """Run JPDA association.

        Returns:
            beta: dict[track_idx][meas_idx] = association probability
                  plus beta[track_idx][-1] = probability of no detection
        """
        M = len(measurements)
        N = len(tracks)

        if N == 0:
            return {}

        # ── Step 1: Build validation matrix Ω ──
        # Ω[j][t] = True if measurement j is in track t's gate
        omega = np.zeros((M, N), dtype=bool)
        for t, track in enumerate(tracks):
            gated = self.gate(track, measurements)
            for j in gated:
                omega[j, t] = True

        # ── Step 2: Enumerate feasible joint events ──
        # Each event is a mapping: track → measurement index (or -1 for no detection)
        feasible_events = self._enumerate_feasible_events(omega)

        if not feasible_events:
            # No feasible events → all measurements are clutter, all tracks undetected
            return {t: {-1: 1.0} for t in range(N)}

        # ── Step 3: Compute event probabilities ──
        event_probs = []
        total = 0.0
        for event in feasible_events:
            prob = self._event_probability(event, tracks, measurements, omega)
            event_probs.append(prob)
            total += prob

        # Normalize
        if total > 0:
            event_probs = [p / total for p in event_probs]

        # ── Step 4: Compute marginal association probabilities β(j,t) ──
        beta = {t: {} for t in range(N)}
        for t in range(N):
            beta[t][-1] = 0.0  # probability of no measurement for this track

        for event, prob in zip(feasible_events, event_probs):
            for t, j in event.items():
                if j not in beta[t]:
                    beta[t][j] = 0.0
                beta[t][j] += prob

        return beta

    def _enumerate_feasible_events(self, omega: np.ndarray) -> List[Dict[int, int]]:
        """Enumerate all feasible joint association events.

        Feasibility constraints:
          - Each measurement assigned to at most 1 track
          - Each track assigned at most 1 measurement
          - Assignment only allowed if ω(j,t) = True

        Uses depth-first enumeration. For M,N ≤ 6 this is fast.
        """
        M, N = omega.shape

        # Build per-track list of valid measurement indices
        valid_meas = [set(np.where(omega[:, t])[0]) for t in range(N)]

        events = []

        def dfs(t: int, assigned_meas: set, current: Dict[int, int]):
            if t == N:
                events.append(current.copy())
                return

            # Option 1: Track t has no detection
            current[t] = -1
            dfs(t + 1, assigned_meas, current)
            del current[t]

            # Option 2: Track t assigned to an available measurement in its gate
            for j in valid_meas[t]:
                if j not in assigned_meas:
                    current[t] = j
                    assigned_meas.add(j)
                    dfs(t + 1, assigned_meas, current)
                    assigned_meas.discard(j)
                    del current[t]

        dfs(0, set(), {})
        return events

    def _event_probability(self, event: Dict[int, int],
                           tracks: List[Track],
                           measurements: List[np.ndarray],
                           omega: np.ndarray) -> float:
        """Compute probability of a single joint association event.

        P(event) ∝ Π_t [P_D · p(z_j|t)]^{δ_t} · [1-P_D]^{1-δ_t} · Π_j λ^{1-τ_j}

        where δ_t = 1 if track t is detected, τ_j = 1 if measurement j is assigned.
        """
        prob = 1.0
        M = len(measurements)

        assigned_meas = set(j for j in event.values() if j >= 0)

        for t, j in event.items():
            track = tracks[t]
            if j >= 0:
                # Track detected, measurement j associated
                _, S, nis = track.kf.innovation(measurements[j])
                # Gaussian likelihood
                det_S = np.linalg.det(S)
                if det_S <= 0:
                    det_S = 1e-10
                likelihood = (np.exp(-0.5 * nis) /
                              (2 * np.pi * math.sqrt(det_S)))
                prob *= self.P_D * likelihood
            else:
                # Track not detected
                prob *= (1.0 - self.P_D * self.P_G)

        # Clutter factor: unassociated measurements
        n_unassigned = M - len(assigned_meas)
        prob *= (self.clutter_density ** n_unassigned)

        return max(prob, 1e-30)


# =============================================================================
# ROS 2 JPDA Tracker Node
# =============================================================================

class JPDATrackerNode(Node):
    """ROS 2 node wrapping the JPDA multi-target tracker."""

    def __init__(self):
        super().__init__('jpda_tracker')

        # ── Parameters ──
        self.declare_parameter('target_names', ['ts01', 'ts02a', 'ts02b'])
        self.declare_parameter('measurement_noise_std', 0.5)      # m
        self.declare_parameter('detection_probability', 0.95)
        self.declare_parameter('gate_probability', 0.99)
        self.declare_parameter('track_confirmation_threshold', 3)
        self.declare_parameter('track_deletion_threshold', 8)
        self.declare_parameter('publish_rate', 5.0)               # Hz
        self.declare_parameter('dt', 1.0)                         # prediction step

        # ── JPDA engine ──
        self.jpda = JPDAEngine(
            detection_prob=self.get_parameter('detection_probability').value,
            gate_prob=self.get_parameter('gate_probability').value,
        )

        # ── Track storage ──
        self.tracks: Dict[int, Track] = {}
        self._next_track_id = 1
        self._last_time = self.get_clock().now().nanoseconds * 1e-9

        # ── Raw measurement buffer ──
        self.measurement_buffer: List[np.ndarray] = []
        self.measurement_names: List[str] = []

        # ── Subscribers ──
        target_names = self.get_parameter('target_names').value
        for name in target_names:
            self.create_subscription(
                Odometry,
                f'/model/{name}/odometry',
                lambda msg, n=name: self._measurement_cb(n, msg),
                10)

        # ── Publishers ──
        self.tracks_pub = self.create_publisher(
            String, '/tracked_targets', 10)

        # Per-track pose publishers (for RViz)
        self.track_pose_pubs: Dict[int, any] = {}

        # ── Timer ──
        rate = max(self.get_parameter('publish_rate').value, 1.0)
        self.timer = self.create_timer(1.0 / rate, self._tracking_cycle)

        self.get_logger().info(
            f'🎯 JPDA Tracker ready: P_D={self.jpda.P_D}, '
            f'gate_chi2={self.jpda.gate_threshold:.1f}')

    # =====================================================================
    # Measurement callback
    # =====================================================================

    def _measurement_cb(self, name: str, msg: Odometry):
        """Buffer raw position measurements (with simulated noise).

        In simulation we have access to ground truth odometry. To simulate
        realistic sensor measurements, we add Gaussian noise.
        """
        noise_std = self.get_parameter('measurement_noise_std').value

        # Ground truth position
        x_gt = msg.pose.pose.position.x
        y_gt = msg.pose.pose.position.y

        # Add measurement noise
        x_meas = x_gt + np.random.normal(0, noise_std)
        y_meas = y_gt + np.random.normal(0, noise_std)

        self.measurement_buffer.append(np.array([x_meas, y_meas]))
        self.measurement_names.append(name)

    # =====================================================================
    # Tracking cycle
    # =====================================================================

    def _tracking_cycle(self):
        """Main JPDA tracking loop: predict → associate → update → manage."""
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = now - self._last_time
        self._last_time = now

        if dt <= 0:
            dt = self.get_parameter('dt').value

        # ── Step 1: Predict all tracks ──
        for track in list(self.tracks.values()):
            track.kf.dt = dt
            track.predict()
            track.beta_sum = 0.0

        # ── Step 2: Gate & associate ──
        measurements = self.measurement_buffer
        confirmed_tracks = [t for t in self.tracks.values()
                           if t.status in ('tentative', 'confirmed', 'coasting')]

        if confirmed_tracks and measurements:
            beta = self.jpda.associate(confirmed_tracks, measurements)

            # ── Step 3: Weighted update ──
            for t_idx, track in enumerate(confirmed_tracks):
                if t_idx in beta:
                    for j, prob in beta[t_idx].items():
                        if j >= 0 and prob > 0.001:
                            track.update_weighted(measurements[j], prob)

            # Track which measurements were associated
            associated_meas = set()
            for t_idx in range(len(confirmed_tracks)):
                if t_idx in beta:
                    for j in beta[t_idx]:
                        if j >= 0 and beta[t_idx].get(j, 0) > 0.3:
                            associated_meas.add(j)

            # Update track lifecycle
            for t_idx, track in enumerate(confirmed_tracks):
                if t_idx in beta:
                    total_beta = sum(p for j, p in beta[t_idx].items() if j >= 0)
                    if total_beta > 0.3:
                        track.hits += 1
                        track.misses = 0
                    else:
                        track.misses += 1
                else:
                    track.misses += 1

                # Lifecycle transitions
                if track.status == "tentative" and track.hits >= \
                   self.get_parameter('track_confirmation_threshold').value:
                    track.status = "confirmed"
                    self.get_logger().info(
                        f'✅ Track {track.track_id} confirmed ({track.hits} hits)')

                elif track.status == "confirmed" and track.misses >= 3:
                    track.status = "coasting"
                    self.get_logger().warn(
                        f'⚠️  Track {track.track_id} coasting ({track.misses} misses)')

            # ── Step 4: Initiate new tentative tracks for unassoc measurements ──
            unassigned = set(range(len(measurements))) - associated_meas
            for j in unassigned:
                self._create_tentative_track(measurements[j],
                                            self.measurement_names[j]
                                            if j < len(self.measurement_names) else "")

        # ── Step 5: Delete stale tracks ──
        deletion_threshold = self.get_parameter('track_deletion_threshold').value
        for t_id in list(self.tracks.keys()):
            track = self.tracks[t_id]
            if track.misses >= deletion_threshold:
                self.get_logger().info(
                    f'🗑️  Track {t_id} deleted ({track.misses} misses)')
                del self.tracks[t_id]
                if t_id in self.track_pose_pubs:
                    del self.track_pose_pubs[t_id]

        # ── Step 6: Publish ──
        self._publish_tracks()

        # Clear measurement buffer for next cycle
        self.measurement_buffer = []
        self.measurement_names = []

    def _create_tentative_track(self, pos: np.ndarray, name: str = ""):
        """Initialize a new tentative track from an unassociated measurement.

        Uses two-point differencing: if there's no velocity info, assume
        near-zero initial velocity.
        """
        dt = self.get_parameter('dt').value
        noise_std = self.get_parameter('measurement_noise_std').value

        kf = ConstantVelocityKF(
            dt=dt,
            process_noise_q=0.05,       # slightly higher for new tracks
            measurement_noise_r=noise_std**2,
        )
        kf.x = np.array([pos[0], pos[1], 0.0, 0.0])

        # Initial covariance: moderate position uncertainty, high velocity uncertainty
        kf.P = np.diag([noise_std**2, noise_std**2, 1.0, 1.0])

        track = Track(
            track_id=self._next_track_id,
            kf=kf,
            status="tentative",
            hits=1,
            name=name,
        )
        self.tracks[self._next_track_id] = track
        self._next_track_id += 1

        self.get_logger().debug(
            f'🆕 Tentative track {track.track_id} at ({pos[0]:.1f}, {pos[1]:.1f})')

    def _publish_tracks(self):
        """Publish all confirmed and coasting tracks as JSON."""
        track_list = []
        for track in self.tracks.values():
            if track.status in ('confirmed', 'coasting', 'tentative'):
                pos = track.kf.get_position()
                vel = track.kf.get_velocity()
                P_diag = np.diag(track.kf.P)
                track_list.append({
                    'track_id': track.track_id,
                    'name': track.name,
                    'status': track.status,
                    'position': {'x': float(pos[0]), 'y': float(pos[1])},
                    'velocity': {'vx': float(vel[0]), 'vy': float(vel[1])},
                    'speed': float(np.linalg.norm(vel)),
                    'heading': float(math.atan2(vel[1], vel[0])),
                    'covariance_diag': [float(v) for v in P_diag[:2]],
                    'hits': track.hits,
                    'misses': track.misses,
                    'age': track.age,
                })

        msg = String()
        msg.data = json.dumps({
            'timestamp': self.get_clock().now().nanoseconds * 1e-9,
            'num_tracks': len(track_list),
            'tracks': track_list,
        }, indent=2)
        self.tracks_pub.publish(msg)

        # Publish confirmed track poses
        for track in self.tracks.values():
            if track.status == 'confirmed':
                if track.track_id not in self.track_pose_pubs:
                    self.track_pose_pubs[track.track_id] = self.create_publisher(
                        PoseStamped,
                        f'/tracked_targets/pose_{track.track_id}',
                        10)

                pose = PoseStamped()
                pose.header = Header()
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.header.frame_id = 'map'
                pos = track.kf.get_position()
                vel = track.kf.get_velocity()
                pose.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=0.0)

                yaw = math.atan2(float(vel[1]), float(vel[0]))
                pose.pose.orientation = Quaternion(
                    x=0.0, y=0.0,
                    z=math.sin(yaw/2), w=math.cos(yaw/2))

                self.track_pose_pubs[track.track_id].publish(pose)


def main():
    rclpy.init()
    node = JPDATrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
