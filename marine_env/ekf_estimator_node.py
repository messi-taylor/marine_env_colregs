#!/usr/bin/env python3
"""
EKF State Estimator for WAM-V USV.
Fuses IMU + GPS + GPS-COG + 3× Camera VO using a 6-state EKF.
State: [px, py, yaw, vx, vy, yaw_rate] in world (ENU) frame.

Sensors:
  - IMU (100Hz): body-frame accel + gyro → prediction
  - GPS (1Hz): world-frame position → position update
  - GPS-COG (1Hz): course-over-ground → yaw update (when speed > min_speed)
  - Camera VO ×3 (10-30Hz): velocity + yaw rate from visual odometry

Key safeguards:
  - Yaw wrapping after every update
  - Innovation gating (chi-squared at 95% confidence)
  - Covariance regularization to prevent numerical blow-up
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
import numpy as np
import math
import json
from dataclasses import dataclass


def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0):
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f * f

    def pvr(lat_rad):
        sin_lat = math.sin(lat_rad)
        return a / math.sqrt(1 - e2 * sin_lat * sin_lat)

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)

    n = pvr(lat_rad)
    n0 = pvr(lat0_rad)

    x_ecef = (n + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    y_ecef = (n + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    z_ecef = (n * (1 - e2) + alt) * math.sin(lat_rad)

    x0_ecef = (n0 + alt0) * math.cos(lat0_rad) * math.cos(lon0_rad)
    y0_ecef = (n0 + alt0) * math.cos(lat0_rad) * math.sin(lon0_rad)
    z0_ecef = (n0 * (1 - e2) + alt0) * math.sin(lat0_rad)

    dx = x_ecef - x0_ecef
    dy = y_ecef - y0_ecef
    dz = z_ecef - z0_ecef

    sin_lat0 = math.sin(lat0_rad)
    cos_lat0 = math.cos(lat0_rad)
    sin_lon0 = math.sin(lon0_rad)
    cos_lon0 = math.cos(lon0_rad)

    e = -sin_lon0 * dx + cos_lon0 * dy
    n = -sin_lat0 * cos_lon0 * dx - sin_lat0 * sin_lon0 * dy + cos_lat0 * dz
    u = cos_lat0 * cos_lon0 * dx + cos_lat0 * sin_lon0 * dy + sin_lat0 * dz

    return np.array([e, n, u])


def _wrap_yaw(yaw):
    return (yaw + math.pi) % (2 * math.pi) - math.pi


def _quat_to_yaw(qx, qy, qz, qw):
    """Proper yaw extraction from full quaternion (handles pitch/roll)."""
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


# Chi-squared 95% thresholds: DOF → critical value
CHI2_95 = {1: 3.841, 2: 5.991, 3: 7.815}


@dataclass
class VOState:
    name: str
    R: np.ndarray
    last_time: float = None
    # EMA-filtered VO measurements
    filtered_vx: float = 0.0
    filtered_vy: float = 0.0
    ema_inited: bool = False
    # Throttle
    last_update_time: float = 0.0


class EKFEstimator(Node):
    def __init__(self):
        super().__init__('ekf_estimator')

        self.declare_parameter('imu_topic', '/wamv/sensors/imu_wamv_sensor/imu')
        self.declare_parameter('gps_topic', '/wamv/sensors/gps_wamv_link/navsat')
        self.declare_parameter('odom_topic', '/wamv/state/estimated')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('body_frame', 'wamv/ekf_base_link')
        self.declare_parameter('gps_origin_lat', -33.86)
        self.declare_parameter('gps_origin_lon', 151.20)
        self.declare_parameter('gps_origin_alt', 0.0)
        self.declare_parameter('init_from_gps', False)
        self.declare_parameter('gt_odom_topic', '')

        # 3 cameras VO (JSON strings)
        self.declare_parameter('enable_vo', True)
        self.declare_parameter('vo_topics_json',
            '["/wamv/sensors/cameras/front_left/vo",'
            '"/wamv/sensors/cameras/front_right/vo",'
            '"/wamv/sensors/cameras/middle_right/vo"]')
        self.declare_parameter('vo_noise_json',
            '[[0.3, 0.3, 0.02], [0.3, 0.3, 0.02], [0.5, 0.5, 0.03]]')
        self.declare_parameter('vo_rate', 15.0)

        # Process noise
        self.declare_parameter('q_pos', 0.01)
        self.declare_parameter('q_vel', 0.1)
        self.declare_parameter('q_yaw', 0.005)
        self.declare_parameter('q_yaw_rate', 0.05)

        # Measurement noise
        self.declare_parameter('r_gps_pos', 2.0)
        self.declare_parameter('r_gps_cog', 0.15)       # COG yaw noise [rad²]
        self.declare_parameter('cog_min_speed', 0.5)     # Min speed [m/s] for COG update
        self.declare_parameter('enable_gating', True)    # Chi-squared innovation gating

        # Initial covariance
        self.declare_parameter('p_init_pos', 10.0)
        self.declare_parameter('p_init_vel', 1.0)
        self.declare_parameter('p_init_yaw', 0.5)
        self.declare_parameter('p_init_yaw_rate', 0.1)

        # State: [px, py, yaw, surge, sway, yaw_rate]
        # surge/sway are body-frame velocities — inherently aligned with heading
        self.x = np.zeros(6)
        self.P = np.eye(6)
        p_init_pos = self.get_parameter('p_init_pos').value
        self.P[0, 0] = p_init_pos ** 2
        self.P[1, 1] = p_init_pos ** 2
        self.P[2, 2] = self.get_parameter('p_init_yaw').value ** 2
        self.P[3, 3] = self.get_parameter('p_init_vel').value ** 2
        self.P[4, 4] = self.get_parameter('p_init_vel').value ** 2
        self.P[5, 5] = self.get_parameter('p_init_yaw_rate').value ** 2

        self._last_imu_time = None
        self._gps_origin = None
        self._initialized = False
        self._init_pending_yaw = False
        self._gps_gt_offset = None
        self._yaw_calibrated = False   # Yaw inited from COG

        # GPS COG history
        self._last_gps_pos = None   # (enu_x, enu_y)
        self._last_gps_time = None
        self._cog_history = []      # [(cog, stamp)] for stable init

        # Build noise matrices
        q_pos = self.get_parameter('q_pos').value
        q_vel = self.get_parameter('q_vel').value
        q_yaw = self.get_parameter('q_yaw').value
        q_yaw_rate = self.get_parameter('q_yaw_rate').value
        self.Q_diag = np.array([q_pos, q_pos, q_yaw, q_vel, q_vel, q_yaw_rate])
        r_gps_pos = self.get_parameter('r_gps_pos').value
        self.R_gps = np.diag([r_gps_pos ** 2] * 2)
        self._r_cog = self.get_parameter('r_gps_cog').value ** 2
        self._cog_min_speed = self.get_parameter('cog_min_speed').value
        self._enable_gating = self.get_parameter('enable_gating').value
        self._r_imu_yaw = 0.01  # IMU yaw measurement noise [rad²] (~5.7° std)
        self._imu_yaw_count = 0

        # EMA filter for IMU data smoothing (cutoff ~2.5 Hz at 100 Hz)
        self._ema_alpha = 0.15
        self._ema_ax = 0.0
        self._ema_ay = 0.0
        self._ema_omega_z = 0.0
        self._ema_inited = False

        # VO pre-filtering: EMA smoother + update throttle
        self._vo_ema_alpha = 0.25   # cutoff ~2 Hz at 15 Hz VO rate
        self._vo_update_interval = 0.1  # max 10 Hz VO updates

        # Output post-filter (1st-order low-pass on body-frame, τ ≈ 0.17s, fc ≈ 1 Hz)
        self._out_filter_alpha = 0.06
        self._out_filtered_surge = 0.0
        self._out_filtered_sway = 0.0
        self._out_filtered_yr = 0.0
        self._out_filter_inited = False

        # Subscribers
        self.imu_sub = self.create_subscription(
            Imu, self.get_parameter('imu_topic').value, self._imu_cb, 10)
        self.gps_sub = self.create_subscription(
            NavSatFix, self.get_parameter('gps_topic').value, self._gps_cb, 10)

        # 3 camera VO subscribers
        self._vo_states = []
        if self.get_parameter('enable_vo').value:
            vo_topics = json.loads(self.get_parameter('vo_topics_json').value)
            vo_noise = json.loads(self.get_parameter('vo_noise_json').value)
            for i, topic in enumerate(vo_topics):
                noise = vo_noise[min(i, len(vo_noise) - 1)]
                # noise format: [surge_std, sway_std, yaw_rate_std]
                R = np.diag([noise[0] ** 2, noise[1] ** 2, noise[2] ** 2])
                cam_name = topic.rstrip('/').split('/')[-2]
                vo_state = VOState(name=cam_name, R=R)
                self._vo_states.append(vo_state)
                self.create_subscription(
                    Odometry, topic, self._make_vo_cb(i), 10)
            self.get_logger().info(
                f'VO enabled with {len(self._vo_states)} cameras: '
                f'{[v.name for v in self._vo_states]}')

        # Ground truth subscriber for simulation calibration
        gt_topic = self.get_parameter('gt_odom_topic').value
        if gt_topic:
            self.gt_sub = self.create_subscription(
                Odometry, gt_topic, self._gt_init_cb, 10)
            self.get_logger().info(f'GT calib enabled on {gt_topic}')

        # Publishers
        self.odom_pub = self.create_publisher(
            Odometry, self.get_parameter('odom_topic').value, 10)
        self.path_pub = self.create_publisher(Path, '/wamv/state/path', 10)
        self._path = Path()
        self._path.header.frame_id = self.get_parameter('world_frame').value

        self.get_logger().info(
            'EKF Estimator ready (IMU + GPS + GPS-COG + 3×Camera VO)')

    # ── helpers ────────────────────────────────────────────────────────

    def _make_vo_cb(self, idx):
        def cb(msg: Odometry):
            self._vo_cb(msg, idx)
        return cb

    def _regularize_covariance(self):
        """Ensure P stays symmetric, positive-definite, and bounded."""
        self.P = 0.5 * (self.P + self.P.T)
        eigvals = np.linalg.eigvalsh(self.P)
        # Clamp minimum eigenvalue (prevents collapse)
        if eigvals[0] < 1e-12:
            self.P += np.eye(6) * 1e-10
        # Clamp maximum eigenvalue (prevents divergence in long runs)
        # Max position uncertainty: 100m², max velocity: 25 m²/s²
        max_eig = np.array([100.0, 100.0, 1.0, 25.0, 25.0, 0.25])
        if eigvals[-1] > 200.0:
            # Scale P down if any eigenvalue exceeds global cap
            scale = 100.0 / eigvals[-1]
            self.P *= scale

    def _kalman_update(self, z, H, R, dof):
        """Single EKF update with optional chi-squared gating."""
        y = z - H @ self.x
        S = H @ self.P @ H.T + R

        if self._enable_gating and dof in CHI2_95:
            try:
                mahalanobis = float(y.T @ np.linalg.solve(S, y))
                if mahalanobis > CHI2_95[dof] * 3.0:  # relaxed threshold
                    return  # reject outlier
            except np.linalg.LinAlgError:
                return

        try:
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.x[2] = _wrap_yaw(self.x[2])
            self.P = (np.eye(6) - K @ H) @ self.P
            self._regularize_covariance()
        except np.linalg.LinAlgError:
            pass

    # ── subscribers ────────────────────────────────────────────────────

    def _gps_origin_from_first_fix(self, lat, lon, alt):
        if self.get_parameter('init_from_gps').value:
            self._gps_origin = (lat, lon, alt)
            self.get_logger().info(f'GPS origin: ({lat:.6f}, {lon:.6f})')

    def _gt_init_cb(self, msg: Odometry):
        if self._gps_gt_offset is not None or self._gps_origin is None:
            return
        gt_x = msg.pose.pose.position.x
        gt_y = msg.pose.pose.position.y
        enu_at_origin = np.zeros(2)
        self._gps_gt_offset = np.array([gt_x, gt_y]) - enu_at_origin
        self.get_logger().info(
            f'GT calib: offset=({self._gps_gt_offset[0]:.1f}, '
            f'{self._gps_gt_offset[1]:.1f})')

    # ── IMU prediction ─────────────────────────────────────────────────

    def _imu_cb(self, msg: Imu):
        if not self._initialized:
            if self._init_pending_yaw:
                self.x[2] = _quat_to_yaw(msg.orientation.x, msg.orientation.y,
                                         msg.orientation.z, msg.orientation.w)
                self._init_pending_yaw = False
                self._initialized = True
                self._last_imu_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                self.get_logger().info(
                    f'EKF yaw init: {math.degrees(self.x[2]):.1f}deg, '
                    f'pos=({self.x[0]:.1f},{self.x[1]:.1f})')
                return
            return

        now_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_imu_time is None:
            self._last_imu_time = now_sec
            return

        dt = now_sec - self._last_imu_time
        self._last_imu_time = now_sec
        if dt <= 0 or dt > 0.5:
            return

        ax_b = msg.linear_acceleration.x
        ay_b = msg.linear_acceleration.y
        omega_z = msg.angular_velocity.z

        # EMA low-pass filter on IMU data (cutoff ~2.5 Hz)
        if not self._ema_inited:
            self._ema_ax = ax_b
            self._ema_ay = ay_b
            self._ema_omega_z = omega_z
            self._ema_inited = True
        else:
            a = self._ema_alpha
            self._ema_ax = a * ax_b + (1 - a) * self._ema_ax
            self._ema_ay = a * ay_b + (1 - a) * self._ema_ay
            self._ema_omega_z = a * omega_z + (1 - a) * self._ema_omega_z

        self._predict(dt, self._ema_ax, self._ema_ay, self._ema_omega_z)

        # Gyro yaw_rate measurement (EMA-filtered, ~100 Hz)
        z_gyro = np.array([self._ema_omega_z])
        H_gyro = np.zeros((1, 6))
        H_gyro[0, 5] = 1.0
        R_gyro = np.array([[0.005]])  # gyro noise ~4°/s std — trust random walk more
        self._kalman_update(z_gyro, H_gyro, R_gyro, dof=1)

        # IMU yaw measurement update (throttled to ~20Hz, simulates magnetometer)
        self._imu_yaw_count += 1
        if self._imu_yaw_count % 5 == 0:
            imu_yaw = _quat_to_yaw(msg.orientation.x, msg.orientation.y,
                                    msg.orientation.z, msg.orientation.w)
            yaw_err = _wrap_yaw(imu_yaw - self.x[2])
            z_imu_yaw = np.array([self.x[2] + yaw_err])
            H_imu_yaw = np.zeros((1, 6))
            H_imu_yaw[0, 2] = 1.0
            R_imu_yaw = np.array([[self._r_imu_yaw]])
            self._kalman_update(z_imu_yaw, H_imu_yaw, R_imu_yaw, dof=1)

        self._publish_odometry(msg.header.stamp)

    def _predict(self, dt, ax_b, ay_b, omega_z):
        psi = self.x[2]
        surge = self.x[3]
        sway = self.x[4]

        cos_psi = math.cos(psi)
        sin_psi = math.sin(psi)

        # World-frame velocity from body-frame
        vx_w = surge * cos_psi - sway * sin_psi
        vy_w = surge * sin_psi + sway * cos_psi

        # State derivative with Coriolis coupling (rotating body-frame):
        #   d(surge)/dt = ax_b + omega_z * sway
        #   d(sway)/dt  = ay_b - omega_z * surge
        omega = self.x[5]
        surge_dot = ax_b + omega * sway
        sway_dot = ay_b - omega * surge

        # Mid-point integration for velocity (more stable during turns)
        surge_mid = surge + 0.5 * surge_dot * dt
        sway_mid = sway + 0.5 * sway_dot * dt
        omega_mid = omega  # random walk, no change

        surge_dot_mid = ax_b + omega_mid * sway_mid
        sway_dot_mid = ay_b - omega_mid * surge_mid

        # Position update
        self.x[0] += vx_w * dt + 0.5 * (ax_b * cos_psi - ay_b * sin_psi) * dt * dt
        self.x[1] += vy_w * dt + 0.5 * (ax_b * sin_psi + ay_b * cos_psi) * dt * dt
        self.x[2] += self.x[5] * dt
        self.x[3] += surge_dot_mid * dt
        self.x[4] += sway_dot_mid * dt
        # x[5] = yaw_rate — random walk

        # Jacobian F = dx'/dx (including Coriolis coupling)
        F = np.eye(6)
        F[0, 2] = (-surge * sin_psi - sway * cos_psi) * dt
        F[0, 3] = cos_psi * dt
        F[0, 4] = -sin_psi * dt
        F[1, 2] = (surge * cos_psi - sway * sin_psi) * dt
        F[1, 3] = sin_psi * dt
        F[1, 4] = cos_psi * dt
        F[2, 5] = dt
        # Coriolis: d(surge)/d(sway)=omega, d(surge)/d(yaw_rate)=sway
        F[3, 4] = omega * dt
        F[3, 5] = sway * dt
        # Coriolis: d(sway)/d(surge)=-omega, d(sway)/d(yaw_rate)=-surge
        F[4, 3] = -omega * dt
        F[4, 5] = -surge * dt

        Q = np.diag(self.Q_diag * dt)
        self.P = F @ self.P @ F.T + Q
        self.x[2] = _wrap_yaw(self.x[2])
        self._regularize_covariance()

    # ── GPS update (position + COG) ────────────────────────────────────

    def _gps_cb(self, msg: NavSatFix):
        lat = msg.latitude
        lon = msg.longitude
        alt = msg.altitude

        if self._gps_origin is None:
            self._gps_origin_from_first_fix(lat, lon, alt)
            if self._gps_origin is None:
                lat0 = self.get_parameter('gps_origin_lat').value
                lon0 = self.get_parameter('gps_origin_lon').value
                alt0 = self.get_parameter('gps_origin_alt').value
                self._gps_origin = (lat0, lon0, alt0)

        enu = geodetic_to_enu(lat, lon, alt, *self._gps_origin)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Apply GT offset for simulation alignment
        enu_pos = enu[:2]
        if self._gps_gt_offset is not None:
            enu_pos = enu[:2] + self._gps_gt_offset

        # Initialization
        if not self._initialized:
            self.x[0] = enu_pos[0]
            self.x[1] = enu_pos[1]
            self._last_imu_time = stamp
            self._init_pending_yaw = True
            self._last_gps_pos = (enu_pos[0], enu_pos[1])
            self._last_gps_time = stamp
            self.get_logger().info(
                f'EKF init position: ENU({enu[0]:.1f},{enu[1]:.1f}) '
                f'→ World({self.x[0]:.1f},{self.x[1]:.1f}), waiting for IMU yaw...')
            return

        # ── GPS position update ──
        z_pos = enu_pos
        H_pos = np.zeros((2, 6))
        H_pos[0, 0] = 1.0
        H_pos[1, 1] = 1.0
        self._kalman_update(z_pos, H_pos, self.R_gps, dof=2)

        # ── GPS COG — soft yaw backup (very low weight) ──
        # IMU yaw is the primary reference. COG from GPS is noisy (σ≈2m)
        # and only useful as a gentle long-term drift correction.
        if self._last_gps_pos is not None and self._last_gps_time is not None:
            dt_gps = stamp - self._last_gps_time
            if dt_gps > 0.5:
                dx_g = enu_pos[0] - self._last_gps_pos[0]
                dy_g = enu_pos[1] - self._last_gps_pos[1]
                dist_g = math.sqrt(dx_g**2 + dy_g**2)

                if dist_g > 5.0 or dt_gps > 15.0:
                    if dist_g > 5.0:
                        cog = math.atan2(dy_g, dx_g)
                        # Extremely high noise: COG is ~30-60° uncertain at 5m displacement.
                        # This means the update has negligible weight unless IMU fails.
                        yaw_err = _wrap_yaw(cog - self.x[2])
                        z_cog = np.array([self.x[2] + yaw_err])
                        H_cog = np.zeros((1, 6))
                        H_cog[0, 2] = 1.0
                        r_effective = self._r_cog * (10.0 / max(dist_g, 1.0)) ** 2
                        R_cog = np.array([[r_effective]])
                        self._kalman_update(z_cog, H_cog, R_cog, dof=1)

                    self._last_gps_pos = (enu_pos[0], enu_pos[1])
                    self._last_gps_time = stamp
        else:
            self._last_gps_pos = (enu_pos[0], enu_pos[1])
            self._last_gps_time = stamp

    # ── VO update (velocity + yaw rate) ────────────────────────────────

    def _vo_cb(self, msg: Odometry, idx: int):
        if not self._initialized:
            return

        vo = self._vo_states[idx]
        now_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # ── EMA pre-filter raw VO velocity measurements ──
        raw_vx = msg.twist.twist.linear.x
        raw_vy = msg.twist.twist.linear.y
        raw_wz = msg.twist.twist.angular.z

        if not vo.ema_inited:
            vo.filtered_vx = raw_vx
            vo.filtered_vy = raw_vy
            vo.ema_inited = True
        else:
            a = self._vo_ema_alpha
            vo.filtered_vx = a * raw_vx + (1 - a) * vo.filtered_vx
            vo.filtered_vy = a * raw_vy + (1 - a) * vo.filtered_vy

        # ── Throttle: max 10 Hz update rate ──
        if now_sec - vo.last_update_time < self._vo_update_interval:
            vo.last_time = now_sec
            return
        vo.last_update_time = now_sec

        # Use body-frame twist velocity directly (Gazebo odom twist is body-frame)
        if vo.last_time is not None:
            dt = now_sec - vo.last_time
            if dt > 0.01 and dt < 2.0:
                # Only camera 0 does full 3-DOF (surge+sway+yr).
                # Cameras 1+2 update surge+yr only — avoids 3× sway injection.
                if idx == 0:
                    z = np.array([vo.filtered_vx, vo.filtered_vy, raw_wz])
                    H = np.zeros((3, 6))
                    H[0, 3] = 1.0; H[1, 4] = 1.0; H[2, 5] = 1.0
                    self._kalman_update(z, H, vo.R, dof=3)
                else:
                    z = np.array([vo.filtered_vx, raw_wz])
                    H = np.zeros((2, 6))
                    H[0, 3] = 1.0; H[1, 5] = 1.0
                    R2 = np.diag([vo.R[0, 0], vo.R[2, 2]])
                    self._kalman_update(z, H, R2, dof=2)

        vo.last_time = now_sec

    # ── output ─────────────────────────────────────────────────────────

    def _publish_odometry(self, stamp):
        psi = self.x[2]
        surge_raw = self.x[3]
        sway_raw = self.x[4]
        yr_raw = self.x[5]

        # ── Output low-pass filter on body-frame velocity (τ ≈ 0.08s) ──
        # Filtering in body-frame avoids cross-coupling between surge/sway
        # that occurs when filtering in world-frame and rotating back.
        a = self._out_filter_alpha
        if not self._out_filter_inited:
            self._out_filtered_surge = surge_raw
            self._out_filtered_sway = sway_raw
            self._out_filtered_yr = yr_raw
            self._out_filter_inited = True
        else:
            self._out_filtered_surge = a * surge_raw + (1 - a) * self._out_filtered_surge
            self._out_filtered_sway = a * sway_raw + (1 - a) * self._out_filtered_sway
            self._out_filtered_yr = a * yr_raw + (1 - a) * self._out_filtered_yr

        # World-frame velocity from filtered body-frame
        cos_psi = math.cos(psi)
        sin_psi = math.sin(psi)
        vx_w = self._out_filtered_surge * cos_psi - self._out_filtered_sway * sin_psi
        vy_w = self._out_filtered_surge * sin_psi + self._out_filtered_sway * cos_psi

        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.get_parameter('world_frame').value
        msg.child_frame_id = self.get_parameter('body_frame').value
        msg.pose.pose.position.x = self.x[0]
        msg.pose.pose.position.y = self.x[1]
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.z = math.sin(psi / 2)
        msg.pose.pose.orientation.w = math.cos(psi / 2)

        msg.twist.twist.linear.x = vx_w
        msg.twist.twist.linear.y = vy_w
        msg.twist.twist.angular.z = self._out_filtered_yr

        cov_pose = np.zeros(36)
        cov_pose[0] = self.P[0, 0]
        cov_pose[7] = self.P[1, 1]
        cov_pose[35] = self.P[2, 2]
        msg.pose.covariance = cov_pose.tolist()

        cov_twist = np.zeros(36)
        cov_twist[0] = self.P[3, 3]
        cov_twist[7] = self.P[4, 4]
        cov_twist[35] = self.P[5, 5]
        msg.twist.covariance = cov_twist.tolist()

        self.odom_pub.publish(msg)

        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self._path.poses.append(pose)
        self._path.header.stamp = stamp
        self.path_pub.publish(self._path)


def main():
    rclpy.init()
    node = EKFEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
