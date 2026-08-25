#!/usr/bin/env python3
"""
ES-EKF State Estimator for WAM-V USV.

Error-State Extended Kalman Filter fusing IMU + GPS + GPS-COG + 3× Camera VO.
Uses the Joan Solà ES-EKF formulation with quaternion attitude propagation
on SO(3) — no Euler angle singularities.

State: 15-dim error state + 16-dim nominal state (see eskf_core.py).
Output: 2D marine pose + body-frame velocity as Odometry message.

Sensors:
  - IMU (100Hz): body-frame accel + gyro → prediction
  - GPS (1Hz): world-frame position → position update
  - GPS-COG (1Hz): course-over-ground → yaw update (gentle drift correction)
  - Camera VO ×3 (10-30Hz): body-frame velocity + yaw rate from visual odometry

Key differences from standard EKF (ekf_estimator_node.py):
  - Quaternion attitude on SO(3) — no Euler wrap issues at ±π
  - Error state propagates in tangent so(3) Lie algebra
  - Full 3D gravity compensation (handles wave-induced roll/pitch)
  - Proper covariance reset after error injection (J·P·J^T)
  - Online IMU bias estimation (accel + gyro biases in state)
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
from concurrent.futures import ThreadPoolExecutor

from marine_env.eskf_core import ESKF, quat_to_yaw, quat_normalize, skew_symmetric


# ═══════════════════════════════════════════════════════════════════════════
# Geodetic → ENU conversion (copied from ekf_estimator_node.py)
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# VO State Bookkeeping
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VOState:
    name: str
    R: np.ndarray          # measurement noise covariance
    last_time: float = None
    # EMA-filtered VO measurements
    filtered_vx: float = 0.0
    filtered_vy: float = 0.0
    filtered_wz: float = 0.0
    ema_inited: bool = False
    # Throttle
    last_update_time: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# ES-EKF Estimator Node
# ═══════════════════════════════════════════════════════════════════════════

class ESKFEstimator(Node):
    def __init__(self):
        super().__init__('eskf_estimator')

        # ── Parameters ──
        self.declare_parameter('imu_topic', '/wamv/sensors/imu_wamv_sensor/imu')
        self.declare_parameter('gps_topic', '/wamv/sensors/gps_wamv_link/navsat')
        self.declare_parameter('odom_topic', '/wamv/state/estimated')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('body_frame', 'wamv/eskf_base_link')
        self.declare_parameter('gps_origin_lat', -33.86)
        self.declare_parameter('gps_origin_lon', 151.20)
        self.declare_parameter('gps_origin_alt', 0.0)
        self.declare_parameter('init_from_gps', False)
        self.declare_parameter('gt_odom_topic', '')

        # Camera VO
        self.declare_parameter('enable_vo', True)
        self.declare_parameter('vo_topics_json',
            '["/wamv/sensors/cameras/front_left/vo",'
            '"/wamv/sensors/cameras/front_right/vo",'
            '"/wamv/sensors/cameras/middle_right/vo"]')
        self.declare_parameter('vo_noise_json',
            '[[0.3, 0.3, 0.02], [0.3, 0.3, 0.02], [0.5, 0.5, 0.03]]')
        self.declare_parameter('vo_rate', 15.0)

        # ES-EKF process noise
        self.declare_parameter('sigma_accel', 0.1)
        self.declare_parameter('sigma_gyro', 0.01)
        self.declare_parameter('sigma_accel_bias', 0.001)
        self.declare_parameter('sigma_gyro_bias', 0.0001)

        # Measurement noise
        self.declare_parameter('r_gps_pos', 2.0)         # GPS position noise [m]
        self.declare_parameter('r_gps_cog', 0.15)        # COG yaw noise [rad²]
        self.declare_parameter('cog_min_speed', 0.5)     # Min speed [m/s] for COG update

        # Initial uncertainty
        self.declare_parameter('p_init_pos', 10.0)
        self.declare_parameter('p_init_vel', 1.0)
        self.declare_parameter('p_init_att', 0.5)
        self.declare_parameter('p_init_ab', 0.1)
        self.declare_parameter('p_init_wb', 0.05)

        # ── ES-EKF ──
        self.eskf = ESKF()
        self.eskf.set_process_noise(
            sigma_a=self.get_parameter('sigma_accel').value,
            sigma_w=self.get_parameter('sigma_gyro').value,
            sigma_ba=self.get_parameter('sigma_accel_bias').value,
            sigma_bw=self.get_parameter('sigma_gyro_bias').value,
        )
        self.eskf.set_initial_uncertainty(
            p_std=math.sqrt(self.get_parameter('p_init_pos').value),
            v_std=math.sqrt(self.get_parameter('p_init_vel').value),
            att_std=math.sqrt(self.get_parameter('p_init_att').value),
            ab_std=math.sqrt(self.get_parameter('p_init_ab').value),
            wb_std=math.sqrt(self.get_parameter('p_init_wb').value),
        )

        # ── GPS origin ──
        self._gps_origin = None
        self._gps_gt_offset = None

        # ── Initialization state ──
        self._initialized = False
        self._init_pending_yaw = False
        self._yaw_inited = False

        # ── IMU bookkeeping ──
        self._last_imu_time = None

        # ── GPS COG bookkeeping ──
        self._last_gps_pos = None
        self._last_gps_time = None

        # ── EMA filter for IMU ──
        self._ema_alpha = 0.15
        self._ema_ax = 0.0
        self._ema_ay = 0.0
        self._ema_az = 0.0
        self._ema_wx = 0.0
        self._ema_wy = 0.0
        self._ema_wz = 0.0
        self._ema_inited = False

        # ── IMU yaw measurement update throttling (simulates magnetometer) ──
        self._r_imu_yaw = 0.01       # ~5.7° std
        self._imu_yaw_count = 0

        # ── Gyro yaw rate update throttling ──
        self._gyro_update_interval = 0.05  # 20 Hz
        self._last_gyro_update_time = 0.0

        # ── VO ──
        self._vo_states = []
        self._vo_ema_alpha = 0.25
        self._vo_update_interval = 0.1

        # ── Output filter ──
        self._out_filter_alpha = 0.06
        self._out_filtered_surge = 0.0
        self._out_filtered_sway = 0.0
        self._out_filtered_yr = 0.0
        self._out_filter_inited = False

        # ── Subscribers ──
        self.imu_sub = self.create_subscription(
            Imu, self.get_parameter('imu_topic').value, self._imu_cb, 10)
        self.gps_sub = self.create_subscription(
            NavSatFix, self.get_parameter('gps_topic').value, self._gps_cb, 10)

        # Camera VO subscribers
        if self.get_parameter('enable_vo').value:
            vo_topics = json.loads(self.get_parameter('vo_topics_json').value)
            vo_noise = json.loads(self.get_parameter('vo_noise_json').value)
            for i, topic in enumerate(vo_topics):
                noise = vo_noise[min(i, len(vo_noise) - 1)]
                R_vo = np.diag([noise[0] ** 2, noise[1] ** 2, noise[2] ** 2])
                cam_name = topic.rstrip('/').split('/')[-2]
                vo_state = VOState(name=cam_name, R=R_vo)
                self._vo_states.append(vo_state)
                self.create_subscription(
                    Odometry, topic, self._make_vo_cb(i), 10)
            self.get_logger().info(
                f'VO enabled with {len(self._vo_states)} cameras: '
                f'{[v.name for v in self._vo_states]}')

        # Ground truth subscriber for calibration
        gt_topic = self.get_parameter('gt_odom_topic').value
        if gt_topic:
            self.gt_sub = self.create_subscription(
                Odometry, gt_topic, self._gt_init_cb, 10)
            self.get_logger().info(f'GT calib enabled on {gt_topic}')

        # ── Publishers ──
        self.odom_pub = self.create_publisher(
            Odometry, self.get_parameter('odom_topic').value, 10)
        self.path_pub = self.create_publisher(Path, '/wamv/state/path', 10)
        self._path = Path()
        self._path.header.frame_id = self.get_parameter('world_frame').value

        self.get_logger().info(
            'ES-EKF Estimator ready (IMU + GPS + GPS-COG + 3×Camera VO) '
            '— quaternion attitude on SO(3)')

    # ── Helpers ─────────────────────────────────────────────────────────

    def _make_vo_cb(self, idx):
        def cb(msg: Odometry):
            self._vo_cb(msg, idx)
        return cb

    # ── GPS Origin ──────────────────────────────────────────────────────

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

    # ── IMU Prediction ──────────────────────────────────────────────────

    def _imu_cb(self, msg: Imu):
        if not self._initialized:
            if self._init_pending_yaw:
                # Initialize yaw from IMU orientation quaternion
                q_imu = np.array([
                    msg.orientation.w,
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                ])
                self.eskf.q = quat_normalize(q_imu)
                self._init_pending_yaw = False
                self._initialized = True
                self._yaw_inited = True
                self._last_imu_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                yaw = quat_to_yaw(self.eskf.q)
                self.get_logger().info(
                    f'ES-EKF yaw init from IMU quaternion: {math.degrees(yaw):.1f}°, '
                    f'pos=({self.eskf.p[0]:.1f},{self.eskf.p[1]:.1f})')
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

        # ── Raw IMU ──
        ax_b = msg.linear_acceleration.x
        ay_b = msg.linear_acceleration.y
        az_b = msg.linear_acceleration.z
        wx_b = msg.angular_velocity.x
        wy_b = msg.angular_velocity.y
        wz_b = msg.angular_velocity.z

        # ── EMA low-pass filter ──
        if not self._ema_inited:
            self._ema_ax = ax_b
            self._ema_ay = ay_b
            self._ema_az = az_b
            self._ema_wx = wx_b
            self._ema_wy = wy_b
            self._ema_wz = wz_b
            self._ema_inited = True
        else:
            a = self._ema_alpha
            self._ema_ax = a * ax_b + (1 - a) * self._ema_ax
            self._ema_ay = a * ay_b + (1 - a) * self._ema_ay
            self._ema_az = a * az_b + (1 - a) * self._ema_az
            self._ema_wx = a * wx_b + (1 - a) * self._ema_wx
            self._ema_wy = a * wy_b + (1 - a) * self._ema_wy
            self._ema_wz = a * wz_b + (1 - a) * self._ema_wz

        # ── ES-EKF prediction ──
        a_meas = np.array([self._ema_ax, self._ema_ay, self._ema_az])
        w_meas = np.array([self._ema_wx, self._ema_wy, self._ema_wz])
        self.eskf.predict(a_meas, w_meas, dt)

        # ── Gyro yaw rate measurement update (throttled to ~20Hz) ──
        if now_sec - self._last_gyro_update_time >= self._gyro_update_interval:
            self.eskf.update_yaw_rate(self._ema_wz, 0.005)  # ~4°/s noise
            self._last_gyro_update_time = now_sec

        # ── IMU yaw measurement update (throttled to ~20Hz) ──
        self._imu_yaw_count += 1
        if self._imu_yaw_count % 5 == 0:
            q_imu = np.array([
                msg.orientation.w,
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
            ])
            imu_yaw = quat_to_yaw(quat_normalize(q_imu))
            self.eskf.update_yaw(imu_yaw, self._r_imu_yaw)

        self._publish_odometry(msg.header.stamp)

    # ── GPS Update ──────────────────────────────────────────────────────

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

        # ── Initialization from first GPS fix ──
        if not self._initialized:
            self.eskf.initialize(p0=np.array([enu_pos[0], enu_pos[1], 0.0]))
            self._last_imu_time = stamp
            self._init_pending_yaw = True
            self._last_gps_pos = (enu_pos[0], enu_pos[1])
            self._last_gps_time = stamp
            self.get_logger().info(
                f'ES-EKF init position from GPS: ENU({enu[0]:.1f},{enu[1]:.1f}), '
                f'waiting for IMU orientation...')
            return

        # ── GPS position update ──
        r_gps_pos = self.get_parameter('r_gps_pos').value
        R_gps = np.diag([r_gps_pos ** 2] * 2)
        self.eskf.update_position(enu_pos, R_gps)

        # ── GPS COG — gentle yaw drift correction ──
        if self._last_gps_pos is not None and self._last_gps_time is not None:
            dt_gps = stamp - self._last_gps_time
            if dt_gps > 0.5:
                dx_g = enu_pos[0] - self._last_gps_pos[0]
                dy_g = enu_pos[1] - self._last_gps_pos[1]
                dist_g = math.sqrt(dx_g ** 2 + dy_g ** 2)

                if dist_g > 5.0 or dt_gps > 15.0:
                    if dist_g > 5.0:
                        cog = math.atan2(dy_g, dx_g)
                        # High noise: COG from 5m displacement is ~30-60° uncertain
                        r_cog = self.get_parameter('r_gps_cog').value
                        r_effective = r_cog * (10.0 / max(dist_g, 1.0)) ** 2
                        self.eskf.update_yaw(cog, r_effective)

                    self._last_gps_pos = (enu_pos[0], enu_pos[1])
                    self._last_gps_time = stamp
        else:
            self._last_gps_pos = (enu_pos[0], enu_pos[1])
            self._last_gps_time = stamp

    # ── VO Update ───────────────────────────────────────────────────────

    def _vo_cb(self, msg: Odometry, idx: int):
        if not self._initialized:
            return

        vo = self._vo_states[idx]
        now_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Raw VO measurements (body-frame velocity)
        raw_vx = msg.twist.twist.linear.x
        raw_vy = msg.twist.twist.linear.y
        raw_wz = msg.twist.twist.angular.z

        # ── EMA pre-filter ──
        if not vo.ema_inited:
            vo.filtered_vx = raw_vx
            vo.filtered_vy = raw_vy
            vo.filtered_wz = raw_wz
            vo.ema_inited = True
        else:
            a = self._vo_ema_alpha
            vo.filtered_vx = a * raw_vx + (1 - a) * vo.filtered_vx
            vo.filtered_vy = a * raw_vy + (1 - a) * vo.filtered_vy
            vo.filtered_wz = a * raw_wz + (1 - a) * vo.filtered_wz

        # ── Throttle: max 10 Hz ──
        if now_sec - vo.last_update_time < self._vo_update_interval:
            vo.last_time = now_sec
            return
        vo.last_update_time = now_sec

        # ── ES-EKF body-frame velocity update ──
        if vo.last_time is not None:
            dt = now_sec - vo.last_time
            if dt > 0.01 and dt < 2.0:
                if idx == 0:
                    # Camera 0: full 3-DOF (surge + sway + yaw_rate)
                    z_body = np.array([vo.filtered_vx, vo.filtered_vy])
                    R_body_2d = np.diag([vo.R[0, 0], vo.R[1, 1]])
                    self.eskf.update_velocity_body(z_body, R_body_2d)

                    # Also update yaw rate
                    self.eskf.update_yaw_rate(vo.filtered_wz, vo.R[2, 2])
                else:
                    # Cameras 1+2: surge + yaw_rate only
                    z_body = np.array([vo.filtered_vx, 0.0])
                    # Only use surge component; zero out sway
                    R_body_1d = np.array([[vo.R[0, 0]]])
                    # Update surge only via a 1D body-frame velocity update
                    R = self.eskf._R()
                    v_body = R.T @ self.eskf.v
                    y = np.array([vo.filtered_vx - v_body[0]])

                    H = np.zeros((1, 15))
                    H[0, 3:6] = R.T[0, :]           # ∂surge/∂v
                    H[0, 6:9] = skew_symmetric(v_body)[0, :]  # ∂surge/∂δθ

                    try:
                        S = H @ self.eskf.P @ H.T + R_body_1d
                        K = self.eskf.P @ H.T @ np.linalg.solve(S, np.eye(1))
                        dx = K @ y
                        I_KH = np.eye(15) - K @ H
                        self.eskf.P = I_KH @ self.eskf.P @ I_KH.T + K @ R_body_1d @ K.T
                        self.eskf._inject_error(dx)
                        self.eskf._regularize_covariance()
                    except np.linalg.LinAlgError:
                        pass

                    # Yaw rate update
                    self.eskf.update_yaw_rate(vo.filtered_wz, vo.R[2, 2])

        vo.last_time = now_sec

    # ── Output ──────────────────────────────────────────────────────────

    def _publish_odometry(self, stamp):
        # Get 2D marine state
        pose_2d = self.eskf.get_pose_2d()
        v_body = self.eskf.get_velocity_body()
        yr = self.eskf.get_yaw_rate()

        # ── Output low-pass filter on body-frame velocity ──
        surge_raw = v_body[0]
        sway_raw = v_body[1]

        a = self._out_filter_alpha
        if not self._out_filter_inited:
            self._out_filtered_surge = surge_raw
            self._out_filtered_sway = sway_raw
            self._out_filtered_yr = yr
            self._out_filter_inited = True
        else:
            self._out_filtered_surge = a * surge_raw + (1 - a) * self._out_filtered_surge
            self._out_filtered_sway = a * sway_raw + (1 - a) * self._out_filtered_sway
            self._out_filtered_yr = a * yr + (1 - a) * self._out_filtered_yr

        # World-frame velocity from filtered body-frame
        psi = pose_2d[2]
        cos_psi = math.cos(psi)
        sin_psi = math.sin(psi)
        vx_w = self._out_filtered_surge * cos_psi - self._out_filtered_sway * sin_psi
        vy_w = self._out_filtered_surge * sin_psi + self._out_filtered_sway * cos_psi

        # Build Odometry message
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.get_parameter('world_frame').value
        msg.child_frame_id = self.get_parameter('body_frame').value

        # Position
        msg.pose.pose.position.x = pose_2d[0]
        msg.pose.pose.position.y = pose_2d[1]
        msg.pose.pose.position.z = 0.0

        # Orientation from full quaternion (not just yaw→sin/cos)
        q = self.eskf.q
        msg.pose.pose.orientation.w = q[0]
        msg.pose.pose.orientation.x = q[1]
        msg.pose.pose.orientation.y = q[2]
        msg.pose.pose.orientation.z = q[3]

        # Velocity
        msg.twist.twist.linear.x = vx_w
        msg.twist.twist.linear.y = vy_w
        msg.twist.twist.linear.z = 0.0
        msg.twist.twist.angular.x = 0.0
        msg.twist.twist.angular.y = 0.0
        msg.twist.twist.angular.z = self._out_filtered_yr

        # Covariance (extract relevant blocks from 15×15 P)
        cov_pose = np.zeros(36)
        cov_pose[0] = self.eskf.P[0, 0]    # px variance
        cov_pose[1] = self.eskf.P[0, 1]    # px-py covariance
        cov_pose[6] = self.eskf.P[1, 0]    # py-px covariance
        cov_pose[7] = self.eskf.P[1, 1]    # py variance
        # Yaw variance: P[8,8] is δθ_z variance (approximation)
        cov_pose[35] = self.eskf.P[8, 8]
        msg.pose.covariance = cov_pose.tolist()

        cov_twist = np.zeros(36)
        cov_twist[0] = self.eskf.P[3, 3]    # vx variance
        cov_twist[1] = self.eskf.P[3, 4]    # vx-vy covariance
        cov_twist[6] = self.eskf.P[4, 3]    # vy-vx covariance
        cov_twist[7] = self.eskf.P[4, 4]    # vy variance
        cov_twist[35] = self.eskf.P[14, 14] # ωb_z variance (yaw rate uncertainty)
        msg.twist.covariance = cov_twist.tolist()

        self.odom_pub.publish(msg)

        # Path
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self._path.poses.append(pose)
        self._path.header.stamp = stamp
        self.path_pub.publish(self._path)


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    rclpy.init()
    node = ESKFEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
