#!/usr/bin/env python3
"""
Error-State Extended Kalman Filter (ES-EKF) for WAM-V USV.

Implements the ES-EKF following Joan Solà's formulation:
  - Nominal state propagates in the full manifold (position, velocity, quaternion)
  - Error state propagates linearly in the tangent space (so(3) Lie algebra)
  - Error injection after each update resets the error state and updates P

Nominal state (16 dims):  p(3), v(3), q(4) ∈ S³, ab(3), ωb(3)
Error state  (15 dims):  δp(3), δv(3), δθ(3) ∈ so(3), δab(3), δωb(3)

Key advantages over standard 6-state [x,y,ψ,vx,vy,ω] EKF:
  - Attitude in SO(3) via unit quaternions — no Euler angle singularities
  - Error state is always small → linear approximation is more accurate
  - Proper handling of the circular topology of orientation (ψ ≈ π crossing)
  - Correct covariance reset after error injection (J·P·J^T)
  - Full 3D formulation correctly handles roll/pitch from waves

Reference:
  Solà, J. (2017). Quaternion kinematics for the error-state Kalman filter.
  arXiv:1711.02508.

Conventions:
  - Hamilton quaternions: q = [qw, qx, qy, qz]
  - Rotation: v_world = R(q) · v_body
  - Error quaternion: q_true = δq ⊗ q_nominal,  δq ≈ [1, ½δθ]
  - World frame: ENU (East-North-Up), gravity = [0, 0, -9.81]
"""

import numpy as np
import math


# ═══════════════════════════════════════════════════════════════════════════
# Quaternion Utilities
# ═══════════════════════════════════════════════════════════════════════════

def quat_multiply(q1, q2):
    """Hamilton product: q1 ⊗ q2 (rotate by q2, then by q1).

    Args:
        q1, q2: [w, x, y, z] quaternions.

    Returns:
        q1 ⊗ q2 as np.ndarray(4).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q):
    """Conjugate (inverse for unit quaternion).

    Args:
        q: [w, x, y, z].

    Returns:
        q* = [w, -x, -y, -z].
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_rotation_vector(phi):
    """Exponential map from so(3) to S³.

    q = exp(½φ) = [cos(‖φ‖/2), sin(‖φ‖/2)·φ/‖φ‖]

    For small φ: q ≈ [1, ½φ] (first-order).

    Args:
        phi: 3-element rotation vector (axis * angle).

    Returns:
        Unit quaternion [w, x, y, z].
    """
    angle = np.linalg.norm(phi)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = phi / angle
    half = 0.5 * angle
    return np.array([
        math.cos(half),
        math.sin(half) * axis[0],
        math.sin(half) * axis[1],
        math.sin(half) * axis[2],
    ])


def quat_to_rotation_matrix(q):
    """Convert unit quaternion to rotation matrix R (body → world).

    Args:
        q: [qw, qx, qy, qz] unit quaternion.

    Returns:
        3×3 rotation matrix R such that v_world = R @ v_body.
    """
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ])


def quat_normalize(q):
    """Normalize quaternion to unit length.

    Falls back to identity quaternion for near-zero input.
    """
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def skew_symmetric(v):
    """Skew-symmetric matrix [v]× ∈ so(3).

    [v]× = [[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]]
    """
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def quat_to_yaw(q):
    """Extract yaw (heading) angle from quaternion.

    Uses the proper full-quaternion formula:
        ψ = atan2(2(qw·qz + qx·qy), 1 - 2(qy² + qz²))

    This correctly handles non-zero roll/pitch.
    """
    siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
    return math.atan2(siny_cosp, cosy_cosp)


# ═══════════════════════════════════════════════════════════════════════════
# ES-EKF
# ═══════════════════════════════════════════════════════════════════════════

class ESKF:
    """Error-State Kalman Filter for 3D INS/GNSS fusion.

    Nominal state (16 dims): p(3), v(3), q(4), ab(3), ωb(3)
    Error state   (15 dims): δp(3), δv(3), δθ(3), δab(3), δωb(3)

    Index map for error state vector dx[0:15]:
        0:2   δp   — position error [ENU, m]
        3:5   δv   — velocity error [ENU, m/s]
        6:8   δθ   — attitude error (rotation vector) [rad]
        9:11  δab  — accelerometer bias error [body frame, m/s²]
        12:14 δωb  — gyroscope bias error [body frame, rad/s]
    """

    # Gravity in ENU frame (East-North-Up)
    GRAVITY = np.array([0.0, 0.0, -9.81])

    def __init__(self):
        # ── Nominal state ──
        self.p = np.zeros(3)                          # position ENU
        self.v = np.zeros(3)                          # velocity ENU
        self.q = np.array([1.0, 0.0, 0.0, 0.0])      # attitude quaternion (body→world)
        self.ab = np.zeros(3)                          # accelerometer bias (body frame)
        self.wb = np.zeros(3)                          # gyroscope bias (body frame)

        # ── Error-state covariance (15×15) ──
        self.P = np.eye(15)

        # Initial uncertainty
        self.P[0:3, 0:3] = np.eye(3) * 10.0      # position: ~3.2 m std
        self.P[3:6, 3:6] = np.eye(3) * 1.0       # velocity: 1 m/s std
        self.P[6:9, 6:9] = np.eye(3) * 0.1       # attitude: ~0.32 rad (18°)
        self.P[9:12, 9:12] = np.eye(3) * 0.01     # accel bias: 0.1 m/s² std
        self.P[12:15, 12:15] = np.eye(3) * 0.001  # gyro bias: ~0.032 rad/s std

        # ── Process noise spectral densities (tunable) ──
        self.sigma_a = 0.1        # accelerometer noise [m/s²/√Hz]
        self.sigma_w = 0.01       # gyroscope noise [rad/s/√Hz]
        self.sigma_ba = 0.001     # accel bias random walk [m/s²/√Hz]
        self.sigma_bw = 0.0001    # gyro bias random walk [rad/s/√Hz]

        # ── Bookkeeping ──
        self._initialized = False
        self._last_gyro = np.zeros(3)   # most recent bias-corrected gyro for yaw_rate output

    # ── Accessors ───────────────────────────────────────────────────────

    def _R(self):
        """Current rotation matrix (body → world)."""
        return quat_to_rotation_matrix(self.q)

    def get_pose_2d(self):
        """Extract 2D marine pose: (x, y, yaw) in ENU.

        Yaw is extracted from the full quaternion using the proper formula,
        handling non-zero roll/pitch correctly.
        """
        yaw = quat_to_yaw(self.q)
        return np.array([self.p[0], self.p[1], yaw])

    def get_velocity_body(self):
        """Get body-frame velocity (surge, sway, heave).

        v_body = R^T · v_world
        """
        return self._R().T @ self.v

    def get_yaw_rate(self):
        """Estimated yaw rate (body-frame angular velocity, z-component).

        Derived from last bias-corrected gyro measurement.
        """
        return self._last_gyro[2]

    def set_process_noise(self, sigma_a=None, sigma_w=None,
                          sigma_ba=None, sigma_bw=None):
        """Configure process noise spectral densities."""
        if sigma_a is not None:
            self.sigma_a = sigma_a
        if sigma_w is not None:
            self.sigma_w = sigma_w
        if sigma_ba is not None:
            self.sigma_ba = sigma_ba
        if sigma_bw is not None:
            self.sigma_bw = sigma_bw

    def set_initial_uncertainty(self, p_std=3.2, v_std=1.0,
                                 att_std=0.32, ab_std=0.1, wb_std=0.032):
        """Configure initial covariance diagonal."""
        self.P[0:3, 0:3] = np.eye(3) * p_std ** 2
        self.P[3:6, 3:6] = np.eye(3) * v_std ** 2
        self.P[6:9, 6:9] = np.eye(3) * att_std ** 2
        self.P[9:12, 9:12] = np.eye(3) * ab_std ** 2
        self.P[12:15, 12:15] = np.eye(3) * wb_std ** 2

    # ── Initialization ──────────────────────────────────────────────────

    def initialize(self, p0, v0=None, q0=None):
        """Initialize nominal state.

        Args:
            p0: Initial position [px, py, pz] or [px, py] (z=0 if 2D).
            v0: Initial velocity (default: zeros).
            q0: Initial attitude quaternion [qw, qx, qy, qz]
                (default: identity = pointing East, level).
        """
        p0 = np.asarray(p0, dtype=float)
        if len(p0) == 2:
            self.p = np.array([p0[0], p0[1], 0.0])
        else:
            self.p = p0.copy()

        if v0 is not None:
            v0 = np.asarray(v0, dtype=float)
            if len(v0) == 2:
                self.v = np.array([v0[0], v0[1], 0.0])
            else:
                self.v = v0.copy()
        else:
            self.v = np.zeros(3)

        if q0 is not None:
            self.q = quat_normalize(np.asarray(q0, dtype=float))
        else:
            self.q = np.array([1.0, 0.0, 0.0, 0.0])

        self._initialized = True

    # ── Prediction ──────────────────────────────────────────────────────

    def predict(self, a_m, w_m, dt):
        """ES-EKF prediction step driven by IMU.

        Propagates nominal state with accelerometer + gyroscope,
        and error-state covariance with the linearized error dynamics.

        Args:
            a_m: Accelerometer measurement [ax, ay, az] in body frame [m/s²].
            w_m: Gyroscope measurement [wx, wy, wz] in body frame [rad/s].
            dt: Time step [s].
        """
        if not self._initialized:
            return
        if dt <= 0 or dt > 0.5:
            return

        # ── 1. Bias-corrected IMU ──
        a_corrected = a_m - self.ab
        w_corrected = w_m - self.wb
        self._last_gyro = w_corrected.copy()

        # ── 2. Nominal state propagation ──
        R = self._R()

        # World-frame acceleration (with gravity)
        a_world = R @ a_corrected + self.GRAVITY

        # Position: 2nd-order integration
        self.p = self.p + self.v * dt + 0.5 * a_world * dt * dt

        # Velocity: 1st-order integration
        self.v = self.v + a_world * dt

        # Attitude: q ← q ⊗ exp(½ · w_corrected · dt)
        dq = quat_from_rotation_vector(w_corrected * dt)
        self.q = quat_multiply(self.q, dq)
        self.q = quat_normalize(self.q)

        # Biases: constant (random walk captured by noise)

        # ── 3. Error-state transition matrix Φ ≈ I + F·dt ──
        # Continuous-time error dynamics:
        #   δṗ   = δv
        #   δv̇   = -R[ã]× δθ - R δab + R n_a
        #   δθ̇   = -[ω̃]× δθ - δωb - n_ω
        #   δȧb  = n_ba
        #   δω̇b  = n_bω
        Phi = np.eye(15)

        # δv contributions from δθ and δab
        R_skew_a = R @ skew_symmetric(a_corrected)
        Phi[3:6, 6:9] = -R_skew_a * dt       # -R[ã]× · dt
        Phi[3:6, 9:12] = -R * dt              # -R · dt

        # δθ contributions from δθ and δωb
        Phi[6:9, 6:9] = np.eye(3) - skew_symmetric(w_corrected) * dt   # I - [ω̃]×·dt
        Phi[6:9, 12:15] = -np.eye(3) * dt      # -I · dt

        # δp contribution from δv
        Phi[0:3, 3:6] = np.eye(3) * dt

        # ── 4. Process noise covariance Q = G·Σ·G^T·dt ──
        # Noise vector: [n_a(3), n_ω(3), n_ba(3), n_bω(3)]
        G = np.zeros((15, 12))
        G[3:6, 0:3] = R              # accel noise → velocity error
        G[6:9, 3:6] = np.eye(3)      # gyro noise → attitude error
        G[9:12, 6:9] = np.eye(3)     # accel bias walk
        G[12:15, 9:12] = np.eye(3)   # gyro bias walk

        Sigma_diag = np.array([
            self.sigma_a ** 2, self.sigma_a ** 2, self.sigma_a ** 2,
            self.sigma_w ** 2, self.sigma_w ** 2, self.sigma_w ** 2,
            self.sigma_ba ** 2, self.sigma_ba ** 2, self.sigma_ba ** 2,
            self.sigma_bw ** 2, self.sigma_bw ** 2, self.sigma_bw ** 2,
        ])
        Q = G @ np.diag(Sigma_diag) @ G.T * dt

        # ── 5. Propagate covariance ──
        self.P = Phi @ self.P @ Phi.T + Q
        self._regularize_covariance()

    # ── Measurement Updates ─────────────────────────────────────────────

    def update_position(self, z_pos, R_pos):
        """GPS position update (2D or 3D).

        Args:
            z_pos: Measured position [x, y] or [x, y, z] in ENU.
            R_pos: Measurement noise covariance [2×2] or [3×3].
        """
        if not self._initialized:
            return

        dim = len(z_pos)
        z_hat = self.p[:dim]
        y = z_pos - z_hat

        # H = [I_dim, 0_{dim × (15-dim)}]
        H = np.zeros((dim, 15))
        H[:dim, :dim] = np.eye(dim)

        # Chi-squared gating for 2D position
        if dim == 2:
            if not self._gating_check(y, H, R_pos, dof=2, multiplier=3.0):
                return

        self._kalman_update(y, H, R_pos)

    def update_velocity_world(self, z_vel, R_vel):
        """Velocity update in world frame (e.g., from GPS Doppler).

        Args:
            z_vel: Measured velocity [vx, vy] or [vx, vy, vz] in ENU.
            R_vel: Measurement noise covariance.
        """
        if not self._initialized:
            return

        dim = len(z_vel)
        z_hat = self.v[:dim]
        y = z_vel - z_hat

        # H = [0, I_dim, 0, ...]
        H = np.zeros((dim, 15))
        H[:dim, 3:3 + dim] = np.eye(dim)

        self._kalman_update(y, H, R_vel)

    def update_velocity_body(self, z_body, R_body):
        """VO velocity update in body frame.

        Measurement model: h(x) = (R^T · v)[0:2]  (surge, sway)

        The full Jacobian accounts for cross-coupling between
        attitude error and body-frame velocity projection.

        Args:
            z_body: Measured body-frame velocity [surge, sway] [m/s].
            R_body: Measurement noise covariance [2×2].
        """
        if not self._initialized:
            return

        R = self._R()
        v_body = R.T @ self.v

        # Innovation (2D)
        z_hat = v_body[:2]
        y = z_body - z_hat

        # Jacobian H (2 × 15)
        H = np.zeros((2, 15))

        # ∂h/∂δv: H[0:2, 3:6] = R^T[0:2, :]
        H[0:2, 3:6] = R.T[0:2, :]

        # ∂h/∂δθ: H[0:2, 6:9] = [v_body]×[0:2, :]
        # Derivation: R(δθ)^T · v ≈ (I + [δθ]×) · R^T · v = v_body + [δθ]× · v_body
        #                                   = v_body - [v_body]× · δθ
        # So ∂h/∂δθ = -[v_body]×[0:2, :] ... wait.
        #
        # R_new = R · Exp(δθ)
        # R_new^T = Exp(-δθ) · R^T ≈ (I - [δθ]×) · R^T
        # R_new^T · v = (I - [δθ]×) · R^T · v = v_body - [δθ]× · v_body
        #            = v_body + [v_body]× · δθ    (since -[δθ]×·v = [v]×·δθ)
        # So ∂(R^T·v)/∂δθ = [v_body]×
        H[0:2, 6:9] = skew_symmetric(v_body)[0:2, :]

        self._kalman_update(y, H, R_body)

    def update_yaw(self, z_yaw, R_yaw):
        """Heading (yaw) measurement update.

        Used for GPS-COG (gentle drift correction) and IMU magnetometer.

        The measurement model maps the quaternion to yaw:
            ψ = atan2(2(qw·qz + qx·qy), 1 - 2(qy² + qz²))

        The Jacobian ∂ψ/∂δθ is computed via quaternion perturbation.

        Args:
            z_yaw: Measured yaw angle [rad].
            R_yaw: Measurement noise variance [rad²].
        """
        if not self._initialized:
            return

        # Predicted yaw from quaternion
        qw, qx, qy, qz = self.q
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw_hat = math.atan2(siny_cosp, cosy_cosp)

        # Innovation with wrapping
        y = z_yaw - yaw_hat
        y = (y + math.pi) % (2.0 * math.pi) - math.pi

        # ── Jacobian ∂ψ/∂δθ ──
        # Step 1: ∂ψ/∂q  (derivative of atan2)
        # d/dθ atan2(f(θ), g(θ)) = (g·f' - f·g') / (g² + f²)
        denom = siny_cosp ** 2 + cosy_cosp ** 2
        if denom < 1e-12:
            return

        # ∂ψ/∂qw = (cosy_cosp · 2qz - siny_cosp · 0) / denom
        d_dqw = cosy_cosp * 2.0 * qz / denom
        # ∂ψ/∂qx = (cosy_cosp · 2qy - siny_cosp · 0) / denom
        d_dqx = cosy_cosp * 2.0 * qy / denom
        # ∂ψ/∂qy = (cosy_cosp · 2qx - siny_cosp · (-4qy)) / denom
        d_dqy = (cosy_cosp * 2.0 * qx + siny_cosp * 4.0 * qy) / denom
        # ∂ψ/∂qz = (cosy_cosp · 2qw - siny_cosp · (-4qz)) / denom
        d_dqz = (cosy_cosp * 2.0 * qw + siny_cosp * 4.0 * qz) / denom

        d_yaw_dq = np.array([d_dqw, d_dqx, d_dqy, d_dqz])

        # Step 2: ∂q/∂δθ for quaternion perturbation q ← q ⊗ [1, ½δθ]
        # At δθ=0:
        #   ∂q_new/∂δθ = ½ · [[-qx, -qy, -qz],
        #                      [ qw,  qz, -qy],
        #                      [-qz,  qw,  qx],
        #                      [ qy, -qx,  qw]]
        dq_dtheta = 0.5 * np.array([
            [-qx, -qy, -qz],
            [qw,  qz, -qy],
            [-qz, qw,  qx],
            [qy, -qx, qw],
        ])

        # Step 3: ∂ψ/∂δθ = ∂ψ/∂q · ∂q/∂δθ
        d_yaw_dtheta = d_yaw_dq @ dq_dtheta

        # H = [0(1×6), d_yaw_dtheta(1×3), 0(1×6)]
        H = np.zeros((1, 15))
        H[0, 6:9] = d_yaw_dtheta

        R = np.array([[R_yaw]])
        self._kalman_update(np.array([y]), H, R)

    def update_yaw_rate(self, z_yr, R_yr):
        """Direct yaw rate measurement update (gyroscope z-axis).

        This directly corrects the gyro bias estimate for the z-axis.

        Args:
            z_yr: Measured yaw rate [rad/s].
            R_yr: Measurement noise variance [rad²/s²].
        """
        if not self._initialized:
            return

        # Predicted yaw rate = last gyro z (bias-corrected)
        z_hat = self._last_gyro[2]
        y = z_yr - z_hat

        # H maps δωb_z to the yaw rate measurement
        # yaw_rate = ω_mz - (ωb_z + δωb_z)
        # ∂(yaw_rate)/∂δωb_z = -1
        H = np.zeros((1, 15))
        H[0, 14] = -1.0   # negative because bias subtracts from measurement

        R = np.array([[R_yr]])
        self._kalman_update(np.array([y]), H, R)

    # ── Core Kalman Machinery ───────────────────────────────────────────

    def _kalman_update(self, y, H, R):
        """Standard EKF update on the error state, followed by injection.

        Args:
            y: Innovation vector (measurement - prediction).
            H: Measurement Jacobian (measurement_dim × 15).
            R: Measurement noise covariance.
        """
        # Innovation covariance
        S = H @ self.P @ H.T + R

        try:
            # Kalman gain
            K = self.P @ H.T @ np.linalg.solve(S, np.eye(len(y)))

            # Error state update
            dx = K @ y

            # Covariance update (Joseph form for numerical stability)
            I_KH = np.eye(15) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

            # Inject error into nominal state
            self._inject_error(dx)

            self._regularize_covariance()

        except np.linalg.LinAlgError:
            pass

    def _inject_error(self, dx):
        """Inject estimated error into nominal state and reset.

        After injection, the error state is reset to zero.
        Covariance is adjusted via P ← J·P·J^T to account for
        the nonlinearity of the quaternion update.

        Args:
            dx: Error state vector [δp, δv, δθ, δab, δωb] (15 elements).
        """
        # Linear injection: p, v, biases
        self.p += dx[0:3]
        self.v += dx[3:6]
        self.ab += dx[9:12]
        self.wb += dx[12:15]

        # Rotational injection: q ← q ⊗ exp(δθ)
        dtheta = dx[6:9]
        if np.linalg.norm(dtheta) > 1e-12:
            dq = quat_from_rotation_vector(dtheta)
            self.q = quat_multiply(self.q, dq)
            self.q = quat_normalize(self.q)

        # ── Covariance reset: P ← J·P·J^T ──
        # J accounts for the fact that the error state definition changes
        # after nominal state update. Most blocks are identity, but the
        # attitude block has: J_θθ = I - [½ δθ]×
        J = np.eye(15)
        J[6:9, 6:9] = np.eye(3) - skew_symmetric(0.5 * dtheta)

        self.P = J @ self.P @ J.T

    def _gating_check(self, y, H, R, dof, multiplier=3.0):
        """Chi-squared innovation gating.

        Returns True if measurement passes the gate.
        Uses a relaxed threshold (multiplier × critical value) to avoid
        rejecting valid measurements during turns.
        """
        S = H @ self.P @ H.T + R
        try:
            mahalanobis = float(y.T @ np.linalg.solve(S, y))
            # Chi-squared critical values
            chi2_95 = {1: 3.841, 2: 5.991, 3: 7.815}
            threshold = chi2_95.get(dof, dof * 4.0) * multiplier
            return mahalanobis <= threshold
        except np.linalg.LinAlgError:
            return False

    def _regularize_covariance(self):
        """Ensure P stays symmetric and positive semi-definite."""
        # Symmetrize
        self.P = 0.5 * (self.P + self.P.T)

        # Clamp minimum eigenvalue
        eigvals = np.linalg.eigvalsh(self.P)
        if eigvals[0] < 1e-12:
            self.P += np.eye(15) * 1e-10

        # Clamp maximum eigenvalue (prevents divergence)
        # Position: 100m², velocity: 25 m²/s², attitude: 1.0 rad²,
        # accel bias: 1.0 m²/s⁴, gyro bias: 0.1 rad²/s²
        max_eig = 200.0
        if eigvals[-1] > max_eig:
            scale = max_eig / eigvals[-1]
            self.P *= scale
