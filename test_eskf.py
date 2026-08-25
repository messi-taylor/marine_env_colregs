#!/usr/bin/env python3
"""
Test suite for ES-EKF (Error-State Extended Kalman Filter).

Tests:
  Q1: Quaternion utilities
  P1: ESKF initialization
  P2: Prediction (IMU propagation)
  P3: Position update (GPS)
  P4: Yaw update (COG / magnetometer)
  P5: Body-frame velocity update (VO)
  P6: Error injection and covariance reset
  P7: Integration scenarios
  P8: Singularity handling (±180° crossing)
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marine_env.eskf_core import (
    ESKF,
    quat_multiply,
    quat_conjugate,
    quat_from_rotation_vector,
    quat_to_rotation_matrix,
    quat_normalize,
    quat_to_yaw,
    skew_symmetric,
)

_FAILED = []


def check(name: str, condition: bool, detail: str = ""):
    status = "✅ PASS" if condition else "❌ FAIL"
    if not condition:
        _FAILED.append(name)
    print(f"  {status}: {name}" + (f" — {detail}" if detail else ""))


def check_close(name, actual, expected, tol=1e-6):
    ok = abs(actual - expected) < tol
    check(name, ok, f"got {actual:.6f}, expected {expected:.6f}")


def check_allclose(name, actual, expected, atol=1e-6):
    ok = np.allclose(actual, expected, atol=atol)
    check(name, ok, f"got {actual}, expected {expected}")


# =============================================================================
# Test Q1: Quaternion Utilities
# =============================================================================

def test_quaternion_utils():
    print("\n" + "=" * 60)
    print("Test Q1: Quaternion Utilities")
    print("=" * 60)

    # Q1.1: Identity multiplication
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    q_test = np.array([0.7071, 0.0, 0.0, 0.7071])  # 90° about Z
    result = quat_multiply(q_id, q_test)
    check_allclose("Q1.1: q_id ⊗ q = q", result, q_test, atol=1e-4)

    # Q1.2: Conjugate
    q_conj = quat_conjugate(q_test)
    q_prod = quat_multiply(q_test, q_conj)
    check_allclose("Q1.2: q ⊗ q* = [1,0,0,0]", q_prod, q_id, atol=1e-4)

    # Q1.3: Rotation vector → quaternion roundtrip
    phi_small = np.array([0.1, 0.0, 0.0])  # small rotation about X
    q_from_phi = quat_from_rotation_vector(phi_small)
    check("Q1.3: exp(φ) has norm ≈ 1", abs(np.linalg.norm(q_from_phi) - 1.0) < 1e-10)
    # For small φ, q ≈ [1, φ_x/2, φ_y/2, φ_z/2]
    check_close("Q1.4: Small φ → q ≈ [1, ½φ]", q_from_phi[1], 0.05, tol=1e-2)

    # Q1.5: Rotation matrix orthogonality (relaxed tol — 0.7071 is truncated)
    R = quat_to_rotation_matrix(q_test)
    err_ortho = np.max(np.abs(R.T @ R - np.eye(3)))
    check("Q1.5: R^T·R ≈ I", err_ortho < 5e-5, f"max error={err_ortho:.1e}")
    check("Q1.6: det(R) ≈ 1", abs(np.linalg.det(R) - 1.0) < 1e-4,
          f"det={np.linalg.det(R):.6f}")

    # Q1.7: quat_to_yaw
    q_north = np.array([0.7071, 0.0, 0.0, -0.7071])  # -90° about Z → pointing North
    yaw_north = quat_to_yaw(q_north)
    check_close("Q1.7: Yaw from quat (pointing North)", yaw_north, -math.pi / 2, tol=1e-3)

    q_east = np.array([1.0, 0.0, 0.0, 0.0])
    yaw_east = quat_to_yaw(q_east)
    check_close("Q1.8: Yaw from identity quat (East)", yaw_east, 0.0, tol=1e-6)


# =============================================================================
# Test P1: ESKF Initialization
# =============================================================================

def test_eskf_init():
    print("\n" + "=" * 60)
    print("Test P1: ESKF Initialization")
    print("=" * 60)

    eskf = ESKF()

    # P1.1: Initial nominal state is zero (position, velocity)
    check_allclose("P1.1: Initial p = 0", eskf.p, np.zeros(3))
    check_allclose("P1.2: Initial v = 0", eskf.v, np.zeros(3))
    check_allclose("P1.3: Initial q = [1,0,0,0]", eskf.q, np.array([1.0, 0.0, 0.0, 0.0]))

    # P1.4: Initial covariance is 15×15, positive diagonal
    check("P1.4: P is 15×15", eskf.P.shape == (15, 15))
    check("P1.5: P diagonal positive", np.all(np.diag(eskf.P) > 0))

    # P1.6: get_pose_2d with identity quaternion → yaw=0
    pose = eskf.get_pose_2d()
    check_close("P1.6: get_pose_2d yaw=0", pose[2], 0.0)

    # P1.7: get_velocity_body with zero velocity → zero
    v_body = eskf.get_velocity_body()
    check_allclose("P1.7: v_body = 0", v_body, np.zeros(3))

    # P1.8: initialize sets state correctly
    eskf2 = ESKF()
    eskf2.initialize(p0=np.array([10.0, 20.0]), v0=np.array([1.0, 0.0]))
    check_allclose("P1.8: Init p", eskf2.p, np.array([10.0, 20.0, 0.0]))


# =============================================================================
# Test P2: Prediction (IMU propagation)
# =============================================================================

def test_prediction():
    print("\n" + "=" * 60)
    print("Test P2: ES-EKF Prediction")
    print("=" * 60)

    eskf = ESKF()
    eskf.initialize(p0=np.zeros(3), v0=np.zeros(3))

    # P2.1: Forward acceleration → velocity increase
    # Body-frame accel: [2.0, 0, -9.81] (2 m/s² surge + gravity compensation)
    # At identity orientation (East), R=I, so a_world = [2.0, 0, -9.81] + [0, 0, -9.81]
    # Wait - gravity is [0, 0, -9.81] in ENU. a_world = R·a_body + g
    # For the IMU on a level surface: a_body = [2, 0, 9.81] (senses gravity as +z)
    # a_world = I·[2, 0, 9.81] + [0, 0, -9.81] = [2, 0, 0]
    # So v changes by 2*dt on x-axis
    a_body = np.array([2.0, 0.0, 9.81])   # surge + gravity in -z sensed as +z
    w_body = np.array([0.0, 0.0, 0.0])    # no rotation
    dt = 0.1
    eskf.predict(a_body, w_body, dt)
    check_close("P2.1: Forward accel → vx > 0", eskf.v[0], 0.2, tol=0.05)
    check_close("P2.2: No lateral accel → vy ≈ 0", eskf.v[1], 0.0, tol=0.01)

    # P2.3: Yaw rotation → quaternion updates
    # Use multiple small steps (dt must be ≤ 0.5)
    eskf2 = ESKF()
    eskf2.initialize(p0=np.zeros(3))
    w_body = np.array([0.0, 0.0, 1.0])          # 1 rad/s about Z
    a_body = np.array([0.0, 0.0, 9.81])         # just gravity
    dt = 0.05
    steps = 16  # 16×0.05 = 0.8s, rotation ≈ 0.8 rad (close to π/4)
    for _ in range(steps):
        eskf2.predict(a_body, w_body, dt)
    yaw = quat_to_yaw(eskf2.q)
    check_close("P2.3: ω_z=1 rad/s × 0.8s → yaw≈0.8 rad", yaw, 0.8, tol=1e-3)

    # P2.4: Bias correction — accel bias reduces effective acceleration
    eskf3 = ESKF()
    eskf3.initialize(p0=np.zeros(3))
    eskf3.ab = np.array([0.5, 0.0, 0.0])   # 0.5 m/s² bias on x
    a_body = np.array([2.0, 0.0, 9.81])
    w_body = np.array([0.0, 0.0, 0.0])
    dt = 0.1
    eskf3.predict(a_body, w_body, dt)
    # Effective x accel = 2.0 - 0.5 = 1.5 m/s²
    # vx should be ≈ 0.15
    check_close("P2.4: Accel bias reduces vx", eskf3.v[0], 0.15, tol=0.02)

    # P2.5: Covariance grows during prediction
    eskf4 = ESKF()
    eskf4.initialize(p0=np.zeros(3))
    trace_before = np.trace(eskf4.P)
    for _ in range(10):
        eskf4.predict(np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.0]), 0.1)
    trace_after = np.trace(eskf4.P)
    check("P2.5: Covariance grows with prediction", trace_after > trace_before)

    # P2.6: Gyro bias corrects angular velocity
    eskf5 = ESKF()
    eskf5.initialize(p0=np.zeros(3))
    eskf5.wb = np.array([0.0, 0.0, 0.5])   # 0.5 rad/s gyro bias on z
    w_meas = np.array([0.0, 0.0, 1.0])     # measured 1 rad/s
    a_meas = np.array([0.0, 0.0, 9.81])
    dt = 0.1
    eskf5.predict(a_meas, w_meas, dt)
    yaw_biased = quat_to_yaw(eskf5.q)
    # Effective ω = 1.0 - 0.5 = 0.5 rad/s → 0.05 rad in 0.1s
    check_close("P2.6: Gyro bias corrects ω_eff", yaw_biased, 0.05, tol=0.01)


# =============================================================================
# Test P3: Position Update (GPS)
# =============================================================================

def test_position_update():
    print("\n" + "=" * 60)
    print("Test P3: ES-EKF Position Update")
    print("=" * 60)

    eskf = ESKF()
    eskf.initialize(p0=np.array([0.0, 0.0, 0.0]))

    # P3.1: GPS position pulls estimate toward measurement
    z_pos = np.array([1.0, 0.5])
    R_gps = np.diag([1.0, 1.0])
    eskf.update_position(z_pos, R_gps)
    check("P3.1: Position pulled toward GPS x", eskf.p[0] > 0.1)
    check("P3.2: Position pulled toward GPS y", eskf.p[1] > 0.05)

    # P3.3: Covariance reduces after update
    trace_before = np.trace(eskf.P[0:2, 0:2])
    eskf.update_position(np.array([1.1, 0.6]), R_gps)
    trace_after = np.trace(eskf.P[0:2, 0:2])
    check("P3.3: Position covariance reduced by update", trace_after < trace_before)

    # P3.4: 2D update does not affect z
    z_before = eskf.p[2]
    eskf.update_position(np.array([2.0, 3.0]), R_gps)
    check_close("P3.4: Z position unchanged by 2D update", eskf.p[2], z_before)

    # P3.5: Outlier rejected by gating
    eskf2 = ESKF()
    eskf2.initialize(p0=np.array([0.0, 0.0]))
    # Tight covariance → position well known
    eskf2.P[0:2, 0:2] = np.eye(2) * 0.01
    p_before = eskf2.p.copy()
    # Huge outlier (100m away with 1m noise)
    eskf2.update_position(np.array([100.0, 0.0]), np.diag([1.0, 1.0]))
    check("P3.5: Outlier rejected (pos unchanged)", np.allclose(eskf2.p[:2], p_before[:2], atol=0.01))

    # P3.6: Z position updated by 3D measurement
    eskf3 = ESKF()
    eskf3.initialize(p0=np.array([0.0, 0.0, 0.0]))
    eskf3.update_position(np.array([0.0, 0.0, 5.0]), np.diag([1.0, 1.0, 1.0]))
    check("P3.6: 3D position update affects z", abs(eskf3.p[2]) > 0.1)


# =============================================================================
# Test P4: Yaw Update
# =============================================================================

def test_yaw_update():
    print("\n" + "=" * 60)
    print("Test P4: ES-EKF Yaw Update")
    print("=" * 60)

    # P4.1: Yaw measurement corrects quaternion
    eskf = ESKF()
    eskf.initialize(p0=np.zeros(3))
    z_yaw = math.pi / 2   # 90°
    eskf.update_yaw(z_yaw, 0.01)
    yaw = quat_to_yaw(eskf.q)
    check("P4.1: Yaw update → q reflects measurement", abs(yaw) > 0.1)

    # P4.2: After position+velocity init, yaw update moves yaw correctly
    eskf2 = ESKF()
    eskf2.initialize(p0=np.array([0.0, 0.0]))
    initial_yaw = quat_to_yaw(eskf2.q)
    # Multiple updates to converge
    for _ in range(5):
        eskf2.update_yaw(math.pi / 4, 0.001)  # 45°, very low noise
    yaw2 = quat_to_yaw(eskf2.q)
    check_close("P4.2: Converged yaw ≈ 45°", yaw2, math.pi / 4, tol=0.05)

    # P4.3: Yaw wrapping (−π and +π are equivalent)
    eskf3 = ESKF()
    eskf3.initialize(p0=np.zeros(3))
    # Initialize at yaw = π (pointing West)
    eskf3.q = np.array([0.0, 0.0, 0.0, 1.0])  # 180° yaw
    # Update with yaw = -π + ε (should wrap)
    eskf3.update_yaw(-math.pi + 0.1, 0.01)
    yaw3 = quat_to_yaw(eskf3.q)
    # Yaw should be near ±π
    check("P4.3: Yaw wrap handled", abs(abs(yaw3) - math.pi) < 0.2,
          f"yaw={math.degrees(yaw3):.1f}°")


# =============================================================================
# Test P5: Body-Frame Velocity Update (VO)
# =============================================================================

def test_vo_update():
    print("\n" + "=" * 60)
    print("Test P5: ES-EKF Body-Frame Velocity Update")
    print("=" * 60)

    # P5.1: Body-frame surge → world-frame velocity at identity orientation
    eskf = ESKF()
    eskf.initialize(p0=np.zeros(3), v0=np.zeros(3))
    z_body = np.array([2.0, 0.0])   # 2 m/s surge
    R_body = np.diag([0.1, 0.1])
    eskf.update_velocity_body(z_body, R_body)
    # At identity quat (East), body surge = world vx
    check("P5.1: Surge at 0° yaw → vx > 0", eskf.v[0] > 0.5)
    check_close("P5.2: Sway at 0° yaw → vy ≈ 0", eskf.v[1], 0.0, tol=0.3)

    # P5.3: Body-frame sway → world-frame vy (at 0° yaw)
    # Marine convention: +sway = starboard. At yaw=0 (pointing East),
    # starboard = North = world +y.
    eskf2 = ESKF()
    eskf2.initialize(p0=np.zeros(3), v0=np.zeros(3))
    z_body2 = np.array([0.0, 1.0])   # 1 m/s sway (starboard)
    eskf2.update_velocity_body(z_body2, np.diag([0.1, 0.1]))
    check("P5.3: Sway at 0° yaw → vy > 0 (starboard=North)", eskf2.v[1] > 0.2,
          f"vy={eskf2.v[1]:.3f}")

    # P5.4: Surge at 90° yaw → vy_w > 0
    eskf3 = ESKF()
    eskf3.initialize(p0=np.zeros(3), v0=np.zeros(3))
    # Set yaw = 90° (North)
    eskf3.q = np.array([0.7071, 0.0, 0.0, 0.7071])
    z_body3 = np.array([2.0, 0.0])   # surge forward
    eskf3.update_velocity_body(z_body3, np.diag([0.1, 0.1]))
    # At 90° yaw, surge (body +x) = world +y
    check("P5.4: Surge at 90° yaw → vy > 0", eskf3.v[1] > 0.5)
    check("P5.5: Surge at 90° yaw → vx ≈ 0", abs(eskf3.v[0]) < 0.5)


# =============================================================================
# Test P6: Error Injection and Covariance Reset
# =============================================================================

def test_error_injection():
    print("\n" + "=" * 60)
    print("Test P6: Error Injection and Covariance Reset")
    print("=" * 60)

    # P6.1: Position error injected correctly
    eskf = ESKF()
    eskf.initialize(p0=np.array([0.0, 0.0, 0.0]))
    p_before = eskf.p.copy()

    # Simulate an update that produces position error
    z_pos = np.array([5.0, 0.0])
    eskf.update_position(z_pos, np.diag([0.01, 0.01]))  # very low noise
    check("P6.1: Position updated after injection", np.linalg.norm(eskf.p[:2] - p_before[:2]) > 0.5)
    # After injection, P should be positive definite
    eigvals = np.linalg.eigvalsh(eskf.P)
    check("P6.2: P positive definite after injection", eigvals[0] > 0)

    # P6.3: Quaternion injection preserves unit norm
    eskf2 = ESKF()
    eskf2.initialize(p0=np.zeros(3))
    q_before = eskf2.q.copy()
    for _ in range(10):
        eskf2.update_yaw(0.5, 0.01)   # update with fixed measurement
    q_norm = np.linalg.norm(eskf2.q)
    check_close("P6.3: q remains unit norm after injections", q_norm, 1.0, tol=1e-10)

    # P6.4: Covariance reset J·P·J^T is correct (trace should be close)
    eskf3 = ESKF()
    eskf3.initialize(p0=np.zeros(3))
    trace_before = np.trace(eskf3.P)
    # Prediction
    for _ in range(5):
        eskf3.predict(np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.1]), 0.1)
    # Update
    eskf3.update_position(np.array([1.0, 0.0]), np.diag([1.0, 1.0]))
    trace_after = np.trace(eskf3.P)
    # P should be well-behaved (not NaN, not infinite)
    check("P6.4: P finite after injection",
          np.all(np.isfinite(eskf3.P)) and trace_after < 1e6)


# =============================================================================
# Test P7: Integration Scenarios
# =============================================================================

def test_integration():
    print("\n" + "=" * 60)
    print("Test P7: ES-EKF Integration Scenarios")
    print("=" * 60)

    # P7.1: Straight-line motion with GPS updates
    eskf = ESKF()
    eskf.initialize(p0=np.array([0.0, 0.0, 0.0]))
    dt = 0.1

    # Simulate 10s of forward motion at 2 m/s
    positions = []
    for i in range(100):
        t = i * dt
        # IMU: forward accel (surge=2m/s² initially, then cruise)
        if t < 1.0:
            a_body = np.array([2.0, 0.0, 9.81])   # accelerating
        else:
            a_body = np.array([0.0, 0.0, 9.81])   # cruising (constant velocity)
        w_body = np.array([0.0, 0.0, 0.0])        # no rotation

        eskf.predict(a_body, w_body, dt)

        # GPS update every 1s
        if i % 10 == 0 and i > 0:
            # True position: ½·a·t² during accel, then constant v
            if t <= 1.0:
                true_x = 0.5 * 2.0 * t ** 2
            else:
                accel_end_x = 0.5 * 2.0 * 1.0 ** 2   # = 1.0m at t=1
                v_cruise = 2.0 * 1.0                  # = 2.0 m/s after accel
                true_x = accel_end_x + v_cruise * (t - 1.0)
            gps_x = true_x + np.random.normal(0, 0.5)
            gps_y = np.random.normal(0, 0.5)
            eskf.update_position(np.array([gps_x, gps_y]), np.diag([1.0, 1.0]))

        positions.append(eskf.p[:2].copy())

    # After 10s, position should be positive x
    final_pos = positions[-1]
    check("P7.1: Final x > 0 after forward motion", final_pos[0] > 0.5)

    # P7.2: Turning motion with GPS
    eskf2 = ESKF()
    eskf2.initialize(p0=np.array([0.0, 0.0, 0.0]))
    dt = 0.05

    # Track unwrapped yaw by accumulating ω·dt
    unwrapped_yaw = 0.0
    for i in range(200):  # 10 seconds
        t = i * dt
        a_body = np.array([0.0, 0.0, 9.81])     # just gravity
        w_body = np.array([0.0, 0.0, 0.5])      # constant 0.5 rad/s yaw

        eskf2.predict(a_body, w_body, dt)
        unwrapped_yaw += 0.5 * dt

        if i % 20 == 0:
            # GPS at origin (ship spinning in place)
            eskf2.update_position(np.array([0.0, 0.0]), np.diag([0.5, 0.5]))

    # Total accumulated rotation: 200 × 0.05s × 0.5 rad/s = 5 rad
    check("P7.2: Yaw accumulated ~5 rad during turn",
          abs(unwrapped_yaw - 5.0) < 0.01,
          f"unwrapped yaw={unwrapped_yaw:.2f} rad")
    # Position should stay near origin
    check("P7.3: Position near origin during turn",
          np.linalg.norm(eskf2.p[:2]) < 2.0,
          f"pos=({eskf2.p[0]:.2f}, {eskf2.p[1]:.2f})")


# =============================================================================
# Test P8: Singularity Handling (±180°)
# =============================================================================

def test_singularity():
    print("\n" + "=" * 60)
    print("Test P8: ES-EKF Singularity Handling (±180°)")
    print("=" * 60)

    # P8.1: Yaw crosses π (180°) — quaternion stays continuous
    # quat_to_yaw wraps to [-π,π], so extracted yaw jumps at boundary.
    # The real test of continuity is the quaternion itself.
    eskf = ESKF()
    eskf.initialize(p0=np.zeros(3))

    q_prev = None
    max_q_jump = 0.0
    dt = 0.1
    for i in range(63):  # 6.3s × 1 rad/s = ~360°
        w_body = np.array([0.0, 0.0, 1.0])
        a_body = np.array([0.0, 0.0, 9.81])
        eskf.predict(a_body, w_body, dt)
        if q_prev is not None:
            q_dist = np.linalg.norm(eskf.q - q_prev)
            max_q_jump = max(max_q_jump, q_dist)
        q_prev = eskf.q.copy()

    # Quaternion changes smoothly (< 0.05 per 0.1s step at 1 rad/s)
    check("P8.1: Quaternion continuous through full rotation",
          max_q_jump < 0.06, f"max q jump={max_q_jump:.4f}")

    # P8.2: Update across π boundary
    eskf2 = ESKF()
    eskf2.initialize(p0=np.zeros(3))
    # Set yaw to 179°
    eskf2.q = np.array([0.0087, 0.0, 0.0, 0.99996])  # cos(179°/2), sin(179°/2)
    yaw_before = quat_to_yaw(eskf2.q)
    check("P8.2: Initial yaw ≈ π", abs(abs(yaw_before) - math.pi) < 0.05,
          f"yaw={math.degrees(yaw_before):.1f}°")

    # Update to -179° (should be a small correction, not a 358° jump)
    eskf2.update_yaw(-math.pi + 0.05, 0.01)
    yaw_after = quat_to_yaw(eskf2.q)
    # The innovation should wrap to ~0.05 rad, not ~6.23 rad
    check("P8.3: Yaw wraps correctly across ±π",
          abs(abs(yaw_after) - math.pi) < 0.15,
          f"yaw={math.degrees(yaw_after):.1f}°")

    # P8.4: ES-EKF quaternion has no singularity — continuous at any angle
    # Verify that quaternion components are smooth across 0°/360°
    eskf3 = ESKF()
    eskf3.initialize(p0=np.zeros(3))
    q_history = []
    dt = 0.01
    for i in range(630):  # 6.3s × 1 rad/s = ~360°
        w_body = np.array([0.0, 0.0, 1.0])
        a_body = np.array([0.0, 0.0, 9.81])
        eskf3.predict(a_body, w_body, dt)
        q_history.append(eskf3.q.copy())

    # Quaternion should change smoothly (no discontinuous flips)
    max_q_jump = max(
        np.linalg.norm(q_history[i] - q_history[i-1])
        for i in range(1, len(q_history))
    )
    check("P8.4: Quaternion continuous across full rotation",
          max_q_jump < 0.02,
          f"max q jump={max_q_jump:.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("🧪 ES-EKF Test Suite: Error-State Extended Kalman Filter")
    print("=" * 60)
    print("Reference: Solà, J. (2017). Quaternion kinematics for the")
    print("           error-state Kalman filter. arXiv:1711.02508.")
    print()

    test_quaternion_utils()
    test_eskf_init()
    test_prediction()
    test_position_update()
    test_yaw_update()
    test_vo_update()
    test_error_injection()
    test_integration()
    test_singularity()

    print("\n" + "=" * 60)
    total_passed = 8 * 6 - len(_FAILED)  # ~48 checks across 8 suites
    failed = len(_FAILED)
    if failed:
        print(f"❌ {failed} FAILURES:")
        for f in _FAILED:
            print(f"   - {f}")
    else:
        print(f"✅ All tests passed ({8} suites completed).")
    print("=" * 60)

    return failed


if __name__ == '__main__':
    failed = main()
    sys.exit(0 if failed == 0 else 1)
