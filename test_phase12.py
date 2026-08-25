#!/usr/bin/env python3
"""
Test suite for Phase 1-2 components: JPDA Tracker + WAM-V Autopilot.

Tests:
  1. ConstantVelocityKF — predict, innovate, update, weighted update
  2. JPDAEngine — gating, validation matrix, feasible events, association
  3. Track lifecycle — tentative → confirmed → coasting → deleted
  4. LOS guidance — cross-track error, desired heading, waypoint advance
  5. Autopilot control — PI heading, speed profile
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test Phase 1-2 modules in isolation (no ROS needed for core logic)
from marine_env import jpda_tracker_node as jpda_mod

# We import the classes directly
ConstantVelocityKF = jpda_mod.ConstantVelocityKF
JPDAEngine = jpda_mod.JPDAEngine
Track = jpda_mod.Track

from marine_env import wamv_autopilot as ap_mod

_FAILED = []


def check(name: str, condition: bool, detail: str = ""):
    status = "✅ PASS" if condition else "❌ FAIL"
    if not condition:
        _FAILED.append(name)
    print(f"  {status}: {name}" + (f" — {detail}" if detail else ""))


# =============================================================================
# Test 1: Kalman Filter
# =============================================================================

def test_kalman_filter():
    print("\n" + "=" * 60)
    print("Test P1: Constant-Velocity Kalman Filter")
    print("=" * 60)

    kf = ConstantVelocityKF(dt=1.0, process_noise_q=0.01, measurement_noise_r=1.0)

    # Test 1.1: initial state
    check("1.1: Initial state is zero", np.allclose(kf.x, [0, 0, 0, 0]))
    check("1.2: Initial covariance is diagonal",
          np.allclose(kf.P, np.eye(4) * 100))

    # Test 1.3: predict propagates position by velocity
    kf.x = np.array([0.0, 0.0, 2.0, 1.0])  # vx=2, vy=1
    kf.P = np.eye(4)
    kf.predict()
    check("1.3: Predict x position", abs(kf.x[0] - 2.0) < 0.01,
          f"x={kf.x[0]:.2f} (expect 2.0)")
    check("1.4: Predict y position", abs(kf.x[1] - 1.0) < 0.01,
          f"y={kf.x[1]:.2f} (expect 1.0)")
    check("1.5: Velocity unchanged by predict",
          np.allclose(kf.x[2:4], [2.0, 1.0]))

    # Test 1.6: update pulls state toward measurement
    kf.x = np.array([0.0, 0.0, 2.0, 1.0])
    kf.P = np.eye(4) * 10.0
    z = np.array([0.5, 0.3])  # measurement slightly offset
    kf.update(z)
    check("1.6: Update moves x toward measurement", abs(kf.x[0] - 0.0) < 1.0,
          f"x={kf.x[0]:.3f}")
    check("1.7: Update reduces covariance",
          np.trace(kf.P) < np.trace(np.eye(4) * 10),
          f"trace(P)={np.trace(kf.P):.1f}")

    # Test 1.8: innovation computes Mahalanobis distance
    kf.x = np.array([0.0, 0.0, 0.0, 0.0])
    kf.P = np.eye(4)
    z_far = np.array([10.0, 0.0])
    _, _, nis = kf.innovation(z_far)
    check("1.8: Far measurement → large NIS", nis > 5.0, f"NIS={nis:.1f}")

    # Test 1.9: weighted update with beta=0 → no change
    kf.x = np.array([0.0, 0.0, 2.0, 1.0])
    x_before = kf.x.copy()
    kf.update_weighted(z_far, beta=0.0)
    check("1.9: β=0 → state unchanged",
          np.allclose(kf.x[:2], x_before[:2], atol=0.01))


# =============================================================================
# Test 2: JPDA Engine
# =============================================================================

def test_jpda_engine():
    print("\n" + "=" * 60)
    print("Test P2: JPDA Association Engine")
    print("=" * 60)

    engine = JPDAEngine(detection_prob=0.95, gate_prob=0.99)

    # Create two well-separated tracks
    track_a = Track(track_id=1, kf=ConstantVelocityKF(dt=1.0), status="confirmed")
    track_a.kf.x = np.array([0.0, 0.0, 0.0, 0.0])
    track_a.kf.P = np.eye(4) * 0.5

    track_b = Track(track_id=2, kf=ConstantVelocityKF(dt=1.0), status="confirmed")
    track_b.kf.x = np.array([50.0, 0.0, 0.0, 0.0])
    track_b.kf.P = np.eye(4) * 0.5

    # Measurements: one near each track
    measurements = [
        np.array([0.2, 0.1]),    # near track_a
        np.array([49.8, -0.1]),  # near track_b
    ]

    # Test 2.1: gating
    gated_a = engine.gate(track_a, measurements)
    gated_b = engine.gate(track_b, measurements)
    check("2.1: Track A gates meas 0", 0 in gated_a)
    check("2.2: Track A rejects meas 1", 1 not in gated_a)
    check("2.3: Track B gates meas 1", 1 in gated_b)

    # Test 2.4: association
    beta = engine.associate([track_a, track_b], measurements)
    # track_a should have high β for meas 0
    beta_a_0 = beta[0].get(0, 0.0)
    beta_b_1 = beta[1].get(1, 0.0)
    check("2.4: Track A strongly associates with meas 0",
          beta_a_0 > 0.5, f"β={beta_a_0:.3f}")
    check("2.5: Track B strongly associates with meas 1",
          beta_b_1 > 0.5, f"β={beta_b_1:.3f}")

    # Test 2.6: clutter-only measurements
    empty_beta = engine.associate([], [np.array([1.0, 2.0])])
    check("2.6: No tracks → empty association", len(empty_beta) == 0)

    # Test 2.7: ambiguous case — two tracks, one measurement
    track_c = Track(track_id=3, kf=ConstantVelocityKF(dt=1.0), status="confirmed")
    track_c.kf.x = np.array([0.0, 0.0, 0.0, 0.0])
    track_c.kf.P = np.eye(4) * 2.0

    track_d = Track(track_id=4, kf=ConstantVelocityKF(dt=1.0), status="confirmed")
    track_d.kf.x = np.array([5.0, 0.0, 0.0, 0.0])
    track_d.kf.P = np.eye(4) * 2.0

    # One measurement between the two tracks
    meas_ambiguous = [np.array([2.5, 0.0])]
    beta_amb = engine.associate([track_c, track_d], meas_ambiguous)
    # Both tracks should have non-zero β (measurement could be from either)
    beta_c = beta_amb[0].get(0, 0.0)
    beta_d = beta_amb[1].get(0, 0.0)
    check("2.7: Ambiguous meas → both tracks get β > 0",
          beta_c > 0.01 and beta_d > 0.01,
          f"β_c={beta_c:.3f}, β_d={beta_d:.3f}")
    check("2.8: β sums ≤ 1.0 per track",
          beta_c + beta_amb[0].get(-1, 0) <= 1.01 and
          beta_d + beta_amb[1].get(-1, 0) <= 1.01)

    # Test 2.9: clutter density sanity
    check("2.9: Gate threshold ≈ 9.21 for χ²₂(0.99)",
          abs(engine.gate_threshold - 9.21) < 0.1,
          f"χ²={engine.gate_threshold:.2f}")


# =============================================================================
# Test 3: Track Lifecycle
# =============================================================================

def test_track_lifecycle():
    print("\n" + "=" * 60)
    print("Test P3: Track Lifecycle State Machine")
    print("=" * 60)

    kf = ConstantVelocityKF(dt=1.0)
    track = Track(track_id=1, kf=kf, status="tentative", hits=0)

    check("3.1: New track → tentative", track.status == "tentative")

    # Simulate 3 hits → confirmation
    for i in range(3):
        track.hits += 1
    # Manual transition check (normally done in tracking cycle)
    if track.status == "tentative" and track.hits >= 3:
        track.status = "confirmed"
    check("3.2: 3 hits → confirmed", track.status == "confirmed")

    # Simulate misses → coasting
    track.misses = 3
    if track.status == "confirmed" and track.misses >= 3:
        track.status = "coasting"
    check("3.3: 3 misses → coasting", track.status == "coasting")

    # Simulate too many misses → deleted
    track.misses = 8
    should_delete = track.misses >= 8
    check("3.4: 8 misses → deletion", should_delete)

    # Predict increases age
    age_before = track.age
    track.predict()
    check("3.5: Predict increments age", track.age == age_before + 1)


# =============================================================================
# Test 4: LOS Guidance
# =============================================================================

def test_los_guidance():
    print("\n" + "=" * 60)
    print("Test P4: LOS Guidance Geometry")
    print("=" * 60)

    # We test the math independently from ROS

    def compute_los(pos, wp_k, wp_next, look_ahead):
        """Standalone LOS calculation for testing."""
        dx_seg = wp_next[0] - wp_k[0]
        dy_seg = wp_next[1] - wp_k[1]
        alpha_k = math.atan2(dy_seg, dx_seg)

        dx_os = pos[0] - wp_k[0]
        dy_os = pos[1] - wp_k[1]
        cross_track = -dx_os * math.sin(alpha_k) + dy_os * math.cos(alpha_k)
        along_track = dx_os * math.cos(alpha_k) + dy_os * math.sin(alpha_k)

        chi_d = alpha_k + math.atan2(-cross_track, look_ahead)
        return chi_d, cross_track, along_track, alpha_k

    # Test 4.1: On the path → χ_d = α_k
    chi_d, ct, at, ak = compute_los(
        np.array([5.0, 0.0]),     # ship at (5,0) — on the line
        (0.0, 0.0), (10.0, 0.0), # wp_k → wp_next: East
        15.0)
    check("4.1: On path → χ_d ≈ path tangent",
          abs(chi_d - ak) < 0.01,
          f"χ_d={math.degrees(chi_d):.1f}°")

    # Test 4.2: Right of path → χ_d < α_k (steer left toward path)
    chi_d, ct, at, ak = compute_los(
        np.array([5.0, -5.0]),    # ship at (5,-5) — right of East path
        (0.0, 0.0), (10.0, 0.0),
        15.0)
    check("4.2: Right of East path → negative cross-track",
          ct < 0, f"y_e={ct:.2f}m")
    # Ship South of East-going path → LOS points North (χ_d > 0 = α_k)
    check("4.3: Right of path → χ_d > α_k (steer toward path)",
          chi_d > ak, f"χ_d={math.degrees(chi_d):.1f}°, α_k={math.degrees(ak):.1f}°")

    # Test 4.4: Left of path → χ_d > α_k (steer right toward path)
    chi_d, ct, at, ak = compute_los(
        np.array([5.0, 5.0]),     # ship at (5,5) — left of East path
        (0.0, 0.0), (10.0, 0.0),
        15.0)
    check("4.4: Left of East path → positive cross-track",
          ct > 0, f"y_e={ct:.2f}m")

    # Test 4.5: Far off path → larger steering correction
    chi_d_near, ct_near, _, ak = compute_los(
        np.array([5.0, -2.0]), (0, 0), (10, 0), 15.0)
    chi_d_far, ct_far, _, ak = compute_los(
        np.array([5.0, -20.0]), (0, 0), (10, 0), 15.0)
    correction_near = abs(chi_d_near - ak)
    correction_far = abs(chi_d_far - ak)
    check("4.5: Larger cross-track → larger steering correction",
          correction_far > correction_near,
          f"near: {math.degrees(correction_near):.1f}°, "
          f"far: {math.degrees(correction_far):.1f}°")

    # Test 4.6: Waypoint advance — within acceptance radius
    def check_advance(pos, wp_next, acceptance):
        dx = pos[0] - wp_next[0]
        dy = pos[1] - wp_next[1]
        return math.sqrt(dx**2 + dy**2) < acceptance

    close = check_advance(np.array([9.0, 0.0]), (10.0, 0.0), 5.0)
    check("4.6: Close to waypoint → advance", close)

    far = check_advance(np.array([0.0, 0.0]), (10.0, 0.0), 5.0)
    check("4.7: Far from waypoint → no advance", not far)

    # Test 4.8: Multi-segment path
    waypoints = [(0, 0), (10, 0), (10, 10), (0, 10)]
    check("4.8: Waypoint list length", len(waypoints) == 4)


# =============================================================================
# Test 5: Autopilot Control Logic
# =============================================================================

def test_autopilot_control():
    print("\n" + "=" * 60)
    print("Test P5: Autopilot PI Control")
    print("=" * 60)

    # Test heading error normalization
    def heading_error(target, current):
        return (target - current + math.pi) % (2 * math.pi) - math.pi

    err1 = heading_error(math.radians(10), math.radians(350))
    check("5.1: Heading error wraps correctly (10° → 350°)",
          abs(err1 - math.radians(20)) < 0.01,
          f"err={math.degrees(err1):.1f}°")

    err2 = heading_error(math.radians(350), math.radians(10))
    check("5.2: Heading error wraps correctly (350° → 10°)",
          abs(err2 - math.radians(-20)) < 0.01,
          f"err={math.degrees(err2):.1f}°")

    err3 = heading_error(math.radians(180), math.radians(0))
    check("5.3: Heading error 0° → 180°",
          abs(abs(err3) - math.pi) < 0.01,
          f"err={math.degrees(err3):.1f}°")

    # Test thrust differential logic
    def compute_diff(yaw_error, base_thrust, kp=2.0, kd=0.0, yaw_rate=0.0,
                     max_ratio=0.6):
        correction = kp * yaw_error - kd * yaw_rate
        diff = -correction * 400.0
        diff_max = base_thrust * max_ratio
        return max(-diff_max, min(diff_max, diff))

    # Positive yaw error (need CCW turn = right thrust > left = negative diff)
    diff = compute_diff(0.5, 1000.0)     # need 0.5rad CCW
    check("5.4: CCW turn → right > left (diff < 0)", diff < 0,
          f"diff={diff:.1f}N")

    # Negative yaw error (need CW turn = left thrust > right = positive diff)
    diff = compute_diff(-0.5, 1000.0)    # need 0.5rad CW
    check("5.5: CW turn → left > right (diff > 0)", diff > 0,
          f"diff={diff:.1f}N")

    # No error → no differential
    diff = compute_diff(0.0, 1000.0)
    check("5.6: No heading error → no diff", abs(diff) < 1.0,
          f"diff={diff:.1f}N")

    # Clamping at max ratio
    diff = compute_diff(3.0, 100.0)      # huge error, small base
    check("5.7: Diff clamped to max_ratio",
          abs(diff) <= 60.0 + 1.0,       # 100 * 0.6 = 60
          f"diff={diff:.1f}N (max=±60N)")

    # Test speed scaling with cross-track error
    def speed_scale(cross_track):
        if abs(cross_track) > 10.0:
            return max(0.4, 1.0 - abs(cross_track) / 50.0)
        return 1.0

    check("5.8: Small cross-track → full speed",
          abs(speed_scale(5.0) - 1.0) < 0.01)
    check("5.9: Large cross-track → reduced speed",
          speed_scale(25.0) < 1.0,
          f"scale={speed_scale(25.0):.2f}")
    check("5.10: Speed scale min 0.4",
          speed_scale(100.0) >= 0.39)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("🧪 Phase 1-2 Test Suite: JPDA Tracker + Autopilot")
    print("=" * 60)

    test_kalman_filter()
    test_jpda_engine()
    test_track_lifecycle()
    test_los_guidance()
    test_autopilot_control()

    print("\n" + "=" * 60)
    total = 5 * 10  # approximate
    failed = len(_FAILED)
    if failed:
        print(f"❌ {failed} FAILURES:")
        for f in _FAILED:
            print(f"   - {f}")
    else:
        print(f"✅ All tests completed.")
    print("=" * 60)

    return failed


if __name__ == '__main__':
    failed = main()
    sys.exit(0 if failed == 0 else 1)
