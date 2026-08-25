#!/usr/bin/env python3
"""
Offline collision geometry verification for all 20 COLREGS scenarios.
======================================================================

Uses Fossen 3DOF kinematics (constant velocity, no environmental forces)
to verify that OS and target ships come within collision detection range
after the thrust model fix.

Compares OLD thrust model (TS=600 N/(m/s), OS boosted 1.35x) vs
NEW thrust model (TS=1200 N/(m/s), OS no boost, PI speed control).

Usage:
  python3 verify_collision_fix.py           # all 20 scenarios
  python3 verify_collision_fix.py --scenario 4   # single scenario
  python3 verify_collision_fix.py --old     # show old model results too
"""
import yaml
import math
import sys
import os
from pathlib import Path

SCENARIO_FILE = Path(__file__).parent / 'config' / 'colregs_20_new_scenarios.yaml'

# ── Physical constants ──
# WAM-V hull drag: k ≈ 700-800 N/(m/s)^2
# With OS PI controller: achieves target speed accurately
# Old TS thrust coefficient = 600 N/(m/s) → ~0.6-0.85 m/s per target m/s
# New TS thrust coefficient = 1200 N/(m/s) → target speed achievable

OLD_TS_COEFF = 600.0   # old thrust-to-speed ratio
NEW_TS_COEFF = 1200.0  # new (matches OS feedforward)
OLD_OS_BOOST = 1.35    # old OS speed multiplier
NEW_OS_BOOST = 1.0     # new (design speed)

STOP_DIST = 5.0        # collision detection threshold (m)
SIM_TIME = 120.0       # simulate up to 120s
DT = 0.1               # simulation step


def compass_to_enu(yaw_compass):
    """罗经方位角 → ENU yaw."""
    ros_yaw = math.pi / 2.0 - yaw_compass
    return math.atan2(math.sin(ros_yaw), math.cos(ros_yaw))


def simulate_ship(x0, y0, yaw_compass, target_speed, thrust_coeff, dt, sim_time):
    """
    Simulate a ship under constant-thrust open-loop control.

    Args:
        thrust_coeff: N per m/s of target speed (1200 → achieves target, 600 → ~0.7x)

    Returns:
        list of (t, x, y) positions
    """
    yaw_enu = compass_to_enu(yaw_compass)
    thrust = target_speed * thrust_coeff
    # Equilibrium speed: thrust = k * v^2, k ≈ 800
    k_drag = 800.0
    eq_speed = math.sqrt(max(thrust, 0.0) / k_drag)

    positions = []
    x, y = x0, y0
    t = 0.0
    while t <= sim_time:
        positions.append((t, x, y))
        # Use equilibrium speed (instantaneous — simplified)
        x += eq_speed * math.cos(yaw_enu) * dt
        y += eq_speed * math.sin(yaw_enu) * dt
        t += dt
    return positions


def simulate_ship_accurate(x0, y0, yaw_compass, target_speed, thrust_coeff, dt, sim_time):
    """
    Simulate with proper speed dynamics (dv/dt = (thrust - k*v^2) / m).

    More accurate than instantaneous equilibrium.
    """
    yaw_enu = compass_to_enu(yaw_compass)
    thrust = target_speed * thrust_coeff
    k_drag = 800.0   # N/(m/s)^2
    mass = 300.0     # kg (WAM-V approximate)

    positions = []
    x, y = x0, y0
    v = 0.0  # start from rest
    t = 0.0
    while t <= sim_time:
        positions.append((t, x, y))
        # Speed dynamics
        drag = k_drag * v * v
        accel = (thrust - drag) / mass
        v += accel * dt
        if v < 0:
            v = 0.0

        x += v * math.cos(yaw_enu) * dt
        y += v * math.sin(yaw_enu) * dt
        t += dt
    return positions


def compute_min_cpa(os_positions, ts_positions_list):
    """
    Compute minimum CPA between OS and each TS over the simulation.

    Returns:
        list of (ts_name, min_cpa, t_at_min, os_xy, ts_xy)
    """
    results = []
    for ts_name, ts_positions in ts_positions_list:
        min_cpa = float('inf')
        t_at_min = 0.0
        os_at_min = (0, 0)
        ts_at_min = (0, 0)

        for (t_os, ox, oy), (t_ts, tx, ty) in zip(os_positions, ts_positions):
            # Times should match since same dt
            d = math.hypot(tx - ox, ty - oy)
            if d < min_cpa:
                min_cpa = d
                t_at_min = (t_os + t_ts) / 2
                os_at_min = (ox, oy)
                ts_at_min = (tx, ty)

        results.append((ts_name, min_cpa, t_at_min, os_at_min, ts_at_min))
    return results


def color_for_cpa(cpa):
    if cpa < 1.0:
        return '\033[91m'  # red - COLLISION
    elif cpa < STOP_DIST:
        return '\033[93m'  # yellow - DETECTED
    elif cpa < 20.0:
        return '\033[96m'  # cyan - CLOSE
    else:
        return '\033[92m'  # green - SAFE


RESET = '\033[0m'
BOLD = '\033[1m'


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Verify collision fix for all 20 scenarios')
    parser.add_argument('--scenario', '-s', type=int, default=0,
                        help='Verify single scenario (1-20)')
    parser.add_argument('--old', action='store_true',
                        help='Show OLD model results for comparison')
    parser.add_argument('--accurate', action='store_true',
                        help='Use accurate speed dynamics (mass + drag)')
    args = parser.parse_args()

    with open(SCENARIO_FILE) as f:
        scenarios = yaml.safe_load(f)

    sim_fn = simulate_ship_accurate if args.accurate else simulate_ship

    total_old_miss = 0
    total_new_miss = 0
    total_old_detect = 0
    total_new_detect = 0

    print(f"{'='*80}")
    print(f"  COLREGS 20-Scenario Collision Verification")
    print(f"  Thrust Fix: TS {OLD_TS_COEFF}→{NEW_TS_COEFF} N/(m/s), OS boost {OLD_OS_BOOST}→{NEW_OS_BOOST}x")
    print(f"  Collision detection threshold: {STOP_DIST}m")
    print(f"  Simulation: {'accurate dynamics' if args.accurate else 'equilibrium speed'}")
    print(f"{'='*80}")

    count = 0
    for key, s in sorted(scenarios.items()):
        if not key.startswith('scenario_'):
            continue
        sid = int(key.split('_')[1])

        if args.scenario > 0 and sid != args.scenario:
            continue

        count += 1
        os_cfg = s['own_ship']
        ts_list = s['target_ships']
        enc_type = s['encounter_type']
        desc = s['description'].strip().split('\n')[0][:60]

        print(f"\n{BOLD}S{sid:02d}{RESET} [{enc_type:<12}] {desc}")

        os_x, os_y = os_cfg['x'], os_cfg['y']
        os_yaw = os_cfg['yaw']
        os_speed = os_cfg['speed']

        # ── NEW model ──
        os_new = sim_fn(os_x, os_y, os_yaw, os_speed * NEW_OS_BOOST, NEW_TS_COEFF, DT, SIM_TIME)
        ts_new_list = []
        for ts in ts_list:
            ts_pos = sim_fn(ts['x'], ts['y'], ts['yaw'], ts['speed'], NEW_TS_COEFF, DT, SIM_TIME)
            ts_new_list.append((ts['name'], ts_pos))

        results_new = compute_min_cpa(os_new, ts_new_list)

        # ── OLD model (if requested) ──
        if args.old:
            os_old = sim_fn(os_x, os_y, os_yaw, os_speed * OLD_OS_BOOST, OLD_TS_COEFF, DT, SIM_TIME)
            ts_old_list = []
            for ts in ts_list:
                ts_pos = sim_fn(ts['x'], ts['y'], ts['yaw'], ts['speed'], OLD_TS_COEFF, DT, SIM_TIME)
                ts_old_list.append((ts['name'], ts_pos))
            results_old = compute_min_cpa(os_old, ts_old_list)

        for i, (name, cpa_new, t_new, os_xy, ts_xy) in enumerate(results_new):
            detected_new = cpa_new < STOP_DIST
            if detected_new:
                total_new_detect += 1
            else:
                total_new_miss += 1

            col = color_for_cpa(cpa_new)
            status_new = '✅ DETECT' if detected_new else '⚠️  MISS'

            if args.old:
                _, cpa_old, t_old, _, _ = results_old[i]
                detected_old = cpa_old < STOP_DIST
                if detected_old:
                    total_old_detect += 1
                else:
                    total_old_miss += 1
                col_old = color_for_cpa(cpa_old)
                status_old = 'DETECT' if detected_old else 'MISS'
                print(f'  {name:<8} {col}NEW: CPA={cpa_new:5.1f}m @ t={t_new:5.1f}s → {status_new}{RESET}  |  '
                      f'{col_old}OLD: CPA={cpa_old:5.1f}m @ t={t_old:5.1f}s → {status_old}{RESET}')
            else:
                print(f'  {name:<8} {col}CPA={cpa_new:5.1f}m @ t={t_new:5.1f}s → {status_new}{RESET}')

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f"{BOLD}SUMMARY{RESET}")
    print(f"{'='*80}")

    if args.old:
        old_total = total_old_detect + total_old_miss
        new_total = total_new_detect + total_new_miss
        print(f"  OLD model: {total_old_detect}/{old_total} detected ({100*total_old_detect/max(old_total,1):.0f}%), "
              f"{total_old_miss} missed")
        print(f"  NEW model: {total_new_detect}/{new_total} detected ({100*total_new_detect/max(new_total,1):.0f}%), "
              f"{total_new_miss} missed")
        improvement = total_new_detect - total_old_detect
        print(f"  Improvement: +{improvement} collisions detected")
        if improvement <= 0:
            print(f"  ⚠️  No improvement — may need further investigation")
    else:
        total = total_new_detect + total_new_miss
        print(f"  Total target ships: {total}")
        print(f"  Detected (<{STOP_DIST}m): {total_new_detect} ({100*total_new_detect/max(total,1):.0f}%)")
        print(f"  Missed (>{STOP_DIST}m): {total_new_miss}")

        if total_new_miss > 0:
            print(f"\n  ⚠️  {total_new_miss} ships still won't trigger collision detection.")
            print(f"  These may need scenario geometry adjustment or higher STOP_DIST.")

    print()
    return 0 if total_new_miss == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
