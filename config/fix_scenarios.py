#!/usr/bin/env python3
"""
修正 colregs_20_new_scenarios.yaml 中的目标船位置.
将过于遥远的初始位置 (150-400m, TCPA 30-400s) 调整为近距离碰撞场景 (TCPA 12-25s).
"""

import yaml, math, copy

YAML_PATH = '/home/xxy/vrx_ws/src/marine_env/config/colregs_20_new_scenarios.yaml'

with open(YAML_PATH) as f:
    data = yaml.safe_load(f)

# 辅助函数
def compass_heading_to_enu_vel(yaw_compass_rad, speed):
    """compass yaw → ENU velocity (vx_east, vy_north)"""
    enu_yaw = math.pi/2 - yaw_compass_rad
    return speed * math.cos(enu_yaw), speed * math.sin(enu_yaw)

def compute_cpa_tcpa(os_x, os_y, os_spd, os_yaw_c,
                     ts_x, ts_y, ts_spd, ts_yaw_c):
    """Compute CPA/TCPA in ENU"""
    os_ve, os_vn = compass_heading_to_enu_vel(os_yaw_c, os_spd)
    ts_ve, ts_vn = compass_heading_to_enu_vel(ts_yaw_c, ts_spd)
    dp_e, dp_n = ts_x - os_x, ts_y - os_y
    dv_e, dv_n = ts_ve - os_ve, ts_vn - os_vn
    v2 = dv_e**2 + dv_n**2
    if v2 < 1e-9:
        return math.sqrt(dp_e**2+dp_n**2), 0
    tcpa = -(dp_e*dv_e + dp_n*dv_n) / v2
    if tcpa <= 0:
        return math.sqrt(dp_e**2+dp_n**2), tcpa
    cpa = math.sqrt((dp_e+tcpa*dv_e)**2 + (dp_n+tcpa*dv_n)**2)
    return cpa, tcpa

# ============================================================
# 修正每个场景的 TS 位置
# ============================================================

# S01 — Rule 14 Head-on: TS ahead with 3m starboard offset, CPA≈3m, TCPA≈15s
# OS: (0,0), compass 0° (N), 1.5 m/s
# TS: heading 185° (S slightly W), 1.2 m/s
# Target TCPA=15s → closing=2.7m/s → dist≈40m
sc = data['scenario_01_starboard_headon_with_offset']
sc['own_ship']['speed'] = 1.5
sc['target_ships'][0]['x'] = 3.0
sc['target_ships'][0]['y'] = 38.0
sc['target_ships'][0]['speed'] = 1.2
sc['target_ships'][0]['yaw'] = math.radians(185)
c, t = compute_cpa_tcpa(0,0,1.5,0, 3,38,1.2,math.radians(185))
print(f"S01: CPA={c:.1f}m, TCPA={t:.1f}s")

# S02 — Rule 15+16 Urgent Starboard Crossing: CPA=0, TCPA≈15s
# OS: (0,0), N, 1.5 m/s. TS: starboard side, W, 1.0 m/s
# OS reaches crossing at y=22.5 in t=15s. TS at x=15 reaches (0,22.5) in t=15s
sc = data['scenario_02_close_starboard_crossing_urgent']
sc['own_ship']['speed'] = 1.5
sc['target_ships'][0]['x'] = 15.0
sc['target_ships'][0]['y'] = 22.5
sc['target_ships'][0]['speed'] = 1.0
sc['target_ships'][0]['yaw'] = math.radians(270)
c, t = compute_cpa_tcpa(0,0,1.5,0, 15,22.5,1.0,math.radians(270))
print(f"S02: CPA={c:.1f}m, TCPA={t:.1f}s")

# S03 — Dual Starboard Crossing: 2 TS both on starboard
# TS1: (10, 15), W 1.5 m/s. TS2: (13, 20), W 1.5 m/s
sc = data['scenario_03_dual_starboard_crossing']
sc['own_ship']['speed'] = 1.5
sc['target_ships'][0]['x'] = 10.0
sc['target_ships'][0]['y'] = 15.0
sc['target_ships'][0]['speed'] = 1.5
sc['target_ships'][0]['yaw'] = math.radians(270)
sc['target_ships'][1]['x'] = 14.0
sc['target_ships'][1]['y'] = 21.0
sc['target_ships'][1]['speed'] = 1.5
sc['target_ships'][1]['yaw'] = math.radians(270)
c1, t1 = compute_cpa_tcpa(0,0,1.5,0, 10,15,1.5,math.radians(270))
c2, t2 = compute_cpa_tcpa(0,0,1.5,0, 14,21,1.5,math.radians(270))
print(f"S03: CPA1={c1:.1f}m TCPA1={t1:.1f}s | CPA2={c2:.1f}m TCPA2={t2:.1f}s")

# S04 — Opposite Crossing Both Sides: TS1 starboard (give-way), TS2 port (stand-on)
sc = data['scenario_04_opposite_crossing_both_sides']
sc['own_ship']['speed'] = 1.5
sc['target_ships'][0]['x'] = 18.0
sc['target_ships'][0]['y'] = 27.0
sc['target_ships'][0]['speed'] = 1.0
sc['target_ships'][0]['yaw'] = math.radians(270)
sc['target_ships'][1]['x'] = -18.0
sc['target_ships'][1]['y'] = 27.0
sc['target_ships'][1]['speed'] = 1.0
sc['target_ships'][1]['yaw'] = math.radians(90)
c1, t1 = compute_cpa_tcpa(0,0,1.5,0, 18,27,1.0,math.radians(270))
c2, t2 = compute_cpa_tcpa(0,0,1.5,0, -18,27,1.0,math.radians(90))
print(f"S04: CPA1={c1:.1f}m TCPA1={t1:.1f}s | CPA2={c2:.1f}m TCPA2={t2:.1f}s")

# S05 — Fishing Vessel Rule 18: TS slow, starboard
sc = data['scenario_05_fishing_vessel_rule18']
sc['own_ship']['speed'] = 1.5
sc['target_ships'][0]['x'] = 12.0
sc['target_ships'][0]['y'] = 18.0
sc['target_ships'][0]['speed'] = 0.5
sc['target_ships'][0]['yaw'] = math.radians(270)
c, t = compute_cpa_tcpa(0,0,1.5,0, 12,18,0.5,math.radians(270))
print(f"S05: CPA={c:.1f}m, TCPA={t:.1f}s")

# S06 — Narrow Channel Head-on Rule 9: close head-on, narrow channel
sc = data['scenario_06_narrow_channel_headon_rule9']
sc['own_ship']['speed'] = 1.2
sc['target_ships'][0]['x'] = 1.0
sc['target_ships'][0]['y'] = 32.0
sc['target_ships'][0]['speed'] = 0.8
sc['target_ships'][0]['yaw'] = math.radians(180)
c, t = compute_cpa_tcpa(0,0,1.2,0, 1,32,0.8,math.radians(180))
print(f"S06: CPA={c:.1f}m, TCPA={t:.1f}s")

# S07 — Narrow Channel Overtaking Rule 9+13: TS ahead, same direction, slower
sc = data['scenario_07_narrow_channel_overtaking_rule9_13']
sc['own_ship']['speed'] = 1.5
sc['target_ships'][0]['x'] = 0.0
sc['target_ships'][0]['y'] = 18.0
sc['target_ships'][0]['speed'] = 0.4
sc['target_ships'][0]['yaw'] = 0.0
c, t = compute_cpa_tcpa(0,0,1.5,0, 0,18,0.4,0)
print(f"S07: CPA={c:.1f}m, TCPA={t:.1f}s")

# S08 — Head-on + Fast Overtaker from Behind
sc = data['scenario_08_headon_with_fast_overtaker']
sc['own_ship']['speed'] = 1.5
# TS1: head-on ahead, S
sc['target_ships'][0]['x'] = 0.0
sc['target_ships'][0]['y'] = 39.0
sc['target_ships'][0]['speed'] = 1.0
sc['target_ships'][0]['yaw'] = math.radians(180)
# TS2: overtaking from behind, N, faster
sc['target_ships'][1]['x'] = 0.0
sc['target_ships'][1]['y'] = -15.0
sc['target_ships'][1]['speed'] = 2.5
sc['target_ships'][1]['yaw'] = 0.0
c1, t1 = compute_cpa_tcpa(0,0,1.5,0, 0,39,1.0,math.radians(180))
c2, t2 = compute_cpa_tcpa(0,0,1.5,0, 0,-15,2.5,0)
print(f"S08: CPA1={c1:.1f}m TCPA1={t1:.1f}s | CPA2={c2:.1f}m TCPA2={t2:.1f}s")

# S09 — Restricted Visibility Crossing Rule 19
sc = data['scenario_09_restricted_vis_crossing_rule19']
sc['own_ship']['speed'] = 1.0
sc['target_ships'][0]['x'] = 16.0
sc['target_ships'][0]['y'] = 16.0
sc['target_ships'][0]['speed'] = 0.7
sc['target_ships'][0]['yaw'] = math.radians(270)
c, t = compute_cpa_tcpa(0,0,1.0,0, 16,16,0.7,math.radians(270))
print(f"S09: CPA={c:.1f}m, TCPA={t:.1f}s")

# S10 — NUC Vessel Rule 18: TS nearly stationary, drifting
sc = data['scenario_10_nuc_vessel_rule18']
sc['own_ship']['speed'] = 1.5
sc['target_ships'][0]['x'] = 6.0
sc['target_ships'][0]['y'] = 24.0
sc['target_ships'][0]['speed'] = 0.2
sc['target_ships'][0]['yaw'] = math.radians(200)
c, t = compute_cpa_tcpa(0,0,1.5,0, 6,24,0.2,math.radians(200))
print(f"S10: CPA={c:.1f}m, TCPA={t:.1f}s")

# S11 — Constrained by Draft Rule 18: head-on
sc = data['scenario_11_constrained_by_draft_rule18']
sc['own_ship']['speed'] = 1.5
sc['target_ships'][0]['x'] = 0.0
sc['target_ships'][0]['y'] = 38.0
sc['target_ships'][0]['speed'] = 0.6
sc['target_ships'][0]['yaw'] = math.radians(180)
c, t = compute_cpa_tcpa(0,0,1.5,0, 0,38,0.6,math.radians(180))
print(f"S11: CPA={c:.1f}m, TCPA={t:.1f}s")

# S12 — Acute Angle Crossing (~20° diff): borderline case
sc = data['scenario_12_acute_angle_crossing_borderline']
sc['own_ship']['speed'] = 1.5
sc['target_ships'][0]['x'] = 5.0
sc['target_ships'][0]['y'] = 32.0
sc['target_ships'][0]['speed'] = 1.0
sc['target_ships'][0]['yaw'] = math.radians(200)  # S slightly W, ~20° from N
c, t = compute_cpa_tcpa(0,0,1.5,0, 5,32,1.0,math.radians(200))
print(f"S12: CPA={c:.1f}m, TCPA={t:.1f}s")

# S13 — Zigzag Dynamic Evasion: TS has waypoints, initial heading East
# Place TS closer, heading toward OS general area
sc = data['scenario_13_zigzag_dynamic_evasion']
sc['own_ship']['speed'] = 1.5
sc['target_ships'][0]['x'] = -40.0
sc['target_ships'][0]['y'] = 30.0
sc['target_ships'][0]['speed'] = 2.0
# Waypoints: zigzag pattern, first waypoint to the right
sc['target_ships'][0]['waypoints'] = [
    [40.0, 30.0, 2.0],
    [40.0, 60.0, 2.0],
    [-40.0, 60.0, 2.0],
]

# S14 — Formation Head-on Two Columns
sc = data['scenario_14_formation_headon_two_columns']
sc['own_ship']['speed'] = 1.5
sc['target_ships'][0]['x'] = 4.0
sc['target_ships'][0]['y'] = 38.0
sc['target_ships'][0]['speed'] = 1.0
sc['target_ships'][0]['yaw'] = math.radians(180)
sc['target_ships'][1]['x'] = -4.0
sc['target_ships'][1]['y'] = 36.0
sc['target_ships'][1]['speed'] = 1.0
sc['target_ships'][1]['yaw'] = math.radians(180)
c1, t1 = compute_cpa_tcpa(0,0,1.5,0, 4,38,1.0,math.radians(180))
c2, t2 = compute_cpa_tcpa(0,0,1.5,0, -4,36,1.0,math.radians(180))
print(f"S14: CPA1={c1:.1f}m TCPA1={t1:.1f}s | CPA2={c2:.1f}m TCPA2={t2:.1f}s")

# S15 — High-Speed Overtaking Rule 13
sc = data['scenario_15_high_speed_overtaking_rule13']
sc['own_ship']['speed'] = 2.0
sc['target_ships'][0]['x'] = 1.0
sc['target_ships'][0]['y'] = 22.0
sc['target_ships'][0]['speed'] = 0.3
sc['target_ships'][0]['yaw'] = 0.0
c, t = compute_cpa_tcpa(0,0,2.0,0, 1,22,0.3,0)
print(f"S15: CPA={c:.1f}m, TCPA={t:.1f}s")

# S16 — Restricted Visibility Four Ships Rule 19
sc = data['scenario_16_restricted_vis_four_ships_rule19']
sc['own_ship']['speed'] = 1.0
# TS1: head-on ahead
sc['target_ships'][0]['x'] = 0.0
sc['target_ships'][0]['y'] = 28.0
sc['target_ships'][0]['speed'] = 1.2
sc['target_ships'][0]['yaw'] = math.radians(180)
# TS2: starboard crossing
sc['target_ships'][1]['x'] = 14.0
sc['target_ships'][1]['y'] = 14.0
sc['target_ships'][1]['speed'] = 0.8
sc['target_ships'][1]['yaw'] = math.radians(270)
# TS3: port crossing
sc['target_ships'][2]['x'] = -14.0
sc['target_ships'][2]['y'] = 14.0
sc['target_ships'][2]['speed'] = 0.8
sc['target_ships'][2]['yaw'] = math.radians(90)
# TS4: slow ahead
sc['target_ships'][3]['x'] = 0.0
sc['target_ships'][3]['y'] = 12.0
sc['target_ships'][3]['speed'] = 0.3
sc['target_ships'][3]['yaw'] = 0.0
for i in range(4):
    ts = sc['target_ships'][i]
    c, t = compute_cpa_tcpa(0,0,1.0,0, ts['x'],ts['y'],ts['speed'],ts['yaw'])
    print(f"S16 TS{i+1}: CPA={c:.1f}m TCPA={t:.1f}s")

# S17 — Sudden Acceleration & Turn: TS dynamic with waypoints
sc = data['scenario_17_crossing_with_sudden_acceleration']
sc['own_ship']['speed'] = 1.5
sc['target_ships'][0]['x'] = -30.0
sc['target_ships'][0]['y'] = 35.0
sc['target_ships'][0]['speed'] = 1.5
sc['target_ships'][0]['waypoints'] = [
    [30.0, 35.0, 3.0],
    [30.0, 70.0, 3.0],
]

# S18 — Triple Crossing Staggered
sc = data['scenario_18_triple_crossing_staggered']
sc['own_ship']['speed'] = 1.5
sc['target_ships'][0]['x'] = 14.0
sc['target_ships'][0]['y'] = 14.0
sc['target_ships'][0]['speed'] = 0.9
sc['target_ships'][0]['yaw'] = math.radians(270)
sc['target_ships'][1]['x'] = 18.0
sc['target_ships'][1]['y'] = 24.0
sc['target_ships'][1]['speed'] = 1.1
sc['target_ships'][1]['yaw'] = math.radians(270)
sc['target_ships'][2]['x'] = -12.0
sc['target_ships'][2]['y'] = 18.0
sc['target_ships'][2]['speed'] = 0.7
sc['target_ships'][2]['yaw'] = math.radians(90)
for i in range(3):
    ts = sc['target_ships'][i]
    c, t = compute_cpa_tcpa(0,0,1.5,0, ts['x'],ts['y'],ts['speed'],ts['yaw'])
    print(f"S18 TS{i+1}: CPA={c:.1f}m TCPA={t:.1f}s")

# S19 — Night Encounter Rules 20-23
sc = data['scenario_19_night_encounter_rules20_23']
sc['own_ship']['speed'] = 1.5
sc['target_ships'][0]['x'] = 0.0
sc['target_ships'][0]['y'] = 36.0
sc['target_ships'][0]['speed'] = 0.9
sc['target_ships'][0]['yaw'] = math.radians(180)
sc['target_ships'][1]['x'] = 14.0
sc['target_ships'][1]['y'] = 20.0
sc['target_ships'][1]['speed'] = 0.8
sc['target_ships'][1]['yaw'] = math.radians(270)
c1, t1 = compute_cpa_tcpa(0,0,1.5,0, 0,36,0.9,math.radians(180))
c2, t2 = compute_cpa_tcpa(0,0,1.5,0, 14,20,0.8,math.radians(270))
print(f"S19: CPA1={c1:.1f}m TCPA1={t1:.1f}s | CPA2={c2:.1f}m TCPA2={t2:.1f}s")

# S20 — Ultimate 6-Ship Convergence
sc = data['scenario_20_ultimate_six_ship_convergence']
sc['own_ship']['speed'] = 1.5
# TS1: head-on
sc['target_ships'][0]['x'] = 1.0
sc['target_ships'][0]['y'] = 30.0
sc['target_ships'][0]['speed'] = 1.2
sc['target_ships'][0]['yaw'] = math.radians(180)
# TS2: starboard crossing
sc['target_ships'][1]['x'] = 12.0
sc['target_ships'][1]['y'] = 18.0
sc['target_ships'][1]['speed'] = 0.9
sc['target_ships'][1]['yaw'] = math.radians(270)
# TS3: port crossing
sc['target_ships'][2]['x'] = -12.0
sc['target_ships'][2]['y'] = 18.0
sc['target_ships'][2]['speed'] = 0.9
sc['target_ships'][2]['yaw'] = math.radians(90)
# TS4: slow ahead (overtaking target)
sc['target_ships'][3]['x'] = 0.0
sc['target_ships'][3]['y'] = 10.0
sc['target_ships'][3]['speed'] = 0.3
sc['target_ships'][3]['yaw'] = 0.0
# TS5: behind, fast (overtaking OS)
sc['target_ships'][4]['x'] = -2.0
sc['target_ships'][4]['y'] = -12.0
sc['target_ships'][4]['speed'] = 2.5
sc['target_ships'][4]['yaw'] = 0.0
# TS6: starboard quarter
sc['target_ships'][5]['x'] = 16.0
sc['target_ships'][5]['y'] = 24.0
sc['target_ships'][5]['speed'] = 0.7
sc['target_ships'][5]['yaw'] = math.radians(250)
for i in range(6):
    ts = sc['target_ships'][i]
    c, t = compute_cpa_tcpa(0,0,1.5,0, ts['x'],ts['y'],ts['speed'],ts['yaw'])
    print(f"S20 TS{i+1}: CPA={c:.1f}m TCPA={t:.1f}s")

# ============================================================
# 保存
# ============================================================
backup_path = YAML_PATH.replace('.yaml', '_BACKUP_FAR.yaml')
import shutil
shutil.copy(YAML_PATH, backup_path)
print(f"\n备份: {backup_path}")

with open(YAML_PATH, 'w') as f:
    yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print(f"修正: {YAML_PATH}")
print("完成! 现在 TCPA 在 12-25s 范围内.")
