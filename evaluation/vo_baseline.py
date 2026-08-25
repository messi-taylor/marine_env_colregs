#!/usr/bin/env python3
"""
Velocity Obstacle (VO) Baseline Controller for COLREGS-Compliant Collision Avoidance.
======================================================================================

Implements a COLREGS-aware Velocity Obstacle method as an external baseline
for comparison against the neuro-symbolic NMPC framework.

Algorithm:
  1. Compute VO cones for all target ships (linear velocity obstacle)
  2. Apply COLREGS constraints on the velocity space:
     - Rule 14 (head-on): bias right turn
     - Rule 15 (crossing): give-way → starboard turn
     - Rule 13 (overtaking): pass astern
     - Rule 19 (restricted visibility): reduced speed, starboard bias
  3. Select optimal feasible velocity from discretized velocity space
  4. Output desired speed + heading for PD tracking

Reference:
  - Fiorini & Shiller (1998), "Motion Planning in Dynamic Environments
    Using Velocity Obstacles"
  - Johansen et al. (2017), "Ship collision avoidance and COLREGS
    compliance using simulation-based control behavior selection"

Usage:
  from evaluation.vo_baseline import VOController

  vo = VOController(visibility="clear")
  desired_speed, desired_heading = vo.compute(
      os_pos, os_heading, os_speed,
      target_states, dt=0.5
  )
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# Velocity space configuration
# =============================================================================

@dataclass
class VOConfig:
    """Configuration for Velocity Obstacle controller."""
    # Velocity search space
    speed_min: float = 0.0        # m/s
    speed_max: float = 3.5        # m/s
    speed_resolution: float = 0.1  # m/s
    heading_resolution: float = math.radians(5)  # rad

    # VO parameters
    time_horizon: float = 60.0    # s — VO lookahead window (ships need long lead time)
    cpa_safe: float = 50.0        # m — safe CPA threshold (for cost penalty, NOT collision)
    r_coll: float = 20.0  # m — VO collision circle (ship physical dims + margin)
    #  WAM-V ~5m + target small boat ~10m + 5m margin = 20m

    # COLREGS parameters
    starboard_bias_weight: float = 5.0    # cost penalty per radian of port turn


# =============================================================================
# Velocity Obstacle Controller
# =============================================================================

class VOController:
    """COLREGS-aware Velocity Obstacle controller for ship collision avoidance.

    Replaces the NMPC + COLREGS referee pipeline with a simpler geometric
    method that directly selects a safe velocity from the discretized
    velocity space.

    This serves as an external baseline for evaluating whether the
    neuro-symbolic architecture provides benefits beyond classical
    geometric collision avoidance.
    """

    def __init__(self, config: VOConfig = None, visibility: str = "clear"):
        self.config = config or VOConfig()
        self.visibility = visibility

        # Pre-compute velocity candidates (polar grid)
        self._build_velocity_grid()

    def _build_velocity_grid(self):
        """Pre-compute discretized velocity candidates in polar coordinates."""
        cfg = self.config
        speeds = np.arange(cfg.speed_min + cfg.speed_resolution,
                          cfg.speed_max + cfg.speed_resolution,
                          cfg.speed_resolution)
        headings = np.arange(-math.pi, math.pi, cfg.heading_resolution)

        self._speed_grid, self._heading_grid = np.meshgrid(speeds, headings)
        # Flatten for easy iteration
        self._speeds_flat = self._speed_grid.ravel()
        self._headings_flat = self._heading_grid.ravel()

        # Pre-compute velocity vectors for each candidate
        n = len(self._speeds_flat)
        self._vx = np.zeros(n)
        self._vy = np.zeros(n)
        for i in range(n):
            h = self._headings_flat[i]
            s = self._speeds_flat[i]
            self._vx[i] = s * math.cos(h)
            self._vy[i] = s * math.sin(h)

        # Also include zero velocity
        self._vx = np.append(self._vx, 0.0)
        self._vy = np.append(self._vy, 0.0)
        self._speeds_flat = np.append(self._speeds_flat, 0.0)
        self._headings_flat = np.append(self._headings_flat, 0.0)

    # =====================================================================
    # Public API
    # =====================================================================

    def compute(self,
                os_pos: np.ndarray,        # [x, y]
                os_heading: float,          # rad (ENU convention)
                os_speed: float,            # m/s
                target_states: List[Dict],  # [{"pos": [x,y], "heading": rad, "speed": m/s}]
                dt: float = 0.5,
                current_heading_change: float = 0.0,  # cumulative heading change (rad, negative=starboard)
                ) -> Tuple[float, float, Dict]:
        """Compute desired speed and heading for collision-free COLREGS-compliant motion.

        Args:
            os_pos: Own ship position [x, y] in ENU.
            os_heading: Own ship heading in radians (ENU, 0=East).
            os_speed: Own ship speed in m/s.
            target_states: List of target ship states.
            dt: Time step for velocity integration.
            current_heading_change: Cumulative heading change so far (radians).
                                    Negative = starboard turn.

        Returns:
            (desired_speed, desired_heading, debug_info)
        """
        cfg = self.config

        if not target_states:
            # No targets — maintain current speed and heading
            return os_speed, os_heading, {"method": "free"}

        n_candidates = len(self._speeds_flat)
        cost = np.full(n_candidates, np.inf)

        # ── Phase 1: Build VO collision masks ──
        in_collision = np.zeros(n_candidates, dtype=bool)
        cpa_violations = np.zeros(n_candidates, dtype=bool)

        for ts in target_states:
            ts_pos = np.array(ts["pos"])
            ts_heading = ts.get("heading", 0.0)
            ts_speed = ts.get("speed", 1.0)

            # Target velocity in world frame
            ts_vx = ts_speed * math.cos(ts_heading)
            ts_vy = ts_speed * math.sin(ts_heading)

            # Relative position
            rel_pos = ts_pos - os_pos
            dist = float(np.linalg.norm(rel_pos))

            if dist < 1e-6:
                continue

            # VO cone: check if relative velocity points toward collision circle
            r_coll = cfg.r_coll

            for i in range(n_candidates):
                if in_collision[i]:
                    continue

                vx_os = self._vx[i]
                vy_os = self._vy[i]

                # Relative velocity
                vx_rel = vx_os - ts_vx
                vy_rel = vy_os - ts_vy

                v_rel_norm = math.sqrt(vx_rel**2 + vy_rel**2)
                if v_rel_norm < 1e-6:
                    if dist < r_coll:
                        in_collision[i] = True
                    continue

                # Time to closest approach: p_rel · v_rel / |v_rel|²
                # p_rel = p_TS - p_OS, v_rel = v_OS - v_TS
                t_cpa = (rel_pos[0] * vx_rel + rel_pos[1] * vy_rel) / (v_rel_norm**2)

                if t_cpa < 0:
                    # Moving away — check if already too close
                    if dist < r_coll:
                        in_collision[i] = True
                    continue

                if t_cpa > cfg.time_horizon:
                    # Too far in the future — not an immediate threat
                    continue

                # CPA distance
                cpa_x = rel_pos[0] + vx_rel * t_cpa
                cpa_y = rel_pos[1] + vy_rel * t_cpa
                cpa = math.sqrt(cpa_x**2 + cpa_y**2)

                if cpa < r_coll:
                    in_collision[i] = True

                # Also check CPA violations (less severe than collision)
                if cpa < cfg.cpa_safe:
                    cpa_violations[i] = True

        # ── Phase 2: COLREGS soft constraints (penalty-based, not hard forbids) ──
        colregs_penalty = np.zeros(n_candidates)

        for ts in target_states:
            ts_pos = np.array(ts["pos"])
            ts_heading = ts.get("heading", 0.0)
            ts_speed = ts.get("speed", 1.0)

            rel_pos = ts_pos - os_pos
            dist = float(np.linalg.norm(rel_pos))
            if dist < 1e-6:
                continue

            rel_bearing = math.atan2(rel_pos[1], rel_pos[0])
            encounter = self._classify_encounter(
                os_heading, os_speed, ts_heading, ts_speed, rel_bearing, dist)

            if encounter == "head_on":
                for i in range(n_candidates):
                    h = self._headings_flat[i]
                    dh = self._angle_diff(h, os_heading)
                    if dh > math.radians(-5):
                        colregs_penalty[i] += 80.0  # port or near-zero → heavy
                    elif dh > math.radians(-20):
                        colregs_penalty[i] += 20.0 * (1.0 + dh / math.radians(20))

            elif encounter == "crossing_give_way":
                for i in range(n_candidates):
                    h = self._headings_flat[i]
                    dh = self._angle_diff(h, os_heading)
                    if dh > math.radians(5):
                        colregs_penalty[i] += 40.0
                    elif abs(dh) < math.radians(10):
                        colregs_penalty[i] += 15.0

            elif encounter == "crossing_stand_on":
                for i in range(n_candidates):
                    h = self._headings_flat[i]
                    s = self._speeds_flat[i]
                    dh = abs(self._angle_diff(h, os_heading))
                    ds = abs(s - os_speed) / max(os_speed, 1e-6)
                    colregs_penalty[i] += 25.0 * (dh / math.radians(10) + ds / 0.3)

            elif encounter == "overtaking":
                for i in range(n_candidates):
                    h = self._headings_flat[i]
                    dh_target = self._angle_diff(h, rel_bearing)
                    if abs(dh_target) < math.radians(30):
                        colregs_penalty[i] += 10.0 * (1.0 - abs(dh_target) / math.radians(30))

        if self.visibility in ("fog", "restricted", "poor"):
            for i in range(n_candidates):
                if self._speeds_flat[i] > 2.5:
                    colregs_penalty[i] += 15.0 * (self._speeds_flat[i] - 2.5)
                h = self._headings_flat[i]
                dh = self._angle_diff(h, os_heading)
                if dh > 0:
                    colregs_penalty[i] += 15.0

        # ── Phase 3: Compute cost for each candidate ──
        for i in range(n_candidates):
            if in_collision[i]:
                cost[i] = np.inf
                continue

            h = self._headings_flat[i]
            s = self._speeds_flat[i]

            dh = abs(self._angle_diff(h, os_heading))
            ds = abs(s - os_speed) / max(os_speed, 1e-6)
            port_penalty = max(0, self._angle_diff(h, os_heading)) * cfg.starboard_bias_weight
            cpa_penalty = 50.0 if cpa_violations[i] else 0.0
            speed_cost = (cfg.speed_max - s) / cfg.speed_max * 5.0

            cost[i] = (3.0 * dh + 1.0 * ds + port_penalty +
                      cpa_penalty + speed_cost + colregs_penalty[i])

        # ── Phase 4: Select best velocity ──
        best_idx = np.argmin(cost)

        if np.isinf(cost[best_idx]):
            emergency_heading = os_heading - math.radians(60)
            emergency_speed = 0.5
            emergency_heading = (emergency_heading + math.pi) % (2 * math.pi) - math.pi
            debug = {
                "method": "vo_emergency",
                "n_feasible": 0,
                "n_total": n_candidates,
                "n_collision": int(np.sum(in_collision)),
            }
            return emergency_speed, emergency_heading, debug

        best_heading = self._headings_flat[best_idx]
        best_speed = self._speeds_flat[best_idx]
        best_heading = (best_heading + math.pi) % (2 * math.pi) - math.pi

        n_feasible = int(np.sum(~in_collision))
        debug = {
            "method": "vo",
            "best_cost": float(cost[best_idx]),
            "n_feasible": n_feasible,
            "n_total": n_candidates,
            "n_collision": int(np.sum(in_collision)),
            "colregs_penalty": float(colregs_penalty[best_idx]),
            "best_heading_deg": math.degrees(best_heading),
            "best_speed": best_speed,
        }

        return best_speed, best_heading, debug

    # =====================================================================
    # Encounter classification (vector-based, same logic as batch_runner)
    # =====================================================================

    def _classify_encounter(self,
                            os_heading: float,
                            os_speed: float,
                            ts_heading: float,
                            ts_speed: float,
                            rel_bearing: float,
                            dist: float) -> str:
        """Classify COLREGS encounter type from relative geometry.

        Returns one of:
          "head_on", "crossing_give_way", "crossing_stand_on",
          "overtaking", "overtaken", "none"
        """
        # Direction vectors
        os_dir = np.array([math.cos(os_heading), math.sin(os_heading)])
        ts_dir = np.array([math.cos(ts_heading), math.sin(ts_heading)])

        # Reciprocal check (dot product)
        course_dot = float(np.dot(os_dir, ts_dir))
        # Is target ahead? (dot of os_dir and relative position direction)
        rel_dir = np.array([math.cos(rel_bearing), math.sin(rel_bearing)])
        ahead = float(np.dot(os_dir, rel_dir))

        # Cross product: os_dir × rel_dir (>0 = port, <0 = starboard)
        cross = float(os_dir[0] * rel_dir[1] - os_dir[1] * rel_dir[0])

        # Head-on (Rule 14): reciprocal courses + target ahead
        if (course_dot < -0.866        # |Δheading| > 150°
                and ahead > 0.866       # within ~30° of dead ahead
                and dist < 200.0):      # within reasonable range
            return "head_on"

        # Overtaking (Rule 13): target abaft the beam + similar courses
        if (ahead < -0.3827             # >112.5° abaft (cos 112.5° = -0.3827)
                and course_dot > 0.342  # |Δheading| < 70°
                and os_speed > ts_speed * 1.05):
            return "overtaking"

        # Being overtaken
        if (ahead > 0.3827              # ahead of us
                and course_dot > 0.342
                and ts_speed > os_speed * 1.05):
            return "overtaken"

        # Crossing (Rule 15)
        if cross < 0:
            # Target to starboard → we are give-way
            return "crossing_give_way"
        else:
            # Target to port → we are stand-on
            return "crossing_stand_on"

    # =====================================================================
    # Helpers
    # =====================================================================

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Signed angle difference a - b, normalized to [-π, π]."""
        return (a - b + math.pi) % (2 * math.pi) - math.pi


# =============================================================================
# Convenience function for batch_runner integration
# =============================================================================

def create_vo_controller(visibility: str = "clear") -> VOController:
    """Factory function for VO controller with scene-appropriate config."""
    cfg = VOConfig()

    if visibility in ("fog", "restricted", "poor"):
        cfg.speed_max = 2.5
        cfg.time_horizon = 40.0  # longer lookahead in low visibility
        cfg.collision_radius = 30.0  # larger safety margin in fog

    return VOController(config=cfg, visibility=visibility)
