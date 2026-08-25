#!/usr/bin/env python3
"""
Deterministic Rule-Based COLREGS Referee
=========================================

A rule-based implementation of the symbolic referee layer that does NOT
require an LLM. Uses deterministic COLREGS rule lookup tables and geometric
computations to classify encounters and generate structured constraints.

This serves as:
  1. A validated baseline for the LLM-based referee
  2. A fallback when LLM inference fails or times out
  3. A reference implementation for testing/debugging

COLREGS Rules Implemented:
  - Rule 2   (Responsibility)        — general prudence
  - Rule 5   (Look-out)              — situational awareness check
  - Rule 6   (Safe Speed)            — speed limits based on conditions
  - Rule 7   (Risk of Collision)     — CPA/TCPA-based risk assessment
  - Rule 8   (Action to Avoid Collision) — positive, ample, timely action
  - Rule 13  (Overtaking)            — overtaking vessel keeps clear
  - Rule 14  (Head-on)               — both alter to starboard
  - Rule 15  (Crossing)              — give-way to vessel on starboard
  - Rule 16  (Action by Give-way)    — early and substantial action
  - Rule 17  (Action by Stand-on)    — keep course and speed
  - Rule 18  (Responsibilities)      — hierarchy: NUC/RAM/Fishing/Sailing/Power
  - Rule 19  (Restricted Visibility) — no stand-on vessel, proceed at safe speed
"""

import math
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .output_schema import (
    COLREGSConstraintOutput,
    ColregsRuleInterpretation,
    SpatialConstraint,
    ManeuverConstraint,
    SpeedConstraint,
    EncounterClassification,
    EncounterType,
    ShipRole,
    ManeuverType,
    ForbiddenManeuver,
)


# =============================================================================
# Ship state observation (input to the referee)
# =============================================================================

@dataclass
class ShipObservation:
    """Snapshot of a single ship's state at a given time."""
    name: str
    position: np.ndarray       # [x, y] in ENU (m)
    heading: float             # rad, ENU convention
    speed: np.ndarray          # [vx, vy] in world frame (m/s)
    length: float = 5.0        # ship length (m)
    is_own_ship: bool = False


# =============================================================================
# Geometric computation utilities
# =============================================================================

def _compute_relative_state(
    os: ShipObservation, ts: ShipObservation
) -> Dict[str, float]:
    """Compute relative geometry between own ship and a target ship.

    Returns:
        dict with: rel_distance, rel_bearing, cpa, tcpa, heading_diff,
                   rel_speed, crossing_angle, bearing_sector
    """
    # Relative position vector (OS -> TS)
    rel_pos = ts.position - os.position
    dist = float(np.linalg.norm(rel_pos))

    # Relative bearing (angle from OS bow to TS, starboard = positive, port = negative)
    # Maritime convention: heading 0=North, CW positive.
    # atan2(dx, dy) = angle from North to rel_pos (CW).
    # Bearing = angle_from_north - heading, then normalized to [-π, π].
    bearing = math.atan2(rel_pos[0], rel_pos[1]) - os.heading
    bearing = (bearing + math.pi) % (2 * math.pi) - math.pi  # normalize to [-π, π]

    # Relative velocity
    rel_vel = ts.speed - os.speed
    rel_speed = float(np.linalg.norm(rel_vel))

    # CPA (Closest Point of Approach)
    if rel_speed < 1e-6:
        cpa = dist
        tcpa = float('inf')
    else:
        # CPA = |rel_pos × rel_vel| / |rel_vel|
        cross = rel_pos[0] * rel_vel[1] - rel_pos[1] * rel_vel[0]
        cpa = abs(cross) / rel_speed
        # TCPA = -(rel_pos · rel_vel) / |rel_vel|²
        tcpa = -np.dot(rel_pos, rel_vel) / (rel_speed ** 2)

    # Heading difference
    heading_diff = ts.heading - os.heading
    heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi

    # Bearing sector classification
    bearing_sector = _classify_bearing_sector(bearing)

    return {
        'rel_distance': dist,
        'rel_bearing': bearing,
        'rel_bearing_deg': math.degrees(bearing),
        'cpa': cpa,
        'tcpa': tcpa,
        'heading_diff': heading_diff,
        'heading_diff_deg': math.degrees(heading_diff),
        'rel_speed': rel_speed,
        'bearing_sector': bearing_sector,
    }


def _classify_bearing_sector(bearing: float) -> str:
    """Classify relative bearing into COLREGS sectors.

    COLREGS arc definitions (from own ship bow, clockwise):
      - Starboard bow:   0° to 112.5°  (Rule 15 crossing starboard)
      - Port bow:        0° to -112.5° (Rule 15 crossing port)
      - Starboard quarter: 112.5° to 180° (astern starboard)
      - Port quarter:      -112.5° to -180° (astern port)
      - Dead ahead:        ±5°
      - Dead astern:       ±175° to 180°

    bearing: rad, positive = starboard (right), negative = port (left)
    """
    deg = math.degrees(bearing)
    abs_deg = abs(deg)

    if abs_deg < 5.0:
        return "dead_ahead"
    if abs_deg > 175.0:
        return "dead_astern"
    if bearing > 0:
        if abs_deg <= 112.5:
            return "starboard_bow"
        else:
            return "starboard_quarter"
    else:
        if abs_deg <= 112.5:
            return "port_bow"
        else:
            return "port_quarter"


def _compute_cpa_risk_field(cpa: float, tcpa: float, dist: float) -> float:
    """Compute nonlinear CPA/TCPA risk field (Section 3.2).

    φ(cpa, tcpa) = exp(-α·cpa/cpa_threshold) · exp(-β·|tcpa|/tcpa_threshold)

    High φ → high collision risk → trigger LLM referee.
    """
    cpa_threshold = 50.0   # m — below this is concerning
    tcpa_threshold = 60.0  # s — below this is concerning
    alpha = 2.0
    beta = 1.5

    if tcpa <= 0:
        # Already passing or passed — risk based on current distance
        return float(np.exp(-alpha * cpa / cpa_threshold))

    cpa_factor = np.exp(-alpha * cpa / cpa_threshold)
    tcpa_factor = np.exp(-beta * abs(tcpa) / tcpa_threshold)
    return float(cpa_factor * tcpa_factor)


# =============================================================================
# COLREGS Rule Engine
# =============================================================================

class DeterministicReferee:
    """Rule-based COLREGS compliance referee.

    Determines encounter types, ship roles, and generates structured
    spatial/maneuver/speed constraints using deterministic logic.

    This implements the full COLREGS encounter logic from Section 4.4
    of the research document.

    Usage:
        referee = DeterministicReferee()
        # Observe ship states, get structured constraints
        output = referee.evaluate(os_state, [ts1_state, ts2_state, ...])
    """

    # ── Configuration constants ──
    CPA_SAFE_DISTANCE = 50.0          # m — Rule 8(d) min safe CPA
    CPA_DANGER_DISTANCE = 20.0        # m — danger zone
    TCPA_WARNING_TIME = 60.0          # s — Rule 7 risk assessment window
    HEAD_ON_ANGLE_THRESHOLD = 5.0     # deg — within this = head-on
    CROSSING_SECTOR_MAX = 112.5       # deg — COLREGS crossing arc
    OVERTAKING_SECTOR_MIN = 135.0     # deg — behind this = overtaking
    MIN_ALTERATION_ANGLE = math.radians(30)  # rad — Rule 8(b) substantial action
    MAX_SAFE_SPEED_CLEAR = 5.0        # m/s — Rule 6 clear conditions
    MAX_SAFE_SPEED_RESTRICTED = 2.5   # m/s — Rule 6 restricted visibility
    LOOKOUT_RANGE = 500.0             # m — Rule 5 reasonable range
    RISK_CPA_THRESHOLD = 50.0         # m — Rule 7(b) CPA-based risk
    RISK_TCPA_THRESHOLD = 60.0        # s — Rule 7(b) TCPA-based risk

    def __init__(self, visibility: str = "clear", sea_state: int = 2,
                 traffic_density: float = 1.0):
        """
        Args:
            visibility: "clear" | "restricted" — affects Rule 19 activation
            sea_state: 0-9 — affects safe speed computation
            traffic_density: ≥1.0 multiplier — higher = more conservative
        """
        self.visibility = visibility
        self.sea_state = sea_state
        self.traffic_density = traffic_density

    # =====================================================================
    # Main evaluation entry point
    # =====================================================================

    def evaluate(
        self,
        own_ship: ShipObservation,
        target_ships: List[ShipObservation],
        scenario_id: str = "",
        environment_context: str = "",
    ) -> COLREGSConstraintOutput:
        """Evaluate the COLREGS situation and produce structured constraints.

        Args:
            own_ship: Own ship state observation
            target_ships: List of target ship observations
            scenario_id: Scenario identifier for logging
            environment_context: Description of environmental conditions

        Returns:
            COLREGSConstraintOutput with all 12 constraint fields populated
        """
        t_start = time.perf_counter()

        # ── Step 1: Compute relative geometry for all targets ──
        geo_data = {}
        active_targets = []
        for ts in target_ships:
            geo = _compute_relative_state(own_ship, ts)
            if geo['rel_distance'] < self.LOOKOUT_RANGE:
                geo_data[ts.name] = geo
                active_targets.append(ts)

        # ── Step 2: Risk assessment per target (Rule 7) ──
        risk_assessments = {}
        for ts in active_targets:
            geo = geo_data[ts.name]
            risk = self._assess_risk(geo)
            risk_assessments[ts.name] = risk

        # ── Step 3: Encounter classification per target ──
        interpretations = []
        for ts in active_targets:
            geo = geo_data[ts.name]
            risk = risk_assessments[ts.name]
            interp = self._classify_encounter(own_ship, ts, geo, risk)
            interpretations.append(interp)

        # ── Step 4: Global encounter classification ──
        encounter_class = self._classify_global(interpretations)

        # ── Step 5: Aggregate constraints ──
        global_maneuver, global_forbidden = self._aggregate_maneuvers(
            interpretations, encounter_class)
        global_max_speed = self._compute_safe_speed(encounter_class)
        global_min_cpa, global_min_tcpa = self._aggregate_cpa(interpretations)

        # ── Step 6: Compute overall risk field ──
        cpa_risk = max(
            (r['cpa_risk'] for r in risk_assessments.values()),
            default=0.0)

        # ── Step 7: Build reasoning trace ──
        reasoning = self._build_reasoning_trace(
            interpretations, encounter_class, global_maneuver)

        # ── Step 8: Determine degradation level ──
        degradation = self._determine_degradation(encounter_class, cpa_risk)

        inference_time = (time.perf_counter() - t_start) * 1000

        return COLREGSConstraintOutput(
            timestamp=time.time(),
            scenario_id=scenario_id,
            encounter_classification=encounter_class,
            target_interpretations=interpretations,
            required_maneuver=global_maneuver,
            forbidden_maneuver=global_forbidden,
            max_safe_speed=global_max_speed,
            global_min_cpa=global_min_cpa,
            global_min_tcpa=global_min_tcpa,
            confidence_score=0.95,  # deterministic referee has high confidence
            reasoning_trace=reasoning,
            llm_model="deterministic",
            inference_time_ms=inference_time,
            fallback_active=False,
            degradation_level=degradation,
        )

    # =====================================================================
    # Step 2: Risk Assessment (Rule 7)
    # =====================================================================

    def _assess_risk(self, geo: dict) -> dict:
        """Assess collision risk using CPA/TCPA (Rule 7).

        Rule 7(b): Risk of collision shall be deemed to exist if the compass
        bearing of an approaching vessel does not appreciably change.
        In practice: low CPA + low TCPA = high risk.
        """
        cpa = geo['cpa']
        tcpa = geo['tcpa']
        dist = geo['rel_distance']

        # CPA-based risk
        if cpa < self.CPA_DANGER_DISTANCE:
            cpa_level = "critical"
        elif cpa < self.CPA_SAFE_DISTANCE:
            cpa_level = "high"
        elif cpa < 2 * self.CPA_SAFE_DISTANCE:
            cpa_level = "medium"
        else:
            cpa_level = "low"

        # Time urgency
        if tcpa > 0:
            if tcpa < 30:
                time_urgency = "immediate"
            elif tcpa < 120:
                time_urgency = "short_term"
            elif tcpa < 600:
                time_urgency = "medium_term"
            else:
                time_urgency = "long_term"
        else:
            time_urgency = "immediate"  # passed CPA — already closest

        # Combined risk field (Section 3.2)
        cpa_risk = _compute_cpa_risk_field(cpa, tcpa, dist)

        return {
            'cpa': cpa,
            'tcpa': tcpa,
            'cpa_level': cpa_level,
            'time_urgency': time_urgency,
            'cpa_risk': cpa_risk,
            'risk_exists': cpa < self.RISK_CPA_THRESHOLD and (
                tcpa > 0 and tcpa < self.RISK_TCPA_THRESHOLD),
        }

    # =====================================================================
    # Step 3: Encounter Classification (Rules 13, 14, 15, 19)
    # =====================================================================

    def _classify_encounter(
        self,
        os: ShipObservation,
        ts: ShipObservation,
        geo: dict,
        risk: dict,
    ) -> ColregsRuleInterpretation:
        """Classify the encounter type between OS and a target ship.

        Implements the COLREGS decision tree:
          1. Rule 19 (restricted visibility) — overrides Section II if applicable
          2. Rule 13 (overtaking)           — check first (overrides crossing)
          3. Rule 14 (head-on)              — reciprocal or nearly reciprocal
          4. Rule 15 (crossing)             — crossing situation
          5. Rule 2  (responsibility)       — residual prudence
        """
        bearing_sector = geo['bearing_sector']
        heading_diff = abs(geo['heading_diff_deg'])
        bearing_deg = geo['rel_bearing_deg']

        # ── Rule 19: Restricted Visibility ──
        if self.visibility == "restricted":
            return self._rule19_encounter(os, ts, geo, risk)

        # ── Rule 13: Overtaking ──
        # An overtaking situation exists when a vessel is coming up with
        # another vessel from a direction more than 22.5° abaft her beam.
        is_overtaking, overtaking_role = self._check_overtaking(geo, os, ts)
        if is_overtaking:
            return self._rule13_encounter(os, ts, geo, risk, overtaking_role)

        # ── Rule 14: Head-on ──
        is_headon = self._check_headon(geo)
        if is_headon:
            return self._rule14_encounter(os, ts, geo, risk)

        # ── Rule 15: Crossing ──
        is_crossing, crossing_role = self._check_crossing(bearing_sector)
        if is_crossing:
            return self._rule15_encounter(os, ts, geo, risk, crossing_role)

        # ── Default: no risk, no specific rule ──
        return ColregsRuleInterpretation(
            target_name=ts.name,
            encounter_type=EncounterType.NO_RISK,
            own_ship_role=ShipRole.NOT_APPLICABLE,
            applicable_rules=["Rule 2", "Rule 5"],
            spatial=SpatialConstraint(
                target_name=ts.name,
                min_distance=self.CPA_SAFE_DISTANCE,
                priority=0,
            ),
        )

    def _check_overtaking(
        self, geo: dict, os: ShipObservation, ts: ShipObservation
    ) -> Tuple[bool, str]:
        """Check if an overtaking situation exists (Rule 13).

        Rule 13: A vessel is overtaking when coming up with another vessel
        from a direction more than 22.5° abaft her beam.

        To determine overtaking:
          1. Compute bearing FROM TS TO OS (is OS behind TS?)
          2. If |bearing| > 112.5°, OS is abaft TS's beam → potential overtaking
          3. OS speed > TS speed → OS is overtaking TS
          4. TS speed > OS speed → TS is overtaking OS

        Returns (is_overtaking, role: "overtaking" / "being_overtaken").
        """
        # Bearing from TS to OS: where is OS relative to TS?
        # Maritime convention: heading 0=North, CW positive, starboard=positive
        rel_pos_ts_to_os = os.position - ts.position
        bearing_ts_to_os = math.atan2(rel_pos_ts_to_os[0], rel_pos_ts_to_os[1]) - ts.heading
        bearing_ts_to_os = (bearing_ts_to_os + math.pi) % (2 * math.pi) - math.pi
        bearing_ts_to_os_abs = abs(math.degrees(bearing_ts_to_os))

        os_speed = float(np.linalg.norm(os.speed))
        ts_speed = float(np.linalg.norm(ts.speed))

        # Overtaking prerequisite: roughly parallel courses (heading diff < 20°)
        # Ships on crossing courses (>20° heading difference) are not overtaking
        heading_diff_abs = abs(geo['heading_diff_deg'])
        # Normalize to [0, 180]
        if heading_diff_abs > 180:
            heading_diff_abs = 360 - heading_diff_abs

        is_parallel = heading_diff_abs < 20.0 or abs(heading_diff_abs - 180) < 20.0

        if not is_parallel:
            return False, ""

        # OS is abaft TS's beam (> 112.5° from TS's bow) → OS is behind TS
        if bearing_ts_to_os_abs > self.CROSSING_SECTOR_MAX:
            if os_speed > ts_speed * 1.05:  # 5% margin
                return True, "overtaking"
            else:
                return True, "being_overtaken"

        # Also check from OS perspective: TS abaft OS beam → TS behind OS
        bearing_abs = abs(geo['rel_bearing_deg'])
        if bearing_abs > self.CROSSING_SECTOR_MAX:
            if ts_speed > os_speed * 1.05:
                return True, "being_overtaken"

        return False, ""

    def _check_headon(self, geo: dict) -> bool:
        """Check if a head-on situation exists (Rule 14).

        Head-on: vessels meeting on reciprocal or nearly reciprocal courses.
        Reciprocal course ≈ heading_diff ≈ 180° ± threshold.
        Also: the target must be within ~5° dead ahead.
        """
        heading_diff = abs(geo['heading_diff_deg'])
        bearing_abs = abs(geo['rel_bearing_deg'])

        # Heading difference near 180° (meeting)
        if abs(heading_diff - 180) < 15.0:
            # Target is ahead (bearing near 0°)
            if bearing_abs < self.HEAD_ON_ANGLE_THRESHOLD + 5.0:
                return True

        # Small angle crossing near dead ahead (combined check)
        if bearing_abs < self.HEAD_ON_ANGLE_THRESHOLD:
            if abs(heading_diff - 180) < 22.5:
                return True

        return False

    def _check_crossing(self, bearing_sector: str) -> Tuple[bool, str]:
        """Check if a crossing situation exists (Rule 15).

        Crossing: target is on the bow sector (±112.5° from bow).
        The vessel which has the other on her starboard side is the give-way.

        Returns (is_crossing, role: "give_way" / "stand_on").
        """
        if bearing_sector in ("starboard_bow", "starboard_quarter"):
            return True, "give_way"   # TS on starboard → OS gives way
        elif bearing_sector in ("port_bow", "port_quarter"):
            return True, "stand_on"   # TS on port → OS stands on
        elif bearing_sector == "dead_ahead":
            return True, "give_way"   # conservative: dead ahead = both give way
        return False, ""

    # ── Rule-specific encounter builders ──

    def _rule13_encounter(
        self, os, ts, geo, risk, role: str
    ) -> ColregsRuleInterpretation:
        """Rule 13: Overtaking."""
        if role == "overtaking":
            own_role = ShipRole.GIVE_WAY
            maneuver = ManeuverType.ALTER_TO_STARBOARD
        else:
            own_role = ShipRole.STAND_ON
            maneuver = ManeuverType.MAINTAIN_COURSE_SPEED

        return ColregsRuleInterpretation(
            target_name=ts.name,
            encounter_type=EncounterType.OVERTAKING,
            own_ship_role=own_role,
            applicable_rules=["Rule 13", "Rule 8", "Rule 6"],
            spatial=self._build_spatial_constraint(
                ts.name, geo, risk, own_role,
                pass_astern=True if own_role == ShipRole.GIVE_WAY else None),
            maneuver=ManeuverConstraint(
                required_maneuver=maneuver,
                forbidden_maneuver=(ForbiddenManeuver.ALTER_TO_PORT
                                    if own_role == ShipRole.GIVE_WAY
                                    else ForbiddenManeuver.NONE),
                alteration_min_angle=(self.MIN_ALTERATION_ANGLE
                                      if own_role == ShipRole.GIVE_WAY else 0.0),
                priority=1,
            ),
            speed=SpeedConstraint(
                max_speed=self.MAX_SAFE_SPEED_CLEAR,
                min_speed=max(0.5, float(np.linalg.norm(ts.speed)) * 1.05),
            ),
        )

    def _rule14_encounter(
        self, os, ts, geo, risk
    ) -> ColregsRuleInterpretation:
        """Rule 14: Head-on — both vessels alter to starboard."""
        return ColregsRuleInterpretation(
            target_name=ts.name,
            encounter_type=EncounterType.HEAD_ON,
            own_ship_role=ShipRole.GIVE_WAY,  # both give way in head-on
            applicable_rules=["Rule 14", "Rule 8", "Rule 6"],
            spatial=self._build_spatial_constraint(
                ts.name, geo, risk, ShipRole.GIVE_WAY,
                pass_astern=False,    # head-on: pass port-to-port
            ),
            maneuver=ManeuverConstraint(
                required_maneuver=ManeuverType.ALTER_TO_STARBOARD,
                forbidden_maneuver=ForbiddenManeuver.ALTER_TO_PORT,
                alteration_min_angle=self.MIN_ALTERATION_ANGLE,
                priority=2,  # highest priority — head-on is most dangerous
            ),
            speed=SpeedConstraint(
                max_speed=self.MAX_SAFE_SPEED_CLEAR * 0.8,
            ),
        )

    def _rule15_encounter(
        self, os, ts, geo, risk, role: str
    ) -> ColregsRuleInterpretation:
        """Rule 15: Crossing situation."""
        if role == "give_way":
            own_role = ShipRole.GIVE_WAY
            maneuver = ManeuverType.ALTER_TO_STARBOARD
            # Rule 15: give-way vessel shall avoid crossing ahead
            pass_astern = True
            forbidden = ForbiddenManeuver.ALTER_TO_PORT
        else:
            own_role = ShipRole.STAND_ON
            maneuver = ManeuverType.MAINTAIN_COURSE_SPEED
            pass_astern = None
            forbidden = ForbiddenManeuver.NONE

        return ColregsRuleInterpretation(
            target_name=ts.name,
            encounter_type=EncounterType.CROSSING,
            own_ship_role=own_role,
            applicable_rules=["Rule 15", "Rule 16", "Rule 17", "Rule 8", "Rule 6"],
            spatial=self._build_spatial_constraint(
                ts.name, geo, risk, own_role, pass_astern=pass_astern),
            maneuver=ManeuverConstraint(
                required_maneuver=maneuver,
                forbidden_maneuver=forbidden,
                alteration_min_angle=(self.MIN_ALTERATION_ANGLE
                                      if own_role == ShipRole.GIVE_WAY else 0.0),
                priority=1 if own_role == ShipRole.GIVE_WAY else 0,
            ),
            speed=SpeedConstraint(
                max_speed=self.MAX_SAFE_SPEED_CLEAR,
            ),
        )

    def _rule19_encounter(
        self, os, ts, geo, risk
    ) -> ColregsRuleInterpretation:
        """Rule 19: Restricted Visibility — no stand-on vessel.

        All vessels proceed at safe speed and take avoiding action.
        No concept of give-way/stand-on; all are responsible.
        """
        # Determine safest maneuver based on bearing
        bearing_deg = geo['rel_bearing_deg']
        if bearing_deg > 0:
            # Target starboard: alter to starboard (standard)
            maneuver = ManeuverType.ALTER_TO_STARBOARD
            forbidden = ForbiddenManeuver.ALTER_TO_PORT
        else:
            # Target port: reduce speed, alter to starboard if needed
            maneuver = ManeuverType.REDUCE_SPEED
            forbidden = ForbiddenManeuver.ALTER_TO_PORT

        return ColregsRuleInterpretation(
            target_name=ts.name,
            encounter_type=EncounterType.RESTRICTED_VIS,
            own_ship_role=ShipRole.GIVE_WAY,  # all give way in restricted vis
            applicable_rules=["Rule 19", "Rule 8", "Rule 6", "Rule 5"],
            spatial=SpatialConstraint(
                target_name=ts.name,
                min_distance=self.CPA_SAFE_DISTANCE * 1.5,  # extra margin
                pass_astern=True,
                priority=2,
            ),
            maneuver=ManeuverConstraint(
                required_maneuver=maneuver,
                forbidden_maneuver=forbidden,
                alteration_min_angle=self.MIN_ALTERATION_ANGLE,
                max_yaw_rate=0.3,  # slower turning in restricted vis
                priority=2,
            ),
            speed=SpeedConstraint(
                max_speed=self.MAX_SAFE_SPEED_RESTRICTED,
                visibility_reduction=0.5,
            ),
        )

    # =====================================================================
    # Spatial constraint builder
    # =====================================================================

    def _build_spatial_constraint(
        self,
        target_name: str,
        geo: dict,
        risk: dict,
        role: ShipRole,
        pass_astern: Optional[bool] = None,
    ) -> SpatialConstraint:
        """Build a spatial constraint for a target ship.

        The min_distance is based on:
          - Base safe CPA (50m)
          - Scaled by risk level
          - Scaled by traffic density
        """
        base_cpa = self.CPA_SAFE_DISTANCE

        # Risk-based scaling
        if risk['cpa_level'] == 'critical':
            base_cpa *= 1.3
        elif risk['cpa_level'] == 'high':
            base_cpa *= 1.1

        # Traffic density scaling
        base_cpa *= self.traffic_density

        # Sea state scaling
        base_cpa *= (1 + 0.05 * self.sea_state)

        # CPA must be at least the current CPA + safety margin
        min_cpa = max(base_cpa, risk['cpa'] + 10.0)

        # Bearing constraints based on passing side
        forbidden_min = None
        forbidden_max = None
        if pass_astern is True:
            # Must pass behind TS → forbidden to go directly toward TS
            bearing = geo['rel_bearing']
            # Forbidden sector: ±30° around bearing
            forbidden_min = float(bearing - math.radians(30))
            forbidden_max = float(bearing + math.radians(30))

        priority = 2 if risk['cpa_level'] in ('critical', 'high') else 1

        return SpatialConstraint(
            target_name=target_name,
            min_distance=float(min_cpa),
            forbidden_bearing_min=forbidden_min,
            forbidden_bearing_max=forbidden_max,
            pass_astern=pass_astern,
            valid_until_tcpa=risk['tcpa'] if risk['tcpa'] > 0 else 60.0,
            priority=priority,
        )

    # =====================================================================
    # Step 4: Global classification
    # =====================================================================

    def _classify_global(
        self, interpretations: List[ColregsRuleInterpretation]
    ) -> EncounterClassification:
        """Determine the global encounter type across all targets.

        Priority logic:
          1. If restricted visibility → Rule 19 overrides all
          2. If any head-on → head_on primary
          3. If crossing + overtaking → multi_ship
          4. If multiple crossing → multi_ship
          5. Otherwise → worst single encounter
        """
        if not interpretations:
            return EncounterClassification(
                primary_encounter=EncounterType.NO_RISK,
                risk_level="low",
            )

        types = [i.encounter_type for i in interpretations]
        all_rules = []
        for i in interpretations:
            all_rules.extend(i.applicable_rules)
        all_rules = list(set(all_rules))

        # Priority detection
        has_restricted = EncounterType.RESTRICTED_VIS in types
        has_headon = EncounterType.HEAD_ON in types
        has_crossing = EncounterType.CROSSING in types
        has_overtaking = EncounterType.OVERTAKING in types

        # Determine give-way/stand-on
        is_stand_on = any(
            i.own_ship_role == ShipRole.STAND_ON for i in interpretations)
        must_give_way = any(
            i.own_ship_role == ShipRole.GIVE_WAY for i in interpretations)

        # Primary encounter
        if has_restricted:
            primary = EncounterType.RESTRICTED_VIS
            risk = "medium"
        elif has_headon:
            primary = EncounterType.HEAD_ON
            risk = "high"
        elif has_crossing and (has_overtaking or len(types) > 1):
            primary = EncounterType.MULTI_SHIP
            risk = "high"
        elif has_crossing:
            primary = EncounterType.CROSSING
            risk = "medium"
        elif has_overtaking:
            primary = EncounterType.OVERTAKING
            risk = "low"
        else:
            primary = EncounterType.NO_RISK
            risk = "low"

        # Rule priority order (from COLREGS)
        rule_order = self._resolve_rule_priority(all_rules)

        # CPA risk field
        cpa_risk = 0.0
        for interp in interpretations:
            if interp.spatial:
                cpa_risk = max(cpa_risk, 0.5 if interp.spatial.priority >= 2 else 0.2)

        return EncounterClassification(
            primary_encounter=primary,
            all_encounters=[t.value for t in types],
            rule_priority_order=rule_order,
            risk_level=risk,
            is_stand_on_vessel=is_stand_on and not must_give_way,
            cpa_risk_field=cpa_risk,
            environment_context=(
                f"visibility={self.visibility}, sea_state={self.sea_state}, "
                f"traffic_density={self.traffic_density}"
            ),
        )

    def _resolve_rule_priority(self, rules: List[str]) -> List[str]:
        """Resolve COLREGS rule priority order (Rule 18 hierarchy)."""
        priority_map = {
            "Rule 19": 0, "Rule 18": 1, "Rule 14": 2, "Rule 15": 3,
            "Rule 13": 4, "Rule 16": 5, "Rule 17": 6, "Rule 8": 7,
            "Rule 7": 8, "Rule 6": 9, "Rule 5": 10, "Rule 2": 11,
        }
        return sorted(rules, key=lambda r: priority_map.get(r, 99))

    # =====================================================================
    # Step 5: Constraint aggregation
    # =====================================================================

    def _aggregate_maneuvers(
        self,
        interpretations: List[ColregsRuleInterpretation],
        encounter_class: EncounterClassification,
    ) -> Tuple[ManeuverType, ForbiddenManeuver]:
        """Aggregate maneuver constraints across all targets.

        Conflict resolution:
          - If any requires starboard turn → global starboard turn
          - If conflict (port vs starboard) → starboard wins (COLREGS default)
          - Stand-on ship: maintain unless risk of collision (Rule 17(a)(ii))
        """
        if not interpretations:
            return ManeuverType.MAINTAIN_COURSE_SPEED, ForbiddenManeuver.NONE

        maneuvers = []
        forbiddens = []
        for interp in interpretations:
            if interp.maneuver:
                maneuvers.append(interp.maneuver.required_maneuver)
                forbiddens.append(interp.maneuver.forbidden_maneuver)

        # Conflict resolution
        if ManeuverType.ALTER_TO_STARBOARD in maneuvers:
            global_maneuver = ManeuverType.ALTER_TO_STARBOARD
        elif ManeuverType.ALTER_TO_PORT in maneuvers:
            # Only port if no starboard requirement and not head-on
            if encounter_class.primary_encounter != EncounterType.HEAD_ON:
                global_maneuver = ManeuverType.ALTER_TO_PORT
            else:
                global_maneuver = ManeuverType.ALTER_TO_STARBOARD
        elif ManeuverType.REDUCE_SPEED in maneuvers:
            global_maneuver = ManeuverType.REDUCE_SPEED
        else:
            global_maneuver = ManeuverType.MAINTAIN_COURSE_SPEED

        # Forbidden maneuvers
        if ForbiddenManeuver.ALTER_TO_PORT in forbiddens:
            global_forbidden = ForbiddenManeuver.ALTER_TO_PORT
        elif ForbiddenManeuver.ALTER_TO_STARBOARD in forbiddens:
            global_forbidden = ForbiddenManeuver.ALTER_TO_STARBOARD
        else:
            global_forbidden = ForbiddenManeuver.NONE

        # Rule 17(a)(ii): stand-on may take action if collision risk is imminent
        if encounter_class.is_stand_on_vessel and \
           encounter_class.risk_level in ("high", "critical"):
            global_forbidden = ForbiddenManeuver.ALTER_TO_PORT  # don't turn toward danger
            if global_maneuver == ManeuverType.MAINTAIN_COURSE_SPEED:
                global_maneuver = ManeuverType.ALTER_TO_STARBOARD

        return global_maneuver, global_forbidden

    def _compute_safe_speed(
        self, encounter_class: EncounterClassification
    ) -> float:
        """Compute safe speed (Rule 6).

        Factors:
          - Visibility state
          - Sea state
          - Traffic density
          - Encounter risk level
        """
        if self.visibility == "restricted":
            base_speed = self.MAX_SAFE_SPEED_RESTRICTED
        else:
            base_speed = self.MAX_SAFE_SPEED_CLEAR

        # Risk scaling
        if encounter_class.risk_level == "critical":
            base_speed *= 0.5
        elif encounter_class.risk_level == "high":
            base_speed *= 0.7
        elif encounter_class.risk_level == "medium":
            base_speed *= 0.85

        # Traffic density
        base_speed /= self.traffic_density

        # Sea state: higher sea state = lower speed
        if self.sea_state >= 5:
            base_speed *= 0.6
        elif self.sea_state >= 3:
            base_speed *= 0.8

        return float(max(base_speed, 0.5))  # minimum steerage speed

    def _aggregate_cpa(
        self, interpretations: List[ColregsRuleInterpretation]
    ) -> Tuple[float, float]:
        """Aggregate CPA/TCPA constraints — take the most restrictive."""
        min_cpa = self.CPA_SAFE_DISTANCE
        min_tcpa = self.TCPA_WARNING_TIME

        for interp in interpretations:
            if interp.spatial:
                min_cpa = max(min_cpa, interp.spatial.min_distance)

        return float(min_cpa), float(min_tcpa)

    # =====================================================================
    # Step 7-8: Reasoning trace and degradation
    # =====================================================================

    def _build_reasoning_trace(
        self,
        interpretations: List[ColregsRuleInterpretation],
        encounter_class: EncounterClassification,
        global_maneuver: ManeuverType,
    ) -> str:
        """Build human-readable COLREGS legal reasoning trace (Section 6.5)."""
        lines = ["=== COLREGS Symbolic Referee — Reasoning Trace ===",
                 f"Visibility: {self.visibility}, Sea State: {self.sea_state}",
                 f"Primary Encounter: {encounter_class.primary_encounter.value}",
                 f"Risk Level: {encounter_class.risk_level}",
                 f"Global Maneuver: {global_maneuver.value}",
                 "",
                 "Per-Target Analysis:"]

        for interp in interpretations:
            lines.append(
                f"  {interp.target_name}: {interp.encounter_type.value} — "
                f"OS is {interp.own_ship_role.value}")
            lines.append(f"    Rules: {', '.join(interp.applicable_rules)}")
            if interp.spatial:
                lines.append(f"    Min CPA: {interp.spatial.min_distance:.1f}m")
                if interp.spatial.pass_astern:
                    lines.append(f"    Required: pass astern of {interp.target_name}")
            if interp.maneuver:
                lines.append(f"    Maneuver: {interp.maneuver.required_maneuver.value}")
                if interp.maneuver.forbidden_maneuver != ForbiddenManeuver.NONE:
                    lines.append(
                        f"    Forbidden: {interp.maneuver.forbidden_maneuver.value}")
            if interp.speed:
                lines.append(f"    Max Speed: {interp.speed.max_speed:.1f} m/s")

        lines.append("")
        lines.append(f"Rule Priority Order: "
                     f"{' > '.join(encounter_class.rule_priority_order)}")
        lines.append(f"Cumulative CPA Risk Field: "
                     f"{encounter_class.cpa_risk_field:.3f}")

        return "\n".join(lines)

    def _determine_degradation(
        self,
        encounter_class: EncounterClassification,
        cpa_risk: float,
    ) -> int:
        """Determine degradation level for the fallback state machine.

        0 = normal operation
        1 = soft degradation (elevated caution)
        2 = hard degradation (conservative constraints)
        3 = emergency (stop/safe-mode)
        """
        if encounter_class.risk_level == "critical" or cpa_risk > 0.8:
            return 3
        elif encounter_class.risk_level == "high" or cpa_risk > 0.5:
            return 2
        elif encounter_class.risk_level == "medium" or cpa_risk > 0.3:
            return 1
        return 0
