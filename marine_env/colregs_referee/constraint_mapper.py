#!/usr/bin/env python3
"""
Symbolic-to-Numeric Constraint Mapper
======================================

Transforms discrete COLREGS semantic constraints from the symbolic referee
into continuous numeric constraints for the NMPC control layer.

Mapping functions (Section 4.4 of the research document):

  Rule 8(d)  → min_cpa  (time-varying non-convex exclusion zone)
  Rule 14/15 → forbidden_maneuver (turn direction: alter_to_port blocked)
  Rule 8(b)  → alteration_min_angle (minimum course change magnitude)
  Rule 6     → max_speed (speed upper bound)

The mapper also implements the hierarchical slack variable mechanism:
  - ε_safety  (priority=2): collision avoidance — NEVER relaxed
  - ε_legal   (priority=1): COLREGS compliance — relaxed under infeasibility
  - ε_smooth  (priority=0): trajectory smoothness — relaxed first

Slack penalty weights: w_safety >> w_legal >> w_smooth
"""

import math
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from .output_schema import (
    COLREGSConstraintOutput,
    ColregsRuleInterpretation,
    SpatialConstraint,
    ManeuverConstraint,
    SpeedConstraint,
    ManeuverType,
    ForbiddenManeuver,
    ShipRole,
    EncounterType,
)


# =============================================================================
# NMPC constraint data structures
# =============================================================================

@dataclass
class SpatialNMPCConstraint:
    """NMPC-compatible spatial constraint for a single target ship.

    Implements the nonlinear exclusion zone (Rule 8(d)):
      ||os_pos[k] - ts_pos[k]||² ≥ min_cpa² + ε_safety

    For the NMPC formulation (Section 4.4):
      g_cpa(x_k) = (x_os - x_ts)² + (y_os - y_ts)² - r_safe² ≥ 0
    """
    target_name: str
    # Target ship predicted trajectory for N prediction steps
    ts_trajectory: Optional[List[np.ndarray]] = None  # [(x, y), ...] for k=0..N-1
    # Exclusion radius (min CPA distance)
    min_distance: float = 50.0
    # Passing side (convex half-plane constraints)
    pass_astern: bool = False
    pass_ahead: bool = False
    # Forbidden bearing sector (for Rule 14/15 maneuver constraints)
    forbidden_bearing_min: Optional[float] = None
    forbidden_bearing_max: Optional[float] = None
    # Slack variable — priority 2 (safety, NEVER relaxed)
    epsilon_safety: float = 0.0
    epsilon_safety_weight: float = 1e6  # w_safety — effectively infinite
    # Constraint type for NMPC solver
    constraint_type: str = "exclusion_circle"  # exclusion_circle | half_plane


@dataclass
class ManeuverNMPCConstraint:
    """NMPC-compatible maneuver constraint.

    Rule 14/15: forbidden_maneuver = alter_to_port
      → u[0] (rudder angle) ≥ 0  (or heading rate ≥ 0)

    Rule 8(b): alteration_min_angle
      → |ψ_N - ψ_0| ≥ min_angle, with ψ_N - ψ_0 ≥ min_angle for starboard
    """
    # Forbidden turn direction (control input bound)
    forbidden_maneuver: ForbiddenManeuver = ForbiddenManeuver.NONE
    # Minimum course alteration (state constraint at terminal prediction step)
    alteration_min_angle: float = 0.0   # rad
    # Turn direction constraint: -1 = port only, +1 = starboard only, 0 = free
    turn_direction_sign: int = 0
    # Control input bounds
    rudder_min: float = -0.5           # rad
    rudder_max: float = 0.5            # rad
    # Slack variable — priority 1 (legal compliance, relaxed under infeasibility)
    epsilon_legal: float = 0.0
    epsilon_legal_weight: float = 1e3  # w_legal
    # Slack variable — priority 0 (smoothness)
    epsilon_smooth: float = 0.0
    epsilon_smooth_weight: float = 1e0  # w_smooth


@dataclass
class SpeedNMPCConstraint:
    """NMPC-compatible speed constraint (Rule 6).

    speed_lower ≤ √(vx² + vy²) ≤ speed_upper

    For the underactuated 3DOF model:
      speed ≈ surge velocity (sway is small)
      → u_surge ∈ [min_speed, max_speed]
    """
    max_speed: float = 5.0        # m/s — Rule 6 upper bound
    min_speed: float = 0.5        # m/s — minimum steerage speed
    # Soft constraint: can violate slightly for collision avoidance
    epsilon_speed: float = 0.0
    epsilon_speed_weight: float = 1e2


@dataclass
class NMPCConstraints:
    """Complete set of NMPC constraints generated from referee output.

    This is what the NMPC layer consumes at each re-planning step.
    """
    timestamp: float = 0.0
    scenario_id: str = ""

    # Spatial constraints (one per target ship)
    spatial_constraints: List[SpatialNMPCConstraint] = field(default_factory=list)

    # Maneuver constraints (global for OS)
    maneuver_constraint: ManeuverNMPCConstraint = field(
        default_factory=ManeuverNMPCConstraint)

    # Speed constraint (global for OS)
    speed_constraint: SpeedNMPCConstraint = field(
        default_factory=SpeedNMPCConstraint)

    # Global prediction horizon metadata
    prediction_horizon_N: int = 20
    time_step_dt: float = 0.5         # s

    # Slack variable hierarchy summary
    max_epsilon_safety: float = 1.0   # max allowed safety constraint relaxation
    max_epsilon_legal: float = 5.0    # max allowed legal compliance relaxation
    max_epsilon_smooth: float = 10.0  # max allowed smoothness relaxation
    max_epsilon_speed: float = 2.0    # max allowed speed constraint relaxation

    def is_feasible(self) -> bool:
        """Quick feasibility check: are safety constraints satisfiable?"""
        return all(sc.epsilon_safety < self.max_epsilon_safety
                   for sc in self.spatial_constraints)

    def to_dict(self) -> dict:
        """Convert to dictionary for YAML/JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'scenario_id': self.scenario_id,
            'num_spatial_constraints': len(self.spatial_constraints),
            'min_cpa_distances': [sc.min_distance for sc in self.spatial_constraints],
            'spatial_targets': [
                {
                    'target_name': sc.target_name,
                    'min_distance': sc.min_distance,
                    'pass_astern': sc.pass_astern,
                }
                for sc in self.spatial_constraints
            ],
            'forbidden_maneuver': self.maneuver_constraint.forbidden_maneuver.value,
            'alteration_min_angle_deg': math.degrees(
                self.maneuver_constraint.alteration_min_angle),
            'turn_direction_sign': self.maneuver_constraint.turn_direction_sign,
            'rudder_bounds': (self.maneuver_constraint.rudder_min,
                              self.maneuver_constraint.rudder_max),
            'speed_bounds': (self.speed_constraint.min_speed,
                             self.speed_constraint.max_speed),
            'prediction_horizon': f"N={self.prediction_horizon_N}, dt={self.time_step_dt}s",
            'slack_weights': {
                'w_safety': self.spatial_constraints[0].epsilon_safety_weight
                if self.spatial_constraints else 1e6,
                'w_legal': self.maneuver_constraint.epsilon_legal_weight,
                'w_smooth': self.maneuver_constraint.epsilon_smooth_weight,
                'w_speed': self.speed_constraint.epsilon_speed_weight,
            },
        }


# =============================================================================
# Constraint Mapper
# =============================================================================

class ConstraintMapper:
    """Transforms symbolic COLREGS constraints into NMPC numerical bounds.

    Usage:
        mapper = ConstraintMapper()
        referee_output = referee.evaluate(os, targets)
        nmpc_constraints = mapper.map(referee_output)
    """

    def __init__(self, prediction_horizon: int = 20, dt: float = 0.5,
                 min_cpa_margin: float = 10.0):
        """
        Args:
            prediction_horizon: NMPC prediction horizon steps
            dt: time step in seconds
            min_cpa_margin: additional safety margin on CPA (m)
        """
        self.N = prediction_horizon
        self.dt = dt
        self.min_cpa_margin = min_cpa_margin

    def map(self, referee_output: COLREGSConstraintOutput) -> NMPCConstraints:
        """Transform referee output to NMPC constraints.

        This is the main entry point. It maps each symbolic constraint
        field to a corresponding numerical constraint, then applies
        rule-specific enhancements for all 9 COLREGS rule mappings:
          Rule 6  -> speed bounds
          Rule 8  -> exclusion circle + min alteration angle
          Rule 13 -> overtaking geometry + continuity
          Rule 14 -> head-on forbidden port turn
          Rule 15 -> crossing give-way/stand-on role constraints
          Rule 16 -> give-way early & substantial action
          Rule 17 -> stand-on maintain + breakthrough condition
          Rule 18 -> responsibility hierarchy priority
          Rule 19 -> restricted visibility override
        """
        # ── Map spatial constraints ──
        spatial_nmpc = []
        for interp in referee_output.target_interpretations:
            if interp.spatial:
                sc = self._map_spatial(interp.spatial)
                spatial_nmpc.append(sc)

        # ── Map maneuver constraints ──
        maneuver_nmpc = self._map_maneuver(referee_output)

        # ── Map speed constraint ──
        speed_nmpc = self._map_speed(referee_output)

        nmpc = NMPCConstraints(
            timestamp=referee_output.timestamp,
            scenario_id=referee_output.scenario_id,
            spatial_constraints=spatial_nmpc,
            maneuver_constraint=maneuver_nmpc,
            speed_constraint=speed_nmpc,
            prediction_horizon_N=self.N,
            time_step_dt=self.dt,
        )

        # ── Apply rule-specific enhancements (Phase 5-6 completion) ──
        nmpc = self._enhance_constraints(referee_output, nmpc)

        return nmpc

    def _map_spatial(self, spatial: SpatialConstraint) -> SpatialNMPCConstraint:
        """Map a spatial constraint to NMPC exclusion zone.

        Rule 8(d): min_cpa maps to:
          g(x_k) = ||OS_pos - TS_pos||² - r_safe² ≥ ε_safety

        Passing side maps to half-plane constraints:
          pass_astern:  OS must be behind TS
          pass_ahead:   OS must be ahead of TS
        """
        return SpatialNMPCConstraint(
            target_name=spatial.target_name,
            min_distance=spatial.min_distance + self.min_cpa_margin,
            pass_astern=spatial.pass_astern if spatial.pass_astern is not None else False,
            pass_ahead=spatial.pass_ahead if spatial.pass_ahead is not None else False,
            forbidden_bearing_min=spatial.forbidden_bearing_min,
            forbidden_bearing_max=spatial.forbidden_bearing_max,
            epsilon_safety=0.0,
            epsilon_safety_weight=1e6,  # effectively infinite — never relax safety
            constraint_type=(
                "half_plane" if spatial.pass_astern or spatial.pass_ahead
                else "exclusion_circle"
            ),
        )

    def _map_maneuver(
        self, referee_output: COLREGSConstraintOutput
    ) -> ManeuverNMPCConstraint:
        """Map global maneuver directive to NMPC control constraints.

        forbidden_maneuver → rudder angle sign constraint
          alter_to_port forbidden:  rudder ≥ 0 (NO port turn)
          alter_to_starboard forbidden: rudder ≤ 0 (NO starboard turn)

        alteration_min_angle → terminal heading constraint
          |ψ_N - ψ_0| ≥ min_angle (with correct sign for starboard)
        """
        forbidden = referee_output.forbidden_maneuver
        required = referee_output.required_maneuver

        # Determine turn direction sign and rudder bounds
        if forbidden == ForbiddenManeuver.ALTER_TO_PORT:
            # Cannot turn to port → rudder must be non-negative (starboard only)
            turn_sign = +1
            rudder_min = 0.0
            rudder_max = 0.5
        elif forbidden == ForbiddenManeuver.ALTER_TO_STARBOARD:
            # Cannot turn to starboard → rudder must be non-positive (port only)
            turn_sign = -1
            rudder_min = -0.5
            rudder_max = 0.0
        else:
            turn_sign = 0
            rudder_min = -0.5
            rudder_max = 0.5

        # Minimum alteration angle from per-target interpretations
        min_alteration = 0.0
        for interp in referee_output.target_interpretations:
            if interp.maneuver and interp.maneuver.alteration_min_angle > min_alteration:
                min_alteration = interp.maneuver.alteration_min_angle

        # If altering to starboard is required, the alteration must be positive
        if required == ManeuverType.ALTER_TO_STARBOARD and min_alteration == 0.0:
            min_alteration = math.radians(30)  # default substantial alteration

        return ManeuverNMPCConstraint(
            forbidden_maneuver=forbidden,
            alteration_min_angle=min_alteration,
            turn_direction_sign=turn_sign,
            rudder_min=rudder_min,
            rudder_max=rudder_max,
            epsilon_legal=0.0,
            epsilon_legal_weight=1e3,
            epsilon_smooth=0.0,
            epsilon_smooth_weight=1e0,
        )

    def _map_speed(
        self, referee_output: COLREGSConstraintOutput
    ) -> SpeedNMPCConstraint:
        """Map speed constraint to NMPC surge velocity bounds.

        Rule 6: max_safe_speed → surge_upper_bound
        Minimum steerage speed → surge_lower_bound

        The speed constraint is soft (can be relaxed slightly for safety).
        """
        # Collect per-target speed limits
        max_speeds = [referee_output.max_safe_speed]
        for interp in referee_output.target_interpretations:
            if interp.speed:
                max_speeds.append(interp.speed.max_speed)

        # Most restrictive
        speed_upper = min(max_speeds)

        return SpeedNMPCConstraint(
            max_speed=speed_upper,
            min_speed=0.5,  # minimum steerage speed for WAM-V
            epsilon_speed=0.0,
            epsilon_speed_weight=1e2,
        )

    # =====================================================================
    # Rule-specific constraint enhancement layer (Section 4.4)
    # =====================================================================

    def _enhance_constraints(
        self, referee_output: COLREGSConstraintOutput, nmpc: NMPCConstraints
    ) -> NMPCConstraints:
        """Apply rule-specific tuning to all NMPC constraints.

        This completes the 9 COLREGS rule mappings:
          Rule 6  -> _map_speed (base layer, already applied)
          Rule 8  -> _map_spatial + _map_maneuver (base layer, already applied)
          Rule 13 -> overtaking geometry + continuity (new)
          Rule 14 -> _map_maneuver: forbidden port turn (base layer, already applied)
          Rule 15 -> crossing role-specific: give-way / stand-on (new)
          Rule 16 -> give-way early & substantial action (new)
          Rule 17 -> stand-on maintain + breakthrough condition (new)
          Rule 18 -> responsibility hierarchy priority (new)
          Rule 19 -> restricted visibility global override (new)
        """
        for interp in referee_output.target_interpretations:
            rules = interp.applicable_rules
            role = interp.own_ship_role
            enc_type = interp.encounter_type

            # Find the corresponding spatial constraint
            sc = next(
                (s for s in nmpc.spatial_constraints
                 if s.target_name == interp.target_name), None)

            # ── Rule 13: Overtaking ──
            if "Rule 13" in rules:
                self._apply_rule13_overtaking(interp, sc, nmpc)

            # ── Rule 15: Crossing ──
            if "Rule 15" in rules:
                self._apply_rule15_crossing(interp, sc, nmpc)

            # ── Rule 16: Give-way vessel action ──
            if "Rule 16" in rules:
                self._apply_rule16_giveway(interp, sc, nmpc)

            # ── Rule 17: Stand-on vessel duty ──
            if "Rule 17" in rules:
                self._apply_rule17_standon(interp, sc, nmpc, referee_output)

            # ── Rule 18: Responsibility hierarchy ──
            if "Rule 18" in rules:
                self._apply_rule18_hierarchy(interp, sc, nmpc)

        # ── Rule 19: Restricted Visibility — global override ──
        if referee_output.encounter_classification.primary_encounter == \
           EncounterType.RESTRICTED_VIS:
            self._apply_rule19_restricted_vis(referee_output, nmpc)

        return nmpc

    # ── Rule 13: Overtaking ────────────────────────────────────────────

    def _apply_rule13_overtaking(
        self, interp: ColregsRuleInterpretation,
        sc: Optional[SpatialNMPCConstraint], nmpc: NMPCConstraints
    ) -> None:
        """Rule 13: Overtaking — forbidden passing from right, maintain continuity.

        When OS is the overtaking vessel (give_way):
          - Must keep out of the way of the vessel being overtaken
          - Must not cross ahead of the overtaken vessel
          - Must pass astern with wider safety margin
          - Maintain giving-way continuity until finally past and clear
          - Forbid alteration to port (stay on starboard passing side)

        When OS is being overtaken (stand_on):
          - Maintain course and speed
          - No sudden maneuvers that could confuse the overtaking vessel
        """
        is_overtaking = interp.own_ship_role == ShipRole.GIVE_WAY

        if is_overtaking:
            # OS is the overtaking vessel — must give way
            if sc:
                # Wider CPA: overtaking requires keeping clear of TS's path
                sc.min_distance = max(sc.min_distance,
                                      self.CPA_SAFE_DISTANCE * 1.5)
                sc.pass_astern = True
                sc.pass_ahead = False
                # Increase safety slack weight — overtaking duty is absolute
                sc.epsilon_safety_weight = 2e6

            # Maneuver: ensure forbids port turn, requires starboard alteration
            nmpc.maneuver_constraint.forbidden_maneuver = \
                ForbiddenManeuver.ALTER_TO_PORT
            nmpc.maneuver_constraint.rudder_min = max(
                nmpc.maneuver_constraint.rudder_min, 0.0)
            nmpc.maneuver_constraint.turn_direction_sign = 1

            # Min alteration angle: must be clearly visible to TS
            nmpc.maneuver_constraint.alteration_min_angle = max(
                nmpc.maneuver_constraint.alteration_min_angle,
                self.MIN_ALTERATION_ANGLE)

            # Speed: must exceed TS speed to maintain overtaking continuity
            # (min speed is set by referee, we enforce it here)
            nmpc.speed_constraint.max_speed = min(
                nmpc.speed_constraint.max_speed, self.MAX_SAFE_SPEED_CLEAR)
        else:
            # OS is being overtaken — stand on, minimize disturbances
            if sc:
                # Tighter spatial: don't make sudden changes
                sc.epsilon_safety_weight = 1.5e6

            # Tighten maneuver bounds (stay on course)
            nmpc.maneuver_constraint.epsilon_legal_weight = 5e3
            nmpc.maneuver_constraint.epsilon_smooth = 0.0

    # ── Rule 15: Crossing ──────────────────────────────────────────────

    def _apply_rule15_crossing(
        self, interp: ColregsRuleInterpretation,
        sc: Optional[SpatialNMPCConstraint], nmpc: NMPCConstraints
    ) -> None:
        """Rule 15: Crossing situation — role-dependent constraint tuning.

        Give-way vessel (TS on starboard bow):
          - Must avoid crossing ahead of the other vessel
          - Shall pass astern of TS
          - Must alter course to starboard (larger alteration for crossing)
          - Must not alter to port

        Stand-on vessel (TS on port bow):
          - Keep course and speed
          - Tight maneuver bounds — don't interfere with give-way vessel
        """
        is_give_way = interp.own_ship_role == ShipRole.GIVE_WAY

        if is_give_way:
            # OS has TS on starboard → OS is give-way
            if sc:
                # Must pass astern of TS (avoid crossing ahead)
                sc.min_distance = max(sc.min_distance,
                                      self.CPA_SAFE_DISTANCE * 1.3)
                sc.pass_astern = True
                sc.pass_ahead = False
                sc.epsilon_safety_weight = 1.5e6

            # Maneuver: alter to starboard, forbid port, substantial alteration
            nmpc.maneuver_constraint.forbidden_maneuver = \
                ForbiddenManeuver.ALTER_TO_PORT
            nmpc.maneuver_constraint.rudder_min = max(
                nmpc.maneuver_constraint.rudder_min, 0.0)
            nmpc.maneuver_constraint.turn_direction_sign = 1

            # Crossing requires more substantial alteration (45° minimum)
            nmpc.maneuver_constraint.alteration_min_angle = max(
                nmpc.maneuver_constraint.alteration_min_angle,
                math.radians(45))

            # Larger legal weight — crossing compliance is strict
            nmpc.maneuver_constraint.epsilon_legal_weight = 5e3
        else:
            # OS is stand-on (TS on port) — maintain course and speed
            if sc:
                # Softer spatial (not required to maneuver, but stay clear)
                sc.min_distance = max(sc.min_distance, self.CPA_SAFE_DISTANCE)
                sc.pass_astern = None  # no specific passing side requirement
                sc.pass_ahead = None
                sc.epsilon_safety_weight = 2e6  # still high for safety

            # Tight maneuver: maintain course
            nmpc.maneuver_constraint.turn_direction_sign = 0
            nmpc.maneuver_constraint.rudder_min = -0.15
            nmpc.maneuver_constraint.rudder_max = 0.15
            nmpc.maneuver_constraint.epsilon_legal_weight = 5e3

    # ── Rule 16: Give-way vessel — early & substantial action ──────────

    def _apply_rule16_giveway(
        self, interp: ColregsRuleInterpretation,
        sc: Optional[SpatialNMPCConstraint], nmpc: NMPCConstraints
    ) -> None:
        """Rule 16: Action by Give-way Vessel.

        Every vessel directed to keep out of the way shall:
          - Take early and substantial action
          - Action must be apparent to the other vessel
          - Large course alteration (≥ 45°) and/or significant speed change

        This enhances the basic forbidden_maneuver mapping with:
          - Larger alteration_min_angle (≥ 45°)
          - Higher legal compliance weight
          - Wider CPA margin for the give-way vessel
        """
        if sc:
            sc.min_distance = max(sc.min_distance,
                                  self.CPA_SAFE_DISTANCE * 1.4)
            sc.epsilon_safety_weight = 2e6

        # Substantial alteration: at least 45° (more than basic 30°)
        nmpc.maneuver_constraint.alteration_min_angle = max(
            nmpc.maneuver_constraint.alteration_min_angle,
            math.radians(45))

        # Higher legal weight — give-way action is mandatory, not optional
        nmpc.maneuver_constraint.epsilon_legal_weight = 8e3

    # ── Rule 17: Stand-on vessel — maintain + breakthrough ─────────────

    def _apply_rule17_standon(
        self, interp: ColregsRuleInterpretation,
        sc: Optional[SpatialNMPCConstraint], nmpc: NMPCConstraints,
        referee_output: COLREGSConstraintOutput,
    ) -> None:
        """Rule 17: Action by Stand-on Vessel.

        (a)(i): Stand-on vessel shall keep her course and speed.
        (a)(ii): May take action when it becomes apparent give-way vessel
                 is not taking appropriate action (BREAKTHROUGH).
        (b): When so close that collision cannot be avoided by give-way
             vessel alone, stand-on SHALL take the most effective action.

        Mapping strategy:
          - Normal: tight course/speed maintenance bounds
          - Breakthrough (high/critical risk): release stand-on, allow
            emergency starboard avoidance maneuver
        """
        is_stand_on = interp.own_ship_role == ShipRole.STAND_ON
        risk_level = referee_output.encounter_classification.risk_level
        degradation = referee_output.degradation_level
        # Breakthrough: soft-form risk_level OR hard risk from degradation state
        is_breakthrough = risk_level in ("high", "critical") or degradation >= 2

        if is_stand_on:
            if not is_breakthrough:
                # Stand-on duty: keep course and speed (normal operation)
                nmpc.maneuver_constraint.turn_direction_sign = 0
                nmpc.maneuver_constraint.epsilon_legal_weight = 5e3
                # Allow minimal rudder for station-keeping (drift correction)
                nmpc.maneuver_constraint.rudder_min = -0.10
                nmpc.maneuver_constraint.rudder_max = 0.10
                # Keep speed steady: narrow speed bounds around current
                nmpc.speed_constraint.epsilon_speed_weight = 5e3
                if sc:
                    sc.epsilon_safety_weight = 2.5e6
            else:
                # ── Breakthrough condition (Rule 17(a)(ii) + 17(b)) ──
                # Release stand-on constraints — allow full starboard maneuver,
                # but forbid port turn (consistent with ALTER_TO_PORT)
                nmpc.maneuver_constraint.turn_direction_sign = 1
                nmpc.maneuver_constraint.rudder_min = 0.0    # forbid port turn
                nmpc.maneuver_constraint.rudder_max = 0.5    # allow full starboard
                nmpc.maneuver_constraint.forbidden_maneuver = \
                    ForbiddenManeuver.ALTER_TO_PORT
                nmpc.maneuver_constraint.alteration_min_angle = max(
                    nmpc.maneuver_constraint.alteration_min_angle,
                    self.MIN_ALTERATION_ANGLE)
                # Legal weight lowered — safety above formality
                nmpc.maneuver_constraint.epsilon_legal_weight = 1e3
                if sc:
                    sc.epsilon_safety_weight = 2.5e6

    # ── Rule 18: Responsibility hierarchy ──────────────────────────────

    def _apply_rule18_hierarchy(
        self, interp: ColregsRuleInterpretation,
        sc: Optional[SpatialNMPCConstraint], nmpc: NMPCConstraints
    ) -> None:
        """Rule 18: Responsibilities Between Vessels.

        COLREGS vessel priority (high to low):
          NUC (Not Under Command) > RAM (Restricted Ability to Maneuver)
          > Fishing > Sailing > Power-driven

        Mapping: higher-priority vessels require larger exclusion zones
        from others. If OS is higher priority, spatial constraints on
        nearby TS are tightened (wider CPA), reflecting their duty to
        keep clear.

        For constraint mapping, this affects:
          - Spatial CPA margin scaling by relative priority
          - Maneuver urgency for lower-priority give-way vessels
        """
        # Rule 18 primarily affects role assignment (handled by referee).
        # In constraint mapper, we scale CPA based on responsibility level.

        # Extract priority level from encounter metadata
        is_high_priority = interp.own_ship_role == ShipRole.STAND_ON
        is_low_priority = interp.own_ship_role == ShipRole.GIVE_WAY

        if sc:
            if is_low_priority:
                # OS is lower priority → must give wider berth
                sc.min_distance = max(sc.min_distance,
                                      self.CPA_SAFE_DISTANCE * 1.5)
                sc.epsilon_safety_weight = 2e6
            elif is_high_priority:
                # OS is higher priority → others should keep clear
                # Maintain standard CPA but with high enforcement
                sc.min_distance = max(sc.min_distance, self.CPA_SAFE_DISTANCE)
                sc.epsilon_safety_weight = 2e6

        # Adjust legal weight for low-priority vessels
        if is_low_priority:
            nmpc.maneuver_constraint.epsilon_legal_weight = 8e3

    # ── Rule 19: Restricted Visibility — global override ───────────────

    def _apply_rule19_restricted_vis(
        self, referee_output: COLREGSConstraintOutput, nmpc: NMPCConstraints
    ) -> None:
        """Rule 19: Restricted Visibility.

        Overrides COLREGS Section II (Rules 11-18 — conduct in sight).
        Key provisions:
          - No stand-on vessel concept — ALL vessels give way
          - All vessels proceed at safe speed (≤ 50% of normal)
          - Avoid altering to port for vessels forward of the beam
          - Take avoiding action in ample time
          - Larger CPA margin (1.5× clear-weather value)

        Mapping:
          - Global speed reduction
          - Wider CPA for all targets
          - Forbid port turns (conservative default)
          - Increase all safety priority weights
          - Reduce max yaw rate (slower maneuvers in fog)
        """
        # ── Speed: restrict to safe speed for poor visibility ──
        nmpc.speed_constraint.max_speed = min(
            nmpc.speed_constraint.max_speed,
            self.MAX_SAFE_SPEED_RESTRICTED)
        nmpc.speed_constraint.epsilon_speed_weight = 5e3

        # ── Spatial: wider exclusion zones for all targets ──
        for sc in nmpc.spatial_constraints:
            sc.min_distance = max(sc.min_distance,
                                  self.CPA_SAFE_DISTANCE * 1.5)
            sc.epsilon_safety_weight = 2e6
            sc.pass_astern = True  # conservative: always pass behind

        # ── Maneuver: no port turn, slower turning rate ──
        nmpc.maneuver_constraint.forbidden_maneuver = \
            ForbiddenManeuver.ALTER_TO_PORT
        nmpc.maneuver_constraint.rudder_min = max(
            nmpc.maneuver_constraint.rudder_min, 0.0)
        nmpc.maneuver_constraint.turn_direction_sign = 1
        nmpc.maneuver_constraint.alteration_min_angle = max(
            nmpc.maneuver_constraint.alteration_min_angle,
            self.MIN_ALTERATION_ANGLE)

        # ── Increase all safety weights ──
        nmpc.maneuver_constraint.epsilon_legal_weight = 1e4

    # =====================================================================
    # Constraint tuning constants
    # =====================================================================

    # Safe CPA distance (Rule 8(d) reference)
    CPA_SAFE_DISTANCE = 50.0           # m
    # Safe speed limits
    MAX_SAFE_SPEED_CLEAR = 5.0         # m/s — clear visibility
    MAX_SAFE_SPEED_RESTRICTED = 2.5    # m/s — restricted visibility (50%)
    # Minimum alteration angle
    MIN_ALTERATION_ANGLE = math.radians(30)  # rad — Rule 8(b)

    # =====================================================================
    # Slack variable management (Section 4.4 — Lemma 1)
    # =====================================================================

    def resolve_infeasibility(
        self, nmpc: NMPCConstraints, solver_status: str
    ) -> NMPCConstraints:
        """Activate slack variables to resolve infeasibility.

        Implements Lemma 1 (Recursive Feasibility): when the LLM-derived
        legal constraints conflict with physical dynamics, relax legal
        constraints while keeping safety hard.

        Priority hierarchy: safety(2) > legal(1) > smoothness(0)
        """
        if solver_status == "SOLVED":
            return nmpc

        # Step 1: Try relaxing smoothness first
        if not any(sc.epsilon_safety > 0
                   for sc in nmpc.spatial_constraints):
            nmpc.maneuver_constraint.epsilon_smooth = min(
                nmpc.maneuver_constraint.epsilon_smooth + 0.5,
                nmpc.max_epsilon_smooth)

        # Step 2: If still infeasible, relax legal compliance
        if solver_status == "INFEASIBLE":
            nmpc.maneuver_constraint.epsilon_legal = min(
                nmpc.maneuver_constraint.epsilon_legal + 1.0,
                nmpc.max_epsilon_legal)
            nmpc.speed_constraint.epsilon_speed = min(
                nmpc.speed_constraint.epsilon_speed + 0.25,
                nmpc.max_epsilon_speed)

        # Step 3: NEVER relax safety — this is the hard guarantee
        # (safety slack stays at 0; if truly infeasible, activate emergency fallback)

        return nmpc
