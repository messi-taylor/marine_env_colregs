#!/usr/bin/env python3
"""
Scene Descriptor — Φ Operator: Numerical State → Natural Language
===================================================================

Implements the Contextual Semantic Abstraction operator (Section 4.2 / Phase 3-4):
  Φ: Rⁿ → Language Manifold

Converts continuous numerical maritime state (positions, headings, speeds, CPA/TCPA)
into discrete natural-language scene descriptions for downstream symbolic reasoning.

Architecture role (per 文档1.doc Section 3.1):
  Environment Perception → State Fusion → **Scene Semanticization (Φ)** → Symbolic Referee → NMPC

This module is independent of any LLM backend — it produces structured text that
can be consumed by either an LLM referee or a deterministic rule engine.

Original location: llm_referee.py (Phase 5-6)
Extracted to standalone module: 2026-06-07 (Phase 3-4 remediation)
"""

import math
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np

from .deterministic_referee import (
    ShipObservation,
    _compute_relative_state,
)


# =============================================================================
# SceneDescription — structured output of the Φ operator
# =============================================================================

@dataclass
class SceneDescription:
    """Natural language description of the maritime scene.

    Implements the Contextual Semantic Abstraction operator (Section 4.2):
      Φ: Rⁿ → Language Manifold

    Attributes:
        scene_text: Human-readable natural language scene description.
        own_ship_state: Dict with own ship position, heading, speed.
        target_ships: List of dicts, one per detected target ship.
        environment_context: String summarising visibility, sea state, etc.
        colregs_rules_applicable: List of COLREGS rule identifiers likely
            applicable to this scene.
    """
    scene_text: str
    own_ship_state: Dict
    target_ships: List[Dict]
    environment_context: str
    colregs_rules_applicable: List[str]


# =============================================================================
# Φ operator — numerical state → natural language
# =============================================================================

def build_scene_description(
    own_ship: ShipObservation,
    target_ships: List[ShipObservation],
    visibility: str = "clear",
    sea_state: int = 2,
    wind_speed: float = 5.0,
    wave_height: float = 0.5,
    current_speed: float = 0.3,
) -> SceneDescription:
    """Build a natural language scene description from numerical maritime state.

    Converts continuous numerical state to discrete semantic facts (Section 4.2):
      - Bearing → port_bow / starboard_bow / dead_ahead / astern
      - CPA → safe / close / dangerous / critical
      - Speed → slow / moderate / fast
      - Heading → northbound / southbound / eastbound / westbound

    Args:
        own_ship: Own ship observation (position, heading, speed, length).
        target_ships: List of target ship observations.
        visibility: Visibility condition string (clear, fog, etc.).
        sea_state: Sea state on the Douglas scale (0-9).
        wind_speed: Wind speed in m/s.
        wave_height: Significant wave height in m (JONSWAP).
        current_speed: Current speed in m/s.

    Returns:
        SceneDescription with structured natural-language scene.
    """
    os_speed = float(np.linalg.norm(own_ship.speed))
    os_heading_deg = math.degrees(own_ship.heading) % 360

    # Own ship description
    heading_cardinal = heading_to_cardinal(os_heading_deg)
    speed_desc = speed_to_text(os_speed)

    scene_lines = [
        f"## Maritime Scene Description",
        f"",
        f"### Own Ship (OS)",
        f"- Position: ({own_ship.position[0]:.1f}, {own_ship.position[1]:.1f}) m ENU",
        f"- Heading: {os_heading_deg:.0f}° ({heading_cardinal})",
        f"- Speed: {os_speed:.2f} m/s ({speed_desc})",
        f"- Ship length: {own_ship.length:.1f} m (WAM-V USV)",
        f"",
        f"### Environment",
        f"- Visibility: {visibility}",
        f"- Sea state: {sea_state}/9",
        f"- Wind: {wind_speed:.1f} m/s",
        f"- Wave height: {wave_height:.2f} m (JONSWAP)",
        f"- Current: {current_speed:.2f} m/s",
        f"",
    ]

    # Target ships
    if target_ships:
        scene_lines.append(f"### Target Ships ({len(target_ships)} detected)")
        scene_lines.append("")
        for ts in target_ships:
            ts_speed = float(np.linalg.norm(ts.speed))
            ts_heading_deg = math.degrees(ts.heading) % 360
            ts_cardinal = heading_to_cardinal(ts_heading_deg)

            # Compute relative geometry
            geo = _compute_relative_state(own_ship, ts)
            bearing_desc = bearing_to_text(geo['rel_bearing_deg'])
            cpa_desc = cpa_to_text(geo['cpa'], geo['tcpa'])
            sector = geo['bearing_sector']

            scene_lines.append(f"**{ts.name}**:")
            scene_lines.append(f"  - Position: ({ts.position[0]:.1f}, {ts.position[1]:.1f}) m")
            scene_lines.append(f"  - Heading: {ts_heading_deg:.0f}° ({ts_cardinal})")
            scene_lines.append(f"  - Speed: {ts_speed:.2f} m/s")
            scene_lines.append(f"  - Distance: {geo['rel_distance']:.1f} m")
            scene_lines.append(f"  - Relative bearing: {geo['rel_bearing_deg']:.1f}° ({bearing_desc}, sector={sector})")
            scene_lines.append(f"  - CPA: {geo['cpa']:.1f} m, TCPA: {geo['tcpa']:.1f} s ({cpa_desc})")
            scene_lines.append(f"  - Heading difference: {geo['heading_diff_deg']:.0f}°")

            # Encounter type hint
            encounter_hint_text = encounter_hint(geo)
            if encounter_hint_text:
                scene_lines.append(f"  - Likely encounter: {encounter_hint_text}")
            scene_lines.append("")

    scene_text = "\n".join(scene_lines)

    return SceneDescription(
        scene_text=scene_text,
        own_ship_state={
            'position': own_ship.position.tolist(),
            'heading': own_ship.heading,
            'speed': os_speed,
        },
        target_ships=[
            {
                'name': ts.name,
                'position': ts.position.tolist(),
                'heading': ts.heading,
                'speed': float(np.linalg.norm(ts.speed)),
                'length': ts.length,
            }
            for ts in target_ships
        ],
        environment_context=f"visibility={visibility}, sea={sea_state}",
        colregs_rules_applicable=suggest_rules(target_ships, own_ship),
    )


# =============================================================================
# Φ helper functions — continuous value → discrete semantic category
# =============================================================================

def heading_to_cardinal(deg: float) -> str:
    """Convert heading degrees to cardinal direction.

    Maps continuous heading angle to 8-point compass direction.

    Args:
        deg: Heading in degrees (0 = East, 90 = North, ENU convention).

    Returns:
        Cardinal direction string: one of N, NE, E, SE, S, SW, W, NW.
    """
    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = round(deg / 45) % 8
    return dirs[idx]


def speed_to_text(speed: float) -> str:
    """Convert speed (m/s) to natural language description.

    Args:
        speed: Speed in metres per second.

    Returns:
        Qualitative speed description.
    """
    if speed < 0.5:
        return "very slow (drifting)"
    elif speed < 1.5:
        return "slow"
    elif speed < 3.0:
        return "moderate"
    elif speed < 5.0:
        return "fast"
    else:
        return "very fast"


def bearing_to_text(bearing_deg: float) -> str:
    """Convert relative bearing to natural language description.

    Maritime convention: starboard bearings are positive, port bearings negative.

    Args:
        bearing_deg: Relative bearing in degrees (starboard +, port -).

    Returns:
        Human-readable bearing description.
    """
    if abs(bearing_deg) < 5:
        return "dead ahead"
    elif bearing_deg > 0:
        if bearing_deg <= 45:
            return "starboard bow"
        elif bearing_deg <= 112.5:
            return "starboard beam to bow"
        else:
            return "starboard quarter (astern)"
    else:
        if abs(bearing_deg) <= 45:
            return "port bow"
        elif abs(bearing_deg) <= 112.5:
            return "port beam to bow"
        else:
            return "port quarter (astern)"


def cpa_to_text(cpa: float, tcpa: float) -> str:
    """Convert CPA/TCPA to risk-level description.

    Args:
        cpa: Closest Point of Approach in metres.
        tcpa: Time to CPA in seconds.

    Returns:
        Risk-level string with CPA/TCPA context.
    """
    if cpa < 5:
        return "CRITICAL — imminent collision risk"
    elif cpa < 20:
        return f"DANGEROUS — close approach in {tcpa:.0f}s"
    elif cpa < 50:
        return f"CONCERNING — moderate risk"
    else:
        return "safe passing distance"


def encounter_hint(geo: dict) -> str:
    """Suggest likely COLREGS encounter type based on relative geometry.

    Args:
        geo: Relative geometry dict from _compute_relative_state() containing
             rel_bearing_deg, heading_diff_deg.

    Returns:
        Encounter hint string (Rule 14/15/13), or empty string if ambiguous.
    """
    bearing_abs = abs(geo['rel_bearing_deg'])
    heading_diff_abs = abs(geo['heading_diff_deg'])

    if bearing_abs < 5 and abs(heading_diff_abs - 180) < 15:
        return "Rule 14 HEAD-ON — nearly reciprocal courses"
    elif bearing_abs > 135:
        return "Rule 13 OVERTAKING — vessel astern"
    elif bearing_abs <= 112.5:
        if geo['rel_bearing_deg'] > 0:
            return "Rule 15 CROSSING — TS on starboard, OS is give-way"
        else:
            return "Rule 15 CROSSING — TS on port, OS is stand-on"
    return ""


def suggest_rules(
    target_ships: List[ShipObservation],
    own_ship: ShipObservation,
) -> List[str]:
    """Suggest applicable COLREGS rules based on scene geometry.

    Always includes Rules 5 (lookout), 6 (safe speed), and 7 (risk assessment).
    Adds encounter-specific rules (8, 13, 14, 15, 16, 17) as geometry indicates.

    Args:
        target_ships: List of target ship observations.
        own_ship: Own ship observation.

    Returns:
        Sorted list of applicable COLREGS rule identifiers (e.g. "Rule 14").
    """
    rules = set()
    for ts in target_ships:
        geo = _compute_relative_state(own_ship, ts)
        hint = encounter_hint(geo)
        if "Rule 14" in hint:
            rules.add("Rule 14")
            rules.add("Rule 8")
        elif "Rule 15" in hint:
            rules.add("Rule 15")
            rules.add("Rule 16" if "give-way" in hint else "Rule 17")
            rules.add("Rule 8")
        elif "Rule 13" in hint:
            rules.add("Rule 13")
            rules.add("Rule 8")
    rules.add("Rule 6")   # safe speed always applies
    rules.add("Rule 7")   # risk assessment always applies
    rules.add("Rule 5")   # lookout always applies
    return sorted(rules)
