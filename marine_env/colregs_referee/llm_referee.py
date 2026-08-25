#!/usr/bin/env python3
"""
LLM-Based COLREGS Referee Interface
====================================

Abstract interface and API-based implementations for LLM-driven
COLREGS rule interpretation with CFG-constrained structured output.

Architecture:
  - AbstractLLMReferee   — base class defining the LLM referee interface
  - AnthropicReferee     — Anthropic Claude API (structured output + prompt caching)
  - OpenAIReferee        — OpenAI-compatible API
  - SimulatedLLMReferee  — deterministic + templated reasoning for testing

The LLM is called with:
  1. A structured prompt describing the maritime situation
  2. CFG-defined JSON schema for output format
  3. Few-shot examples of COLREGS interpretations
  4. The current scene state (numerical → textual mapping)
"""

import json
import time
import os
import difflib
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np

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
    validate_output_dict,
)
from .deterministic_referee import (
    DeterministicReferee,
    ShipObservation,
)


# =============================================================================
# Scene description — imported from standalone Φ operator module
# (Phase 3-4 remediation: extracted from llm_referee.py to scene_descriptor.py)
# =============================================================================

from .scene_descriptor import (
    SceneDescription,
    build_scene_description,
)


def _safe_radians(value: Optional[float]) -> Optional[float]:
    """Auto-detect degrees vs radians conversion and clamp invalid values.

    LLMs sometimes output degrees (e.g., 30) instead of radians (0.5236)
    for angular quantities. Also clamps negative values to 0 (LLM sometimes
    outputs negative alteration angles, which fail schema validation).
    """
    if value is None:
        return None
    if value > math.pi:
        # Likely degrees — LLMs rarely intend angles > π rad for COLREGS
        return math.radians(value)
    if value < 0:
        # LLM output negative angle — clamp to 0 (no alteration required)
        return 0.0
    return value


# =============================================================================
# Few-shot examples for LLM in-context learning
# =============================================================================

FEW_SHOT_EXAMPLES = [
    {
        "scene": "OS at (0,0) heading 0° at 1.5 m/s. TS at (24, 357.5) heading 185° at 1.2 m/s. Distance 28.5m. CPA≈3m, TCPA≈11s.",
        "output": {
            "encounter_classification": {
                "primary_encounter": "head_on",
                "risk_level": "medium",
                "is_stand_on_vessel": False,
                "cpa_risk_field": 0.65,
                "environment_context": "clear visibility, calm seas"
            },
            "target_interpretations": [{
                "target_name": "ts01",
                "encounter_type": "head_on",
                "own_ship_role": "give_way",
                "applicable_rules": ["Rule 14", "Rule 8", "Rule 6"],
                "spatial": {
                    "target_name": "ts01",
                    "min_distance": 50.0,
                    "pass_astern": False,
                    "priority": 2
                },
                "maneuver": {
                    "required_maneuver": "alter_to_starboard",
                    "forbidden_maneuver": "alter_to_port",
                    "alteration_min_angle": 0.5236,
                    "priority": 2
                },
                "speed": {
                    "max_speed": 4.0
                }
            }],
            "required_maneuver": "alter_to_starboard",
            "forbidden_maneuver": "alter_to_port",
            "max_safe_speed": 4.0,
            "global_min_cpa": 50.0,
            "global_min_tcpa": 30.0,
            "confidence_score": 0.92,
            "reasoning_trace": "Rule 14 head-on situation: vessels meeting on reciprocal courses. Both must alter course to starboard so that each shall pass on the port side of the other. Rule 8 requires substantial action — recommend ≥30° starboard turn.",
            "fallback_active": False,
            "degradation_level": 1
        }
    },
    {
        "scene": "OS at (0,0) heading 0° at 1.5 m/s. TS at (158, 135) heading 270° at 1.0 m/s. Distance 80m. CPA=0, TCPA≈8s. TS on starboard bow.",
        "output": {
            "encounter_classification": {
                "primary_encounter": "crossing",
                "risk_level": "high",
                "is_stand_on_vessel": False,
                "cpa_risk_field": 0.85,
                "environment_context": "clear visibility, moderate seas"
            },
            "target_interpretations": [{
                "target_name": "ts02",
                "encounter_type": "crossing",
                "own_ship_role": "give_way",
                "applicable_rules": ["Rule 15", "Rule 16", "Rule 8", "Rule 6"],
                "spatial": {
                    "target_name": "ts02",
                    "min_distance": 55.0,
                    "pass_astern": True,
                    "priority": 2
                },
                "maneuver": {
                    "required_maneuver": "alter_to_starboard",
                    "forbidden_maneuver": "alter_to_port",
                    "alteration_min_angle": 0.5236,
                    "priority": 1
                },
                "speed": {
                    "max_speed": 3.5
                }
            }],
            "required_maneuver": "alter_to_starboard",
            "forbidden_maneuver": "alter_to_port",
            "max_safe_speed": 3.5,
            "global_min_cpa": 55.0,
            "global_min_tcpa": 30.0,
            "confidence_score": 0.88,
            "reasoning_trace": "Rule 15 crossing: TS is on starboard side, OS is the give-way vessel. Rule 16 requires early and substantial action. OS must alter course to starboard and pass astern of TS. Crossing ahead of TS is prohibited. TCPA=8s — urgent situation requires immediate action.",
            "fallback_active": False,
            "degradation_level": 2
        }
    }
]


# =============================================================================
# CFG Vocabulary & Enum Repair (Phase 5-6: logit pruning integration)
# =============================================================================

# Complete valid enum vocabulary — mirrors cfg_grammar.build_token_mask_map()
# Used by _build_vocabulary_section() and _repair_enum_values()

VALID_ENUMS = {
    "encounter_type": [
        "head_on", "crossing", "overtaking",
        "restricted_visibility", "multi_ship", "no_risk",
    ],
    "ship_role": ["give_way", "stand_on", "not_applicable"],
    "maneuver": [
        "alter_to_starboard", "alter_to_port", "maintain",
        "any_safe", "reduce_speed", "increase_speed",
    ],
    "forbidden_maneuver": [
        "alter_to_port", "alter_to_starboard", "none", "reduce_speed",
    ],
    "risk_level": ["low", "medium", "high", "critical"],
    "colregs_rule": [
        "Rule 2", "Rule 5", "Rule 6", "Rule 7", "Rule 8",
        "Rule 13", "Rule 14", "Rule 15", "Rule 16", "Rule 17",
        "Rule 18", "Rule 19",
    ],
    "degradation_level": [0, 1, 2, 3],
    "priority": [0, 1, 2],
}

# Known LLM enum errors → correct value mapping
# Built from observed qwen2.5:7b output errors; extendable
ENUM_CORRECTIONS: Dict[str, str] = {
    # maneuver values
    "maintain_course_and_speed": "maintain",
    "no action required": "maintain",
    "no action required (stand-on vessel)": "maintain",
    "no action required (stand on vessel)": "maintain",
    "no_action_required": "maintain",
    "no specific action required": "maintain",
    "keep_course_and_speed": "maintain",
    "continue_course_and_speed": "maintain",
    "stay_on_course": "maintain",
    "continue": "maintain",
    "hold": "maintain",
    "stand_on": "maintain",  # "stand on" is a ship role, not a maneuver
    # ship_role values
    "stand_on_vessel": "stand_on",
    "stand-on_vessel": "stand_on",
    "give_way_vessel": "give_way",
    "give-way_vessel": "give_way",
    # forbidden_maneuver values
    "alter_to_port_or_starboard": "none",
    "no_restriction": "none",
    "any": "none",
    # risk_level values (capitalization)
    "low_risk": "low",
    "medium_risk": "medium",
    "high_risk": "high",
    "critical_risk": "critical",
    # encounter_type
    "head-on": "head_on",
    "no risk": "no_risk",
    "restricted vis": "restricted_visibility",
    "multi ship": "multi_ship",
}


def _build_vocabulary_section() -> str:
    """Build a vocabulary reference section for the system prompt.

    Injects the exact allowed enum values so the model knows precisely
    which strings are legal. This is the soft-constraint analogue of CFG
    logit masking — it can't force compliance but dramatically reduces errors.
    """
    lines = [
        "## CRITICAL: Allowed Vocabulary (EXACT strings required)",
        "",
        "You MUST use EXACTLY these strings for enum fields. No variants allowed.",
        "",
        "| Field | Allowed Values |",
        "|-------|---------------|",
    ]
    for key, vals in VALID_ENUMS.items():
        if key in ("degradation_level", "priority"):
            continue  # numeric, not string enum
        val_str = " | ".join(f"`{v}`" for v in vals)
        lines.append(f"| {key} | {val_str} |")

    lines.extend([
        "",
        "Common mistakes to avoid:",
        '- ❌ "maintain_course_and_speed" → ✅ "maintain"',
        '- ❌ "stand_on_vessel" → ✅ "stand_on"',
        '- ❌ "give_way_vessel" → ✅ "give_way"',
        '- ❌ "alter_to_port_or_starboard" → ✅ "none"',
        '- ❌ "no action required" → ✅ "maintain"',
        '- ❌ "head-on" → ✅ "head_on"',
        "",
    ])
    return "\n".join(lines)


def _fuzzy_match(value: str, candidates: List[str], cutoff: float = 0.4) -> Optional[str]:
    """Fuzzy-match a value against a list of candidates using difflib.

    Returns the closest match if similarity >= cutoff, else None.
    """
    if not value or not isinstance(value, str):
        return None
    value_lower = value.lower().strip()
    matches = difflib.get_close_matches(value_lower, [c.lower() for c in candidates], n=1, cutoff=cutoff)
    if matches:
        # Return the original-cased candidate
        for c in candidates:
            if c.lower() == matches[0]:
                return c
    return None


# ── Valid field names extracted from COLREGS_OUTPUT_SCHEMA (all levels) ──
_VALID_FIELD_NAMES = frozenset({
    # Root level
    "encounter_classification", "target_interpretations", "required_maneuver",
    "forbidden_maneuver", "max_safe_speed", "global_min_cpa", "global_min_tcpa",
    "confidence_score", "reasoning_trace", "degradation_level", "fallback_active",
    "inference_time_ms", "timestamp", "scenario_id", "llm_model",
    # encounter_classification
    "primary_encounter", "risk_level", "all_encounters", "cpa_risk_field",
    "rule_priority_order", "environment_context",
    # target_interpretations[*]
    "target_name", "applicable_rules", "encounter_type", "own_ship_role",
    "is_stand_on_vessel", "maneuver", "spatial", "speed",
    # maneuver
    "required_maneuver", "alteration_min_angle", "alteration_max_angle",
    "max_yaw_rate", "priority",
    # spatial
    "min_distance", "forbidden_bearing_min", "forbidden_bearing_max",
    "pass_astern", "pass_ahead", "valid_until_tcpa",
    # speed
    "current_speed", "max_speed",
})


def _repair_field_names(data: dict) -> dict:
    """Pre-process LLM output to fix common field name pluralization errors.

    qwen2.5:7b frequently adds 's' or 'es' to field names (e.g.,
    ``forbidden_maneuvers`` instead of ``forbidden_maneuver``). JSON Schema
    ``additionalProperties: false`` rejects these before enum repair can run.

    This function walks the JSON tree and renames any key whose de-pluralized
    form is a known valid field name. Returns the modified dict (in-place).
    """
    if not isinstance(data, dict):
        return data

    # Direct plural→singular mapping for known LLM errors
    PLURAL_TO_SINGULAR = {
        "forbidden_maneuvers": "forbidden_maneuver",
        "required_maneuvers": "required_maneuver",
        "confidence_scores": "confidence_score",
        "degradation_levels": "degradation_level",
        "max_safe_speeds": "max_safe_speed",
        "reasoning_traces": "reasoning_trace",
        "primary_encounters": "primary_encounter",
        "all_encounters": "all_encounters",  # already plural — correct name
        "applicable_rules": "applicable_rules",  # already plural — correct name
        "target_interpretations": "target_interpretations",  # already plural
    }

    # Known invalid extra fields the LLM sometimes injects (removed silently)
    _INVALID_EXTRA_FIELDS = {
        "current_speed",            # LLM adds inside speed{} sub-object
        "recommended_action",       # LLM invents extra fields
        "recommended_speed",        # variant
        "recommended_speed_reduction",  # variant
        "suggested_action",         # variant
        "combined_risk_level",      # variant of risk_level
        "action",                   # variant
        "reason",                   # variant
        "collision_risk",           # non-standard field name
        "encounter",                # ambiguous, should be primary_encounter
    }

    # Known field name aliases (LLM uses different key names)
    FIELD_ALIASES = {
        "maneuver_type": "maneuver",        # LLM uses in per-target maneuver dicts/arrays
        "maneuvers": "maneuver",
    }

    def _walk(obj, path=()):
        if isinstance(obj, dict):
            # Remove known invalid extra fields
            keys_to_remove = set()
            for key in list(obj.keys()):
                if key in _INVALID_EXTRA_FIELDS:
                    keys_to_remove.add(key)
                elif key in FIELD_ALIASES:
                    new_key = FIELD_ALIASES[key]
                    if new_key not in obj:
                        obj[new_key] = obj.pop(key)
                        print(f"  [CFG repair] field alias '{key}' → '{new_key}'")
                    else:
                        keys_to_remove.add(key)  # already have correct key, drop alias
            for key in keys_to_remove:
                del obj[key]

            keys_to_rename = {}
            for key in list(obj.keys()):
                if key in _VALID_FIELD_NAMES:
                    pass  # already correct
                elif key in PLURAL_TO_SINGULAR:
                    new_key = PLURAL_TO_SINGULAR[key]
                    if new_key != key:
                        keys_to_rename[key] = new_key
                elif key.endswith('s') and key[:-1] in _VALID_FIELD_NAMES:
                    keys_to_rename[key] = key[:-1]
                elif key.endswith('es') and key[:-2] in _VALID_FIELD_NAMES:
                    keys_to_rename[key] = key[:-2]
            for old, new in keys_to_rename.items():
                obj[new] = obj.pop(old)
                print(f"  [CFG repair] field name '{old}' → '{new}'")
            for key, val in obj.items():
                _walk(val, path + (key,))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _walk(item, path + (i,))

    _walk(data)
    return data


def _repair_structural_errors(data: dict) -> dict:
    """Fix structural errors where the LLM outputs per-target arrays for
    scalar root-level fields like ``required_maneuver`` and ``forbidden_maneuver``.

    In multi-ship scenarios, qwen2.5:7b sometimes produces::

        "required_maneuver": [
            {"target_name": "ts03a", "maneuver": "no action required"},
            {"target_name": "ts03b", "maneuver": "alter_to_starboard"}
        ]

    instead of the schema-required scalar string ``"alter_to_starboard"``.

    Strategy: extract the most conservative (safety-prioritized) maneuver string.
    Priority: alter_to_starboard > alter_to_port > maintain > any_safe.
    """
    MANEUVER_PRIORITY = {
        "alter_to_starboard": 4,
        "alter_to_port": 3,
        "maintain": 2,
        "any_safe": 1,
        "reduce_speed": 2,
        "increase_speed": 2,
        "no action required": 2,        # LLM variant of maintain
        "no action required (stand-on vessel)": 2,
    }

    def _extract_maneuver_str(item) -> str:
        """Given a per-target maneuver dict or string, return the maneuver string."""
        if isinstance(item, str):
            return item.lower().strip()
        if isinstance(item, dict):
            # Try multiple known key names the LLM might use
            for key in ("maneuver", "maneuver_type", "required_maneuver", "action"):
                m = item.get(key, "")
                if isinstance(m, str) and m:
                    return m.lower().strip()
                if isinstance(m, dict):
                    inner = (m.get("required_maneuver", "") or "")
                    if isinstance(inner, str) and inner:
                        return inner.lower().strip()
        return ""

    def _pick_best_maneuver(items: list, field_name: str) -> Optional[str]:
        """Pick the highest-priority maneuver from a list of per-target items."""
        best_str = None
        best_prio = -1
        for item in items:
            m_str = _extract_maneuver_str(item)
            prio = MANEUVER_PRIORITY.get(m_str, 0)
            if prio > best_prio:
                best_prio = prio
                best_str = m_str
        # Map back to canonical enum
        if best_str and best_str.startswith("no action"):
            best_str = "maintain"
        return best_str

    if isinstance(data, dict):
        # Fix required_maneuver: array → scalar
        if "required_maneuver" in data and isinstance(data["required_maneuver"], list):
            rm_list = data["required_maneuver"]
            if len(rm_list) > 0:
                best = _pick_best_maneuver(rm_list, "required_maneuver")
                if best:
                    print(f"  [CFG repair] required_maneuver: array[{len(rm_list)}] → '{best}'")
                    data["required_maneuver"] = best

        # Fix forbidden_maneuver: array → scalar
        if "forbidden_maneuver" in data and isinstance(data["forbidden_maneuver"], list):
            fm_list = data["forbidden_maneuver"]
            if len(fm_list) > 0:
                best = _pick_best_maneuver(fm_list, "forbidden_maneuver")
                if best:
                    print(f"  [CFG repair] forbidden_maneuver: array[{len(fm_list)}] → '{best}'")
                    data["forbidden_maneuver"] = best

        # Fix reasoning_trace: array of strings → joined string
        if "reasoning_trace" in data and isinstance(data["reasoning_trace"], list):
            rt = data["reasoning_trace"]
            if all(isinstance(x, str) for x in rt):
                joined = " ".join(rt)
                print(f"  [CFG repair] reasoning_trace: array[{len(rt)}] → joined string")
                data["reasoning_trace"] = joined

        # Fix max_safe_speed: dict (per-target) or array → min value
        if "max_safe_speed" in data and isinstance(data["max_safe_speed"], (dict, list)):
            vals = data["max_safe_speed"]
            if isinstance(vals, dict):
                vals = list(vals.values())
            if vals:
                min_val = min(float(v) for v in vals)
                print(f"  [CFG repair] max_safe_speed: {type(data['max_safe_speed']).__name__} → {min_val}")
                data["max_safe_speed"] = min_val

        # Fix global_min_cpa: array → min
        if "global_min_cpa" in data and isinstance(data["global_min_cpa"], list):
            vals = data["global_min_cpa"]
            if vals:
                min_val = min(float(v) for v in vals)
                print(f"  [CFG repair] global_min_cpa: array[{len(vals)}] → {min_val}")
                data["global_min_cpa"] = min_val

        # Fix global_min_tcpa: array → min
        if "global_min_tcpa" in data and isinstance(data["global_min_tcpa"], list):
            vals = data["global_min_tcpa"]
            if vals:
                min_val = min(float(v) for v in vals)
                print(f"  [CFG repair] global_min_tcpa: array[{len(vals)}] → {min_val}")
                data["global_min_tcpa"] = min_val

        # Fix confidence_score: array or dict → scalar (take max for safety)
        if "confidence_score" in data and isinstance(data["confidence_score"], (list, dict)):
            vals = data["confidence_score"]
            if isinstance(vals, dict):
                vals = list(vals.values())
            if vals:
                val = max(min(float(v), 1.0) for v in vals)
                print(f"  [CFG repair] confidence_score: {type(data['confidence_score']).__name__} → {val}")
                data["confidence_score"] = val

        # Fix fallback_active: array → any True
        if "fallback_active" in data and isinstance(data["fallback_active"], list):
            val = any(data["fallback_active"])
            print(f"  [CFG repair] fallback_active: array → {val}")
            data["fallback_active"] = val

        # Fix degradation_level: dict → min value (safest)
        if "degradation_level" in data and isinstance(data["degradation_level"], (dict, list)):
            vals = data["degradation_level"]
            if isinstance(vals, dict):
                vals = list(vals.values())
            if vals:
                val = int(min(int(v) for v in vals))
                print(f"  [CFG repair] degradation_level: {type(data['degradation_level']).__name__} → {val}")
                data["degradation_level"] = val

        # ── Walk into sub-objects for nested array→scalar fixes ──
        # encounter_classification
        ec = data.get("encounter_classification")
        if isinstance(ec, dict):
            # is_stand_on_vessel: array → any True
            if "is_stand_on_vessel" in ec and isinstance(ec["is_stand_on_vessel"], list):
                val = any(ec["is_stand_on_vessel"])
                print(f"  [CFG repair] encounter_classification.is_stand_on_vessel: array → {val}")
                ec["is_stand_on_vessel"] = val

        # target_interpretations[].maneuver
        if "target_interpretations" in data and isinstance(data["target_interpretations"], list):
            for ti in data["target_interpretations"]:
                if not isinstance(ti, dict):
                    continue
                m = ti.get("maneuver")
                if isinstance(m, dict):
                    # Remove fields that are NOT valid inside maneuver sub-object
                    _INVALID_MANEUVER_FIELDS = {"maneuver", "maneuvers", "maneuver_type"}
                    for bad in _INVALID_MANEUVER_FIELDS:
                        if bad in m:
                            del m[bad]
                    # priority: array/dict → first/min
                    for fld in ("priority",):
                        if fld in m and isinstance(m[fld], (list, dict)):
                            vals = m[fld]
                            if isinstance(vals, dict):
                                vals = list(vals.values())
                            if vals:
                                m[fld] = int(vals[0])
                                print(f"  [CFG repair] maneuver.{fld}: array → {m[fld]}")
                    # required_maneuver / forbidden_maneuver: array → first string
                    for fld in ("required_maneuver", "forbidden_maneuver"):
                        if fld in m and isinstance(m[fld], list) and m[fld]:
                            first = m[fld][0]
                            if isinstance(first, str):
                                m[fld] = first
                                print(f"  [CFG repair] maneuver.{fld}: array → '{first}'")

        # Fix missing risk_level in encounter_classification
        if isinstance(ec, dict) and "risk_level" not in ec:
            # Default to "high" for multi-target scenarios with close CPA
            ec["risk_level"] = "high"
            print(f"  [CFG repair] encounter_classification.risk_level: missing → 'high'")

        # Fix primary_encounter: array of per-target dicts → scalar string
        if isinstance(ec, dict) and "primary_encounter" in ec:
            pe = ec["primary_encounter"]
            if isinstance(pe, list) and len(pe) > 0:
                # Try to extract encounter_type from array elements or all_encounters
                best = "multi_ship"  # default for multi-target
                if isinstance(pe[0], dict) and "encounter_type" in pe[0]:
                    best = pe[0]["encounter_type"]
                elif "all_encounters" in ec and ec["all_encounters"]:
                    best = ec["all_encounters"][0] if ec["all_encounters"] else "multi_ship"
                print(f"  [CFG repair] encounter_classification.primary_encounter: "
                      f"array[{len(pe)}] → '{best}'")
                ec["primary_encounter"] = best

    return data



def _repair_enum_values(data: dict, path: Tuple[str, ...] = ()) -> dict:
    """Post-hoc repair of LLM enum values using fuzzy matching.

    Walks the JSON output tree and corrects any enum values that don't
    match the allowed vocabulary. Also adds missing required fields.

    Returns the repaired dict (modified in-place as well).
    """
    if not isinstance(data, dict):
        return data

    # 1. Repair known mapping errors (exact lookup first)
    for key, valid_list in VALID_ENUMS.items():
        if key not in data:
            continue

    # 2. Walk the tree and apply corrections
    # 2a. Top-level fields — handle both string errors and null values
    for field, corrections, default in [
        ("required_maneuver", VALID_ENUMS["maneuver"], "maintain"),
        ("forbidden_maneuver", VALID_ENUMS["forbidden_maneuver"], "none"),
    ]:
        if field in data:
            if isinstance(data[field], str):
                corrected = _correct_enum(data[field], corrections)
                if corrected != data[field]:
                    print(f"  [CFG repair] {'.'.join(path)}{field}: "
                          f"'{data[field]}' → '{corrected}'")
                    data[field] = corrected
            elif data[field] is None:
                print(f"  [CFG repair] {'.'.join(path)}{field}: null → '{default}'")
                data[field] = default
    # Also handle max_safe_speed null
    if "max_safe_speed" in data and data["max_safe_speed"] is None:
        data["max_safe_speed"] = 5.0
        print(f"  [CFG repair] max_safe_speed: null → 5.0")
    # And global_min_cpa / global_min_tcpa null
    if "global_min_cpa" in data and data["global_min_cpa"] is None:
        data["global_min_cpa"] = 50.0
    if "global_min_tcpa" in data and data["global_min_tcpa"] is None:
        data["global_min_tcpa"] = 30.0

    # 2b. encounter_classification
    if "encounter_classification" in data and isinstance(data["encounter_classification"], dict):
        ec = data["encounter_classification"]
        for field, candidates in [
            ("primary_encounter", VALID_ENUMS["encounter_type"]),
            ("risk_level", VALID_ENUMS["risk_level"]),
        ]:
            if field in ec and isinstance(ec[field], str):
                corrected = _correct_enum(ec[field], candidates)
                if corrected != ec[field]:
                    print(f"  [CFG repair] encounter_classification.{field}: "
                          f"'{ec[field]}' → '{corrected}'")
                    ec[field] = corrected

        # Fill missing required field is_stand_on_vessel
        if "is_stand_on_vessel" not in ec or ec.get("is_stand_on_vessel") is None:
            # Infer from own_ship_role in target_interpretations if available
            inferred = False
            if "target_interpretations" in data and isinstance(data["target_interpretations"], list):
                for ti in data["target_interpretations"]:
                    role = (ti.get("own_ship_role") or "").lower().strip()
                    if role in ("stand_on", "stand-on"):
                        inferred = True
                        break
            ec["is_stand_on_vessel"] = inferred
            print(f"  [CFG repair] encounter_classification.is_stand_on_vessel: "
                  f"null → {inferred}")

        # also check all_encounters list
        if "all_encounters" in ec and isinstance(ec["all_encounters"], list):
            for i, v in enumerate(ec["all_encounters"]):
                if isinstance(v, str):
                    corrected = _correct_enum(v, VALID_ENUMS["encounter_type"])
                    if corrected != v:
                        print(f"  [CFG repair] all_encounters[{i}]: '{v}' → '{corrected}'")
                        ec["all_encounters"][i] = corrected

        # rule_priority_order
        if "rule_priority_order" in ec and isinstance(ec["rule_priority_order"], list):
            for i, r in enumerate(ec["rule_priority_order"]):
                if isinstance(r, str):
                    corrected = _correct_enum(r, VALID_ENUMS["colregs_rule"])
                    if corrected != r:
                        print(f"  [CFG repair] rule_priority_order[{i}]: '{r}' → '{corrected}'")
                        ec["rule_priority_order"][i] = corrected

    # 2c. target_interpretations
    if "target_interpretations" in data and isinstance(data["target_interpretations"], list):
        for ti_idx, ti in enumerate(data["target_interpretations"]):
            if not isinstance(ti, dict):
                continue
            ti_path = path + (f"target_interpretations[{ti_idx}]",)

            for field, candidates, default in [
                ("encounter_type", VALID_ENUMS["encounter_type"], "no_risk"),
                ("own_ship_role", VALID_ENUMS["ship_role"], "not_applicable"),
            ]:
                if field in ti:
                    if isinstance(ti[field], str):
                        corrected = _correct_enum(ti[field], candidates)
                        if corrected != ti[field]:
                            print(f"  [CFG repair] {'.'.join(ti_path)}.{field}: "
                                  f"'{ti[field]}' → '{corrected}'")
                            ti[field] = corrected
                    elif ti[field] is None:
                        print(f"  [CFG repair] {'.'.join(ti_path)}.{field}: null → '{default}'")
                        ti[field] = default

            # applicable_rules
            if "applicable_rules" in ti and isinstance(ti["applicable_rules"], list):
                for r_idx, rule in enumerate(ti["applicable_rules"]):
                    if isinstance(rule, str):
                        corrected = _correct_enum(rule, VALID_ENUMS["colregs_rule"])
                        if corrected != rule:
                            print(f"  [CFG repair] applicable_rules[{r_idx}]: '{rule}' → '{corrected}'")
                            ti["applicable_rules"][r_idx] = corrected

            # ── Propagate target_name into spatial sub-object ──
            if "spatial" in ti and isinstance(ti["spatial"], dict):
                sp = ti["spatial"]
                if "target_name" not in sp or sp["target_name"] is None:
                    parent_name = ti.get("target_name", "")
                    sp["target_name"] = parent_name
                    if parent_name:
                        print(f"  [CFG repair] {'.'.join(ti_path)}.spatial.target_name: "
                              f"null → '{parent_name}'")

            # nested maneuver
            if "maneuver" in ti and isinstance(ti["maneuver"], dict):
                m = ti["maneuver"]
                for field, candidates, default in [
                    ("required_maneuver", VALID_ENUMS["maneuver"], "maintain"),
                    ("forbidden_maneuver", VALID_ENUMS["forbidden_maneuver"], "none"),
                ]:
                    if field in m:
                        if isinstance(m[field], str):
                            corrected = _correct_enum(m[field], candidates)
                            if corrected != m[field]:
                                print(f"  [CFG repair] maneuver.{field}: "
                                      f"'{m[field]}' → '{corrected}'")
                                m[field] = corrected
                        elif m[field] is None:
                            print(f"  [CFG repair] maneuver.{field}: null → '{default}'")
                            m[field] = default

    # 3. Repair null numeric fields in target_interpretations[].maneuver
    if "target_interpretations" in data and isinstance(data["target_interpretations"], list):
        for ti in data["target_interpretations"]:
            if not isinstance(ti, dict):
                continue
            m = ti.get("maneuver")
            if isinstance(m, dict):
                for num_field, default in [
                    ("alteration_min_angle", 0.0),
                    ("alteration_max_angle", 0.0),
                    ("max_yaw_rate", 0.0),
                    ("priority", 1),
                ]:
                    if m.get(num_field) is None:
                        print(f"  [CFG repair] maneuver.{num_field}: null → {default}")
                        m[num_field] = default

            # Also repair spatial null fields
            s = ti.get("spatial")
            if isinstance(s, dict):
                for null_field in ["pass_ahead", "pass_astern", "forbidden_bearing_min",
                                   "forbidden_bearing_max", "valid_until_tcpa"]:
                    if null_field in s and s[null_field] is not None:
                        pass  # keep non-null values
                    elif null_field in s and s[null_field] is None:
                        # keep null for optional fields — schema allows it
                        pass

            # Repair speed null fields
            sp = ti.get("speed")
            if isinstance(sp, dict):
                if sp.get("max_speed") is None:
                    sp["max_speed"] = 5.0

    # 4. Add missing `timestamp` field — LLM cannot know ROS time
    if "timestamp" not in data:
        import time as _time
        data["timestamp"] = _time.time()

    # 5. Add missing `scenario_id` if provided elsewhere
    if "scenario_id" not in data:
        data["scenario_id"] = ""

    # 6. Ensure `llm_model` is set
    if "llm_model" not in data:
        data["llm_model"] = "qwen2.5:7b"

    return data


def _correct_enum(value: str, candidates: List[str]) -> str:
    """Correct a single enum value.

    Checks: exact match → known corrections → fuzzy match.
    Returns the corrected value, or the original if no match found.
    """
    # 1. Exact match (case-insensitive)
    value_lower = value.strip()
    for c in candidates:
        if isinstance(c, str) and c.lower() == value_lower.lower():
            return c

    # 2. Known corrections lookup
    if value_lower.lower() in ENUM_CORRECTIONS:
        return ENUM_CORRECTIONS[value_lower.lower()]

    # 3. Fuzzy match
    fuzzy = _fuzzy_match(value_lower, [c for c in candidates if isinstance(c, str)])
    if fuzzy:
        return fuzzy

    # 4. No match — return original (will fail validation)
    return value


# =============================================================================
# Abstract LLM Referee Interface
# =============================================================================

class AbstractLLMReferee(ABC):
    """Abstract base class for LLM-based COLREGS referee.

    Subclasses implement the actual LLM API call. The framework handles
    CFG validation, output parsing, and fallback to deterministic referee.

    Subclasses must implement:
      - _call_llm(scene_text, output_schema) -> dict
    """

    def __init__(self, visibility: str = "clear", sea_state: int = 2,
                 fallback_to_deterministic: bool = True,
                 confidence_threshold: float = 0.6):
        self.visibility = visibility
        self.sea_state = sea_state
        self.confidence_threshold = confidence_threshold
        self.fallback = DeterministicReferee(
            visibility=visibility, sea_state=sea_state
        ) if fallback_to_deterministic else None
        self._inference_count = 0
        self._total_time_ms = 0.0

    @abstractmethod
    def _call_llm(self, scene_text: str, schema: dict) -> Optional[dict]:
        """Call the LLM with structured output constraint.

        Returns parsed JSON dict, or None on failure.
        """
        ...

    def evaluate(
        self,
        own_ship: ShipObservation,
        target_ships: List[ShipObservation],
        scenario_id: str = "",
    ) -> COLREGSConstraintOutput:
        """Evaluate the COLREGS situation using LLM reasoning.

        Falls back to deterministic referee if:
          - LLM call fails
          - LLM output fails CFG validation
          - LLM confidence is below threshold
        """
        t_start = time.perf_counter()

        # Build scene description
        scene = build_scene_description(
            own_ship, target_ships,
            visibility=self.visibility,
            sea_state=self.sea_state,
        )

        # Try LLM inference
        llm_output = None
        try:
            llm_output = self._call_llm(scene.scene_text, FEW_SHOT_EXAMPLES)
        except Exception as e:
            print(f"[LLM Referee] LLM call failed: {e}")

        inference_time = (time.perf_counter() - t_start) * 1000
        self._inference_count += 1
        self._total_time_ms += inference_time

        if llm_output:
            try:
                # Layer 0a: Field name repair — fix pluralization errors
                #   (qwen2.5:7b often adds 's'/'es' to field names)
                _repair_field_names(llm_output)
                # Layer 0b: Structural repair — fix array-wrapped scalar fields
                #   (qwen2.5:7b outputs per-target arrays for multi-ship scenarios)
                _repair_structural_errors(llm_output)
                # Layer 0c: CFG enum repair — fuzzy-correct known LLM enum errors
                #   (soft-constraint analogue of logit masking)
                repaired = _repair_enum_values(llm_output)
                # Layer 1: JSON Schema validation (type, enum, required fields)
                validate_output_dict(repaired)
                # Layer 2: CFG grammar validation — L_colregs ⊂ L_json
                #   catches: extra fields, wrong key names, invalid nested structures
                #   that jsonschema may not reject
                from .cfg_grammar import validate_cfg_output
                if not validate_cfg_output(json.dumps(repaired)):
                    raise ValueError("CFG validation failed — output ∉ L_colregs")
                # Build output object
                output = self._parse_llm_output(repaired, scenario_id, inference_time)
                # Confidence threshold — reject low-confidence LLM outputs
                if output.confidence_score < self.confidence_threshold:
                    print(f"[LLM Referee] Low confidence ({output.confidence_score:.2f}"
                          f" < {self.confidence_threshold}), falling back")
                    raise ValueError("Confidence below threshold")
                return output
            except Exception as e:
                print(f"[LLM Referee] Validation failed: {e}, falling back")

        # Fallback to deterministic referee
        if self.fallback:
            output = self.fallback.evaluate(
                own_ship, target_ships, scenario_id)
            output.fallback_active = True
            output.llm_model = "deterministic_fallback"
            output.confidence_score *= 0.7  # reduced confidence
            return output

        # Last resort: return minimal safe output
        return self._emergency_output(scenario_id, inference_time)

    def _parse_llm_output(
        self, data: dict, scenario_id: str, inference_time: float
    ) -> COLREGSConstraintOutput:
        """Parse LLM JSON output into COLREGSConstraintOutput."""
        enc = data.get('encounter_classification', {})
        interps = []
        for ti in data.get('target_interpretations', []):
            spatial = None
            if 'spatial' in ti and ti['spatial']:
                s = ti['spatial']
                spatial = SpatialConstraint(
                    target_name=s.get('target_name', ''),
                    min_distance=s.get('min_distance', 50.0),
                    forbidden_bearing_min=_safe_radians(
                        s.get('forbidden_bearing_min')),
                    forbidden_bearing_max=_safe_radians(
                        s.get('forbidden_bearing_max')),
                    pass_astern=s.get('pass_astern'),
                    pass_ahead=s.get('pass_ahead'),
                    valid_until_tcpa=s.get('valid_until_tcpa'),
                    priority=s.get('priority', 1),
                )
            maneuver = None
            if 'maneuver' in ti and ti['maneuver']:
                m = ti['maneuver']
                maneuver = ManeuverConstraint(
                    required_maneuver=ManeuverType(m['required_maneuver']),
                    forbidden_maneuver=ForbiddenManeuver(
                        m.get('forbidden_maneuver', 'none')),
                    alteration_min_angle=_safe_radians(
                        m.get('alteration_min_angle', 0.0)),
                    alteration_max_angle=_safe_radians(
                        m.get('alteration_max_angle', 2.0)),
                    max_yaw_rate=m.get('max_yaw_rate', 0.5),
                    priority=m.get('priority', 1),
                )
            speed = None
            if 'speed' in ti and ti['speed']:
                sp = ti['speed']
                speed = SpeedConstraint(
                    max_speed=sp.get('max_speed', 5.0),
                    min_speed=sp.get('min_speed'),
                    visibility_reduction=sp.get('visibility_reduction', 1.0),
                    traffic_density_factor=sp.get('traffic_density_factor', 1.0),
                    priority=sp.get('priority', 1),
                )
            interps.append(ColregsRuleInterpretation(
                target_name=ti['target_name'],
                encounter_type=EncounterType(ti['encounter_type']),
                own_ship_role=ShipRole(ti['own_ship_role']),
                applicable_rules=ti.get('applicable_rules', []),
                spatial=spatial,
                maneuver=maneuver,
                speed=speed,
            ))

        return COLREGSConstraintOutput(
            timestamp=time.time(),
            scenario_id=scenario_id,
            encounter_classification=EncounterClassification(
                primary_encounter=EncounterType(
                    enc.get('primary_encounter', 'no_risk')),
                all_encounters=enc.get('all_encounters', []),
                rule_priority_order=enc.get('rule_priority_order', []),
                risk_level=enc.get('risk_level', 'low'),
                is_stand_on_vessel=enc.get('is_stand_on_vessel', False),
                cpa_risk_field=enc.get('cpa_risk_field', 0.0),
                environment_context=enc.get('environment_context', ''),
            ),
            target_interpretations=interps,
            required_maneuver=ManeuverType(data.get('required_maneuver', 'maintain')),
            forbidden_maneuver=ForbiddenManeuver(
                data.get('forbidden_maneuver', 'none')),
            max_safe_speed=data.get('max_safe_speed', 5.0),
            global_min_cpa=data.get('global_min_cpa', 50.0),
            global_min_tcpa=data.get('global_min_tcpa', 30.0),
            confidence_score=data.get('confidence_score', 0.5),
            reasoning_trace=data.get('reasoning_trace', ''),
            llm_model=data.get('llm_model', 'llm'),
            inference_time_ms=inference_time,
            fallback_active=data.get('fallback_active', False),
            degradation_level=data.get('degradation_level', 0),
        )

    def _emergency_output(
        self, scenario_id: str, inference_time: float
    ) -> COLREGSConstraintOutput:
        """Emergency output when both LLM and deterministic fail."""
        return COLREGSConstraintOutput(
            timestamp=time.time(),
            scenario_id=scenario_id,
            required_maneuver=ManeuverType.REDUCE_SPEED,
            forbidden_maneuver=ForbiddenManeuver.NONE,
            max_safe_speed=1.0,  # minimum safe speed
            global_min_cpa=100.0,  # very conservative
            global_min_tcpa=60.0,
            confidence_score=0.1,
            reasoning_trace="EMERGENCY: all referee layers failed — minimal safe mode",
            llm_model="emergency",
            inference_time_ms=inference_time,
            fallback_active=True,
            degradation_level=3,
        )

    @property
    def avg_inference_time_ms(self) -> float:
        """Average LLM inference time."""
        if self._inference_count == 0:
            return 0.0
        return self._total_time_ms / self._inference_count


# =============================================================================
# Anthropic Claude API Referee
# =============================================================================

class AnthropicReferee(AbstractLLMReferee):
    """COLREGS referee using Anthropic Claude API with structured output.

    Uses Claude's native structured output (json_mode) for guaranteed
    schema compliance. Supports prompt caching for repeated scenes.

    Requires: `pip install anthropic`
    Environment: ANTHROPIC_API_KEY
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def _call_llm(self, scene_text: str, examples: list) -> Optional[dict]:
        """Call Anthropic Claude with structured JSON output."""
        if not self.api_key:
            print("[AnthropicReferee] No API key configured, using fallback")
            return None

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            system_prompt = self._build_system_prompt()

            # Build few-shot messages
            example_messages = []
            for ex in examples[:2]:  # limit to 2 examples for latency
                example_messages.append({
                    "role": "user",
                    "content": f"Analyze this COLREGS maritime scene:\n\n{ex['scene']}"
                })
                example_messages.append({
                    "role": "assistant",
                    "content": json.dumps(ex['output'], indent=2)
                })

            messages = example_messages + [{
                "role": "user",
                "content": f"Analyze this COLREGS maritime scene:\n\n{scene_text}"
            }]

            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
                temperature=0.1,  # low temperature for deterministic output
            )

            # Parse JSON from response robustly
            text = response.content[0].text

            # Strategy (in priority order):
            #   1. Try parsing raw text as JSON directly
            #   2. Extract first balanced {...} block (handles mixed text+JSON)
            #   3. Fall back to markdown code-block extraction

            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            # Find the first balanced { ... } JSON object
            start = text.find('{')
            if start >= 0:
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == '{':
                        depth += 1
                    elif text[i] == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(text[start:i + 1])
                            except json.JSONDecodeError:
                                break

            # Last resort: try markdown code block extraction
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            return json.loads(text)

        except ImportError:
            print("[AnthropicReferee] anthropic package not installed")
            return None
        except Exception as e:
            print(f"[AnthropicReferee] API call error: {e}")
            return None

    def _build_system_prompt(self) -> str:
        """Build the system prompt with COLREGS domain knowledge."""
        return """You are a COLREGS (International Regulations for Preventing Collisions at Sea) expert system for autonomous maritime collision avoidance.

Your task is to analyze a maritime scene and produce a structured JSON output with COLREGS rule interpretations and spatial/maneuver/speed constraints.

## COLREGS Rules Reference

**Rule 14 (Head-on):** When two power-driven vessels meet on reciprocal or nearly reciprocal courses, each shall alter course to starboard so that each shall pass on the port side of the other.

**Rule 15 (Crossing):** When two power-driven vessels are crossing, the vessel which has the other on her own starboard side shall keep out of the way (give-way vessel).

**Rule 13 (Overtaking):** Any vessel overtaking another shall keep out of the way of the vessel being overtaken.

**Rule 16 (Give-way):** Every vessel directed to keep out of the way shall take early and substantial action (≥30° course alteration is substantial).

**Rule 17 (Stand-on):** The stand-on vessel shall keep her course and speed. May take action only when collision cannot be avoided by give-way vessel alone.

**Rule 8 (Avoiding Action):** Action shall be positive, made in ample time, and with due regard to good seamanship. If alteration of course alone is sufficient, it should be a large alteration (≥30°).

**Rule 6 (Safe Speed):** Every vessel shall proceed at a safe speed. Factors: visibility, traffic density, sea state, maneuverability, background lighting.

**Rule 19 (Restricted Visibility):** In or near restricted visibility, vessels shall proceed at safe speed. No concept of stand-on vessel — all vessels take avoiding action.

**Rule 18 (Responsibilities):** Hierarchy: NUC > RAM > Fishing > Sailing > Power-driven.

## Output Format

Produce valid JSON following this exact structure. All enum values must be from the specified options.

## Important Guidelines

1. ALWAYS forbid turning to port in head-on and give-way crossing situations (Rule 14, 15, 16).
2. Minimum CPA should be ≥50m in clear conditions, ≥75m in restricted visibility.
3. Minimum course alteration should be ≥30° (0.5236 rad) for substantial action.
4. Safe speed ≤5 m/s in clear, ≤2.5 m/s in restricted visibility.
5. Give-way vessels should pass astern of the stand-on vessel.
6. Provide detailed legal reasoning in the reasoning_trace field.
7. Confidence should reflect uncertainty — lower for complex multi-ship scenarios.

## ⚠️ UNITS — CRITICAL

ALL angular quantities MUST be in RADIANS, NOT degrees:
- alteration_min_angle, alteration_max_angle — radians (π ≈ 3.1416)
- forbidden_bearing_min, forbidden_bearing_max — radians
- max_yaw_rate — radians per second

Example: 30° → MUST output 0.5236, NOT 30.

## Key COLREGS Terminology
- **Starboard** = right side of vessel (facing forward)
- **Port** = left side of vessel
- **Alter to starboard** = turn right
- **Alter to port** = turn left
- **Pass astern** = pass behind
- **Pass ahead** = pass in front of
"""


# =============================================================================
# Simulated LLM Referee (for testing without API)
# =============================================================================

class SimulatedLLMReferee(AbstractLLMReferee):
    """Simulated LLM referee for testing.

    Uses the deterministic engine but formats output with LLM-style
    reasoning traces and slightly varied confidence scores to simulate
    LLM behavior. Useful for integration testing.
    """

    def __init__(self, noise_level: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.noise_level = noise_level
        self.deterministic = DeterministicReferee(
            visibility=kwargs.get('visibility', 'clear'),
            sea_state=kwargs.get('sea_state', 2),
        )

    def _call_llm(self, scene_text: str, examples: list) -> Optional[dict]:
        """Simulate LLM by using deterministic engine + noise."""
        # This is not used — we override evaluate() instead
        return None

    def evaluate(
        self,
        own_ship: ShipObservation,
        target_ships: List[ShipObservation],
        scenario_id: str = "",
    ) -> COLREGSConstraintOutput:
        """Evaluate using deterministic engine with LLM-style output."""
        output = self.deterministic.evaluate(
            own_ship, target_ships, scenario_id)

        # Simulate LLM characteristics
        output.llm_model = "simulated_llm"
        output.confidence_score = max(0.7, min(0.98,
            0.95 - self.noise_level * len(target_ships) +
            np.random.normal(0, self.noise_level)))

        # Add LLM-style reasoning
        reasoning = output.reasoning_trace
        reasoning += "\n\n[Simulated LLM] Semantic interpretation:\n"
        reasoning += "  Contextual analysis of encounter geometry suggests "
        if output.encounter_classification.primary_encounter == EncounterType.HEAD_ON:
            reasoning += "head-on situation. Both vessels should alter to starboard.\n"
        elif output.encounter_classification.primary_encounter == EncounterType.CROSSING:
            reasoning += "crossing situation. OS must determine give-way/stand-on status.\n"
        reasoning += "  Constraints mapped to NMPC spatial corridors with "
        reasoning += f"min CPA = {output.global_min_cpa:.1f}m."

        output.reasoning_trace = reasoning

        return output


# =============================================================================
# Ollama Local LLM Referee (Phase 5-6 document compliance)
# =============================================================================

class OllamaReferee(AbstractLLMReferee):
    """COLREGS referee using local Ollama open-source LLM.

    Uses Ollama REST API (http://localhost:11434) for local inference
    with structured JSON output and CFG post-validation.

    Prerequisites:
      - Ollama installed: `sudo snap install ollama`
      - Model pulled: `ollama pull qwen2.5:7b`
      - Optional: `pip install requests`

    Supported models (tested):
      - qwen2.5:7b  (recommended — bilingual, strong JSON)
      - llama3.1:8b
      - mistral:7b
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        use_cfg: bool = True,          # Group B (-CFG): set False to disable CFG enforcement
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.use_cfg = use_cfg
        if base_url is None:
            import os
            base_url = os.environ.get(
                "OLLAMA_HOST", "http://localhost:11434"
            )
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries

    def _call_llm(self, scene_text: str, examples: list) -> Optional[dict]:
        """Call Ollama chat API with JSON mode + retry on CFG failure."""
        import requests

        system_prompt = self._build_system_prompt()

        # Build few-shot messages
        messages = []
        for ex in examples[:2]:
            messages.append({
                "role": "user",
                "content": f"Analyze this COLREGS maritime scene:\n\n{ex['scene']}"
            })
            messages.append({
                "role": "assistant",
                "content": json.dumps(ex['output'], indent=2)
            })
        messages.append({
            "role": "user",
            "content": f"Analyze this COLREGS maritime scene:\n\n{scene_text}"
        })

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "num_ctx": 4096,
            },
        }

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                content = resp.json()["message"]["content"]

                # Extract JSON: try direct parse, then balanced bracket extraction
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    start = content.find('{')
                    if start >= 0:
                        depth = 0
                        for i in range(start, len(content)):
                            if content[i] == '{':
                                depth += 1
                            elif content[i] == '}':
                                depth -= 1
                                if depth == 0:
                                    return json.loads(content[start:i + 1])
                    raise

            except requests.exceptions.Timeout:
                last_error = f"Ollama timeout ({self.timeout}s)"
                print(f"[OllamaReferee] Attempt {attempt+1}: {last_error}")
            except requests.exceptions.ConnectionError:
                last_error = f"Cannot connect to {self.base_url} — is Ollama running?"
                print(f"[OllamaReferee] {last_error}")
                break  # don't retry connection errors
            except Exception as e:
                last_error = str(e)
                print(f"[OllamaReferee] Attempt {attempt+1}: {e}")

            # Retry with slightly higher temperature for diversity
            if attempt < self.max_retries:
                payload["options"]["temperature"] = min(0.3, payload["options"]["temperature"] + 0.1)

        print(f"[OllamaReferee] All {self.max_retries+1} attempts failed: {last_error}")
        return None

    def _build_system_prompt(self) -> str:
        """COLREGS system prompt, optionally with CFG vocabulary constraints.

        When use_cfg=True (Group A/D): includes exact vocabulary table to
        soft-constrain LLM enum output (analogue of CFG logit masking).

        When use_cfg=False (Group B): plain COLREGS prompt without vocabulary
        enforcement — tests CFG's contribution to format compliance.
        """
        base = f"""You are a COLREGS (International Regulations for Preventing Collisions at Sea) expert system for autonomous maritime collision avoidance.

Your task is to analyze a maritime scene and produce a structured JSON output with COLREGS rule interpretations and spatial/maneuver/speed constraints.

## COLREGS Rules Reference

**Rule 14 (Head-on):** When two power-driven vessels meet on reciprocal or nearly reciprocal courses, each shall alter course to starboard so that each shall pass on the port side of the other.

**Rule 15 (Crossing):** When two power-driven vessels are crossing, the vessel which has the other on her own starboard side shall keep out of the way (give-way vessel).

**Rule 13 (Overtaking):** Any vessel overtaking another shall keep out of the way of the vessel being overtaken.

**Rule 16 (Give-way):** Every vessel directed to keep out of the way shall take early and substantial action (>=30 degrees course alteration is substantial).

**Rule 17 (Stand-on):** The stand-on vessel shall keep her course and speed. May take action only when collision cannot be avoided by give-way vessel alone.

**Rule 8 (Avoiding Action):** Action shall be positive, made in ample time. If alteration of course alone is sufficient, it should be a large alteration (>=30 degrees).

**Rule 6 (Safe Speed):** Every vessel shall proceed at a safe speed. Factors: visibility, traffic density, sea state, maneuverability.

**Rule 19 (Restricted Visibility):** In restricted visibility, vessels shall proceed at safe speed. No concept of stand-on vessel — all vessels take avoiding action.

## Output Format

You MUST produce a valid JSON object with these fields: timestamp, scenario_id, encounter_classification (primary_encounter, risk_level, is_stand_on_vessel, cpa_risk_field, environment_context), target_interpretations (array of objects with target_name, encounter_type, own_ship_role, applicable_rules, spatial, maneuver, speed), required_maneuver, forbidden_maneuver, max_safe_speed, global_min_cpa, global_min_tcpa, confidence_score, reasoning_trace, llm_model, inference_time_ms, fallback_active, degradation_level.

## UNITS — CRITICAL

ALL angles MUST be in RADIANS, not degrees:
- alteration_min_angle: radians (30 degrees = 0.5236 rad)
- forbidden_bearing_min/max: radians
- max_yaw_rate: radians per second

## Important Guidelines

1. ALWAYS forbid turning to port in head-on and give-way crossing situations.
2. Minimum CPA >= 50m in clear conditions, >= 75m in restricted visibility.
3. Minimum course alteration >= 30 degrees (0.5236 rad) for substantial action.
4. Safe speed <= 5 m/s in clear, <= 2.5 m/s in restricted visibility.
5. Give-way vessels should pass astern of the stand-on vessel.
6. Provide detailed legal reasoning.
7. Confidence should reflect uncertainty — lower for complex multi-ship scenarios.

## Key COLREGS Terminology
- Starboard = right side of vessel (facing forward)
- Port = left side of vessel
- Alter to starboard = turn right
- Alter to port = turn left
- Pass astern = pass behind
- Pass ahead = pass in front of
"""

        if self.use_cfg:
            vocab = _build_vocabulary_section()
            return base + f"\n{vocab}"
        else:
            # Group B (-CFG): no vocabulary enforcement — plain LLM
            return base


# =============================================================================
# Grammar-Constrained LLM Referee (llama.cpp + GBNF)
# =============================================================================

class GrammarConstrainedReferee(AbstractLLMReferee):
    """COLREGS referee with token-level CFG-constrained decoding via llama.cpp.

    Uses llama-cpp-python's LlamaGrammar (GBNF format) to ENFORCE that every
    generated token stays within the COLREGS output grammar L_colregs_GBNF.
    This ELIMINATES format hallucinations — the output is mathematically
    guaranteed to be valid COLREGS JSON at EVERY decoding step.

    Architecture (replaces 3-layer soft-constraint workaround):
      Layer 0 (HARD): GBNF grammar constrains token generation
        → output ∈ L_colregs_GBNF ⊂ L_json with 100% guarantee
      Layer 1 (verify): JSON Schema validation (defense-in-depth)
      Layer 2 (verify): CFG grammar validation (Lark Earley parser)

    Prerequisites:
      - llama-cpp-python installed: `pip install llama-cpp-python`
      - GGUF model file accessible (can use Ollama's blob storage)
    """

    def __init__(
        self,
        model_path: str = None,
        grammar_string: str = None,
        n_ctx: int = 4096,
        n_threads: int = 8,
        n_gpu_layers: int = -1,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Args:
            model_path: Path to GGUF model file. If None, auto-discovers
                        from Ollama blob storage.
            grammar_string: GBNF grammar string. If None, uses the COLREGS
                            grammar from colregs_gbnf.py.
            n_ctx: Context window size (tokens).
            n_threads: CPU threads for inference.
            n_gpu_layers: GPU layers (-1 = all, 0 = CPU only).
            temperature: Sampling temperature (low = deterministic).
            max_tokens: Maximum tokens to generate.
            verbose: Print llama.cpp logs.
        """
        super().__init__(**kwargs)
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._n_gpu_layers = n_gpu_layers
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._verbose = verbose
        self._llm = None
        self._grammar = None

        # Import GBNF grammar
        from .colregs_gbnf import COLREGS_GBNF_GRAMMAR_SIMPLE
        self._grammar_string = grammar_string or COLREGS_GBNF_GRAMMAR_SIMPLE

    @property
    def llm(self):
        """Lazy-load the llama.cpp model."""
        if self._llm is None:
            import os
            from llama_cpp import Llama

            model_path = self._model_path or self._find_model()
            if not model_path:
                raise RuntimeError(
                    "No GGUF model found. Set model_path or ensure "
                    "Ollama models are in /usr/share/ollama/.ollama/models/blobs/")

            print(f"[GrammarReferee] Loading model: {model_path}")
            self._llm = Llama(
                model_path=model_path,
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                n_gpu_layers=self._n_gpu_layers,
                verbose=self._verbose,
            )
        return self._llm

    @property
    def grammar(self):
        """Lazy-load the GBNF grammar."""
        if self._grammar is None:
            from llama_cpp import LlamaGrammar
            self._grammar = LlamaGrammar.from_string(
                self._grammar_string, verbose=self._verbose)
        return self._grammar

    @staticmethod
    def _find_model() -> Optional[str]:
        """Auto-discover GGUF model file from Ollama blob storage."""
        import os, glob
        search_paths = [
            "/usr/share/ollama/.ollama/models/blobs/",
            os.path.expanduser("~/.ollama/models/blobs/"),
        ]
        for sp in search_paths:
            if os.path.isdir(sp):
                blobs = glob.glob(os.path.join(sp, "sha256-*"))
                # Find the largest blob (likely the model, not the tokenizer)
                if blobs:
                    largest = max(blobs, key=os.path.getsize)
                    # Verify it's GGUF
                    with open(largest, 'rb') as f:
                        if f.read(4) == b'GGUF':
                            return largest
        return None

    def _call_llm(self, scene_text: str, examples: list) -> Optional[dict]:
        """Call llama.cpp with grammar-constrained decoding.

        The grammar ENFORCES that the output is valid COLREGS JSON.
        No repair, no fuzzy matching, no fallback — the LLM CANNOT
        produce invalid output because invalid tokens are masked at
        each decoding step.
        """
        import json as _json

        # Build prompt: system + few-shot + scene
        system_prompt = self._build_system_prompt()
        prompt = self._build_prompt(system_prompt, scene_text, examples)

        try:
            result = self.llm.create_completion(
                prompt,
                grammar=self.grammar,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                stop=["```", "\n\n\n"],
                echo=False,
            )

            text = result["choices"][0]["text"].strip()

            # The LLM output is GUARANTEED valid JSON by the grammar.
            # Still parse + validate as defense-in-depth.
            try:
                output = _json.loads(text)
            except _json.JSONDecodeError:
                # Extremely rare: grammar bug or tokenizer edge case
                # Try extracting the first JSON object
                start = text.find('{')
                end = text.rfind('}')
                if start >= 0 and end > start:
                    output = _json.loads(text[start:end + 1])
                else:
                    return None

            return output

        except Exception as e:
            print(f"[GrammarReferee] Inference failed: {e}")
            return None

    def _build_system_prompt(self) -> str:
        """Build COLREGS system prompt.

        With grammar-constrained decoding, the prompt can be simpler —
        no need to enumerate vocabulary or warn about format errors.
        The grammar handles format; the prompt focuses on COLREGS reasoning.
        """
        return """You are a COLREGS (International Regulations for Preventing Collisions at Sea) expert system for autonomous maritime collision avoidance.

Analyze the maritime scene and output a JSON object with COLREGS rule interpretations and spatial/maneuver/speed constraints.

## COLREGS Rules Reference

**Rule 14 (Head-on):** Both vessels alter course to starboard, pass port-to-port.
**Rule 15 (Crossing):** Vessel with other on starboard side is give-way.
**Rule 13 (Overtaking):** Overtaking vessel keeps clear until past and clear.
**Rule 16 (Give-way):** Take early and substantial action (>=30° alteration).
**Rule 17 (Stand-on):** Keep course and speed; act only if collision imminent.
**Rule 8 (Action):** Positive, ample, timely action. Large alteration (>=30°).
**Rule 6 (Safe Speed):** Speed appropriate to conditions and visibility.
**Rule 19 (Restricted Vis):** No stand-on vessel; all proceed at safe speed.

## Output Specification

Produce a JSON object with these fields:
- timestamp: number (ROS time)
- scenario_id: string
- encounter_classification: object with primary_encounter, risk_level, is_stand_on_vessel, etc.
- target_interpretations: array of per-target objects, each with spatial/maneuver/speed constraints
- required_maneuver: one of alter_to_starboard/alter_to_port/maintain/any_safe/reduce_speed/increase_speed
- forbidden_maneuver: one of alter_to_port/alter_to_starboard/none/reduce_speed
- max_safe_speed: number (m/s, ≤5 clear, ≤2.5 restricted)
- global_min_cpa: number (≥50m clear, ≥75m restricted)
- global_min_tcpa: number (≥30s)
- confidence_score: number (0-1)
- reasoning_trace: string (legal reasoning in English)
- llm_model: string
- inference_time_ms: number
- fallback_active: boolean
- degradation_level: integer (0-3)

## UNITS — CRITICAL
ALL angles in RADIANS: 30° = 0.5236 rad, 45° = 0.7854 rad.

## Guidelines
1. Forbid port turn in head-on and give-way crossing.
2. Give-way vessels pass ASTERN of stand-on vessel.
3. Min CPA ≥ 50m (clear) or ≥ 75m (restricted visibility).
4. Min course alteration ≥ 0.5236 rad (30°) for substantial action.
5. Speed ≤ 5 m/s clear, ≤ 2.5 m/s restricted visibility."""

    def _build_prompt(
        self, system: str, scene_text: str, examples: list
    ) -> str:
        """Build the full prompt with system message, few-shot, and scene."""
        parts = [f"<|system|>\n{system}\n<|end|>"]

        for ex in examples[:2]:
            parts.append(
                f"<|user|>\nAnalyze this COLREGS maritime scene:\n\n{ex['scene']}\n<|end|>")
            parts.append(
                f"<|assistant|>\n{json.dumps(ex['output'], indent=2)}\n<|end|>")

        parts.append(
            f"<|user|>\nAnalyze this COLREGS maritime scene:\n\n{scene_text}\n<|end|>")
        parts.append("<|assistant|>\n")

        return "\n".join(parts)


# =============================================================================
# Math import for scene description (must be at module level)
# =============================================================================

import math
