#!/usr/bin/env python3
"""
COLREGS Referee Output Schema
==============================

Defines the 12-field structured JSON output schema for the symbolic referee layer.
This schema is the contract between the LLM referee and the NMPC control layer.

The 12 constraint fields (Section 4.3):
  1. encounter_type          — COLREGS encounter classification
  2. applicable_rules        — List of applicable COLREGS rules
  3. own_ship_role           — give_way / stand_on / not_applicable
  4. required_maneuver       — alter_to_starboard / alter_to_port / maintain / any_safe
  5. min_cpa                 — Minimum CPA distance (Rule 8(d))
  6. min_tcpa                — Minimum TCPA threshold
  7. forbidden_maneuver      — alter_to_port / alter_to_starboard / none
  8. alteration_min_angle    — Minimum course alteration magnitude (Rule 8(b))
  9. max_speed               — Maximum safe speed (Rule 6)
  10. spatial_constraints    — Geometric half-space / corridor constraints for each TS
  11. confidence_score       — LLM confidence [0, 1]
  12. reasoning_trace        — Human-readable legal reasoning chain
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict
import json
import jsonschema


# =============================================================================
# Enum types
# =============================================================================

class EncounterType(str, Enum):
    HEAD_ON = "head_on"           # Rule 14
    CROSSING = "crossing"         # Rule 15
    OVERTAKING = "overtaking"     # Rule 13
    RESTRICTED_VIS = "restricted_visibility"  # Rule 19
    MULTI_SHIP = "multi_ship"     # Multiple rules
    NO_RISK = "no_risk"


class ShipRole(str, Enum):
    GIVE_WAY = "give_way"
    STAND_ON = "stand_on"
    NOT_APPLICABLE = "not_applicable"


class ManeuverType(str, Enum):
    ALTER_TO_STARBOARD = "alter_to_starboard"
    ALTER_TO_PORT = "alter_to_port"
    MAINTAIN_COURSE_SPEED = "maintain"
    ANY_SAFE = "any_safe"         # Emergency situations
    REDUCE_SPEED = "reduce_speed"
    INCREASE_SPEED = "increase_speed"


class ForbiddenManeuver(str, Enum):
    ALTER_TO_PORT = "alter_to_port"
    ALTER_TO_STARBOARD = "alter_to_starboard"
    NONE = "none"
    REDUCE_SPEED = "reduce_speed"


# =============================================================================
# Data classes — structured referee output
# =============================================================================

@dataclass
class SpatialConstraint:
    """Geometric constraint for a single target ship.

    Maps to NMPC spatial constraints (Section 4.4):
      - Rule 8(d): min_cpa exclusion zone
      - Rule 14/15: forbidden port/starboard corridor
    """
    target_name: str
    # Exclusion circle centered at target ship with radius min_cpa
    min_distance: float              # meters — CPA exclusion radius
    # Relative bearing corridor (forbidden angular sectors in ENU)
    forbidden_bearing_min: Optional[float] = None  # rad, lower bound
    forbidden_bearing_max: Optional[float] = None  # rad, upper bound
    # Passing side constraint
    pass_astern: Optional[bool] = None      # True = pass behind TS
    pass_ahead: Optional[bool] = None       # True = pass ahead of TS
    # Time window
    valid_until_tcpa: Optional[float] = None  # seconds
    # Priority weight (for slack variable hierarchy)
    priority: int = 1                 # 0=soft, 1=compliance, 2=safety(hard)


@dataclass
class ManeuverConstraint:
    """Maneuver constraint for own ship.

    Maps to NMPC input constraints:
      - Rule 8(b): minimum course alteration magnitude
      - Rule 14/15: forbidden turn direction
    """
    required_maneuver: ManeuverType
    forbidden_maneuver: ForbiddenManeuver = ForbiddenManeuver.NONE
    alteration_min_angle: float = 0.0     # rad, minimum course change (Rule 8(b))
    alteration_max_angle: float = 2.0     # rad, max course change (physical limit)
    # Turn rate limits
    max_yaw_rate: float = 0.5             # rad/s
    priority: int = 1


@dataclass
class SpeedConstraint:
    """Speed constraint (Rule 6 — safe speed).

    Safe speed is context-dependent: visibility, traffic density, sea state.
    """
    max_speed: float                     # m/s — upper bound (Rule 6)
    min_speed: Optional[float] = None    # m/s — lower bound (maintain steerage)
    # Context factors
    visibility_reduction: float = 1.0    # [0, 1] — 1.0 = clear, <1 = reduced
    traffic_density_factor: float = 1.0  # ≥1 — higher = more restrictive
    priority: int = 1


@dataclass
class ColregsRuleInterpretation:
    """Per-target-ship COLREGS rule interpretation."""
    target_name: str
    encounter_type: EncounterType
    own_ship_role: ShipRole
    applicable_rules: List[str] = field(default_factory=list)
    # Derived constraints
    spatial: Optional[SpatialConstraint] = None
    maneuver: Optional[ManeuverConstraint] = None
    speed: Optional[SpeedConstraint] = None


@dataclass
class EncounterClassification:
    """Global encounter classification across all target ships."""
    primary_encounter: EncounterType
    all_encounters: List[str] = field(default_factory=list)
    # Rule priority resolution (Rule 18 hierarchy if applicable)
    rule_priority_order: List[str] = field(default_factory=list)
    # Overall risk assessment
    risk_level: str = "low"          # low / medium / high / critical
    is_stand_on_vessel: bool = False
    # Trigger state for event-triggered activation
    cpa_risk_field: float = 0.0      # nonlinear CPA/TCPA risk scalar
    environment_context: str = ""     # visibility, traffic density context


@dataclass
class COLREGSConstraintOutput:
    """Complete structured output from the symbolic referee layer.

    This is the 12-field output that the NMPC layer consumes.
    Validated against the JSON schema defined below.
    """
    # Metadata
    timestamp: float = 0.0
    scenario_id: str = ""

    # Scene understanding (fields 1-3)
    encounter_classification: EncounterClassification = field(
        default_factory=lambda: EncounterClassification(
            primary_encounter=EncounterType.NO_RISK))

    # Per-target interpretations (fields 4-9 embodied)
    target_interpretations: List[ColregsRuleInterpretation] = field(
        default_factory=list)

    # Global maneuver directive (field 4)
    required_maneuver: ManeuverType = ManeuverType.MAINTAIN_COURSE_SPEED
    forbidden_maneuver: ForbiddenManeuver = ForbiddenManeuver.NONE

    # Global speed constraint (field 5)
    max_safe_speed: float = 5.0         # m/s (Rule 6)

    # Aggregate spatial constraints (field 6)
    global_min_cpa: float = 50.0        # m — strictest across all TS
    global_min_tcpa: float = 30.0       # s

    # LLM metadata (fields 10-12)
    confidence_score: float = 1.0       # [0, 1]
    reasoning_trace: str = ""            # human-readable legal reasoning
    llm_model: str = "deterministic"     # model identifier
    inference_time_ms: float = 0.0       # inference latency

    # Fallback/error state
    fallback_active: bool = False
    degradation_level: int = 0           # 0=normal, 1=soft, 2=hard, 3=emergency

    def to_dict(self) -> dict:
        """Serialize to dictionary (for JSON/YAML output)."""
        return _serialize(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# =============================================================================
# JSON Schema for validation
# =============================================================================

COLREGS_OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://marine-env/colregs-referee-output.schema.json",
    "title": "COLREGS Symbolic Referee Output",
    "description": "12-field structured output from the COLREGS symbolic referee layer",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "timestamp",
        "encounter_classification",
        "target_interpretations",
        "required_maneuver",
        "forbidden_maneuver",
        "max_safe_speed",
        "global_min_cpa",
        "global_min_tcpa",
        "confidence_score",
        "reasoning_trace",
    ],
    "properties": {
        "timestamp": {
            "type": "number",
            "description": "ROS time in seconds"
        },
        "scenario_id": {
            "type": "string",
            "description": "Scenario identifier (e.g., S01)"
        },
        "encounter_classification": {
            "type": "object",
            "additionalProperties": False,
            "required": ["primary_encounter", "risk_level", "is_stand_on_vessel"],
            "properties": {
                "primary_encounter": {
                    "type": "string",
                    "enum": ["head_on", "crossing", "overtaking",
                             "restricted_visibility", "multi_ship", "no_risk"]
                },
                "all_encounters": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "rule_priority_order": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "risk_level": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"]
                },
                "is_stand_on_vessel": {"type": "boolean"},
                "cpa_risk_field": {"type": "number", "minimum": 0},
                "environment_context": {"type": "string"},
            }
        },
        "target_interpretations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["target_name", "encounter_type", "own_ship_role"],
                "properties": {
                    "target_name": {"type": "string"},
                    "encounter_type": {
                        "type": "string",
                        "enum": ["head_on", "crossing", "overtaking",
                                 "restricted_visibility", "multi_ship", "no_risk"]
                    },
                    "own_ship_role": {
                        "type": "string",
                        "enum": ["give_way", "stand_on", "not_applicable"]
                    },
                    "applicable_rules": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "spatial": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["target_name", "min_distance"],
                        "properties": {
                            "target_name": {"type": "string"},
                            "min_distance": {"type": "number", "minimum": 0},
                            "forbidden_bearing_min": {"type": ["number", "null"]},
                            "forbidden_bearing_max": {"type": ["number", "null"]},
                            "pass_astern": {"type": ["boolean", "null"]},
                            "pass_ahead": {"type": ["boolean", "null"]},
                            "valid_until_tcpa": {"type": ["number", "null"]},
                            "priority": {
                                "type": "integer",
                                "enum": [0, 1, 2]
                            },
                        }
                    },
                    "maneuver": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["required_maneuver"],
                        "properties": {
                            "required_maneuver": {
                                "type": "string",
                                "enum": ["alter_to_starboard", "alter_to_port",
                                         "maintain", "any_safe",
                                         "reduce_speed", "increase_speed"]
                            },
                            "forbidden_maneuver": {
                                "type": "string",
                                "enum": ["alter_to_port", "alter_to_starboard",
                                         "none", "reduce_speed"]
                            },
                            "alteration_min_angle": {
                                "type": "number", "minimum": 0, "maximum": 3.1416
                            },
                            "alteration_max_angle": {
                                "type": "number", "minimum": 0, "maximum": 3.1416
                            },
                            "max_yaw_rate": {"type": "number", "minimum": 0},
                            "priority": {"type": "integer", "enum": [0, 1, 2]},
                        }
                    },
                    "speed": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["max_speed"],
                        "properties": {
                            "max_speed": {"type": "number", "minimum": 0},
                            "min_speed": {"type": "number", "minimum": 0},
                            "visibility_reduction": {
                                "type": "number", "minimum": 0, "maximum": 1
                            },
                            "traffic_density_factor": {
                                "type": "number", "minimum": 1
                            },
                            "priority": {"type": "integer", "enum": [0, 1, 2]},
                        }
                    },
                }
            }
        },
        "required_maneuver": {
            "type": "string",
            "enum": ["alter_to_starboard", "alter_to_port",
                     "maintain", "any_safe", "reduce_speed", "increase_speed"]
        },
        "forbidden_maneuver": {
            "type": "string",
            "enum": ["alter_to_port", "alter_to_starboard", "none", "reduce_speed"]
        },
        "max_safe_speed": {
            "type": "number", "minimum": 0,
            "description": "Maximum safe speed in m/s (Rule 6)"
        },
        "global_min_cpa": {
            "type": "number", "minimum": 0,
            "description": "Global minimum CPA distance across all targets"
        },
        "global_min_tcpa": {
            "type": "number",
            "description": "Global minimum TCPA threshold"
        },
        "confidence_score": {
            "type": "number", "minimum": 0, "maximum": 1
        },
        "reasoning_trace": {"type": "string"},
        "llm_model": {"type": "string"},
        "inference_time_ms": {"type": "number"},
        "fallback_active": {"type": "boolean"},
        "degradation_level": {
            "type": "integer", "minimum": 0, "maximum": 3
        },
    }
}


def validate_output(output: COLREGSConstraintOutput) -> bool:
    """Validate a COLREGSConstraintOutput against the JSON schema.

    Returns True if valid, raises jsonschema.ValidationError otherwise.
    """
    data = output.to_dict()
    jsonschema.validate(data, COLREGS_OUTPUT_SCHEMA)
    return True


def validate_output_dict(data: dict) -> bool:
    """Validate a raw dictionary against the COLREGS output schema."""
    jsonschema.validate(data, COLREGS_OUTPUT_SCHEMA)
    return True


# =============================================================================
# Serialization helpers
# =============================================================================

def _serialize(obj):
    """Recursively convert dataclass/enum objects to dicts/lists/primitives."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if hasattr(obj, '__dataclass_fields__'):
        result = {}
        for f_name in obj.__dataclass_fields__:
            value = getattr(obj, f_name)
            if value is not None:
                result[f_name] = _serialize(value)
        return result
    return str(obj)
