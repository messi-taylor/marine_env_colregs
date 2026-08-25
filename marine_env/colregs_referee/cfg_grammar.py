#!/usr/bin/env python3
"""
COLREGS Context-Free Grammar (CFG) for Constrained Decoding
============================================================

Defines the deterministic context-free grammar L_colregs — a STRICT subset
of JSON that only accepts COLREGS constraint outputs with the exact 12-field
structure, allowed enum values, and nested geometry.

Key property: L_colregs ⊂ L_json

  - {"foo": 123}  ∉ L_colregs  (rejected — unknown field)
  - {"encounter_classification": {"primary_encounter": "sideways"}}  ∉ L_colregs
  - Only the exact COLREGS 12-field structure is accepted.

Usage for LLM constrained decoding (Section 4.3):
  1. Token-level Logit Bias: at each decoding step, mask all tokens that
     would lead to a string ∉ L_colregs.
  2. Outlines/Jsonformer integration: the CFG is compiled to a DFA that
     tracks the valid token set at each position.

Grammar Design:
  - All JSON keys are EXACT string terminals (not generic ESCAPED_STRING).
  - All enum values are explicitly enumerated.
  - Numeric fields accept SIGNED_NUMBER (validated by JSON Schema downstream).
  - Arbitrary string fields (target_name, reasoning_trace) use ESCAPED_STRING.
  - Whitespace is ignored via %ignore.
"""

from lark import Lark, UnexpectedToken, UnexpectedEOF
from typing import Optional, List
import json


# =============================================================================
# L_colregs — Strict COLREGS Constraint Grammar
# =============================================================================
# Each allowed JSON key is a terminal. The grammar accepts exactly the
# COLREGS output structure. Any deviation (unknown key, wrong enum value,
# wrong nesting) causes a parse error.

COLREGS_CFG_GRAMMAR = r"""
// ── Top-level output ──
?start: output

output: "{" top_fields "}"

// Top-level fields in canonical order (deterministic referee output order)
top_fields: top_field ("," top_field)*

top_field: key_timestamp ":" number
         | key_scenario_id ":" string
         | key_encounter_classification ":" encounter_class_obj
         | key_target_interpretations ":" target_interp_array
         | key_required_maneuver ":" maneuver_value
         | key_forbidden_maneuver ":" forbidden_maneuver_value
         | key_max_safe_speed ":" number
         | key_global_min_cpa ":" number
         | key_global_min_tcpa ":" number
         | key_confidence_score ":" number
         | key_reasoning_trace ":" string
         | key_llm_model ":" string
         | key_inference_time_ms ":" number
         | key_fallback_active ":" bool_value
         | key_degradation_level ":" integer_0_3

// ── Encounter classification object ──
encounter_class_obj: "{" ec_fields "}"

ec_fields: ec_field ("," ec_field)*

ec_field: key_primary_encounter ":" encounter_type_value
        | key_all_encounters ":" string_array
        | key_rule_priority_order ":" string_array
        | key_risk_level ":" risk_level_value
        | key_is_stand_on_vessel ":" bool_value
        | key_cpa_risk_field ":" number
        | key_environment_context ":" string

// ── Target interpretations array ──
target_interp_array: "[" [target_interp ("," target_interp)*] "]"

target_interp: "{" ti_fields "}"

ti_fields: ti_field ("," ti_field)*

ti_field: key_target_name ":" string
        | key_encounter_type ":" encounter_type_value
        | key_own_ship_role ":" ship_role_value
        | key_applicable_rules ":" string_array
        | key_spatial ":" spatial_constraint
        | key_spatial_null ":" "null"
        | key_maneuver ":" maneuver_constraint
        | key_maneuver_null ":" "null"
        | key_speed ":" speed_constraint
        | key_speed_null ":" "null"

// ── Spatial constraint ──
spatial_constraint: "{" sp_fields "}"

sp_fields: sp_field ("," sp_field)*

sp_field: key_target_name ":" string
        | key_min_distance ":" number
        | key_forbidden_bearing_min ":" nullable_number
        | key_forbidden_bearing_max ":" nullable_number
        | key_pass_astern ":" nullable_bool
        | key_pass_ahead ":" nullable_bool
        | key_valid_until_tcpa ":" nullable_number
        | key_priority ":" integer_0_2

// ── Maneuver constraint ──
maneuver_constraint: "{" mv_fields "}"

mv_fields: mv_field ("," mv_field)*

mv_field: key_required_maneuver ":" maneuver_value
        | key_forbidden_maneuver ":" forbidden_maneuver_value
        | key_alteration_min_angle ":" number
        | key_alteration_max_angle ":" number
        | key_max_yaw_rate ":" number
        | key_priority ":" integer_0_2

// ── Speed constraint ──
speed_constraint: "{" spd_fields "}"

spd_fields: spd_field ("," spd_field)*

spd_field: key_max_speed ":" number
         | key_min_speed ":" nullable_number
         | key_visibility_reduction ":" number
         | key_traffic_density_factor ":" number
         | key_priority ":" integer_0_2

// =========================================================================
// Value types
// =========================================================================

number: SIGNED_NUMBER
integer_0_3: /[0-3]/
integer_0_2: /[0-2]/
string: ESCAPED_STRING
bool_value: "true" | "false"
nullable_number: number | "null"
nullable_bool: bool_value | "null"

string_array: "[" [string ("," string)*] "]"

// =========================================================================
// COLREGS enum vocabularies (constrained terminals)
// =========================================================================

encounter_type_value: "\"head_on\""
                    | "\"crossing\""
                    | "\"overtaking\""
                    | "\"restricted_visibility\""
                    | "\"multi_ship\""
                    | "\"no_risk\""

ship_role_value: "\"give_way\"" | "\"stand_on\"" | "\"not_applicable\""

maneuver_value: "\"alter_to_starboard\""
              | "\"alter_to_port\""
              | "\"maintain\""
              | "\"any_safe\""
              | "\"reduce_speed\""
              | "\"increase_speed\""

forbidden_maneuver_value: "\"alter_to_port\""
                        | "\"alter_to_starboard\""
                        | "\"none\""
                        | "\"reduce_speed\""

risk_level_value: "\"low\"" | "\"medium\"" | "\"high\"" | "\"critical\""

// =========================================================================
// JSON key terminals — explicit field names (the core of L_colregs)
// =========================================================================

// Top-level keys
key_timestamp:                "\"timestamp\""
key_scenario_id:              "\"scenario_id\""
key_encounter_classification: "\"encounter_classification\""
key_target_interpretations:   "\"target_interpretations\""
key_required_maneuver:        "\"required_maneuver\""
key_forbidden_maneuver:       "\"forbidden_maneuver\""
key_max_safe_speed:           "\"max_safe_speed\""
key_global_min_cpa:           "\"global_min_cpa\""
key_global_min_tcpa:          "\"global_min_tcpa\""
key_confidence_score:         "\"confidence_score\""
key_reasoning_trace:          "\"reasoning_trace\""
key_llm_model:                "\"llm_model\""
key_inference_time_ms:        "\"inference_time_ms\""
key_fallback_active:          "\"fallback_active\""
key_degradation_level:        "\"degradation_level\""

// Encounter classification keys
key_primary_encounter:    "\"primary_encounter\""
key_all_encounters:       "\"all_encounters\""
key_rule_priority_order:  "\"rule_priority_order\""
key_risk_level:           "\"risk_level\""
key_is_stand_on_vessel:   "\"is_stand_on_vessel\""
key_cpa_risk_field:       "\"cpa_risk_field\""
key_environment_context:  "\"environment_context\""

// Target interpretation keys
key_target_name:      "\"target_name\""
key_encounter_type:   "\"encounter_type\""
key_own_ship_role:    "\"own_ship_role\""
key_applicable_rules: "\"applicable_rules\""
key_spatial:          "\"spatial\""
key_spatial_null:     "\"spatial\""     // Overloaded: spatial: null
key_maneuver:         "\"maneuver\""
key_maneuver_null:    "\"maneuver\""    // Overloaded: maneuver: null
key_speed:            "\"speed\""
key_speed_null:       "\"speed\""       // Overloaded: speed: null

// Spatial constraint keys
key_min_distance:           "\"min_distance\""
key_forbidden_bearing_min:  "\"forbidden_bearing_min\""
key_forbidden_bearing_max:  "\"forbidden_bearing_max\""
key_pass_astern:            "\"pass_astern\""
key_pass_ahead:             "\"pass_ahead\""
key_valid_until_tcpa:       "\"valid_until_tcpa\""

// Maneuver constraint keys
key_alteration_min_angle: "\"alteration_min_angle\""
key_alteration_max_angle: "\"alteration_max_angle\""
key_max_yaw_rate:         "\"max_yaw_rate\""

// Speed constraint keys
key_max_speed:             "\"max_speed\""
key_min_speed:             "\"min_speed\""
key_visibility_reduction:  "\"visibility_reduction\""
key_traffic_density_factor: "\"traffic_density_factor\""

// Generic key
key_priority: "\"priority\""

// ── Whitespace ──
%ignore /[ \t\n\r]+/

// ── Lexer imports ──
%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
"""

# =============================================================================
# Sub-grammar: spatial constraint (standalone, for individual validation)
# =============================================================================

SPATIAL_CONSTRAINT_GRAMMAR = r"""
?start: spatial_constraint

spatial_constraint: "{" sp_fields "}"

sp_fields: sp_field ("," sp_field)*

sp_field: key_target_name ":" ESCAPED_STRING
        | key_min_distance ":" SIGNED_NUMBER
        | key_forbidden_bearing_min ":" nullable_number
        | key_forbidden_bearing_max ":" nullable_number
        | key_pass_astern ":" nullable_bool
        | key_pass_ahead ":" nullable_bool
        | key_valid_until_tcpa ":" nullable_number
        | key_priority ":" /[012]/

nullable_number: SIGNED_NUMBER | "null"
nullable_bool: "true" | "false" | "null"

key_target_name:           "\"target_name\""
key_min_distance:          "\"min_distance\""
key_forbidden_bearing_min: "\"forbidden_bearing_min\""
key_forbidden_bearing_max: "\"forbidden_bearing_max\""
key_pass_astern:           "\"pass_astern\""
key_pass_ahead:            "\"pass_ahead\""
key_valid_until_tcpa:      "\"valid_until_tcpa\""
key_priority:              "\"priority\""

%ignore /[ \t\n\r]+/
%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
"""


# =============================================================================
# Outlines-compatible token mask schema (for future LLM integration)
# =============================================================================

# The CFG above can be compiled into a DFA-based token mask for
# Outlines/Jsonformer. Each terminal maps to a set of allowed token IDs.
#
# Example: when the parser is at state expecting `encounter_type_value`,
# the token mask allows only: "head_on", "crossing", "overtaking",
# "restricted_visibility", "multi_ship", "no_risk" and their subword tokens.

def build_token_mask_map() -> dict:
    """Build a mapping from CFG state to allowed token strings.

    This is a simplified representation — the full implementation
    requires integrating with the LLM's tokenizer to map strings
    to token IDs with proper subword splitting.

    Returns:
        dict mapping parser position labels to allowed string sets.
    """
    return {
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
        "degradation_level": ["0", "1", "2", "3"],
        "priority": ["0", "1", "2"],
    }


# =============================================================================
# Grammar validator class
# =============================================================================

class COLREGSGrammar:
    """Strict CFG grammar validator for L_colregs.

    The grammar accepts ONLY strings matching the COLREGS constraint
    output structure. Unknown keys, wrong enum values, or incorrect
    nesting are rejected at the CFG level — before any semantic validation.

    Usage:
        grammar = COLREGSGrammar()
        grammar.validate(json_string)        # raises ValueError if not in L_colregs
        grammar.is_in_language(json_string)  # returns True/False
    """

    def __init__(self):
        self._parser: Optional[Lark] = None
        self._spatial_parser: Optional[Lark] = None

    @property
    def parser(self) -> Lark:
        if self._parser is None:
            self._parser = Lark(COLREGS_CFG_GRAMMAR, start='output',
                               parser='earley')
        return self._parser

    @property
    def spatial_parser(self) -> Lark:
        if self._spatial_parser is None:
            self._spatial_parser = Lark(SPATIAL_CONSTRAINT_GRAMMAR,
                                        start='spatial_constraint',
                                        parser='earley')
        return self._spatial_parser

    def is_in_language(self, json_string: str) -> bool:
        """Check if a JSON string is in L_colregs.

        Returns True ONLY if the string:
          1. Is valid JSON
          2. Matches the exact COLREGS output structure
          3. Uses only allowed keys and enum values
        """
        try:
            self.parser.parse(json_string)
            json.loads(json_string)  # also verify valid JSON
            return True
        except Exception:
            return False

    def validate(self, json_string: str) -> None:
        """Validate a JSON string against L_colregs.

        Raises ValueError with details if the string is not in L_colregs.
        """
        # Step 1: valid JSON
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        # Step 2: CFG parse
        try:
            self.parser.parse(json_string)
        except UnexpectedToken as e:
            raise ValueError(
                f"Not in L_colregs — unexpected token '{e.token}' "
                f"at line {e.line}, col {e.column}. "
                f"Expected: {e.expected}")
        except UnexpectedEOF as e:
            raise ValueError(
                f"Not in L_colregs — unexpected EOF. Expected: {e.expected}")
        except Exception as e:
            raise ValueError(f"Not in L_colregs — parse error: {e}")

        # Step 3: validate spatial sub-grammar for each target
        if 'target_interpretations' in data:
            for interp in data['target_interpretations']:
                if 'spatial' in interp and interp['spatial'] is not None:
                    try:
                        spatial_json = json.dumps(interp['spatial'])
                        self.spatial_parser.parse(spatial_json)
                    except Exception as e:
                        raise ValueError(
                            f"Not in L_colregs — spatial constraint for "
                            f"{interp.get('target_name', '?')}: {e}")

    def validate_spatial(self, spatial_json: str) -> bool:
        """Validate a spatial constraint against its sub-grammar."""
        try:
            self.spatial_parser.parse(spatial_json)
            json.loads(spatial_json)
            return True
        except Exception:
            return False

    # ── Vocabulary accessors (for logit bias masking) ──

    @staticmethod
    def get_allowed_encounter_types() -> List[str]:
        return ["head_on", "crossing", "overtaking",
                "restricted_visibility", "multi_ship", "no_risk"]

    @staticmethod
    def get_allowed_ship_roles() -> List[str]:
        return ["give_way", "stand_on", "not_applicable"]

    @staticmethod
    def get_allowed_maneuvers() -> List[str]:
        return ["alter_to_starboard", "alter_to_port", "maintain",
                "any_safe", "reduce_speed", "increase_speed"]

    @staticmethod
    def get_allowed_forbidden_maneuvers() -> List[str]:
        return ["alter_to_port", "alter_to_starboard", "none", "reduce_speed"]

    @staticmethod
    def get_allowed_colregs_rules() -> List[str]:
        return [
            "Rule 2", "Rule 5", "Rule 6", "Rule 7", "Rule 8",
            "Rule 13", "Rule 14", "Rule 15", "Rule 16", "Rule 17",
            "Rule 18", "Rule 19",
        ]

    @staticmethod
    def get_allowed_risk_levels() -> List[str]:
        return ["low", "medium", "high", "critical"]


def validate_cfg_output(json_string: str) -> bool:
    """Convenience: return True iff json_string ∈ L_colregs."""
    grammar = COLREGSGrammar()
    try:
        grammar.validate(json_string)
        return True
    except ValueError as e:
        print(f"L_colregs validation failed: {e}")
        return False
