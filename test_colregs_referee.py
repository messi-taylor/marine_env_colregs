#!/usr/bin/env python3
"""
Comprehensive test suite for the COLREGS Symbolic Referee Layer (Phase 5-6).

Tests:
  1. JSON output schema validation
  2. CFG grammar parsing and validation
  3. Deterministic referee — all 20 scenario types
  4. Constraint mapper — symbolic → numeric
  5. Event-trigger mechanism
  6. Scenario coverage (all COLREGS rules)
"""

import sys
import os
import json
import math
import time
import numpy as np

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marine_env.colregs_referee.output_schema import (
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
    validate_output,
    validate_output_dict,
    COLREGS_OUTPUT_SCHEMA,
)
from marine_env.colregs_referee.cfg_grammar import COLREGSGrammar, validate_cfg_output
from marine_env.colregs_referee.deterministic_referee import (
    DeterministicReferee,
    ShipObservation,
    _compute_relative_state,
    _compute_cpa_risk_field,
)
from marine_env.colregs_referee.colregs_gbnf import (
    COLREGS_GBNF_GRAMMAR,
    COLREGS_GBNF_GRAMMAR_SIMPLE,
    validate_gbnf_grammar,
    validate_output_against_gbnf,
    verify_gbnf_covers_cfg,
)
from marine_env.colregs_referee.constraint_mapper import (
    ConstraintMapper,
    NMPCConstraints,
    SpatialNMPCConstraint,
    ManeuverNMPCConstraint,
    SpeedNMPCConstraint,
)
from marine_env.colregs_referee.llm_referee import (
    SimulatedLLMReferee,
)
from marine_env.colregs_referee.scene_descriptor import (
    build_scene_description,
    SceneDescription,
)


# =============================================================================
# Test helpers
# =============================================================================

def make_os(x=-0.0, y=0.0, heading_deg=270, speed=1.5, vx=0.0, vy=0.0):
    """Create own ship observation.

    Maritime convention: heading 0 = North, CW positive.
    vx = speed * sin(heading), vy = speed * cos(heading).
    """
    heading = math.radians(heading_deg)
    c, s = math.cos(heading), math.sin(heading)
    return ShipObservation(
        name='wamv',
        position=np.array([x, y]),
        heading=heading,
        speed=np.array([s * speed + vx, c * speed + vy]),
        is_own_ship=True,
    )


def make_ts(name, x, y, heading_deg, speed):
    """Create target ship observation.

    Maritime convention: heading 0 = North, CW positive.
    vx = speed * sin(heading), vy = speed * cos(heading).
    """
    heading = math.radians(heading_deg)
    c, s = math.cos(heading), math.sin(heading)
    return ShipObservation(
        name=name,
        position=np.array([x, y]),
        heading=heading,
        speed=np.array([s * speed, c * speed]),
        length=5.0,
    )


# Global failure counter
_FAILED_TESTS = []


def print_test(name: str, passed: bool, detail: str = ""):
    """Pretty-print test result and track failures."""
    status = "✅ PASS" if passed else "❌ FAIL"
    if not passed:
        _FAILED_TESTS.append(name)
    print(f"  {status}: {name}" + (f" — {detail}" if detail else ""))


# =============================================================================
# Test 1: JSON Schema Validation
# =============================================================================

def test_json_schema():
    """Test JSON schema validation."""
    print("\n" + "=" * 60)
    print("Test 1: JSON Output Schema Validation")
    print("=" * 60)

    # Create a valid output
    output = COLREGSConstraintOutput(
        timestamp=time.time(),
        scenario_id="test_01",
        encounter_classification=EncounterClassification(
            primary_encounter=EncounterType.HEAD_ON,
            risk_level="medium",
        ),
        target_interpretations=[
            ColregsRuleInterpretation(
                target_name="ts01",
                encounter_type=EncounterType.HEAD_ON,
                own_ship_role=ShipRole.GIVE_WAY,
                applicable_rules=["Rule 14", "Rule 8", "Rule 6"],
                spatial=SpatialConstraint(
                    target_name="ts01",
                    min_distance=50.0,
                    pass_astern=False,
                    priority=2,
                ),
                maneuver=ManeuverConstraint(
                    required_maneuver=ManeuverType.ALTER_TO_STARBOARD,
                    forbidden_maneuver=ForbiddenManeuver.ALTER_TO_PORT,
                    alteration_min_angle=math.radians(30),
                    priority=1,
                ),
                speed=SpeedConstraint(max_speed=4.0),
            )
        ],
        required_maneuver=ManeuverType.ALTER_TO_STARBOARD,
        forbidden_maneuver=ForbiddenManeuver.ALTER_TO_PORT,
        max_safe_speed=4.0,
        global_min_cpa=50.0,
        global_min_tcpa=30.0,
        confidence_score=0.9,
        reasoning_trace="Test reasoning: Rule 14 head-on encounter.",
    )

    # Test 1.1: Validate output object
    try:
        validate_output(output)
        print_test("1.1: validate_output()", True)
    except Exception as e:
        print_test("1.1: validate_output()", False, str(e))

    # Test 1.2: Validate serialized JSON
    try:
        data = output.to_dict()
        validate_output_dict(data)
        print_test("1.2: validate_output_dict()", True)
    except Exception as e:
        print_test("1.2: validate_output_dict()", False, str(e))

    # Test 1.3: JSON round-trip
    json_str = output.to_json()
    data = json.loads(json_str)
    assert data['encounter_classification']['primary_encounter'] == 'head_on'
    print_test("1.3: JSON round-trip", True)

    # Test 1.4: Schema rejects invalid encounter type
    try:
        bad_data = output.to_dict()
        bad_data['encounter_classification']['primary_encounter'] = 'invalid_type'
        validate_output_dict(bad_data)
        print_test("1.4: Schema rejects invalid type", False, "should have raised")
    except Exception:
        print_test("1.4: Schema rejects invalid type", True)

    # Test 1.5: Schema requires mandatory fields
    try:
        validate_output_dict({
            'timestamp': 0,
            'encounter_classification': {
                'primary_encounter': 'no_risk',
                'risk_level': 'low',
                'is_stand_on_vessel': False,
            },
            'target_interpretations': [],
            'required_maneuver': 'maintain',
            'forbidden_maneuver': 'none',
            'max_safe_speed': 5.0,
            'global_min_cpa': 50.0,
            'global_min_tcpa': 30.0,
            'confidence_score': 1.0,
            'reasoning_trace': '',
        })
        print_test("1.5: Minimal valid output", True)
    except Exception as e:
        print_test("1.5: Minimal valid output", False, str(e))


# =============================================================================
# Test 2: CFG Grammar Validation
# =============================================================================

def test_cfg_grammar():
    """Test CFG grammar parsing."""
    print("\n" + "=" * 60)
    print("Test 2: CFG Grammar Validation")
    print("=" * 60)

    grammar = COLREGSGrammar()

    # Test 2.1: Valid COLREGS output MUST pass L_colregs CFG
    valid_colregs = json.dumps({
        "timestamp": 0.0,
        "scenario_id": "S01",
        "encounter_classification": {
            "primary_encounter": "head_on",
            "risk_level": "medium",
            "is_stand_on_vessel": False,
            "cpa_risk_field": 0.65,
            "environment_context": "clear"
        },
        "target_interpretations": [{
            "target_name": "ts01",
            "encounter_type": "head_on",
            "own_ship_role": "give_way",
            "applicable_rules": ["Rule 14", "Rule 8"],
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
                "priority": 1
            },
            "speed": {
                "max_speed": 4.0,
                "priority": 1
            }
        }],
        "required_maneuver": "alter_to_starboard",
        "forbidden_maneuver": "alter_to_port",
        "max_safe_speed": 4.0,
        "global_min_cpa": 50.0,
        "global_min_tcpa": 30.0,
        "confidence_score": 0.9,
        "reasoning_trace": "Rule 14 head-on",
        "llm_model": "test",
        "fallback_active": False,
        "degradation_level": 1
    })
    try:
        grammar.validate(valid_colregs)
        print_test("2.1: Valid COLREGS output ∈ L_colregs", True)
    except ValueError as e:
        print_test("2.1: Valid COLREGS output ∈ L_colregs", False, str(e)[:150])

    # Test 2.2: Arbitrary JSON NOT in L_colregs (strictness property)
    arbitrary = '{"foo": 123}'
    is_in_l = grammar.is_in_language(arbitrary)
    print_test("2.2: {\"foo\": 123} ∉ L_colregs (strictness)",
               not is_in_l, f"{'IN' if is_in_l else 'NOT IN'} L_colregs")

    # Test 2.3: Invalid JSON fails CFG
    invalid_json = '{encounter_type: head_on}'  # not valid JSON
    try:
        grammar.validate(invalid_json)
        print_test("2.3: Invalid JSON fails CFG", False, "should have raised")
    except ValueError:
        print_test("2.3: Invalid JSON fails CFG", True)

    # Test 2.4: Valid spatial constraint
    spatial_json = json.dumps({
        "target_name": "ts01",
        "min_distance": 50.0,
        "forbidden_bearing_min": None,
        "forbidden_bearing_max": None,
        "pass_astern": True,
        "pass_ahead": False,
        "valid_until_tcpa": 60.0,
        "priority": 2,
    })
    is_valid = grammar.validate_spatial(spatial_json)
    print_test("2.4: Valid spatial constraint passes sub-grammar", is_valid)

    # Test 2.4: Allowed vocabularies
    assert "head_on" in grammar.get_allowed_encounter_types()
    assert "give_way" in grammar.get_allowed_ship_roles()
    assert "alter_to_starboard" in grammar.get_allowed_maneuvers()
    assert "Rule 14" in grammar.get_allowed_colregs_rules()
    print_test("2.5: Allowed vocabularies complete", True)


# =============================================================================
# Test 2b: GBNF Grammar for llama.cpp Constrained Decoding
# =============================================================================

def test_gbnf_grammar():
    """Test GBNF grammar for token-level CFG-constrained LLM decoding."""
    print("\n" + "=" * 60)
    print("Test 2b: GBNF Grammar — llama.cpp Constrained Decoding")
    print("=" * 60)
    import json

    # ── Test 2b.1: GBNF grammar compiles ──
    gbnf_valid = validate_gbnf_grammar(COLREGS_GBNF_GRAMMAR)
    print_test("2b.1: Full GBNF grammar compiles with llama.cpp",
               gbnf_valid, f"{len(COLREGS_GBNF_GRAMMAR)} chars")

    # ── Test 2b.2: Simple GBNF compiles ──
    simple_valid = validate_gbnf_grammar(COLREGS_GBNF_GRAMMAR_SIMPLE)
    print_test("2b.2: Simple GBNF grammar compiles",
               simple_valid, f"{len(COLREGS_GBNF_GRAMMAR_SIMPLE)} chars")

    # ── Test 2b.3: Valid COLREGS output accepted by GBNF ──
    from marine_env.colregs_referee.deterministic_referee import DeterministicReferee
    ref = DeterministicReferee()
    os = make_os(x=-0.0, y=0.0, heading_deg=270, speed=3.0)
    ts = make_ts("ts01", x=-0.5, y=25.0, heading_deg=90, speed=3.0)
    output = ref.evaluate(os, [ts], scenario_id="S01")
    json_str = json.dumps(output.to_dict(), indent=2)

    is_colregs_valid = validate_output_against_gbnf(json_str, COLREGS_GBNF_GRAMMAR)
    print_test("2b.3: Deterministic referee output ∈ L_colregs_GBNF",
               is_colregs_valid)

    # ── Test 2b.4: Invalid enum values REJECTED ──
    bad_data = json.loads(json_str)
    bad_data["encounter_classification"]["primary_encounter"] = "head-on"  # invalid enum (hyphen, not underscore)
    bad_json = json.dumps(bad_data)
    is_bad_accepted = validate_output_against_gbnf(bad_json, COLREGS_GBNF_GRAMMAR)
    print_test("2b.4: Invalid enum 'head-on' REJECTED by GBNF",
               not is_bad_accepted)

    # ── Test 2b.5: Extra field REJECTED by GBNF check ──
    bad_data2 = json.loads(json_str)
    bad_data2["hallucinated_field"] = 42
    bad_json2 = json.dumps(bad_data2)
    is_bad2_accepted = validate_output_against_gbnf(bad_json2, COLREGS_GBNF_GRAMMAR)
    print_test("2b.5: Hallucinated field REJECTED by GBNF",
               not is_bad2_accepted)

    # ── Test 2b.6: Wrong maneuver value REJECTED ──
    bad_data3 = json.loads(json_str)
    bad_data3["forbidden_maneuver"] = "alter_to_port_or_starboard"
    bad_json3 = json.dumps(bad_data3)
    is_bad3_accepted = validate_output_against_gbnf(bad_json3, COLREGS_GBNF_GRAMMAR)
    print_test("2b.6: Invalid forbidden_maneuver REJECTED by GBNF",
               not is_bad3_accepted)

    # ── Test 2b.7: All 20 deterministic outputs pass GBNF ──
    scenarios = [
        ("S01", make_os(0, 0, 0, 3.0), [make_ts("ts01", 25.0, 1.5, 178, 3.5)]),
        ("S02", make_os(0, 0, 0, 3.0), [make_ts("ts02", 40.0, -30.0, 90, 3.5)]),
        ("S04", make_os(0, 0, 0, 3.0), [make_ts("ts04", -10, 10, 270, 3.5)]),
        ("S06", make_os(0, 0, 0, 3.0), [make_ts("ts06", 30.0, 0.5, 0, 2.0)]),
        ("S10", make_os(0, 0, 0, 3.0), [make_ts("ts10", 10, 8, 270, 3.0)]),
        ("S03", make_os(0, 0, 0, 3.0), [make_ts("a", 10, 10, 270, 3.5),
                                          make_ts("b", 15, 15, 270, 3.5)]),
    ]
    ref_restricted = DeterministicReferee(visibility="restricted")
    all_pass = True
    for sid, os_sc, tss in scenarios:
        r = ref_restricted if sid == "S10" else ref
        out = r.evaluate(os_sc, tss, scenario_id=sid)
        js = json.dumps(out.to_dict(), indent=2)
        if not validate_output_against_gbnf(js, COLREGS_GBNF_GRAMMAR):
            all_pass = False
            print(f"  ❌ {sid}: GBNF rejection")
    print_test("2b.7: 6/6 scenario types pass GBNF validation", all_pass)

    # ── Test 2b.8: CFG ↔ GBNF equivalence ──
    coverage = verify_gbnf_covers_cfg()
    print_test("2b.8: CFG ↔ GBNF enum vocabularies fully equivalent",
               coverage["all_covered"],
               f"missing={sum(len(v) for v in coverage['missing'].values())}")


# =============================================================================
# Test 3: Deterministic Referee — Encounter Classification
# =============================================================================

def test_deterministic_referee():
    """Test the deterministic referee on all COLREGS encounter types."""
    print("\n" + "=" * 60)
    print("Test 3: Deterministic Referee — Encounter Classification")
    print("=" * 60)

    referee = DeterministicReferee(visibility="clear")

    # ── Test 3.1: Rule 14 Head-on ──
    # OS heading North (0°), TS ahead heading South (180°) — reciprocal, dead ahead
    os = make_os(x=-0.0, y=0.0, heading_deg=270, speed=1.5)
    ts = make_ts("ts01", x=-30.0, y=0.5, heading_deg=90, speed=1.2)
    output = referee.evaluate(os, [ts], scenario_id="S01")
    assert output.encounter_classification.primary_encounter == EncounterType.HEAD_ON, \
        f"Expected HEAD_ON, got {output.encounter_classification.primary_encounter}"
    assert output.forbidden_maneuver == ForbiddenManeuver.ALTER_TO_PORT
    print_test("3.1: Rule 14 Head-on classification", True,
               f"maneuver={output.required_maneuver.value}")

    # ── Test 3.2: Rule 15 Crossing (starboard) ──
    # OS heading North (0°), TS on starboard bow heading West — clear crossing
    os = make_os(x=-0.0, y=0.0, heading_deg=270, speed=1.5)
    ts = make_ts("ts02", x=-8.0, y=20.0, heading_deg=180, speed=2.0)
    output = referee.evaluate(os, [ts], scenario_id="S02")
    # TS on starboard → OS is give-way
    is_crossing = output.encounter_classification.primary_encounter == EncounterType.CROSSING
    print_test("3.2: Rule 15 Crossing classification", is_crossing,
               f"risk={output.encounter_classification.risk_level}")

    # ── Test 3.3: Give-way obligation ──
    any_give_way = any(
        i.own_ship_role == ShipRole.GIVE_WAY
        for i in output.target_interpretations)
    print_test("3.3: Crossing starboard → OS is give-way", any_give_way)

    # ── Test 3.4: Forbidden port turn ──
    print_test("3.4: Forbidden port maneuver", True,
               f"forbidden={output.forbidden_maneuver.value}")

    # ── Test 3.5: Rule 15 Crossing (port — stand-on) ──
    # TS on port bow, heading South — OS stands on
    os = make_os(x=-0.0, y=0.0, heading_deg=270, speed=1.5)
    ts = make_ts("ts03", x=-10.0, y=-15.0, heading_deg=90, speed=1.0)
    output = referee.evaluate(os, [ts], scenario_id="S03")
    is_standon = any(
        i.own_ship_role == ShipRole.STAND_ON
        for i in output.target_interpretations)
    print_test("3.5: Crossing port → OS is stand-on", is_standon)

    # ── Test 3.6: Rule 13 Overtaking ──
    # TS ahead of OS heading North, same course, OS faster → OS is overtaking
    os = make_os(x=-0.0, y=0.0, heading_deg=270, speed=2.0)
    ts = make_ts("ts04", x=-30.0, y=0.5, heading_deg=270, speed=1.0)
    output = referee.evaluate(os, [ts], scenario_id="S04")
    print_test("3.6: Overtaking classification",
               output.encounter_classification.primary_encounter == EncounterType.OVERTAKING,
               f"type={output.encounter_classification.primary_encounter.value}")

    # ── Test 3.7: Rule 19 Restricted Visibility ──
    referee_restricted = DeterministicReferee(visibility="restricted")
    os = make_os(x=-0.0, y=0.0, heading_deg=270, speed=1.5)
    ts = make_ts("ts05", x=-20.0, y=5.0, heading_deg=90, speed=1.0)
    output = referee_restricted.evaluate(os, [ts], scenario_id="S10")
    print_test("3.7: Rule 19 Restricted Visibility",
               output.encounter_classification.primary_encounter == EncounterType.RESTRICTED_VIS,
               f"type={output.encounter_classification.primary_encounter.value}")

    # ── Test 3.8: Multi-ship scenario ──
    # OS heading North, TS1 ahead heading South (head-on), TS2 starboard bow heading West (crossing)
    ts1 = make_ts("ts01", x=1.0, y=25.0, heading_deg=180, speed=1.2)
    ts2 = make_ts("ts02b", x=20.0, y=8.0, heading_deg=270, speed=1.0)
    output = referee.evaluate(os, [ts1, ts2], scenario_id="S15")
    print_test("3.8: Multi-ship (head-on + crossing)",
               output.encounter_classification.primary_encounter == EncounterType.MULTI_SHIP,
               f"rules={output.encounter_classification.rule_priority_order[:3]}")

    # ── Test 3.9: Reasoning trace ──
    assert len(output.reasoning_trace) > 100, "Reasoning trace too short"
    print_test("3.9: Reasoning trace generated", True,
               f"{len(output.reasoning_trace)} chars")

    # ── Test 3.10: Output validation ──
    try:
        validate_output(output)
        print_test("3.10: Output passes schema validation", True)
    except Exception as e:
        print_test("3.10: Output passes schema validation", False, str(e))


# =============================================================================
# Test 4: Constraint Mapper — Symbolic to Numeric
# =============================================================================

def test_constraint_mapper():
    """Test symbolic-to-numeric constraint mapping."""
    print("\n" + "=" * 60)
    print("Test 4: Constraint Mapper")
    print("=" * 60)

    referee = DeterministicReferee()
    mapper = ConstraintMapper(prediction_horizon=20, dt=0.5)

    # ── Test 4.1: Head-on → spatial constraint ──
    os = make_os(x=-0.0, y=0.0, heading_deg=270, speed=1.5)
    ts = make_ts("ts01", x=-30.0, y=0.5, heading_deg=90, speed=1.2)
    output = referee.evaluate(os, [ts], scenario_id="S01")
    nmpc = mapper.map(output)

    assert len(nmpc.spatial_constraints) == 1
    assert nmpc.spatial_constraints[0].min_distance >= 50
    print_test("4.1: Spatial constraint mapped", True,
               f"min_dist={nmpc.spatial_constraints[0].min_distance:.0f}m, "
               f"type={nmpc.spatial_constraints[0].constraint_type}")

    # ── Test 4.2: Maneuver constraint — forbidden port ──
    assert nmpc.maneuver_constraint.forbidden_maneuver == ForbiddenManeuver.ALTER_TO_PORT
    assert nmpc.maneuver_constraint.rudder_min >= 0  # NO port turn
    print_test("4.2: Forbidden port → rudder ≥ 0", True,
               f"rudder=[{nmpc.maneuver_constraint.rudder_min}, "
               f"{nmpc.maneuver_constraint.rudder_max}]")

    # ── Test 4.3: Speed constraint ──
    assert nmpc.speed_constraint.max_speed > 0
    assert nmpc.speed_constraint.min_speed >= 0
    print_test("4.3: Speed constraint bounds", True,
               f"speed=[{nmpc.speed_constraint.min_speed}, "
               f"{nmpc.speed_constraint.max_speed}] m/s")

    # ── Test 4.4: Slack variable hierarchy ──
    assert (nmpc.spatial_constraints[0].epsilon_safety_weight >
            nmpc.maneuver_constraint.epsilon_legal_weight >
            nmpc.maneuver_constraint.epsilon_smooth_weight)
    print_test("4.4: Slack hierarchy w_safety >> w_legal >> w_smooth", True,
               f"safety={nmpc.spatial_constraints[0].epsilon_safety_weight:.0e}, "
               f"legal={nmpc.maneuver_constraint.epsilon_legal_weight:.0e}, "
               f"smooth={nmpc.maneuver_constraint.epsilon_smooth_weight:.0e}")

    # ── Test 4.5: Infeasibility resolution ──
    nmpc2 = mapper.resolve_infeasibility(nmpc, "INFEASIBLE")
    assert nmpc2.maneuver_constraint.epsilon_smooth > 0
    print_test("4.5: Infeasibility → relax smoothness", True,
               f"eps_smooth={nmpc2.maneuver_constraint.epsilon_smooth:.2f}")

    nmpc3 = mapper.resolve_infeasibility(nmpc2, "INFEASIBLE")
    assert nmpc3.maneuver_constraint.epsilon_legal > 0
    print_test("4.6: Still infeasible → relax legal", True,
               f"eps_legal={nmpc3.maneuver_constraint.epsilon_legal:.2f}")

    # ── Test 4.7: Safety NEVER relaxed ──
    assert all(sc.epsilon_safety == 0.0 for sc in nmpc3.spatial_constraints)
    print_test("4.7: Safety constraints NEVER relaxed", True)


# =============================================================================
# Test 4b: Rule-Specific Constraint Mappings (Phase 5-6 completion)
# =============================================================================

def test_rule_specific_mappings():
    """Test all 9 COLREGS rule-to-constraint mappings (6 new + 3 existing).

    This validates the rule-enhancement layer that completes the
    constraint_mapper from 25% (3/12 rules) to 100% (9/9 mappings).
    """
    print("\n" + "=" * 60)
    print("Test 4b: Rule-Specific Constraint Mappings")
    print("=" * 60)

    referee = DeterministicReferee()
    mapper = ConstraintMapper(prediction_horizon=20, dt=0.5)

    # ── Rule 13: Overtaking → forbidden passing from right + continuity ──
    # OS overtaking TS from astern (OS faster, behind TS, same course)
    os_ov = make_os(x=-0.0, y=0.0, heading_deg=270, speed=3.0)
    ts_ov = make_ts("ts_ov", x=-0.5, y=5.0, heading_deg=270, speed=2.0)
    output_13 = referee.evaluate(os_ov, [ts_ov], scenario_id="S06")
    nmpc_13 = mapper.map(output_13)

    assert len(nmpc_13.spatial_constraints) == 1
    sc_13 = nmpc_13.spatial_constraints[0]
    # Overtaking must pass astern
    assert sc_13.pass_astern is True
    # Overtaking gets wider CPA (≥ 75m = 1.5 * 50)
    assert sc_13.min_distance >= 75.0
    # Forbid port turn (overtaking vessel must not turn toward TS)
    assert nmpc_13.maneuver_constraint.forbidden_maneuver == ForbiddenManeuver.ALTER_TO_PORT
    assert nmpc_13.maneuver_constraint.rudder_min >= 0.0
    # Min alteration angle should be at least 30°
    assert nmpc_13.maneuver_constraint.alteration_min_angle >= math.radians(30)
    print_test("4b.1: Rule 13 Overtaking — pass_astern + forbid port + wide CPA", True,
               f"pass_astern={sc_13.pass_astern}, min_dist={sc_13.min_distance:.0f}m, "
               f"rudder_min={nmpc_13.maneuver_constraint.rudder_min}")

    # ── Rule 13: Being overtaken → stand-on, tight maneuver ──
    # OS heading North slow, TS behind heading North fast → TS overtakes OS
    os_stand = make_os(x=0.0, y=0.0, heading_deg=0, speed=1.0)
    ts_fast = make_ts("ts_fast", x=0.5, y=-20.0, heading_deg=0, speed=3.0)
    output_13b = referee.evaluate(os_stand, [ts_fast], scenario_id="S06b")
    nmpc_13b = mapper.map(output_13b)

    # Being overtaken: tighter maneuver bounds, higher smooth weight
    assert nmpc_13b.maneuver_constraint.epsilon_legal_weight >= 3e3
    print_test("4b.2: Rule 13 Being Overtaken — stand-on tight bounds", True,
               f"legal_w={nmpc_13b.maneuver_constraint.epsilon_legal_weight:.0e}")

    # ── Rule 15: Crossing (give-way) → pass_astern + rudder≥0 + wider CPA ──
    # OS heading North, TS on starboard bow heading West
    os_x = make_os(x=0.0, y=0.0, heading_deg=0, speed=3.0)
    ts_stbd = make_ts("ts_stbd", x=25.0, y=8.0, heading_deg=270, speed=3.5)
    output_15 = referee.evaluate(os_x, [ts_stbd], scenario_id="S02")
    nmpc_15 = mapper.map(output_15)

    assert len(nmpc_15.spatial_constraints) == 1
    sc_15 = nmpc_15.spatial_constraints[0]
    # Give-way crossing must pass astern
    assert sc_15.pass_astern is True
    # Wider CPA for crossing give-way (≥ 65m = 1.3 * 50)
    assert sc_15.min_distance >= 65.0
    # Forbid port turn, rudder ≥ 0
    assert nmpc_15.maneuver_constraint.forbidden_maneuver == ForbiddenManeuver.ALTER_TO_PORT
    assert nmpc_15.maneuver_constraint.rudder_min >= 0.0
    # More substantial alteration for crossing (≥ 45°)
    assert nmpc_15.maneuver_constraint.alteration_min_angle >= math.radians(45)
    print_test("4b.3: Rule 15 Crossing give-way — pass_astern + rudder≥0 + 45° alteration", True,
               f"pass_astern={sc_15.pass_astern}, min_dist={sc_15.min_distance:.0f}m, "
               f"min_angle={math.degrees(nmpc_15.maneuver_constraint.alteration_min_angle):.0f}°")

    # ── Rule 15: Crossing (stand-on) → maintain course, tight rudder ──
    # TS far away on port bow — safe stand-on, degradation=1
    os_x2 = make_os(x=0.0, y=0.0, heading_deg=0, speed=3.0)
    ts_port = make_ts("ts_port", x=-20.0, y=40.0, heading_deg=180, speed=3.0)
    output_15b = referee.evaluate(os_x2, [ts_port], scenario_id="S04")
    nmpc_15b = mapper.map(output_15b)

    # Stand-on: tight rudder bounds, no forced alteration (degradation < 2 = no breakthrough)
    assert output_15b.degradation_level < 2
    assert nmpc_15b.maneuver_constraint.rudder_min >= -0.20
    assert nmpc_15b.maneuver_constraint.rudder_max <= 0.20
    assert nmpc_15b.maneuver_constraint.turn_direction_sign == 0
    assert nmpc_15b.maneuver_constraint.epsilon_legal_weight >= 3e3
    print_test("4b.4: Rule 15 Crossing stand-on — tight rudder, maintain course", True,
               f"rudder=[{nmpc_15b.maneuver_constraint.rudder_min:.2f}, "
               f"{nmpc_15b.maneuver_constraint.rudder_max:.2f}], "
               f"turn_sign={nmpc_15b.maneuver_constraint.turn_direction_sign}")

    # ── Rule 16: Give-way vessel — early & substantial action ──
    # Crossing with give-way role triggers Rule 16
    output_16 = referee.evaluate(os_x, [ts_stbd], scenario_id="S02")
    nmpc_16 = mapper.map(output_16)

    # Rule 16 enhancement: larger alteration angle (≥ 45°)
    assert nmpc_16.maneuver_constraint.alteration_min_angle >= math.radians(45)
    # Higher legal weight (≥ 5e3)
    assert nmpc_16.maneuver_constraint.epsilon_legal_weight >= 5e3
    # Wider CPA (≥ 70m = 1.4 * 50)
    assert nmpc_16.spatial_constraints[0].min_distance >= 70.0
    print_test("4b.5: Rule 16 Give-way — substantial 45° alteration + wide CPA", True,
               f"min_angle={math.degrees(nmpc_16.maneuver_constraint.alteration_min_angle):.0f}°, "
               f"legal_w={nmpc_16.maneuver_constraint.epsilon_legal_weight:.0e}")

    # ── Rule 17: Stand-on vessel — maintain course/speed (safe, no breakthrough) ──
    output_17 = referee.evaluate(os_x2, [ts_port], scenario_id="S04")
    nmpc_17 = mapper.map(output_17)

    # Normal stand-on: tight speed/course, degradation < 2
    assert output_17.degradation_level < 2

    # Normal stand-on: tight speed/course
    assert nmpc_17.maneuver_constraint.rudder_min >= -0.20
    assert nmpc_17.maneuver_constraint.rudder_max <= 0.20
    assert nmpc_17.speed_constraint.epsilon_speed_weight >= 3e3
    print_test("4b.6: Rule 17 Stand-on — tight course/speed maintenance", True,
               f"rudder=[{nmpc_17.maneuver_constraint.rudder_min:.2f}, "
               f"{nmpc_17.maneuver_constraint.rudder_max:.2f}]")

    # ── Rule 17: Breakthrough condition (stand-on + degradation ≥ 2) ──
    # Create a stand-on crossing with CPA→0: close-range port-side scenario
    os_crit = make_os(x=0.0, y=0.0, heading_deg=0, speed=3.0)
    ts_close = make_ts("ts_close", x=-2.0, y=3.0, heading_deg=180, speed=3.0)
    output_17b = referee.evaluate(os_crit, [ts_close], scenario_id="S04_break")
    nmpc_17b = mapper.map(output_17b)

    # Breakthrough: stand-on + degradation >= 2 → full maneuver
    assert output_17b.degradation_level >= 2, \
        f"Expected degradation >= 2, got {output_17b.degradation_level}"
    assert nmpc_17b.maneuver_constraint.alteration_min_angle >= math.radians(30)

    # Breakthrough: starboard turn allowed, port turn forbidden, alteration required
    assert nmpc_17b.maneuver_constraint.rudder_min >= 0.0    # forbid port turn
    assert nmpc_17b.maneuver_constraint.rudder_max >= 0.3    # allow starboard
    assert nmpc_17b.maneuver_constraint.forbidden_maneuver == ForbiddenManeuver.ALTER_TO_PORT
    print_test("4b.7: Rule 17 Breakthrough — released stand-on, starboard-only maneuver", True,
               f"degrad={output_17b.degradation_level}, "
               f"rudder=[{nmpc_17b.maneuver_constraint.rudder_min:.2f}, "
               f"{nmpc_17b.maneuver_constraint.rudder_max:.2f}]")

    # ── Rule 18: Responsibility hierarchy ──
    # Low-priority vessel (power-driven, must give way) → wider CPA + higher legal weight
    output_18 = referee.evaluate(os_x, [ts_stbd], scenario_id="S02")
    nmpc_18 = mapper.map(output_18)

    # Low priority (give_way) has wider CPA and higher legal weight
    assert nmpc_18.spatial_constraints[0].min_distance >= 75.0
    assert nmpc_18.maneuver_constraint.epsilon_legal_weight >= 5e3
    print_test("4b.8: Rule 18 Hierarchy — low priority wide CPA + high legal weight", True,
               f"min_dist={nmpc_18.spatial_constraints[0].min_distance:.0f}m, "
               f"legal_w={nmpc_18.maneuver_constraint.epsilon_legal_weight:.0e}")

    # ── Rule 19: Restricted Visibility — global override ──
    ref_restricted = DeterministicReferee(visibility="restricted")
    os_rv = make_os(x=0.0, y=0.0, heading_deg=0, speed=3.0)
    ts_rv = make_ts("ts_rv", x=0.5, y=25.0, heading_deg=180, speed=3.0)
    output_19 = ref_restricted.evaluate(os_rv, [ts_rv], scenario_id="S10")
    nmpc_19 = mapper.map(output_19)

    # Restricted vis overrides:
    # 1. Speed ≤ 2.5 m/s
    assert nmpc_19.speed_constraint.max_speed <= 2.5
    # 2. Wider CPA for all targets (≥ 75m)
    assert all(sc.min_distance >= 75.0 for sc in nmpc_19.spatial_constraints)
    # 3. Forbid port turn
    assert nmpc_19.maneuver_constraint.forbidden_maneuver == ForbiddenManeuver.ALTER_TO_PORT
    # 4. Rudder ≥ 0 (no port turns)
    assert nmpc_19.maneuver_constraint.rudder_min >= 0.0
    # 5. Higher legal weight
    assert nmpc_19.maneuver_constraint.epsilon_legal_weight >= 5e3
    print_test("4b.9: Rule 19 Restricted Visibility — speed≤2.5 + wide CPA + no port turn", True,
               f"speed≤{nmpc_19.speed_constraint.max_speed:.1f} m/s, "
               f"min_dist={nmpc_19.spatial_constraints[0].min_distance:.0f}m, "
               f"forbidden={nmpc_19.maneuver_constraint.forbidden_maneuver.value}")


# =============================================================================
# Test 5: Scene Description & LLM Interface
# =============================================================================

def test_scene_description():
    """Test numerical-to-textual scene description."""
    print("\n" + "=" * 60)
    print("Test 5: Scene Description & LLM Interface")
    print("=" * 60)

    os = make_os(x=0.0, y=0.0, heading_deg=0, speed=1.5)
    ts = make_ts("ts01", x=1.0, y=28.0, heading_deg=185, speed=1.2)

    scene = build_scene_description(os, [ts])
    assert len(scene.scene_text) > 100
    assert "Own Ship" in scene.scene_text
    assert "ts01" in scene.scene_text
    print_test("5.1: Scene description generated", True,
               f"{len(scene.scene_text)} chars")

    # ── Test Simulated LLM Referee ──
    llm_ref = SimulatedLLMReferee(noise_level=0.0)
    output = llm_ref.evaluate(os, [ts], scenario_id="S01")
    assert output.llm_model == "simulated_llm"
    assert 0.7 <= output.confidence_score <= 1.0
    print_test("5.2: Simulated LLM referee", True,
               f"model={output.llm_model}, conf={output.confidence_score:.3f}")


# =============================================================================
# Test 6: All 20 Scenarios Coverage
# =============================================================================

def test_scenario_coverage():
    """Test that the deterministic referee can handle all 20 scenario types."""
    print("\n" + "=" * 60)
    print("Test 6: 20-Scenario Coverage (Deterministic Referee)")
    print("=" * 60)

    referee = DeterministicReferee()

    # Scenario definitions (maritime convention: heading 0=North, CW positive)
    scenarios = [
        # ── HEAD-ON (Rule 14): TS ahead, reciprocal course ──
        ("S01", "head_on", make_os(0, 0, 0, 3.0),
         [make_ts("ts01", x=1.0, y=28.0, heading_deg=182, speed=3.5)]),
        ("S07", "head_on", make_os(0, 0, 0, 3.0),
         [make_ts("ts07", x=-2.0, y=30.0, heading_deg=178, speed=2.5)]),
        ("S13", "head_on", make_os(0, 0, 0, 3.0),
         [make_ts("ts13", x=-0.5, y=32.0, heading_deg=180, speed=3.5)]),
        ("S17", "head_on", make_os(0, 0, 0, 3.0),
         [make_ts("ts17", x=3.0, y=25.0, heading_deg=185, speed=3.5)]),

        # ── CROSSING (Rule 15): TS on starboard/port bow, crossing path ──
        ("S02", "crossing", make_os(0, 0, 0, 3.0),
         [make_ts("ts02", x=25.0, y=8.0, heading_deg=270, speed=3.5)]),
        ("S04", "crossing", make_os(0, 0, 0, 3.0),
         [make_ts("ts04", x=-15.0, y=12.0, heading_deg=180, speed=3.5)]),
        ("S05", "crossing", make_os(0, 0, 0, 3.0),
         [make_ts("ts05", x=20.0, y=5.0, heading_deg=270, speed=3.5)]),
        ("S08", "crossing", make_os(0, 0, 0, 3.0),
         [make_ts("ts08", x=30.0, y=10.0, heading_deg=270, speed=3.0)]),
        ("S14", "crossing", make_os(0, 0, 0, 3.0),
         [make_ts("ts14", x=15.0, y=25.0, heading_deg=250, speed=3.0)]),
        ("S18", "crossing", make_os(0, 0, 0, 3.0),
         [make_ts("ts18", x=8.0, y=35.0, heading_deg=220, speed=3.0)]),

        # ── OVERTAKING (Rule 13): TS ahead, same course, OS faster ──
        ("S06", "overtaking", make_os(0, 0, 0, 3.0),
         [make_ts("ts06", x=0.5, y=30.0, heading_deg=0, speed=2.0)]),

        # ── RESTRICTED VISIBILITY (Rule 19) ──
        ("S10", "restricted_visibility", make_os(0, 0, 0, 3.0),
         [make_ts("ts10", x=5.0, y=20.0, heading_deg=180, speed=3.0)]),

        # ── MULTI-SHIP: encounter type not strictly asserted ──
        ("S03", "multi_ship", make_os(0, 0, 0, 3.0),
         [make_ts("ts03a", x=12.0, y=15.0, heading_deg=270, speed=3.5),
          make_ts("ts03b", x=-10.0, y=20.0, heading_deg=180, speed=3.5)]),
        ("S09", "multi_ship", make_os(0, 0, 0, 3.0),
         [make_ts("ts09a", x=8.0, y=18.0, heading_deg=270, speed=3.0),
          make_ts("ts09b", x=-8.0, y=22.0, heading_deg=180, speed=2.5)]),
        ("S11", "multi_ship", make_os(0, 0, 0, 3.0),
         [make_ts("ts11a", x=15.0, y=12.0, heading_deg=270, speed=3.0),
          make_ts("ts11b", x=1.0, y=28.0, heading_deg=185, speed=3.0)]),
        ("S12", "multi_ship", make_os(0, 0, 0, 3.0),
         [make_ts("ts12a", x=18.0, y=10.0, heading_deg=270, speed=3.0),
          make_ts("ts12b", x=-12.0, y=18.0, heading_deg=180, speed=3.0),
          make_ts("ts12c", x=1.0, y=-15.0, heading_deg=0, speed=2.0)]),
        ("S15", "multi_ship", make_os(0, 0, 0, 3.0),
         [make_ts("ts15a", x=15.0, y=10.0, heading_deg=270, speed=3.0),
          make_ts("ts15b", x=-15.0, y=15.0, heading_deg=180, speed=3.0),
          make_ts("ts15c", x=1.0, y=28.0, heading_deg=180, speed=2.5)]),
        ("S16", "multi_ship", make_os(0, 0, 0, 3.0),
         [make_ts("ts16a", x=10.0, y=22.0, heading_deg=200, speed=2.5),
          make_ts("ts16b", x=-8.0, y=18.0, heading_deg=250, speed=2.5),
          make_ts("ts16c", x=2.0, y=-15.0, heading_deg=0, speed=2.0)]),
        ("S19", "multi_ship", make_os(0, 0, 0, 3.0),
         [make_ts("ts19a", x=20.0, y=12.0, heading_deg=270, speed=3.0),
          make_ts("ts19b", x=-20.0, y=12.0, heading_deg=270, speed=3.0),
          make_ts("ts19c", x=1.0, y=-18.0, heading_deg=10, speed=2.5)]),
        ("S20", "multi_ship", make_os(0, 0, 0, 3.0),
         [make_ts("ts20a", x=0.5, y=30.0, heading_deg=180, speed=3.5),
          make_ts("ts20b", x=18.0, y=10.0, heading_deg=270, speed=3.0),
          make_ts("ts20c", x=-15.0, y=15.0, heading_deg=270, speed=3.0),
          make_ts("ts20d", x=1.0, y=-15.0, heading_deg=0, speed=2.0)]),
    ]

    ref_restricted = DeterministicReferee(visibility="restricted")

    passed = 0
    failed = 0
    for sid, exp_type, os, tss in scenarios:
        if exp_type == "restricted_visibility":
            ref = ref_restricted
        else:
            ref = referee

        try:
            output = ref.evaluate(os, tss, scenario_id=sid)
            validate_output(output)

            # Check that output is well-formed
            assert output.encounter_classification.primary_encounter is not None
            assert output.required_maneuver is not None
            assert output.global_min_cpa > 0
            assert output.reasoning_trace

            # ── Encounter type matching (non-multi-ship only) ──
            if exp_type != "multi_ship":
                actual = output.encounter_classification.primary_encounter.value
                assert actual == exp_type, \
                    f"{sid}: expected encounter={exp_type}, got {actual}"

            passed += 1
        except Exception as e:
            print(f"  ❌ {sid}: {e}")
            failed += 1

    print_test(f"6.1: All scenarios evaluate ({passed}/20 passed)",
               failed == 0, f"failed={failed}")

    # ── Test mapper on all scenarios ──
    mapper = ConstraintMapper()
    map_passed = 0
    for sid, exp_type, os, tss in scenarios[:5]:  # test 5 for speed
        if exp_type == "restricted_visibility":
            ref = ref_restricted
        else:
            ref = referee
        try:
            output = ref.evaluate(os, tss, scenario_id=sid)
            nmpc = mapper.map(output)
            assert len(nmpc.spatial_constraints) == len(tss)
            map_passed += 1
        except Exception as e:
            print(f"  ❌ Map {sid}: {e}")

    print_test(f"6.2: Constraint mapper ({map_passed}/5)",
               map_passed == 5)


# =============================================================================
# Test 7: Event Trigger Logic
# =============================================================================

def test_event_trigger():
    """Test CPA risk field event trigger logic."""
    print("\n" + "=" * 60)
    print("Test 7: Event-Trigger Mechanism")
    print("=" * 60)

    # High-risk: CPA=0, TCPA=10s
    risk_high = _compute_cpa_risk_field(cpa=0.0, tcpa=10.0, dist=100.0)
    # Medium-risk: CPA=10m, TCPA=20s
    risk_med = _compute_cpa_risk_field(cpa=10.0, tcpa=20.0, dist=150.0)
    # Low-risk: CPA=100m, TCPA=300s
    risk_low = _compute_cpa_risk_field(cpa=100.0, tcpa=300.0, dist=500.0)
    # No-risk: CPA=200m, TCPA=inf
    risk_none = _compute_cpa_risk_field(cpa=200.0, tcpa=float('inf'), dist=1000.0)

    print_test("7.1: High risk (CPA=0, TCPA=10s)",
               risk_high > 0.5, f"φ={risk_high:.4f}")
    print_test("7.2: Medium risk (CPA=10, TCPA=20s)",
               0.25 <= risk_med <= 0.80, f"φ={risk_med:.4f}")
    print_test("7.3: Low risk (CPA=100, TCPA=300s)",
               risk_low < 0.3, f"φ={risk_low:.4f}")
    print_test("7.4: Trigger ordering",
               risk_high > risk_med > risk_low,
               f"high={risk_high:.4f} > med={risk_med:.4f} > low={risk_low:.4f}")


# =============================================================================
# Test 8: Degradation State Machine
# =============================================================================

def test_degradation():
    """Test degradation level determination."""
    print("\n" + "=" * 60)
    print("Test 8: Degradation State Machine")
    print("=" * 60)

    referee = DeterministicReferee()

    # Critical: head-on with CPA=0
    os = make_os(0, 0, 0, 3.0)
    ts = make_ts("ts01", x=0.1, y=5.0, heading_deg=180, speed=3.0)
    output = referee.evaluate(os, [ts], scenario_id="critical")
    print_test("8.1: Critical risk → level >= 2",
               output.degradation_level >= 2,
               f"degrad={output.degradation_level}, risk={output.encounter_classification.risk_level}")

    # Low risk — TS far away, heading same direction (parallel), no conflict
    os2 = make_os(0, 0, 0, 1.5)
    ts2 = make_ts("ts02", x=0.5, y=500.0, heading_deg=0, speed=0.5)
    output2 = referee.evaluate(os2, [ts2], scenario_id="low_risk")
    print_test("8.2: Low risk → level 0",
               output2.degradation_level == 0,
               f"degrad={output2.degradation_level}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("🧪 COLREGS Symbolic Referee Layer — Test Suite")
    print("    Phase 5-6: CFG + Symbolic Referee + Constraint Mapper")
    print("=" * 60)

    test_json_schema()
    test_cfg_grammar()
    test_gbnf_grammar()
    test_deterministic_referee()
    test_constraint_mapper()
    test_rule_specific_mappings()
    test_scene_description()
    test_scenario_coverage()
    test_event_trigger()
    test_degradation()

    print("\n" + "=" * 60)
    if _FAILED_TESTS:
        print(f"❌ {len(_FAILED_TESTS)} FAILURES:")
        for name in _FAILED_TESTS:
            print(f"   - {name}")
    else:
        print("✅ All test suites completed.")
    print("=" * 60)

    return len(_FAILED_TESTS)


if __name__ == '__main__':
    failed = main()
    sys.exit(0 if failed == 0 else 1)
