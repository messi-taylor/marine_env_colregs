"""
COLREGS Symbolic Referee Layer — Phase 5-6 Implementation
==========================================================

Dual-Loop Disentangled Neuro-Symbolic Control Architecture:
  - Symbolic Referee Layer (this package): LLM-based COLREGS interpretation
    with CFG-constrained structured output
  - Numerical Control Layer: Resilient NMPC with recursive feasibility guarantees

Subpackages:
  - output_schema:    JSON schema + data classes for structured referee output
  - cfg_grammar:      Lark CFG grammar (L_colregs) + GBNF grammar (llama.cpp)
  - scene_descriptor: Φ operator — numerical state → natural language (Phase 3-4)
  - deterministic_referee: Rule-based fallback referee (no LLM required)
  - llm_referee:      Abstract LLM interface + Ollama/Anthropic/Simulated/Grammar backends
  - constraint_mapper: Symbolic-to-numeric constraint mapping for NMPC (9 rules)
  - referee_node:     ROS 2 node with event-triggered activation
"""

from .output_schema import (
    COLREGSConstraintOutput,
    ColregsRuleInterpretation,
    SpatialConstraint,
    ManeuverConstraint,
    SpeedConstraint,
    EncounterClassification,
)
from .scene_descriptor import SceneDescription, build_scene_description
from .cfg_grammar import COLREGSGrammar, validate_cfg_output
from .deterministic_referee import DeterministicReferee
from .constraint_mapper import ConstraintMapper, NMPCConstraints
from .colregs_gbnf import (
    COLREGS_GBNF_GRAMMAR,
    COLREGS_GBNF_GRAMMAR_SIMPLE,
    validate_gbnf_grammar,
    verify_gbnf_covers_cfg,
)

__all__ = [
    "COLREGSConstraintOutput",
    "ColregsRuleInterpretation",
    "SpatialConstraint",
    "ManeuverConstraint",
    "SpeedConstraint",
    "EncounterClassification",
    "SceneDescription",
    "build_scene_description",
    "COLREGSGrammar",
    "validate_cfg_output",
    "DeterministicReferee",
    "ConstraintMapper",
    "NMPCConstraints",
    "COLREGS_GBNF_GRAMMAR",
    "COLREGS_GBNF_GRAMMAR_SIMPLE",
    "validate_gbnf_grammar",
    "verify_gbnf_covers_cfg",
]
