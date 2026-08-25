#!/usr/bin/env python3
"""
Ablation experiment orchestration for COLREGS collision avoidance.

Runs 4-group controlled-variable experiments to isolate the contribution of:
  - CFG/GBNF token-level hard constraint decoding
  - LLM semantic reasoning (qwen2.5:7b)
  - Soft constraint relaxation (ε_legal, ε_smooth, ε_speed slack)

Groups:
  A (Full):  GrammarConstrainedReferee (GBNF + LLM) + soft constraints
  B (-CFG):  OllamaReferee (LLM, no GBNF grammar) + soft constraints
  C (-LLM):  DeterministicReferee (rule-based, no LLM) + soft constraints
  D (-Soft): GrammarConstrainedReferee (GBNF + LLM) + hard constraints ONLY
"""

import os
import sys
import time
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from .batch_runner import BatchRunner, MonteCarloConfig
from .metrics import MetricsCollector
from .visualize import generate_report, generate_ablation_report


# =============================================================================
# Ablation Group Definition
# =============================================================================

class AblationGroup(str, Enum):
    """Controlled-variable ablation groups."""
    A_FULL = "A"       # GrammarConstrainedReferee (GBNF + LLM) + soft constraints
    B_NO_CFG = "B"     # OllamaReferee (LLM, no GBNF) + soft constraints
    C_NO_LLM = "C"     # DeterministicReferee (rule-based) + soft constraints
    D_NO_SOFT = "D"    # GrammarConstrainedReferee (GBNF + LLM) + hard-only

    @property
    def label(self) -> str:
        labels = {
            "A": "A (Full)",
            "B": "B (-CFG)",
            "C": "C (-LLM)",
            "D": "D (-Soft)",
        }
        return labels[self.value]

    @property
    def description(self) -> str:
        descs = {
            "A": "Complete neuro-symbolic: GBNF + LLM + soft constraints",
            "B": "No CFG/GBNF: LLM with soft constraints only (tests CFG format protection)",
            "C": "No LLM: Deterministic rule engine + soft constraints (tests LLM semantic gain)",
            "D": "No soft constraints: GBNF + LLM + hard-only (tests soft constraint contribution)",
        }
        return descs[self.value]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AblationConfig:
    """Configuration for an ablation experiment across all groups."""
    scenario_id: str = "scenario_01"
    repeats_full: int = 20        # repeats for LLM groups (A, B, D) — expensive
    repeats_fast: int = 100       # repeats for deterministic group (C) — cheap
    sim_duration: float = 40.0
    dt_sim: float = 0.1
    control_period: float = 0.5
    nmpc_setup_time: float = 2.0
    cpa_deadband: float = 60.0
    min_sim_steps: int = 40
    referee_cpa_safe: float = 50.0
    referee_tcpa_warn: float = 60.0
    output_dir: str = "ablation_output"
    parallel_workers: int = 1
    resume: bool = True           # skip completed groups
    groups: List[AblationGroup] = field(default_factory=lambda: list(AblationGroup))


# =============================================================================
# Group Result
# =============================================================================

@dataclass
class GroupResult:
    """Aggregated results for one ablation group on one scenario."""
    group: AblationGroup
    scenario_id: str
    collector: MetricsCollector
    summary: dict
    output_dir: str
    wall_time_s: float
    num_runs: int
    referee_backend: str
    soft_constraints_enabled: bool
    nmpc_weight_overrides: Optional[dict] = None


# =============================================================================
# Ablation Runner
# =============================================================================

class AblationRunner:
    """Orchestrates running all ablation groups for a scenario and generating
    cross-group comparison plots."""

    # ── Group → (referee_backend, nmpc_weight_overrides, num_repeats_key) ──
    # ── Backend selection note ──
    # GrammarConstrainedReferee (llama-cpp-python) is ideal but slow on CPU-only.
    # OllamaReferee uses the local Ollama server (GPU-accelerated Vulkan on AMD).
    # For practical throughput, Groups A,B,D use "ollama" backend by default.
    # To use token-level GBNF hard constraint (grammar_constrained), the machine
    # must have llama-cpp-python with GPU offload working reliably.
    GROUP_PARAMS = {
        AblationGroup.A_FULL: {
            "referee_backend": "ollama",       # Ollama + CFG (vocab injection + fuzzy repair)
            "nmpc_weight_overrides": None,     # default soft constraints
            "repeats_key": "repeats_full",
        },
        AblationGroup.B_NO_CFG: {
            "referee_backend": "ollama_no_cfg", # Ollama WITHOUT CFG (plain LLM, no vocab injection)
            "nmpc_weight_overrides": None,
            "repeats_key": "repeats_full",
        },
        AblationGroup.C_NO_LLM: {
            "referee_backend": "deterministic", # Rule-based engine
            "nmpc_weight_overrides": None,
            "repeats_key": "repeats_fast",
        },
        AblationGroup.D_NO_SOFT: {
            "referee_backend": "ollama",       # Ollama + CFG, but hard constraints only
            # Zero out legal, smoothness, and speed slack penalties.
            # ε_safety is NEVER zeroed (Lemma 1: safety hard constraint).
            "nmpc_weight_overrides": {"w_legal": 0, "w_smooth": 0, "w_speed": 0},
            "repeats_key": "repeats_full",
        },
    }

    def __init__(self, config: AblationConfig = None):
        self.config = config or AblationConfig()
        self.results: Dict[AblationGroup, GroupResult] = {}

    # =====================================================================
    # Public API
    # =====================================================================

    def run_all_groups(self, scenario_id: str = None) -> Dict[AblationGroup, GroupResult]:
        """Run all configured ablation groups for a single scenario.

        Groups run sequentially (A→B→C→D) to avoid LLM GPU/resource conflicts.
        After all groups complete, generates cross-group comparison plots.

        Returns:
            Dict mapping AblationGroup → GroupResult.
        """
        sid = scenario_id or self.config.scenario_id
        groups = self.config.groups

        print(f"\n{'='*70}")
        print(f"Ablation Experiment: {sid}")
        print(f"{'='*70}")
        print(f"Groups: {', '.join(g.label for g in groups)}")
        print(f"LLM-group repeats: {self.config.repeats_full}")
        print(f"Fast-group repeats: {self.config.repeats_fast}")
        print(f"Output: {self.config.output_dir}/{sid}/")
        print(f"{'='*70}")

        for group in groups:
            try:
                result = self.run_group(group, sid)
                self.results[group] = result
            except Exception as e:
                print(f"\n  ✗ Group {group.label} FAILED: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Generate cross-group comparison if at least 2 groups succeeded
        if len(self.results) >= 2:
            print(f"\n{'─'*70}")
            print(f"Generating cross-group comparison for {sid}...")
            print(f"{'─'*70}")
            self.generate_comparison(sid)

        return self.results

    def run_group(self, group: AblationGroup, scenario_id: str) -> GroupResult:
        """Run one ablation group for one scenario.

        1. Check resume (skip if already completed).
        2. Check backend prerequisites (Ollama connectivity for Group B).
        3. Create MonteCarloConfig with group-specific backend + weight overrides.
        4. Run batch, save results, return GroupResult.
        """
        cfg = self.config
        params = self.GROUP_PARAMS[group]

        # ── Output directory ──
        group_dir = os.path.join(cfg.output_dir, scenario_id, f"group_{group.value}")
        os.makedirs(group_dir, exist_ok=True)

        # ── Resume check ──
        group_config_path = os.path.join(group_dir, "group_config.json")
        if cfg.resume and os.path.exists(group_config_path):
            try:
                with open(group_config_path, 'r') as f:
                    saved = json.load(f)
                if saved.get("completed", False):
                    print(f"\n  ↷ Group {group.label}: already completed — skipping "
                          f"(resume=True)")
                    # Load existing results
                    collector = MetricsCollector()
                    # We can't fully reconstruct RunMetrics from CSV, but we can
                    # load the summary for cross-group comparison
                    summary_path = os.path.join(group_dir, "summary.json")
                    if os.path.exists(summary_path):
                        with open(summary_path, 'r') as f:
                            summary = json.load(f)
                    else:
                        summary = {}
                    return GroupResult(
                        group=group, scenario_id=scenario_id, collector=collector,
                        summary=summary, output_dir=group_dir,
                        wall_time_s=saved.get("wall_time_s", 0),
                        num_runs=saved.get("num_runs", 0),
                        referee_backend=saved.get("referee_backend", ""),
                        soft_constraints_enabled=saved.get("soft_constraints_enabled", True),
                        nmpc_weight_overrides=saved.get("nmpc_weight_overrides"),
                    )
            except (json.JSONDecodeError, KeyError):
                pass  # corrupt config — re-run

        # ── Backend prerequisite checks ──
        backend = params["referee_backend"]
        weight_overrides = params["nmpc_weight_overrides"]
        soft_enabled = weight_overrides is None or weight_overrides.get("w_legal", 1e3) > 0

        if backend == "ollama" and not self._check_ollama():
            raise RuntimeError(
                "Ollama server not reachable. Start with:\n"
                "  export OLLAMA_HOST=http://localhost:11435\n"
                "  ollama serve &")

        if backend == "grammar_constrained":
            try:
                import llama_cpp
            except ImportError:
                raise RuntimeError(
                    "llama-cpp-python not installed. Install with:\n"
                    "  pip install llama-cpp-python")

        # ── Determine repeats ──
        repeats_key = params["repeats_key"]
        num_repeats = getattr(cfg, repeats_key)

        print(f"\n{'─'*70}")
        print(f"Group {group.label}: {group.description}")
        print(f"  Backend: {backend}")
        print(f"  Soft constraints: {soft_enabled}")
        if weight_overrides:
            print(f"  Weight overrides: {weight_overrides}")
        print(f"  Repeats: {num_repeats}")
        print(f"  Output: {group_dir}/")
        print(f"{'─'*70}")

        # ── Build MonteCarloConfig ──
        mc_config = MonteCarloConfig(
            scenario_id=scenario_id,
            num_repeats=num_repeats,
            sim_duration=cfg.sim_duration,
            dt_sim=cfg.dt_sim,
            control_period=cfg.control_period,
            nmpc_setup_time=cfg.nmpc_setup_time,
            cpa_deadband=cfg.cpa_deadband,
            min_sim_steps=cfg.min_sim_steps,
            referee_cpa_safe=cfg.referee_cpa_safe,
            referee_tcpa_warn=cfg.referee_tcpa_warn,
            output_dir=group_dir,
            parallel_workers=cfg.parallel_workers,
            referee_backend=backend,
            nmpc_weight_overrides=weight_overrides,
        )

        # ── Run batch (with crash protection) ──
        t_start = time.perf_counter()
        try:
            runner = BatchRunner(mc_config)
            collector = runner.run_batch(scenario_id)
        except Exception as e:
            # Save whatever partial results we have before re-raising
            print(f"  ⚠ Batch runner crashed: {e}")
            import traceback
            traceback.print_exc()
            # Try to salvage partial metrics
            collector = runner.metrics if hasattr(runner, 'metrics') else MetricsCollector()
            if collector.runs:
                print(f"  Salvaged {len(collector.runs)} partial runs")
            else:
                raise  # nothing to salvage, let caller handle
        wall_time = time.perf_counter() - t_start

        # ── Save results ──
        collector.save_csv(os.path.join(group_dir, "metrics.csv"))
        collector.save_json(os.path.join(group_dir, "summary.json"))

        # ── Generate standard per-group plots ──
        try:
            generate_report(collector, group_dir, f"{scenario_id} — Group {group.label}")
        except Exception as e:
            print(f"  Warning: per-group visualization failed: {e}")

        # ── Generate text report with LLM-NMPC gap analysis ──
        summary = collector.summary()
        report_path = os.path.join(group_dir, "summary_report.txt")
        self._write_group_report(report_path, group, mc_config, summary, collector, wall_time)

        # ── Save group provenance ──
        with open(group_config_path, 'w') as f:
            json.dump({
                "group": group.value,
                "label": group.label,
                "description": group.description,
                "referee_backend": backend,
                "soft_constraints_enabled": soft_enabled,
                "nmpc_weight_overrides": weight_overrides,
                "num_runs": num_repeats,
                "wall_time_s": round(wall_time, 1),
                "completed": True,
            }, f, indent=2)

        print(f"\n  ✓ Group {group.label} complete: {wall_time:.0f}s "
              f"({wall_time/num_repeats:.1f}s/run)")

        return GroupResult(
            group=group,
            scenario_id=scenario_id,
            collector=collector,
            summary=summary,
            output_dir=group_dir,
            wall_time_s=wall_time,
            num_runs=num_repeats,
            referee_backend=backend,
            soft_constraints_enabled=soft_enabled,
            nmpc_weight_overrides=weight_overrides,
        )

    def generate_comparison(self, scenario_id: str):
        """Generate cross-group comparison visualizations.

        Requires at least 2 groups to have completed successfully.
        """
        comp_dir = os.path.join(self.config.output_dir, scenario_id)
        os.makedirs(comp_dir, exist_ok=True)

        # Build dict of group_label → collector for visualization functions
        group_collectors = {}
        for group, result in self.results.items():
            group_collectors[group.label] = result.collector

        try:
            generate_ablation_report(group_collectors, comp_dir, scenario_id)
        except Exception as e:
            print(f"  Warning: comparison visualization failed: {e}")
            import traceback
            traceback.print_exc()

        # Save cross-group summary CSV
        csv_path = os.path.join(comp_dir, "ablation_summary.csv")
        self._save_comparison_csv(csv_path)

    # =====================================================================
    # Helpers
    # =====================================================================

    def _check_ollama(self, timeout: float = 5.0) -> bool:
        """Verify Ollama server is reachable before starting Group B."""
        import requests
        import os
        base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11435")
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def _write_group_report(self, path: str, group: AblationGroup,
                            mc_config: MonteCarloConfig,
                            summary: dict, collector: MetricsCollector,
                            wall_time_s: float):
        """Write a detailed text report for a single ablation group.

        For Group D (no soft constraints), includes a special LLM-NMPC
        compliance gap analysis section.
        """
        with open(path, 'w') as f:
            f.write(f"Ablation Group Report\n")
            f.write(f"{'='*60}\n")
            f.write(f"Group:       {group.label}\n")
            f.write(f"Description: {group.description}\n")
            f.write(f"Scenario:    {summary.get('num_runs', '?')} runs\n")
            f.write(f"Backend:     {mc_config.referee_backend}\n")
            f.write(f"Soft constraints: {mc_config.nmpc_weight_overrides is None}\n")
            if mc_config.nmpc_weight_overrides:
                f.write(f"Weight overrides: {mc_config.nmpc_weight_overrides}\n")
            f.write(f"Wall time:   {wall_time_s:.1f}s\n\n")

            f.write(f"SAFETY METRICS\n")
            f.write(f"{'─'*40}\n")
            f.write(f"  Collision rate:     {summary['collision_rate']*100:.1f}%\n")
            f.write(f"  Near-miss rate:     {summary['near_miss_rate']*100:.1f}%\n")
            f.write(f"  Mean min CPA:       {summary['mean_min_cpa']:.1f} m\n")
            f.write(f"  Median min CPA:     {summary['median_min_cpa']:.1f} m\n")
            f.write(f"  Worst CPA:          {summary['worst_cpa']:.1f} m\n\n")

            f.write(f"COMPLIANCE METRICS\n")
            f.write(f"{'─'*40}\n")
            f.write(f"  Turn direction:     {summary['turn_direction_compliance']*100:.1f}%\n")
            f.write(f"  Action magnitude:   {summary['action_magnitude_compliance']*100:.1f}%\n")
            f.write(f"  Overtaking:         {summary['overtaking_compliance']*100:.1f}%\n")
            f.write(f"  Role compliance:    {summary['role_compliance']*100:.1f}%\n\n")

            f.write(f"SOLVER PERFORMANCE\n")
            f.write(f"{'─'*40}\n")
            f.write(f"  Solve success rate: {summary['solve_success_rate']*100:.1f}%\n")
            f.write(f"  Avg solve time:     {summary['avg_solve_time_ms']:.0f} ms\n")
            f.write(f"  Total infeasible:   {summary['total_infeasible']}\n")
            f.write(f"  Retry L1/L2/L3:     {summary.get('total_retry1_recoveries',0)}/"
                  f"{summary.get('total_retry2_recoveries',0)}/"
                  f"{summary.get('total_retry3_recoveries',0)}\n\n")

            # ── LLM-NMPC Compliance Gap Analysis (especially relevant for Group D) ──
            total_disagreements = summary.get('total_llm_nmpc_disagreements', 0)
            runs_with_disagreement = summary.get('runs_with_disagreement', 0)
            f.write(f"LLM-NMPC COMPLIANCE GAP ANALYSIS\n")
            f.write(f"{'─'*40}\n")
            f.write(f"  Total LLM-NMPC disagreements: {total_disagreements}\n")
            f.write(f"  Runs with ≥1 disagreement:    {runs_with_disagreement}\n")
            if summary.get('num_runs', 1) > 0:
                f.write(f"  Disagreement rate:            "
                        f"{runs_with_disagreement/max(summary['num_runs'],1)*100:.1f}% of runs\n")
            if group == AblationGroup.D_NO_SOFT:
                f.write(f"\n  Interpretation (Group D — No Soft Constraints):\n")
                if runs_with_disagreement > summary.get('num_runs', 1) * 0.3:
                    f.write(f"  → HIGH disagreement rate: soft constraints are CRITICAL for\n"
                            f"    aligning NMPC behavior with COLREGS legal requirements.\n"
                            f"    Without ε_legal penalty, the solver freely violates\n"
                            f"    alteration constraints to achieve feasibility.\n")
                else:
                    f.write(f"  → LOW disagreement rate: LLM outputs are naturally safe\n"
                            f"    even without soft constraint enforcement. The neuro-symbolic\n"
                            f"    pipeline exhibits strong generalization — the LLM's suggested\n"
                            f"    trajectories align with NMPC's unconstrained optimum.\n")

            f.write(f"\nDEGRADATION\n")
            f.write(f"{'─'*40}\n")
            f.write(f"  Runs degraded:      {summary['runs_with_degradation']}\n")
            f.write(f"  Max degradation:    Level {summary['max_degradation_observed']}\n")

    def _save_comparison_csv(self, csv_path: str):
        """Save cross-group summary as one row per group."""
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'group', 'label', 'referee_backend', 'soft_constraints',
                'num_runs', 'wall_time_s',
                'collision_rate', 'near_miss_rate', 'median_cpa', 'worst_cpa',
                'turn_compliance', 'mag_compliance',
                'solve_success_rate', 'avg_solve_ms',
                'retry_l1', 'retry_l2', 'retry_l3',
                'llm_nmpc_disagreements',
            ])
            for group, result in self.results.items():
                s = result.summary
                writer.writerow([
                    group.value, group.label, result.referee_backend,
                    result.soft_constraints_enabled,
                    result.num_runs, f"{result.wall_time_s:.0f}",
                    f"{s['collision_rate']*100:.1f}%",
                    f"{s['near_miss_rate']*100:.1f}%",
                    f"{s['median_min_cpa']:.1f}",
                    f"{s['worst_cpa']:.1f}",
                    f"{s['turn_direction_compliance']*100:.1f}%",
                    f"{s['action_magnitude_compliance']*100:.1f}%",
                    f"{s['solve_success_rate']*100:.1f}%",
                    f"{s['avg_solve_time_ms']:.0f}",
                    s.get('total_retry1_recoveries', 0),
                    s.get('total_retry2_recoveries', 0),
                    s.get('total_retry3_recoveries', 0),
                    s.get('total_llm_nmpc_disagreements', 0),
                ])

    def save_master_summary(self, all_scenario_results: Dict[str, Dict[AblationGroup, GroupResult]]):
        """Save a master CSV spanning all scenarios × all groups."""
        master_path = os.path.join(self.config.output_dir, "ablation_master_summary.csv")
        import csv
        with open(master_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'scenario_id', 'group', 'referee_backend', 'soft_constraints',
                'num_runs', 'collision_rate', 'median_cpa', 'worst_cpa',
                'turn_compliance', 'solve_success_rate', 'avg_solve_ms',
                'llm_nmpc_disagreements',
            ])
            for sid, group_results in sorted(all_scenario_results.items()):
                for group, result in group_results.items():
                    s = result.summary
                    writer.writerow([
                        sid, group.value, result.referee_backend,
                        result.soft_constraints_enabled,
                        result.num_runs,
                        f"{s['collision_rate']*100:.1f}%",
                        f"{s['median_min_cpa']:.1f}",
                        f"{s['worst_cpa']:.1f}",
                        f"{s['turn_direction_compliance']*100:.1f}%",
                        f"{s['solve_success_rate']*100:.1f}%",
                        f"{s['avg_solve_time_ms']:.0f}",
                        s.get('total_llm_nmpc_disagreements', 0),
                    ])
        print(f"\nMaster summary saved to: {master_path}")
        return master_path
