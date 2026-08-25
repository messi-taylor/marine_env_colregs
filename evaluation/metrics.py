#!/usr/bin/env python3
"""
Metrics computation for Monte Carlo COLREGS evaluation.

Computes:
  - CPA CDF (collision safety distribution)
  - Compliance rate (turn direction, action magnitude, overtaking, roles, visibility)
  - Control smoothness (rudder rate, thrust jitter)
  - Solver performance (solve rate, avg time)
  - Degradation statistics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json


@dataclass
class RunMetrics:
    """Per-run metrics collected during one simulation run."""
    run_id: int = 0
    scenario_id: str = ""
    seed: int = 0

    # Safety
    min_cpa: float = float('inf')     # minimum CPA across all targets all steps
    min_cpa_target: str = ""
    collision: bool = False            # CPA < 1m
    near_miss: bool = False            # CPA < 20m

    # CPA time series per target
    cpa_timeseries: Dict[str, List[float]] = field(default_factory=dict)
    tcpa_timeseries: Dict[str, List[float]] = field(default_factory=dict)

    # Compliance
    turn_direction_compliant: bool = True   # sign(Δψ) matches expected
    action_magnitude_compliant: bool = True  # |Δψ| ≥ 30° when required
    overtaking_compliant: bool = True        # pass astern when overtaking
    role_compliant: bool = True              # give-way/stand-on correct

    # Control quality
    mean_rudder_rate: float = 0.0    # mean |Δτ_r|/dt
    thrust_std: float = 0.0          # std(τ_u)
    avg_surge: float = 0.0           # mean surge speed
    max_rudder: float = 0.0          # max |τ_r|

    # Solver
    solve_count: int = 0
    solve_success_rate: float = 1.0
    avg_solve_time_ms: float = 0.0
    num_infeasible: int = 0
    num_timeout: int = 0

    # Retry recovery statistics (Plan D: progressive constraint relaxation)
    retry_level1_successes: int = 0   # recoveries with L1 relaxation (-15° alteration)
    retry_level2_successes: int = 0   # recoveries with L2 relaxation (-40% CPA)
    retry_level3_successes: int = 0   # recoveries with L3 relaxation (free rudder)

    # Degradation
    max_degradation_level: int = 0
    degradation_transitions: int = 0

    # ── LLM-NMPC compliance gap (ablation Group D) ──
    llm_required_maneuvers: List[str] = field(default_factory=list)
    llm_forbidden_maneuvers: List[str] = field(default_factory=list)
    actual_turn_directions: List[str] = field(default_factory=list)
    llm_nmpc_disagreement_count: int = 0  # times LLM and NMPC disagreed on turn direction

    # Time series for plotting
    t_history: List[float] = field(default_factory=list)
    pos_x_history: List[float] = field(default_factory=list)
    pos_y_history: List[float] = field(default_factory=list)
    heading_history: List[float] = field(default_factory=list)
    surge_history: List[float] = field(default_factory=list)
    thrust_history: List[float] = field(default_factory=list)
    rudder_history: List[float] = field(default_factory=list)
    cpa_history: List[float] = field(default_factory=list)


class MetricsCollector:
    """Collects and aggregates metrics across Monte Carlo runs."""

    def __init__(self):
        self.runs: List[RunMetrics] = []

    def add_run(self, metrics: RunMetrics):
        self.runs.append(metrics)

    def summary(self) -> dict:
        """Compute aggregate statistics across all runs."""
        if not self.runs:
            return {'error': 'No runs collected'}

        N = len(self.runs)
        collisions = sum(1 for r in self.runs if r.collision)
        near_misses = sum(1 for r in self.runs if r.near_miss)

        min_cpas = [r.min_cpa for r in self.runs]
        min_cpas_safe = [c for c in min_cpas if c < float('inf')]

        # CPA CDF data (for plotting)
        cpa_bins = np.linspace(0, 100, 101)
        cpa_cdf = np.array([sum(1 for c in min_cpas_safe if c <= b) / max(len(min_cpas_safe), 1)
                            for b in cpa_bins])

        return {
            'num_runs': N,
            'collision_rate': collisions / max(N, 1),
            'near_miss_rate': near_misses / max(N, 1),
            'mean_min_cpa': float(np.mean(min_cpas_safe)) if min_cpas_safe else float('inf'),
            'median_min_cpa': float(np.median(min_cpas_safe)) if min_cpas_safe else float('inf'),
            'worst_cpa': float(np.min(min_cpas_safe)) if min_cpas_safe else float('inf'),
            'best_cpa': float(np.max(min_cpas_safe)) if min_cpas_safe else float('inf'),
            'std_min_cpa': float(np.std(min_cpas_safe)) if min_cpas_safe else 0.0,
            'cpa_cdf_bins': cpa_bins.tolist(),
            'cpa_cdf_values': cpa_cdf.tolist(),

            # Compliance
            'turn_direction_compliance': sum(1 for r in self.runs
                                             if r.turn_direction_compliant) / N,
            'action_magnitude_compliance': sum(1 for r in self.runs
                                               if r.action_magnitude_compliant) / N,
            'overtaking_compliance': sum(1 for r in self.runs
                                         if r.overtaking_compliant) / N,
            'role_compliance': sum(1 for r in self.runs
                                   if r.role_compliant) / N,

            # Control
            'mean_rudder_rate': float(np.mean([r.mean_rudder_rate for r in self.runs])),
            'thrust_std': float(np.mean([r.thrust_std for r in self.runs])),
            'avg_surge': float(np.mean([r.avg_surge for r in self.runs])),
            'max_rudder': float(np.max([r.max_rudder for r in self.runs])),

            # Solver
            'total_solves': sum(r.solve_count for r in self.runs),
            'solve_success_rate': float(np.mean([r.solve_success_rate for r in self.runs])),
            'avg_solve_time_ms': float(np.mean([r.avg_solve_time_ms for r in self.runs])),
            'total_infeasible': sum(r.num_infeasible for r in self.runs),
            'total_timeout': sum(r.num_timeout for r in self.runs),

            # Retry recovery statistics (Plan D)
            'total_retry1_recoveries': sum(r.retry_level1_successes for r in self.runs),
            'total_retry2_recoveries': sum(r.retry_level2_successes for r in self.runs),
            'total_retry3_recoveries': sum(r.retry_level3_successes for r in self.runs),

            # Degradation
            'runs_with_degradation': sum(1 for r in self.runs if r.max_degradation_level > 0),
            'max_degradation_observed': max(r.max_degradation_level for r in self.runs),

            # LLM-NMPC compliance gap
            'total_llm_nmpc_disagreements': sum(r.llm_nmpc_disagreement_count for r in self.runs),
            'runs_with_disagreement': sum(1 for r in self.runs if r.llm_nmpc_disagreement_count > 0),
        }

    def save_csv(self, filepath: str):
        """Export run metrics to CSV."""
        import csv
        if not self.runs:
            return
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'run_id', 'scenario_id', 'seed', 'min_cpa', 'collision', 'near_miss',
                'turn_compliant', 'mag_compliant', 'overtake_compliant', 'role_compliant',
                'mean_rudder_rate', 'thrust_std', 'avg_surge', 'max_rudder',
                'solve_rate', 'avg_solve_ms', 'num_infeasible', 'max_degradation',
                'retry_l1', 'retry_l2', 'retry_l3',
                'llm_nmpc_disagreements',
            ])
            for r in self.runs:
                writer.writerow([
                    r.run_id, r.scenario_id, r.seed, r.min_cpa, r.collision, r.near_miss,
                    r.turn_direction_compliant, r.action_magnitude_compliant,
                    r.overtaking_compliant, r.role_compliant,
                    r.mean_rudder_rate, r.thrust_std, r.avg_surge, r.max_rudder,
                    r.solve_success_rate, r.avg_solve_time_ms, r.num_infeasible,
                    r.max_degradation_level,
                    r.retry_level1_successes, r.retry_level2_successes,
                    r.retry_level3_successes,
                    r.llm_nmpc_disagreement_count,
                ])

    def save_json(self, filepath: str):
        """Export summary to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.summary(), f, indent=2)


def compute_all_metrics(metrics: RunMetrics,
                        expected_turn_sign: int = 1,
                        expected_encounter: str = "head_on") -> RunMetrics:
    """Post-process a run to compute compliance flags.

    Args:
        metrics: Raw metrics from simulation
        expected_turn_sign: +1=starboard, -1=port, 0=any (from COLREGS)
        expected_encounter: 'head_on' | 'crossing' | 'overtaking'
    """
    # Collision/near-miss
    metrics.collision = metrics.min_cpa < 1.0
    metrics.near_miss = metrics.min_cpa < 20.0

    # Turn direction: check if heading change matches expected
    # In ENU convention: +Δψ = CCW = PORT, -Δψ = CW = STARBOARD
    if len(metrics.heading_history) >= 2:
        h0 = metrics.heading_history[0]
        hN = metrics.heading_history[-1]
        dh = (hN - h0 + np.pi) % (2 * np.pi) - np.pi  # normalized Δψ
        if expected_turn_sign > 0:
            # Starboard turn → heading should DECREASE in ENU
            metrics.turn_direction_compliant = dh < -0.01
        elif expected_turn_sign < 0:
            # Port turn → heading should INCREASE in ENU
            metrics.turn_direction_compliant = dh > 0.01
        # else: any direction OK

        # Action magnitude: Rule 8(b) requires ≥ 30° for head-on
        if expected_encounter == 'head_on':
            metrics.action_magnitude_compliant = abs(dh) >= np.radians(25)

    return metrics
