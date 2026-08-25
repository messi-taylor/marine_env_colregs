#!/usr/bin/env python3
"""
Offline batch Monte Carlo runner for COLREGS collision avoidance evaluation.

Runs closed-loop simulation: FossenShip + DeterministicReferee + NMPCSolver,
all offline (no ROS/Gazebo needed).

Single-run wall time: ~2s. 100 runs @ 8 cores: ~30s.
"""

import numpy as np
import time
import sys
import os
import math
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from marine_env.ship_dynamics import FossenShip, ShipParams
from marine_env.colregs_referee.deterministic_referee import (
    DeterministicReferee, ShipObservation, _compute_relative_state
)
from marine_env.colregs_referee.constraint_mapper import ConstraintMapper
from marine_env.nmpc_solver import NMPCSolver, NMPCParams
from marine_env.colregs_referee.output_schema import (
    EncounterType, ShipRole, ManeuverType, ForbiddenManeuver
)

from .noise_sampler import NoiseSampler, NoiseConfig
from .metrics import RunMetrics, MetricsCollector, compute_all_metrics


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo evaluation."""
    scenario_id: str = "scenario_01"
    num_repeats: int = 100
    sim_duration: float = 40.0       # seconds
    dt_sim: float = 0.1              # simulation step (100ms for RK4 accuracy)
    control_period: float = 0.5      # NMPC solve period (5 Hz)
    nmpc_setup_time: float = 2.0     # seconds before first NMPC solve
    cpa_deadband: float = 60.0       # m — stop simulation when CPA > this and timesteps > min_steps
    min_sim_steps: int = 40          # minimum simulation steps before early exit

    # Referee configuration
    referee_cpa_safe: float = 50.0   # m
    referee_tcpa_warn: float = 60.0  # s

    # Output
    output_dir: str = "evaluation_output"
    parallel_workers: int = 1        # set >1 for multiprocessing

    # ── Ablation support (backward-compatible defaults) ──
    referee_backend: str = "deterministic"   # "deterministic" | "ollama" | "ollama_no_cfg" | "grammar_constrained"
    referee_model: str = "qwen2.5:7b"       # LLM model name (for ollama backend)
    referee_model_path: str = ""             # path to GGUF file (for grammar_custom backend)
    referee_min_interval: float = 10.0       # s — minimum interval between LLM calls (cache otherwise)
    nmpc_weight_overrides: Optional[dict] = None  # e.g. {"w_legal": 0, "w_smooth": 0, "w_speed": 0}

    # ── Control backend ──
    control_backend: str = "nmpc"  # "nmpc" (default) | "vo" (velocity obstacle baseline)


class BatchRunner:
    """Runs batch Monte Carlo simulations for a given scenario."""

    def __init__(self, config: MonteCarloConfig = None):
        self.config = config or MonteCarloConfig()
        self.metrics = MetricsCollector()
        self._pairwise_referee = DeterministicReferee(visibility="clear")
        # LLM referee cache — GPU model loaded once, reused across all repeats
        self._llm_referee = None
        # VO controller cache — created once per BatchRunner instance
        self._vo_controller = None
        # LLM output cache — avoid re-querying every NMPC solve step
        self._cached_ref_output = None
        self._last_referee_time = -999.0

    # =====================================================================
    # Scenario loading
    # =====================================================================

    def load_scenario(self, scenario_id: str = None) -> dict:
        """Load scenario definition from YAML."""
        sid = scenario_id or self.config.scenario_id
        yaml_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'colregs_20_new_scenarios.yaml'
        )
        with open(yaml_path, 'r') as f:
            all_scenarios = yaml.safe_load(f)

        # Find by ID
        for key, val in all_scenarios.items():
            if key.startswith(sid.replace('scenario_', 'scenario_')):
                return {'key': key, 'data': val}
        # Fallback: match first
        for key, val in all_scenarios.items():
            if sid in key or key.startswith('scenario_01'):
                return {'key': key, 'data': val}
        raise ValueError(f"Scenario {sid} not found in YAML")

    # =====================================================================
    # Target-target COLREGS avoidance
    # =====================================================================

    # Tunable constants for target-target avoidance
    TGT_CPA_THRESHOLD = 15.0           # m — 3× ship length, "danger zone" for 5m ships
    TGT_TCPA_LOOKAHEAD = 25.0          # s — look-ahead window for risk assessment
    TGT_AVOIDANCE_STARBOARD = math.radians(35)  # rad — "substantial action" per Rule 8(b)
    TGT_RECOVERY_RATE = math.radians(3)         # rad/s — per-step recovery toward intended heading
    TGT_RECOVERY_DISTANCE = 30.0       # m — distance beyond which recovery is safe (2× CPA threshold)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _resolve_target_avoidance(self, targets: dict, dt: float) -> None:
        """Check all target-target pairs for collision risk and apply COLREGS starboard avoidance.

        Modifies targets[name]['heading'] in-place for ships that are giving way.
        Also manages avoidance state: initiation, sustain, and recovery.

        Design:
          - Only applies starboard turns (never port) — COLREGS default safe action.
          - A ship can only avoid ONE other ship at a time (most urgent, by min CPA).
          - Avoidance heading is set once at initiation and held until safe to recover.
          - Recovery gradually returns to intended_heading at TGT_RECOVERY_RATE rad/s.

        Args:
            targets: Dict of target ship states (modified in-place).
            dt: Simulation timestep in seconds.
        """
        num_targets = len(targets)
        if num_targets < 2:
            self._recover_targets(targets, dt)
            return

        names = list(targets.keys())

        # ── Phase 1: Check all pairs for collision risk ──
        risks = []  # list of (cpa, name_a, name_b, give_way_ship, stand_on_ship)

        for i in range(num_targets):
            for j in range(i + 1, num_targets):
                name_a, name_b = names[i], names[j]
                ts_a, ts_b = targets[name_a], targets[name_b]

                # Build ShipObservation objects for this pair.
                # Convert ENU heading → maritime heading for referee bearing formula.
                h_a_enu = ts_a['heading']
                h_a_mar = (math.pi/2 - h_a_enu + math.pi) % (2*math.pi) - math.pi
                h_b_enu = ts_b['heading']
                h_b_mar = (math.pi/2 - h_b_enu + math.pi) % (2*math.pi) - math.pi
                obs_a = ShipObservation(
                    name=name_a,
                    position=np.array([ts_a['x'], ts_a['y']]),
                    heading=h_a_mar,
                    speed=np.array([math.cos(h_a_enu) * ts_a['speed'],
                                    math.sin(h_a_enu) * ts_a['speed']]),
                    length=5.0,
                    is_own_ship=False,
                )
                obs_b = ShipObservation(
                    name=name_b,
                    position=np.array([ts_b['x'], ts_b['y']]),
                    heading=h_b_mar,
                    speed=np.array([math.cos(h_b_enu) * ts_b['speed'],
                                    math.sin(h_b_enu) * ts_b['speed']]),
                    length=5.0,
                    is_own_ship=False,
                )

                # Compute relative geometry from both perspectives
                geo_ab = _compute_relative_state(obs_a, obs_b)  # a as "OS", b as "TS"
                geo_ba = _compute_relative_state(obs_b, obs_a)  # b as "OS", a as "TS"

                # Risk check: CPA below threshold AND TCPA within lookahead AND TCPA > 0
                if geo_ab['cpa'] >= self.TGT_CPA_THRESHOLD:
                    continue
                if geo_ab['tcpa'] <= 0 or geo_ab['tcpa'] > self.TGT_TCPA_LOOKAHEAD:
                    continue

                # ── Vector-based encounter classification ──
                # Uses velocity-vector dot/cross products instead of angle subtraction
                # to avoid ±π boundary issues in heading angle comparisons.
                p_a = np.array([ts_a['x'], ts_a['y']])
                p_b = np.array([ts_b['x'], ts_b['y']])
                h_a, h_b = ts_a['heading'], ts_b['heading']
                v_a = ts_a['speed'] * np.array([math.cos(h_a), math.sin(h_a)])
                v_b = ts_b['speed'] * np.array([math.cos(h_b), math.sin(h_b)])

                # Direction unit vectors
                d_a = v_a / (ts_a['speed'] + 1e-10)
                d_b = v_b / (ts_b['speed'] + 1e-10)

                # Reciprocal course check: dot product of heading directions
                course_dot = float(np.dot(d_a, d_b))  # cos(Δheading), -1 = opposite

                # Relative position from a to b
                dp_ab = p_b - p_a
                dist_ab = float(np.linalg.norm(dp_ab))
                dp_ab_unit = dp_ab / (dist_ab + 1e-10)

                # From a's perspective: is b ahead?
                ahead_a = float(np.dot(d_a, dp_ab_unit))  # 1 = dead ahead, -1 = astern

                # From a's perspective: is b to starboard or port?
                # 2D cross product: d_a × dp_ab_unit (>0 = port/left, <0 = starboard/right)
                cross_a = float(d_a[0] * dp_ab_unit[1] - d_a[1] * dp_ab_unit[0])

                # ── Classify encounter (head-on > overtaking > crossing) ──

                # Head-on (Rule 14): reciprocal courses + target ahead
                is_headon = (course_dot < -0.866        # |Δheading| > 150°
                             and ahead_a > 0.966         # within ~15° of dead ahead
                             and dist_ab < 100.0)        # within reasonable range

                # Overtaking (Rule 13): target abaft the beam + similar courses
                is_overtaking_ab = (ahead_a < -0.3827    # >112.5° abaft (cos 112.5° = -0.3827)
                                    and course_dot > 0.342  # |Δheading| < 70° (similar courses)
                                    and ts_a['speed'] > ts_b['speed'] * 1.05)  # faster
                is_overtaking_ba = (
                    float(np.dot(d_b, -dp_ab_unit)) < -0.3827  # a abaft b's beam
                    and course_dot > 0.342
                    and ts_b['speed'] > ts_a['speed'] * 1.05)

                if is_headon:
                    # Rule 14: both shall alter to starboard
                    give_way_ship = 'both'
                    stand_on_ship = ''
                elif is_overtaking_ab:
                    # Rule 13: a is overtaking b → a gives way
                    give_way_ship = name_a
                    stand_on_ship = name_b
                elif is_overtaking_ba:
                    # Rule 13: b is overtaking a → b gives way
                    give_way_ship = name_b
                    stand_on_ship = name_a
                else:
                    # Rule 15: crossing — check starboard/port from both perspectives
                    # From a's perspective: cross_a < 0 → b to starboard → a gives way
                    cross_b = float(d_b[0] * (-dp_ab_unit)[1] - d_b[1] * (-dp_ab_unit)[0])
                    # cross_b > 0 → a is to starboard of b → b gives way
                    # (the sign flips because we use -dp_ab_unit, which is p_a - p_b)

                    a_to_starboard_of_b = (cross_b < 0)  # a is on b's starboard → b gives way
                    b_to_starboard_of_a = (cross_a < 0)  # b is on a's starboard → a gives way

                    if b_to_starboard_of_a:
                        give_way_ship = name_a
                        stand_on_ship = name_b
                    elif a_to_starboard_of_b:
                        give_way_ship = name_b
                        stand_on_ship = name_a
                    else:
                        # No classified encounter — skip this pair
                        continue

                risks.append((geo_ab['cpa'], name_a, name_b,
                              give_way_ship, stand_on_ship))

        if not risks:
            # No risks found. Handle recovery for any currently-avoiding ships.
            self._recover_targets(targets, dt)
            return

        # ── Phase 2: Sort by urgency (smallest CPA first) ──
        risks.sort(key=lambda r: r[0])

        # ── Phase 3: Apply avoidance (one-pair-per-ship constraint) ──
        committed_a = set()
        committed_b = set()

        for cpa, name_a, name_b, give_way_ship, stand_on_ship in risks:
            # Skip if either ship is already committed to an avoidance this step
            if name_a in committed_a or name_a in committed_b:
                continue
            if name_b in committed_a or name_b in committed_b:
                continue

            # Skip if either ship is actively avoiding a DIFFERENT ship
            ts_a, ts_b = targets[name_a], targets[name_b]
            if ts_a['avoiding'] and ts_a['avoidance_target'] not in (name_b, ''):
                continue
            if ts_b['avoiding'] and ts_b['avoidance_target'] not in (name_a, ''):
                continue

            committed_a.add(name_a)
            committed_b.add(name_b)

            # ── Apply starboard avoidance to give-way ship(s) ──
            if give_way_ship == 'both':
                # Head-on: both ships alter to starboard
                for name in (name_a, name_b):
                    ts = targets[name]
                    if not ts['avoiding']:
                        ts['intended_heading'] = ts['heading']
                        ts['avoidance_heading'] = self._normalize_angle(
                            ts['intended_heading'] - self.TGT_AVOIDANCE_STARBOARD)
                        ts['avoiding'] = True
                        ts['avoidance_target'] = (name_b if name == name_a else name_a)
                        ts['heading'] = ts['avoidance_heading']
            else:
                # Crossing or overtaking: only the give-way ship alters course
                gw_name = give_way_ship
                gw = targets[gw_name]
                if not gw['avoiding']:
                    gw['intended_heading'] = gw['heading']
                    gw['avoidance_heading'] = self._normalize_angle(
                        gw['intended_heading'] - self.TGT_AVOIDANCE_STARBOARD)
                    gw['avoiding'] = True
                    gw['avoidance_target'] = stand_on_ship
                    gw['heading'] = gw['avoidance_heading']

        # ── Phase 4: Check recovery for ships that have passed their avoidance target ──
        self._recover_targets(targets, dt)

    def _recover_targets(self, targets: dict, dt: float) -> None:
        """Recover avoiding targets back to their intended heading.

        A target is eligible for recovery when:
          1. It is currently avoiding.
          2. Its avoidance target is at a distance > TGT_RECOVERY_DISTANCE (30m).

        Recovery rotates heading toward intended_heading at TGT_RECOVERY_RATE rad/s.
        Once within 1 degree of intended, avoidance is cleared.
        """
        for name, ts in targets.items():
            if not ts['avoiding']:
                continue

            # Check if the avoided ship is far enough away
            other_name = ts['avoidance_target']
            if other_name and other_name in targets:
                other = targets[other_name]
                dx = other['x'] - ts['x']
                dy = other['y'] - ts['y']
                dist = math.sqrt(dx * dx + dy * dy)

                # Also check if ships have genuinely separated:
                # the avoidance target should be receding (behind us)
                # Compute dot product of (own heading) · (vector to other)
                hx, hy = math.cos(ts['heading']), math.sin(ts['heading'])
                ahead = hx * dx + hy * dy  # >0: other ahead, <0: other behind

                if dist < self.TGT_RECOVERY_DISTANCE or ahead > -5.0:
                    # Still too close or other ship is still ahead — maintain avoidance
                    ts['heading'] = ts['avoidance_heading']
                    continue

            # Safe to recover: rotate toward intended_heading
            current = ts['heading']
            intended = ts['intended_heading']
            diff = self._normalize_angle(intended - current)

            if abs(diff) < math.radians(1.0):
                # Close enough — clear avoidance state
                ts['heading'] = intended
                ts['avoiding'] = False
                ts['avoidance_target'] = ''
                ts['avoidance_heading'] = 0.0
            else:
                # Step toward intended at the recovery rate
                step_size = self.TGT_RECOVERY_RATE * dt
                sign = 1.0 if diff > 0 else -1.0
                ts['heading'] = self._normalize_angle(
                    current + sign * min(abs(diff), step_size))

    # =====================================================================
    # Referee factory (ablation support)
    # =====================================================================

    def _create_referee(self, visibility: str, cfg: MonteCarloConfig):
        """Create the appropriate COLREGS referee backend based on config.

        Backends:
          - deterministic:        Rule-based engine (fast, always available)
          - ollama:              Ollama REST API (qwen2.5:7b, no GBNF grammar)
          - grammar_constrained:  llama-cpp-python + GBNF token-level hard constraint

        GrammarConstrainedReferee is cached on the BatchRunner instance because
        the underlying llama.cpp model (multi-GB GGUF + GPU memory) is expensive
        to load and non-picklable.
        """
        backend = cfg.referee_backend
        if backend == "deterministic":
            return DeterministicReferee(visibility=visibility)
        elif backend in ("ollama", "ollama_no_cfg"):
            from marine_env.colregs_referee.llm_referee import OllamaReferee
            base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11435")
            use_cfg = (backend == "ollama")  # Group A/D: CFG on; Group B: CFG off
            # Group B (-CFG): no retries — fail fast, fallback to deterministic.
            # The failure rate itself IS the experimental result for Group B.
            max_retries = 2 if use_cfg else 0
            return OllamaReferee(visibility=visibility, base_url=base_url,
                                use_cfg=use_cfg, max_retries=max_retries,
                                model=cfg.referee_model)
        elif backend == "grammar_constrained":
            from marine_env.colregs_referee.llm_referee import GrammarConstrainedReferee
            if self._llm_referee is None:
                print("  [BatchRunner] Loading GrammarConstrainedReferee model (one-time)...")
                self._llm_referee = GrammarConstrainedReferee(visibility=visibility)
            return self._llm_referee
        elif backend == "grammar_custom":
            from marine_env.colregs_referee.llm_referee import GrammarConstrainedReferee
            model_path = cfg.referee_model_path
            if not model_path or not os.path.exists(model_path):
                raise RuntimeError(f"GGUF model not found: {model_path}")
            # Don't cache — different models may be used per group
            print(f"  [BatchRunner] Loading model: {model_path}")
            return GrammarConstrainedReferee(model_path=model_path, visibility=visibility)
        else:
            raise ValueError(f"Unknown referee backend: {backend}")

    # =====================================================================
    # Single-run simulation
    # =====================================================================

    def run_single(self, run_id: int, scenario: dict, seed: int) -> RunMetrics:
        """Run one closed-loop simulation and return metrics.

        Args:
            run_id: Unique run identifier
            scenario: Parsed scenario dict with 'key' and 'data'
            seed: Random seed for reproducibility

        Returns:
            RunMetrics with all collected data
        """
        cfg = self.config
        # Reset LLM cache for each run
        self._cached_ref_output = None
        self._last_referee_time = -999.0
        sc_data = scenario['data']
        noise = NoiseSampler(seed=seed)

        # ── Initialize own ship with noise ──
        os_cfg = sc_data['own_ship']
        os_offset = noise.sample_own_ship_offset()

        # Convert compass bearing to ENU: ros_yaw = π/2 - compass_yaw
        compass_to_enu = lambda y: (math.pi/2 - y + math.pi) % (2*math.pi) - math.pi

        os_yaw_compass = os_cfg['yaw']
        os_yaw_enu = compass_to_enu(os_yaw_compass) + os_offset['dheading']

        os_eta = np.array([
            os_cfg['x'] + os_offset['dx'],
            os_cfg['y'] + os_offset['dy'],
            os_yaw_enu,
        ])
        os_speed = os_cfg['speed'] + os_offset['dspeed']
        os_nu = np.array([os_speed, 0.0, 0.0])  # body-frame surge aligned with heading

        ship = FossenShip(dt=cfg.dt_sim)
        ship.set_state(os_eta, os_nu)

        # ── Initialize target ships with noise (kinematic CV — constant speed & heading) ──
        targets = {}
        target_init_positions = {}
        for ts_cfg in sc_data.get('target_ships', []):
            name = ts_cfg['name']
            toff = noise.sample_target_offset(name)
            ts_yaw_enu = compass_to_enu(ts_cfg['yaw']) + toff['dheading']
            ts_x = ts_cfg['x'] + toff['dx']
            ts_y = ts_cfg['y'] + toff['dy']
            ts_speed = ts_cfg['speed'] + toff['dspeed']
            targets[name] = {
                'x': ts_x,
                'y': ts_y,
                'heading': ts_yaw_enu,
                'speed': ts_speed,
                'model': ts_cfg.get('model', 'wamv'),
                # COLREGS target-target avoidance state
                'intended_heading': ts_yaw_enu,
                'intended_speed': ts_speed,
                'avoiding': False,
                'avoidance_target': '',
                'avoidance_heading': 0.0,
            }
            target_init_positions[name] = np.array([ts_x, ts_y, ts_yaw_enu])

        # ── Environment (after target sampling to match manual test RNG order) ──
        noise.sample_environment(sc_data.get('environment', {}))
        tau_env_base = np.zeros(3)

        # ── Initialize referee, mapper, solver ──
        visibility = sc_data.get('visibility', 'clear')
        referee = self._create_referee(visibility, cfg)
        mapper = ConstraintMapper(
            prediction_horizon=NMPCParams().N,
            dt=NMPCParams().dt,
            min_cpa_margin=10.0,
        )
        # ── Scenario-specific NMPC tuning overrides (backward-compatible) ──
        nmpc_tuning = sc_data.get('nmpc_tuning', {})
        cpa_floor_default = nmpc_tuning.get('cpa_floor', 15.0)
        starboard_bias_deg = nmpc_tuning.get('starboard_bias_deg', 40.0)

        nmpc_params = NMPCParams()
        if 'gauss_weight' in nmpc_tuning:
            nmpc_params.gauss_weight = nmpc_tuning['gauss_weight']
        if 'horizon_N' in nmpc_tuning:
            nmpc_params.N = nmpc_tuning['horizon_N']

        # ── Ablation: soft constraint toggle (Group D: w_legal=w_smooth=w_speed=0) ──
        if cfg.nmpc_weight_overrides:
            for attr_name, value in cfg.nmpc_weight_overrides.items():
                if hasattr(nmpc_params, attr_name):
                    setattr(nmpc_params, attr_name, value)
        solver = NMPCSolver(params=nmpc_params)
        solver.setup()

        # Waypoints: default straight-ahead (will be overridden by NMPC starboard bias)
        base_waypoints = [(os_eta[0] + os_speed * 60.0 * math.cos(os_yaw_enu),
                           os_eta[1] + os_speed * 60.0 * math.sin(os_yaw_enu),
                           os_speed)]
        ref_waypoints = base_waypoints
        nmpc_ref_bias_applied = False  # set True once starboard bias added

        # ── Metrics collection ──
        run_metrics = RunMetrics(run_id=run_id, scenario_id=scenario['key'], seed=seed)
        run_metrics.cpa_timeseries = {name: [] for name in targets}
        run_metrics.tcpa_timeseries = {name: [] for name in targets}

        # Time stepping
        sim_steps = int(cfg.sim_duration / cfg.dt_sim)
        control_interval = int(cfg.control_period / cfg.dt_sim)
        solve_count = 0
        infeasible_count = 0
        solve_times = []
        last_u_opt = None  # last valid NMPC solution
        consecutive_infeasible = 0  # track for progressive relaxation
        retry_stats = {1: 0, 2: 0, 3: 0}  # successful recoveries per retry level

        # PI speed controller to maintain nominal speed
        target_speed = os_cfg['speed']  # m/s
        speed_integral = 0.0
        Kp_speed = 800.0   # N per m/s error
        Ki_speed = 200.0   # N/s per m/s error

        prev_rudder = 0.0
        prev_thrust = 0.0
        prev_tau_u = 0.0
        # Fossen ENU: τ = [surge, sway(=0), yaw]. +τ_r → +heading (PORT), -τ_r → STARBOARD
        target_heading_change = math.radians(40)  # starboard turn target
        prev_heading = os_yaw_enu                 # for unwrapped heading tracking
        cum_heading_change = 0.0                  # unwrapped cumulative heading change

        # ── Recovery phase state ──
        cpa_min_reached = False          # True once min CPA has been recorded
        recovery_steps = 0               # consecutive steps with target receding
        RECOVERY_THRESHOLD = 10          # 1.0s @ 0.1s dt — target must recede this long
        avoidance_complete = False       # True when PD should disengage

        for step in range(sim_steps):
            t = step * cfg.dt_sim

            # ── Resolve target-target COLREGS avoidance ──
            self._resolve_target_avoidance(targets, cfg.dt_sim)

            # ── Step target ships (kinematic CV — avoidance heading already applied) ──
            for name, ts_data in targets.items():
                ts_data['x'] += ts_data['speed'] * math.cos(ts_data['heading']) * cfg.dt_sim
                ts_data['y'] += ts_data['speed'] * math.sin(ts_data['heading']) * cfg.dt_sim

            # ── Build ship observations ──
            os_eta_now = ship.eta.copy()
            os_nu_now = ship.nu.copy()
            os_heading = os_eta_now[2]
            # Unwrap heading for cumulative tracking
            dh_step = float((os_heading - prev_heading + math.pi) % (2*math.pi) - math.pi)
            cum_heading_change += dh_step
            prev_heading = os_heading
            os_vel_world = np.array([
                math.cos(os_heading) * os_nu_now[0] - math.sin(os_heading) * os_nu_now[1],
                math.sin(os_heading) * os_nu_now[0] + math.cos(os_heading) * os_nu_now[1],
            ])

            # Convert ENU heading (FossenShip internal) → maritime heading
            # required by deterministic_referee bearing formula (0=North, CW+).
            os_heading_maritime = (math.pi/2 - os_heading + math.pi) % (2*math.pi) - math.pi
            os_obs = ShipObservation(
                name="OS",
                position=os_eta_now[:2].copy(),
                heading=os_heading_maritime,
                speed=os_vel_world.copy(),
                length=5.0,
                is_own_ship=True,
            )

            ts_obs_list = []
            for name, ts_data in targets.items():
                ts_h_enu = ts_data['heading']
                ts_h_maritime = (math.pi/2 - ts_h_enu + math.pi) % (2*math.pi) - math.pi
                ts_sp = ts_data['speed']
                ts_vel_world = np.array([
                    math.cos(ts_h_enu) * ts_sp,
                    math.sin(ts_h_enu) * ts_sp,
                ])
                ts_pos = np.array([ts_data['x'], ts_data['y']])
                ts_obs = ShipObservation(
                    name=name,
                    position=ts_pos.copy(),
                    heading=ts_h_maritime,
                    speed=ts_vel_world.copy(),
                    length=5.0,
                    is_own_ship=False,
                )
                ts_obs_list.append(ts_obs)

            # ── Track actual distance and geometric CPA ──
            all_dists = []
            for name, ts_data in targets.items():
                ts_pos = np.array([ts_data['x'], ts_data['y']])
                actual_dist = float(np.linalg.norm(ts_pos - os_eta_now[:2]))
                all_dists.append(actual_dist)
                run_metrics.cpa_timeseries[name].append(actual_dist)

            if all_dists:
                run_metrics.cpa_history.append(min(all_dists))
                prev_min_cpa_val = run_metrics.min_cpa
                if min(all_dists) < run_metrics.min_cpa:
                    run_metrics.min_cpa = min(all_dists)
                    idx = np.argmin(all_dists)
                    run_metrics.min_cpa_target = list(targets.keys())[idx]

                # Recovery detection: CPA reached and target consistently receding
                if run_metrics.min_cpa < float('inf') and min(all_dists) > run_metrics.min_cpa:
                    recovery_steps += 1
                else:
                    recovery_steps = max(0, recovery_steps - 1)

                if recovery_steps >= RECOVERY_THRESHOLD and not avoidance_complete:
                    avoidance_complete = True

            # ── Record state time series ──
            run_metrics.t_history.append(t)
            run_metrics.pos_x_history.append(float(os_eta_now[0]))
            run_metrics.pos_y_history.append(float(os_eta_now[1]))
            run_metrics.heading_history.append(float(os_heading))
            run_metrics.surge_history.append(float(os_nu_now[0]))

            # ── VO control path (external baseline, runs before PI/rudder) ──
            vo_desired_speed = None
            vo_desired_heading = None
            if (cfg.control_backend == "vo"
                    and not avoidance_complete
                    and t >= cfg.nmpc_setup_time
                    and step % control_interval == 0):
                if self._vo_controller is None:
                    from .vo_baseline import create_vo_controller
                    self._vo_controller = create_vo_controller(visibility=visibility)

                ts_states = []
                for name, ts_data in targets.items():
                    ts_states.append({
                        "pos": np.array([ts_data['x'], ts_data['y']]),
                        "heading": ts_data['heading'],
                        "speed": ts_data['speed'],
                    })

                vo_speed, vo_heading, vo_debug = self._vo_controller.compute(
                    os_pos=os_eta_now[:2],
                    os_heading=os_heading,
                    os_speed=float(os_nu_now[0]),
                    target_states=ts_states,
                    dt=cfg.control_period,
                    current_heading_change=cum_heading_change,
                )

                vo_desired_speed = vo_speed
                vo_desired_heading = vo_heading

                # Record VO metrics
                run_metrics.llm_required_maneuvers.append("vo_baseline")
                run_metrics.llm_forbidden_maneuvers.append("none")
                actual_dir = "starboard" if vo_debug.get("best_heading_deg", 0) < 0 else "port"
                run_metrics.actual_turn_directions.append(actual_dir)

            # ── PI speed control ──
            speed_target = vo_desired_speed if vo_desired_speed is not None else target_speed
            speed_error = speed_target - float(os_nu_now[0])
            speed_integral += speed_error * cfg.dt_sim
            speed_integral = max(-2.0, min(speed_integral, 2.0))
            tau_u_pi = Kp_speed * speed_error + Ki_speed * speed_integral
            tau_u_pi = max(0.0, min(tau_u_pi, 3000.0))

            # ── Rudder control ──
            if vo_desired_heading is not None:
                # VO heading tracking — PD on heading error to desired heading
                dh = (vo_desired_heading - os_heading + math.pi) % (2*math.pi) - math.pi
                r_yaw = float(os_nu_now[2])
                if abs(dh) < math.radians(2):
                    tau_r = max(-60.0, min(60.0, -60.0 * r_yaw))  # damping only
                else:
                    tau_r_raw = 40.0 * dh - 50.0 * r_yaw
                    tau_r = max(-60.0, min(60.0, tau_r_raw))
            elif not avoidance_complete:
                # PHASE 1: Starboard PD for COLREGS avoidance (NMPC path).
                r_yaw = float(os_nu_now[2])
                MAX_TURN = math.radians(60)
                if cum_heading_change < -MAX_TURN:
                    avoidance_complete = True
                    tau_r = max(-60.0, min(60.0, -60.0 * r_yaw))
                else:
                    dh_err = -target_heading_change - cum_heading_change
                    if dh_err < 0:
                        tau_r_raw = 35.0 * dh_err - 50.0 * r_yaw
                        tau_r = max(-60.0, min(0.0, tau_r_raw))
                    else:
                        if abs(r_yaw) < 0.01:
                            tau_r = 0.0
                        elif r_yaw < 0:
                            tau_r = 60.0
                        else:
                            tau_r = -60.0
            else:
                # PHASE 2: Recovery — bang-bang yaw damping.
                r_yaw = float(os_nu_now[2])
                if abs(r_yaw) < 0.01:
                    tau_r = 0.0
                elif r_yaw < 0:
                    tau_r = 60.0
                else:
                    tau_r = -60.0

            tau_u = tau_u_pi
            tau = np.array([tau_u, 0.0, tau_r])  # [surge, sway(=0), yaw]

            # ── NMPC solve (only during avoidance phase; recovery uses LOS only) ──
            if (cfg.control_backend != "vo"
                    and not avoidance_complete
                    and t >= cfg.nmpc_setup_time
                    and step % control_interval == 0):
                try:
                    # ── LLM cache: only re-query if min_interval elapsed ──
                    if (cfg.referee_min_interval > 0
                            and self._cached_ref_output is not None
                            and t - self._last_referee_time < cfg.referee_min_interval):
                        ref_output = self._cached_ref_output
                    else:
                        ref_output = referee.evaluate(os_obs, ts_obs_list,
                                                      scenario_id=cfg.scenario_id)
                        if cfg.referee_min_interval > 0:
                            self._cached_ref_output = ref_output
                            self._last_referee_time = t
                    nmpc_constraints = mapper.map(ref_output)

                    # ── Apply scenario-specific target overrides ──
                    target_overrides = nmpc_tuning.get('target_overrides', {})
                    for tgt_name, ovr in target_overrides.items():
                        for sc in nmpc_constraints.spatial_constraints:
                            if sc.target_name == tgt_name:
                                if 'pass_astern' in ovr:
                                    sc.pass_astern = ovr['pass_astern']

                    # ── Starboard-biased reference on first COLREGS right-turn requirement ──
                    # Triggers when referee OR mapper forbids port turn (head-on Rule 14,
                    # crossing give-way Rule 15, overtaking Rule 13, breakthrough Rule 17)
                    mapper_forbids_port = (
                        nmpc_constraints.maneuver_constraint.forbidden_maneuver
                        == ForbiddenManeuver.ALTER_TO_PORT
                    )
                    ref_forbids_port = (
                        ref_output.forbidden_maneuver.value == 'alter_to_port'
                    )
                    if not avoidance_complete:
                        if not nmpc_ref_bias_applied and (ref_forbids_port or mapper_forbids_port):
                            starboard_heading = os_yaw_enu - math.radians(starboard_bias_deg)  # CW in ENU
                            wp_mid_x = os_eta_now[0] + 50.0 * math.cos(starboard_heading)
                            wp_mid_y = os_eta_now[1] + 50.0 * math.sin(starboard_heading)
                            wp_end_x = os_eta_now[0] + 100.0 * math.cos(starboard_heading)
                            wp_end_y = os_eta_now[1] + 100.0 * math.sin(starboard_heading)
                            ref_waypoints = [(wp_mid_x, wp_mid_y, os_speed),
                                            (wp_end_x, wp_end_y, os_speed)]
                            nmpc_ref_bias_applied = True
                    else:
                        # Recovery: steer toward original waypoint
                        ref_waypoints = base_waypoints

                    target_trajs = {}
                    cpa_radius = {}
                    for name, ts_data in targets.items():
                        ts_h = ts_data['heading']
                        ts_sp = ts_data['speed']
                        ts_pos = np.array([ts_data['x'], ts_data['y']])
                        traj = np.zeros((2, nmpc_params.N + 1))
                        traj[0,0], traj[1,0] = ts_pos[0], ts_pos[1]
                        for k in range(1, nmpc_params.N + 1):
                            traj[0,k] = traj[0,k-1] + ts_sp*math.cos(ts_h)*nmpc_params.dt
                            traj[1,k] = traj[1,k-1] + ts_sp*math.sin(ts_h)*nmpc_params.dt
                        target_trajs[name] = traj
                        # CPA: scale with distance. Floor 15m (relaxed for half-plane feasibility),
                        # ramp 0.82×dist toward full CPA (~60m for 71.5m target).
                        # Balances NMPC feasibility against meaningful CPA requirement.
                        sc = next((s for s in nmpc_constraints.spatial_constraints
                                   if s.target_name == name), None)
                        full_cpa = sc.min_distance if sc else cfg.referee_cpa_safe
                        current_dist = float(np.linalg.norm(ts_pos - os_eta_now[:2]))
                        tgt_cpa_floor = target_overrides.get(name, {}).get('cpa_floor', cpa_floor_default)
                        cpa_radius[name] = max(tgt_cpa_floor, min(full_cpa, current_dist * 0.82))

                    # ── Compute half-plane normals per target ──
                    # Half-plane normal n̂ determines the safe-passing direction:
                    #   n̂ · (p_OS - p_TS) >= r_hp   (convex linear constraint)
                    # COLREGS-aware: use stern/bow direction when available,
                    # fallback to relative bearing for robustness.
                    hp_normals = {}
                    for name, ts_data in targets.items():
                        ts_pos = np.array([ts_data['x'], ts_data['y']])
                        rel_vec = os_eta_now[:2] - ts_pos  # TS → OS direction
                        dist_ts = float(np.linalg.norm(rel_vec))
                        n_rel = rel_vec / max(dist_ts, 1e-6) if dist_ts > 1e-6 else np.array([1.0, 0.0])

                        sc = next((s for s in nmpc_constraints.spatial_constraints
                                   if s.target_name == name), None)
                        if sc is not None and sc.pass_astern:
                            ts_h_vec = np.array([math.cos(ts_data['heading']),
                                                  math.sin(ts_data['heading'])])
                            n_stern = -ts_h_vec  # normal pointing to TS stern
                            # How far OS is from TS projected onto stern axis.
                            # proj_stern > 0 → OS is on TS bow side (wrong for pass_astern).
                            proj_stern = np.dot(n_stern, -rel_vec)
                            r_cpa = cpa_radius.get(name, 20.0)
                            if proj_stern > r_cpa * 0.3:
                                # OS is on the bow side — pure stern normal is hard
                                # to reach within the horizon. Blend with relative
                                # bearing to create an achievable diagonal constraint.
                                blend = min(0.6, proj_stern / (r_cpa * 2 + 1.0))
                                n_blended = (1 - blend) * n_stern + blend * n_rel
                                n_blended /= max(np.linalg.norm(n_blended), 1e-6)
                                hp_normals[name] = n_blended
                            elif proj_stern > -r_cpa * 0.5:
                                # OS is near neutral — stern normal is reachable
                                hp_normals[name] = n_stern
                            else:
                                # OS is on stern side — use relative bearing
                                hp_normals[name] = n_rel
                        else:
                            hp_normals[name] = n_rel

                    rm, rM = nmpc_constraints.maneuver_constraint.rudder_min, \
                             nmpc_constraints.maneuver_constraint.rudder_max
                    # Flip: mapper +rudder=starboard → Fossen -τ_r=starboard
                    solver_c = {
                        'tau_r_min': -(rM * nmpc_params.max_yaw_moment),
                        'tau_r_max': -(rm * nmpc_params.max_yaw_moment),
                        'alteration_min_angle': nmpc_constraints.maneuver_constraint.alteration_min_angle,
                        'alteration_active': nmpc_constraints.maneuver_constraint.alteration_min_angle > 0.01,
                        'v_min': nmpc_constraints.speed_constraint.min_speed,
                        'v_max': nmpc_constraints.speed_constraint.max_speed,
                        'cpa_radius_per_target': cpa_radius,
                        'hp_normals_per_target': hp_normals,
                    }

                    x0 = np.array([os_eta_now[0], os_eta_now[1], os_heading,
                                   os_nu_now[0], os_nu_now[1], os_nu_now[2]])
                    x_ref = solver.generate_reference(x0, ref_waypoints, 0)

                    result = solver.solve(x0=x0, x_ref=x_ref, target_trajs=target_trajs,
                                         constraints=solver_c, tau_env=tau_env_base,
                                         warm_start=True)
                    solve_count += 1
                    solve_times.append(result['solve_time_ms'])

                    # ── Progressive constraint relaxation (Plan D) ──
                    # When primary solve fails, retry with up to 3 levels of
                    # progressive relaxation. Only uses the relaxed solution
                    # if it SOLVES. Relaxation order (safest first):
                    #   L1: reduce alteration by 15°, allow -0.1 rudder
                    #   L2: reduce CPA by 40%, remove alteration
                    #   L3: free rudder, CPA floor at 12m
                    retry_level = 0
                    MAX_RETRY = 3
                    while result['status'] != 'SOLVED' and retry_level < MAX_RETRY:
                        retry_level += 1
                        rc = deepcopy(solver_c)
                        if retry_level == 1:
                            rc['alteration_min_angle'] = max(
                                0.0, rc['alteration_min_angle'] - math.radians(15))
                            rc['alteration_active'] = rc['alteration_min_angle'] > 0.01
                            rc['tau_r_min'] = max(
                                rc['tau_r_min'], -0.1 * nmpc_params.max_yaw_moment)
                        elif retry_level == 2:
                            for name in rc['cpa_radius_per_target']:
                                rc['cpa_radius_per_target'][name] *= 0.6
                            rc['alteration_min_angle'] = 0.0
                            rc['alteration_active'] = False
                        elif retry_level == 3:
                            rc['tau_r_min'] = -nmpc_params.max_yaw_moment
                            rc['tau_r_max'] = nmpc_params.max_yaw_moment
                            rc['alteration_min_angle'] = 0.0
                            rc['alteration_active'] = False
                            for name in rc['cpa_radius_per_target']:
                                rc['cpa_radius_per_target'][name] = max(
                                    12.0, rc['cpa_radius_per_target'][name] * 0.5)
                        rr = solver.solve(x0=x0, x_ref=x_ref, target_trajs=target_trajs,
                                          constraints=rc, tau_env=tau_env_base,
                                          warm_start=False)
                        solve_count += 1
                        solve_times.append(rr['solve_time_ms'])
                        if rr['status'] == 'SOLVED':
                            result = rr
                            retry_stats[retry_level] += 1

                    if result['status'] == 'SOLVED':
                        tau_u = float(result['u_opt'][0, 0])
                        tau_r_nmpc = float(result['u_opt'][1, 0])
                        last_u_opt = (tau_u, tau_r_nmpc)

                        # ── LLM-NMPC compliance gap recording (ablation) ──
                        # Record what the referee/LLM suggested vs what NMPC actually did.
                        # In Fossen convention: +τ_r = PORT, -τ_r = STARBOARD
                        run_metrics.llm_required_maneuvers.append(
                            str(ref_output.required_maneuver.value) if hasattr(
                                ref_output.required_maneuver, 'value') else str(ref_output.required_maneuver))
                        run_metrics.llm_forbidden_maneuvers.append(
                            str(ref_output.forbidden_maneuver.value) if hasattr(
                                ref_output.forbidden_maneuver, 'value') else str(ref_output.forbidden_maneuver))
                        actual_dir = "starboard" if tau_r_nmpc < 0 else ("port" if tau_r_nmpc > 0 else "none")
                        run_metrics.actual_turn_directions.append(actual_dir)
                        # Check disagreement: LLM says "alter_to_starboard" but NMPC gives port rudder (or vice versa)
                        ref_forbids = str(ref_output.forbidden_maneuver.value) if hasattr(
                            ref_output.forbidden_maneuver, 'value') else str(ref_output.forbidden_maneuver)
                        if ref_forbids == 'alter_to_port' and actual_dir == 'port':
                            run_metrics.llm_nmpc_disagreement_count += 1
                        elif ref_forbids == 'alter_to_starboard' and actual_dir == 'starboard':
                            run_metrics.llm_nmpc_disagreement_count += 1

                        # ── Clamp NMPC yaw moment to PD-safe range to prevent circling ──
                        # Without this, NMPC's ±400 Nm yaw moments overpower PD braking
                        # (±60 Nm) and cause excessive turn (>150°) — looks like circling.
                        # ±60 Nm cap preserves NMPC steering guidance while keeping
                        # total turn within ~50-90° (single controlled avoidance arc).
                        tau_r = max(-60.0, min(60.0, tau_r_nmpc))
                        tau = np.array([tau_u, 0.0, tau_r])
                    else:
                        infeasible_count += 1
                except Exception:
                    infeasible_count += 1

            # ── Record control effort ──
            tau_u_val = float(tau[0])
            tau_r_val = float(tau[2])  # tau = [surge, sway(=0), yaw]
            run_metrics.thrust_history.append(tau_u_val)
            run_metrics.rudder_history.append(tau_r_val)

            if step > 0:
                run_metrics.mean_rudder_rate += abs(tau_r_val - prev_rudder) / cfg.control_period
            prev_rudder = tau_r_val
            prev_thrust = tau_u_val

            # ── Step own ship ──
            ship.step(tau)

            # ── Early exit: all ships safely past ──
            if step > cfg.min_sim_steps and all_dists:
                if min(all_dists) > cfg.cpa_deadband:
                    # All targets are far away → safe, terminate early
                    # But check if any target is still approaching
                    approaching = False
                    for name, ts_data in targets.items():
                        ts_pos = np.array([ts_data['x'], ts_data['y']])
                        rel_pos = ts_pos - ship.eta[:2]
                        dist = float(np.linalg.norm(rel_pos))
                        if dist < 2 * cfg.cpa_deadband:
                            approaching = True
                            break
                    if not approaching:
                        break

        # ── Finalize metrics ──
        run_metrics.solve_count = solve_count
        run_metrics.num_infeasible = infeasible_count
        run_metrics.solve_success_rate = (1 - infeasible_count / max(solve_count, 1))
        run_metrics.avg_solve_time_ms = np.mean(solve_times) if solve_times else 0.0
        run_metrics.retry_level1_successes = retry_stats[1]
        run_metrics.retry_level2_successes = retry_stats[2]
        run_metrics.retry_level3_successes = retry_stats[3]

        if run_metrics.thrust_history:
            run_metrics.thrust_std = float(np.std(run_metrics.thrust_history))
            run_metrics.max_rudder = float(np.max(np.abs(run_metrics.rudder_history)))
        if run_metrics.surge_history:
            run_metrics.avg_surge = float(np.mean(run_metrics.surge_history))
        if len(run_metrics.t_history) > 1:
            run_metrics.mean_rudder_rate /= (len(run_metrics.t_history) - 1)

        # Compute compliance for head-on
        run_metrics = compute_all_metrics(
            run_metrics, expected_turn_sign=1, expected_encounter='head_on'
        )

        return run_metrics

    # =====================================================================
    # Batch execution
    # =====================================================================

    def run_batch(self, scenario_id: str = None) -> MetricsCollector:
        """Run all Monte Carlo repeats for a scenario."""
        cfg = self.config
        if scenario_id:
            cfg.scenario_id = scenario_id

        scenario = self.load_scenario(cfg.scenario_id)
        print(f"Scenario: {scenario['key']}")
        print(f"Description: {scenario['data']['description'][:100]}...")
        print(f"Repeats: {cfg.num_repeats}")
        print(f"Duration: {cfg.sim_duration}s")
        print(f"{'='*60}")

        t_start = time.perf_counter()

        # ── LLM backends are NOT compatible with multiprocessing ──
        # GrammarConstrainedReferee holds GPU memory (non-picklable).
        # OllamaReferee targets a single Ollama server — concurrent requests contend.
        if cfg.parallel_workers > 1 and cfg.referee_backend in ("ollama", "ollama_no_cfg", "grammar_constrained"):
            print(f"  WARNING: LLM backend '{cfg.referee_backend}' incompatible with "
                  f"multiprocessing. Forcing parallel=1.")
            cfg.parallel_workers = 1

        if cfg.parallel_workers > 1:
            # Multiprocess
            with ProcessPoolExecutor(max_workers=cfg.parallel_workers) as executor:
                futures = {}
                for i in range(cfg.num_repeats):
                    seed = 42 + i * 137
                    fut = executor.submit(
                        self._run_single_worker, i, scenario, seed, cfg)
                    futures[fut] = i

                done_count = 0
                for fut in as_completed(futures):
                    i = futures[fut]
                    done_count += 1
                    try:
                        metrics = fut.result(timeout=120)
                        self.metrics.add_run(metrics)
                    except Exception as e:
                        print(f"  Run {i}: FAILED — {e}")
                    if done_count % 10 == 0:
                        print(f"  Progress: {done_count}/{cfg.num_repeats}")
        else:
            # Single process
            for i in range(cfg.num_repeats):
                seed = 42 + i * 137
                metrics = self.run_single(i, scenario, seed)
                self.metrics.add_run(metrics)

                if (i + 1) % 10 == 0:
                    t_elapsed = time.perf_counter() - t_start
                    eta = t_elapsed / (i + 1) * (cfg.num_repeats - i - 1)
                    print(f"  [{i+1}/{cfg.num_repeats}] "
                          f"min_cpa={metrics.min_cpa:.1f}m "
                          f"| elapsed={t_elapsed:.0f}s | ETA={eta:.0f}s "
                          f"| solve_rate={metrics.solve_success_rate:.2f} "
                          f"| avg_solve={metrics.avg_solve_time_ms:.0f}ms")

        t_total = time.perf_counter() - t_start
        print(f"\n{'='*60}")
        print(f"Completed {cfg.num_repeats} runs in {t_total:.1f}s "
              f"({t_total/cfg.num_repeats:.2f}s/run)")

        return self.metrics

    @staticmethod
    def _run_single_worker(run_id: int, scenario: dict, seed: int,
                           config: MonteCarloConfig = None) -> RunMetrics:
        """Worker function for multiprocessing (must be picklable)."""
        runner = BatchRunner(config)
        return runner.run_single(run_id, scenario, seed)
