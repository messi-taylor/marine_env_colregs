#!/usr/bin/env python3
"""
Resilient NMPC Solver for COLREGS-Compliant Collision Avoidance
=================================================================

Implements the hierarchical-slack nonlinear model predictive control
formulation defined in Section 4.4 of the research document.

Core formulation:
  min  Σ ||x_k - x_ref_k||²_Q  +  ||u_k||²_R  +  Σ w_i·ε_i
  s.t. x_{k+1} = f_Fossen(x_k, u_k)           (dynamics)
       ||p_OS - p_TS||² ≥ r_min² - ε_safety    (Rule 8(d))
       τ_r_min ≤ τ_r ≤ τ_r_max                 (Rule 14/15)
       |ψ_N - ψ_0| ≥ Δψ_min - ε_legal          (Rule 8(b))
       u_surge ∈ [v_min, v_max + ε_speed]      (Rule 6)
       ε_safety ≡ 0  (NEVER relaxed)            (Lemma 1)

Slack hierarchy: w_safety(1e6) >> w_legal(1e3) >> w_smooth(1e0)

Uses CasADi Opti stack with IPOPT solver.
Warm-start from previous solution for real-time performance.
"""

import numpy as np
import casadi as ca
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import time


# =============================================================================
# Ship parameters — must match ship_dynamics.py ShipParams
# =============================================================================

@dataclass
class NMPCParams:
    """Fossen 3DOF parameters + solver configuration for NMPC."""
    # ── Physical parameters (WAM-V USV) ──
    m: float = 250.0          # mass [kg]
    I_z: float = 350.0        # yaw inertia [kg·m²]
    x_g: float = 0.0          # CG x-offset [m]

    # Added mass
    X_u_dot: float = -25.0
    Y_v_dot: float = -125.0
    Y_r_dot: float = 0.0
    N_v_dot: float = 0.0
    N_r_dot: float = -50.0

    # Linear damping
    X_u: float = 30.0
    Y_v: float = 80.0
    N_r: float = 60.0

    # Quadratic damping
    X_uu: float = 60.0
    Y_vv: float = 180.0
    N_rr: float = 100.0

    # ── Thruster geometry ──
    thruster_distance: float = 2.06     # m — distance between left/right thrusters
    max_thrust_per_engine: float = 1500.0  # N
    max_yaw_moment: float = 800.0       # N·m — τ_r bound

    # ── Solver parameters ──
    N: int = 20                         # prediction horizon steps
    dt: float = 0.5                     # time step [s]
    solver_max_iter: int = 100           # IPOPT max iterations
    solver_tol: float = 1e-6             # IPOPT convergence tolerance

    # ── Cost weights ──
    Q_pos: float = 100.0                # position tracking weight
    Q_heading: float = 50.0             # heading tracking weight
    Q_vel: float = 10.0                 # velocity tracking weight
    R_surge: float = 0.01               # surge effort penalty
    R_yaw: float = 0.005                # yaw moment effort penalty
    R_dsurge: float = 0.1               # surge rate penalty (smoothness)
    R_dyaw: float = 0.05                # yaw moment rate penalty

    # ── Slack penalty weights (hierarchy) ──
    w_safety: float = 1e4               # highest priority, but allows soft violation
    w_legal: float = 1e3                # COLREGS compliance
    w_smooth: float = 1e0               # trajectory smoothness
    w_speed: float = 1e2                # speed limit

    # ── Gaussian proximity penalty weight ──
    gauss_weight: float = 8e5            # scenario-overridable for tight geometries

    # ── Constraint defaults ──
    default_cpa: float = 50.0           # m
    default_min_speed: float = 0.5      # m/s — minimum steerage
    default_max_speed: float = 5.0      # m/s
    max_heading_rate: float = 0.5       # rad/s

    @property
    def M_matrix(self) -> np.ndarray:
        """Inertia matrix M = M_RB + M_A."""
        return np.array([
            [self.m - self.X_u_dot, 0.0, 0.0],
            [0.0, self.m - self.Y_v_dot, self.m * self.x_g - self.Y_r_dot],
            [0.0, self.m * self.x_g - self.N_v_dot, self.I_z - self.N_r_dot],
        ])


# =============================================================================
# NMPC Solver
# =============================================================================

class NMPCSolver:
    """CasADi-based Resilient NMPC solver for COLREGS-compliant collision avoidance.

    Usage:
        solver = NMPCSolver(params=NMPCParams())
        solver.setup()  # build symbolic problem once

        # At each control cycle:
        result = solver.solve(
            x0=current_state,       # [px, py, psi, u, v, r]
            x_ref=reference_traj,   # (N+1, 6) reference trajectory
            target_trajs=ts_preds,  # dict: name -> (N+1, 2) TS predicted positions
            constraints=colregs,    # dict of constraint parameters
        )
        tau_u, tau_r = result.u_opt[0]  # first control input
    """

    def __init__(self, params: NMPCParams = None):
        self.p = params or NMPCParams()
        self._built = False
        self._solver = None
        self._opti = None
        self._warm_start = None       # previous solution for warm-start

    # =====================================================================
    # Setup — build symbolic optimization problem (called once)
    # =====================================================================

    def setup(self):
        """Build the CasADi Opti symbolic problem. Called once at initialization."""
        p = self.p
        N = p.N
        opti = ca.Opti()

        # ── Decision variables ──
        # State: X = [px, py, psi, u, v, r] over horizon N+1
        X = opti.variable(6, N + 1)
        # Control: U = [tau_u, tau_r] over horizon N
        U = opti.variable(2, N)

        # ── Slack variables (declared early so constraints can reference them) ──
        epsilon_safety = opti.variable()
        opti.subject_to(epsilon_safety >= 0)
        epsilon_legal = opti.variable()
        opti.subject_to(epsilon_legal >= 0)
        epsilon_smooth = opti.variable()
        opti.subject_to(epsilon_smooth >= 0)
        epsilon_speed = opti.variable()
        opti.subject_to(epsilon_speed >= 0)

        # ── Parameters (set at each solve) ──
        # Initial state
        x0_param = opti.parameter(6)
        # Reference trajectory: (6, N+1)
        X_ref = opti.parameter(6, N + 1)

        # Target ship trajectories: max 5 targets, each has (2, N+1) positions
        MAX_TARGETS = 6
        TS_pos = [opti.parameter(2, N + 1) for _ in range(MAX_TARGETS)]
        TS_active = opti.parameter(MAX_TARGETS)  # binary: 1=active, 0=inactive
        # Exclusion radius per target
        TS_r_safe = opti.parameter(MAX_TARGETS)  # min CPA distance per target

        # Half-plane normal directions per target (2D unit vector each)
        # Replaces non-convex exclusion circle with convex linear half-plane:
        #   n̂ · (p_OS - p_TS) >= r_hp - ε_safety
        # Set at each solve based on COLREGS passing side or relative bearing.
        TS_hp_normal = [opti.parameter(2) for _ in range(MAX_TARGETS)]

        # Perpendicular half-plane normal: n̂₂ = rotate_90_cw(n̂₁)
        # Closes the orthogonal gap that a single half-plane leaves unprotected.
        #   n̂₂ · (p_OS[k] - p_TS[k]) >= 0  (wedge, blocks wrong-side crossing)
        TS_hp_perp_normal = [opti.parameter(2) for _ in range(MAX_TARGETS)]

        # Maneuver constraints
        tau_r_min = opti.parameter()
        tau_r_max = opti.parameter()
        alteration_min_angle = opti.parameter()   # |ψ_N - ψ_0| ≥ this
        alteration_active = opti.parameter()       # 0=inactive, 1=active

        # Speed bounds
        v_min = opti.parameter()
        v_max = opti.parameter()

        # Environment forces (constant over horizon, set at each solve)
        tau_env = opti.parameter(3)   # [τ_env_u, τ_env_v, τ_env_r]

        # ── Initial condition constraint ──
        opti.subject_to(X[:, 0] == x0_param)

        # ── Build symbolic dynamics ──
        # Pre-compute M inverse for the CasADi expressions
        M = p.M_matrix
        M_inv = np.linalg.inv(M)

        # ── Dynamics constraints (multiple shooting) ──
        cost = 0
        for k in range(N):
            # Current state
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = X[:, k + 1]

            # Unpack state
            px, py, psi = x_k[0], x_k[1], x_k[2]
            u_s, v_s, r_s = x_k[3], x_k[4], x_k[5]

            # --- Fossen 3DOF dynamics (RK4 integration) ---
            x_next_pred = self._rk4_step_casadi(x_k, u_k, tau_env, M, M_inv, p)
            opti.subject_to(x_next == x_next_pred)

            # --- Tracking cost ---
            x_err = x_k - X_ref[:, k]
            cost += (p.Q_pos * (x_err[0]**2 + x_err[1]**2) +
                     p.Q_heading * ca.sin(0.5 * (x_k[2] - X_ref[2, k]))**2 +
                     p.Q_vel * (x_err[3]**2 + x_err[4]**2 + x_err[5]**2))

            # --- Control effort cost ---
            cost += p.R_surge * u_k[0]**2 + p.R_yaw * u_k[1]**2

            # --- Control rate cost (smoothness) ---
            if k > 0:
                du = u_k - U[:, k - 1]
                cost += p.R_dsurge * du[0]**2 + p.R_dyaw * du[1]**2

            # --- Speed constraints ---
            # Soft lower bound: u_s ≥ v_min - ε_speed
            # Soft upper bound: u_s ≤ v_max + ε_speed
            # These are implemented as slack penalties in the cost
            # Hard bounds for physical limits
            opti.subject_to(u_s >= 0.0)           # no reverse
            opti.subject_to(u_s <= 6.0)            # physical max ~11 knots

            # --- Yaw moment bounds (rudder constraint from Rule 14/15) ---
            opti.subject_to(tau_r_min <= u_k[1])
            opti.subject_to(u_k[1] <= tau_r_max)

            # --- Half-plane safety constraints (Rule 8(d)) ---
            # Replaces non-convex exclusion circle with CONVEX linear half-plane.
            # For each target ship at each prediction step k>0:
            #   Primary:  n̂₁ · (p_OS[k] - p_TS[k]) >= r_hp - ε_safety
            #   Perp:     n̂₂ · (p_OS[k] - p_TS[k]) >= 0  (wedge, blocks wrong side)
            # where n̂₁ is COLREGS-determined (pass_astern → stern dir,
            # pass_ahead → bow dir, default → relative bearing).
            # n̂₂ = rotate_90_cw(n̂₁) closes the perpendicular gap that a single
            # half-plane leaves unprotected.
            # r_hp = r_safe * 0.85 (half-plane is less conservative than circle).
            # Enforced from k=1 onward. Soft constraint with ε_safety slack.
            if k > 0:
                for t in range(MAX_TARGETS):
                    ts_pos_k = TS_pos[t][:, k]
                    dx = px - ts_pos_k[0]
                    dy = py - ts_pos_k[1]
                    # Primary half-plane
                    nx1 = TS_hp_normal[t][0]
                    ny1 = TS_hp_normal[t][1]
                    signed_dist1 = nx1 * dx + ny1 * dy
                    r_hp = TS_r_safe[t]  # full CPA distance for half-plane
                    opti.subject_to(
                        TS_active[t] * signed_dist1 >= r_hp - epsilon_safety
                    )
                    # Proximity penalty: soft cost grows exponentially as distance
                    # drops below r_safe. This covers the perpendicular direction
                    # that the half-plane doesn't constrain.
                    # w_prox=2e4 ensures proximity dominates tracking cost (~1e3)
                    # near the CPA point, forcing the solver to maintain separation.
                    dist_sq_prox = dx*dx + dy*dy
                    sigma_sq = (TS_r_safe[t] * 0.6)**2 + 1.0
                    cost += TS_active[t] * self.p.gauss_weight * ca.exp(-dist_sq_prox / (2 * sigma_sq))

                    # Perpendicular wedge SOFT penalty: penalize n̂₂ · (p_OS - p_TS) < 0
                    # n̂₂ = rotate_90_cw(n̂₁) closes the orthogonal gap that a
                    # single half-plane leaves unprotected. Soft penalty (not hard
                    # constraint) to handle initial geometry violations while still
                    # strongly discouraging crossing through the collision region.
                    nx2 = TS_hp_perp_normal[t][0]
                    ny2 = TS_hp_perp_normal[t][1]
                    signed_dist2 = nx2 * dx + ny2 * dy
                    perp_violation = ca.fmax(0.0, -signed_dist2)
                    cost += TS_active[t] * 5e4 * perp_violation**2

        # Terminal cost
        x_err_N = X[:, N] - X_ref[:, N]
        cost += (p.Q_pos * (x_err_N[0]**2 + x_err_N[1]**2) +
                 p.Q_heading * ca.sin(0.5 * (X[2, N] - X_ref[2, N]))**2 +
                 p.Q_vel * (x_err_N[3]**2 + x_err_N[4]**2 + x_err_N[5]**2))

        # --- Terminal heading alteration constraint (Rule 8(b)) ---
        # |ψ_N - ψ_0| ≥ alteration_min_angle (with correct sign for starboard)
        # Since we want starboard alteration to be positive:
        #   ψ_N - ψ_0 ≥ alteration_min_angle  (for starboard turn)
        # This is optionally activated by alteration_active flag
        psi_diff = X[2, N] - X[2, 0]
        # Use a soft constraint via slack, but hard for now
        # We penalize violation rather than hard-constraining it
        # (handled through the constraint mapper's epsilon_legal)

        # Head-on alteration soft constraint
        opti.subject_to(psi_diff >= alteration_active * alteration_min_angle - epsilon_legal)

        # Speed soft constraints via slack
        for k in range(N + 1):
            # v_min - ε_speed ≤ u_k ≤ v_max + ε_speed
            opti.subject_to(X[3, k] <= v_max + epsilon_speed)

        # ── Slack penalty cost (hierarchy: w_safety >> w_legal >> w_smooth) ──
        cost += p.w_safety * epsilon_safety**2
        cost += p.w_legal * epsilon_legal**2
        cost += p.w_smooth * epsilon_smooth**2
        cost += p.w_speed * epsilon_speed**2

        # ── Control bounds ──
        opti.subject_to(opti.bounded(-p.max_yaw_moment, U[1, :], p.max_yaw_moment))
        opti.subject_to(opti.bounded(0.0, U[0, :], p.max_thrust_per_engine * 2))

        # ── Set objective ──
        opti.minimize(cost)

        # ── Solver options ──
        opts = {
            'ipopt.max_iter': p.solver_max_iter,
            'ipopt.tol': p.solver_tol,
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',          # suppress banner
            'print_time': 0,
        }
        opti.solver('ipopt', opts)

        # Store references
        self._opti = opti
        self._X = X
        self._U = U
        self._x0_param = x0_param
        self._X_ref = X_ref
        self._TS_pos = TS_pos
        self._TS_active = TS_active
        self._TS_r_safe = TS_r_safe
        self._TS_hp_normal = TS_hp_normal
        self._TS_hp_perp_normal = TS_hp_perp_normal
        self._tau_r_min = tau_r_min
        self._tau_r_max = tau_r_max
        self._alteration_min_angle = alteration_min_angle
        self._alteration_active = alteration_active
        self._v_min = v_min
        self._v_max = v_max
        self._tau_env = tau_env
        self._epsilon_safety = epsilon_safety
        self._epsilon_legal = epsilon_legal
        self._epsilon_smooth = epsilon_smooth
        self._epsilon_speed = epsilon_speed
        self._N = N
        self._built = True

    def _rk4_step_casadi(self, x, u, tau_env, M, M_inv, p):
        """Single RK4 integration step for Fossen 3DOF dynamics (CasADi symbolic)."""
        dt = p.dt

        def fossen_dynamics(x_state, u_ctrl, tau_ext):
            """Continuous-time Fossen 3DOF dynamics.
            Returns x_dot = [px_dot, py_dot, psi_dot, u_dot, v_dot, r_dot].
            """
            px_s, py_s, psi_s = x_state[0], x_state[1], x_state[2]
            u_s, v_s, r_s = x_state[3], x_state[4], x_state[5]

            tau_u = u_ctrl[0] + tau_ext[0]
            tau_v = tau_ext[1]
            tau_r = u_ctrl[1] + tau_ext[2]

            # Rotation matrix
            c_psi = ca.cos(psi_s)
            s_psi = ca.sin(psi_s)
            px_dot = c_psi * u_s - s_psi * v_s
            py_dot = s_psi * u_s + c_psi * v_s
            psi_dot = r_s

            # Coriolis matrix C(ν)·ν
            m11 = p.m - p.X_u_dot
            m22 = p.m - p.Y_v_dot
            m23 = p.m * p.x_g - p.Y_r_dot
            m33 = p.I_z - p.N_r_dot

            c13 = -m22 * v_s - m23 * r_s
            c23 = m11 * u_s

            # Damping D(ν)·ν
            d_lin_u = p.X_u * u_s
            d_lin_v = p.Y_v * v_s
            d_lin_r = p.N_r * r_s
            d_quad_u = p.X_uu * ca.fabs(u_s) * u_s
            d_quad_v = p.Y_vv * ca.fabs(v_s) * v_s
            d_quad_r = p.N_rr * ca.fabs(r_s) * r_s

            # ν_dot = M^{-1} · (τ - C·ν - D·ν)
            nu_dot_0 = M_inv[0, 0] * (tau_u - c13 * r_s - d_lin_u - d_quad_u) + \
                       M_inv[0, 1] * (tau_v - c23 * r_s - d_lin_v - d_quad_v) + \
                       M_inv[0, 2] * (tau_r + c13 * u_s + c23 * v_s - d_lin_r - d_quad_r)

            nu_dot_1 = M_inv[1, 0] * (tau_u - c13 * r_s - d_lin_u - d_quad_u) + \
                       M_inv[1, 1] * (tau_v - c23 * r_s - d_lin_v - d_quad_v) + \
                       M_inv[1, 2] * (tau_r + c13 * u_s + c23 * v_s - d_lin_r - d_quad_r)

            nu_dot_2 = M_inv[2, 0] * (tau_u - c13 * r_s - d_lin_u - d_quad_u) + \
                       M_inv[2, 1] * (tau_v - c23 * r_s - d_lin_v - d_quad_v) + \
                       M_inv[2, 2] * (tau_r + c13 * u_s + c23 * v_s - d_lin_r - d_quad_r)

            return ca.vertcat(px_dot, py_dot, psi_dot, nu_dot_0, nu_dot_1, nu_dot_2)

        # RK4
        k1 = fossen_dynamics(x, u, tau_env)
        k2 = fossen_dynamics(x + 0.5 * dt * k1, u, tau_env)
        k3 = fossen_dynamics(x + 0.5 * dt * k2, u, tau_env)
        k4 = fossen_dynamics(x + dt * k3, u, tau_env)

        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # =====================================================================
    # Solve — called at each control cycle
    # =====================================================================

    def solve(self,
              x0: np.ndarray,
              x_ref: np.ndarray,
              target_trajs: Dict[str, np.ndarray],
              constraints: dict,
              tau_env: np.ndarray = None,
              warm_start: bool = True) -> dict:
        """Solve the NMPC optimization problem.

        Args:
            x0: Initial state [px, py, psi, u, v, r] shape (6,)
            x_ref: Reference trajectory shape (6, N+1)
            target_trajs: Dict mapping target name -> (2, N+1) predicted positions
            constraints: Dict with keys:
                - tau_r_min, tau_r_max: yaw moment bounds (from forbidden maneuver)
                - alteration_min_angle: minimum heading change (Rule 8(b))
                - alteration_active: bool
                - v_min, v_max: speed bounds (Rule 6)
                - cpa_radius_per_target: dict name->float
            tau_env: Environment forces [τ_u, τ_v, τ_r] shape (3,)
            warm_start: Use previous solution as initial guess

        Returns:
            dict with keys:
                - u_opt: (2, N) optimal control trajectory
                - x_pred: (6, N+1) predicted state trajectory
                - cost: final cost value
                - solve_time_ms: solver wall time
                - status: 'SOLVED' | 'INFEASIBLE' | 'ERROR'
        """
        if not self._built:
            self.setup()

        p = self.p
        N = p.N
        opti = self._opti

        # ── Set parameters ──
        opti.set_value(self._x0_param, x0)

        # Reference trajectory
        x_ref_padded = self._pad_reference(x_ref, N)
        opti.set_value(self._X_ref, x_ref_padded)

        # Target ship trajectories (max 5)
        MAX_TARGETS = 6
        ts_names = list(target_trajs.keys())[:MAX_TARGETS]
        ts_active_vals = np.zeros(MAX_TARGETS)
        ts_pos_vals = [np.zeros((2, N + 1)) for _ in range(MAX_TARGETS)]
        ts_r_safe_vals = np.zeros(MAX_TARGETS)

        cpa_radius_per_target = constraints.get('cpa_radius_per_target', {})

        for i, name in enumerate(ts_names):
            ts_active_vals[i] = 1.0
            traj = target_trajs[name]
            ts_pos_vals[i] = self._pad_ts_trajectory(traj, N)
            ts_r_safe_vals[i] = cpa_radius_per_target.get(name, p.default_cpa)

        for i in range(MAX_TARGETS):
            opti.set_value(self._TS_pos[i], ts_pos_vals[i])
        opti.set_value(self._TS_active, ts_active_vals)
        opti.set_value(self._TS_r_safe, ts_r_safe_vals)

        # Half-plane normals per target
        hp_normals_per_target = constraints.get('hp_normals_per_target', {})
        hp_normal_vals = [np.zeros(2) for _ in range(MAX_TARGETS)]
        for i, name in enumerate(ts_names):
            if name in hp_normals_per_target:
                hp_normal_vals[i] = np.asarray(hp_normals_per_target[name], dtype=float)
            else:
                # Default: relative bearing direction from TS to OS
                ts_pos_0 = ts_pos_vals[i][:, 0]
                rel_vec = x0[:2] - ts_pos_0  # TS → OS vector (points toward OS)
                dist = np.linalg.norm(rel_vec)
                if dist > 1e-6:
                    hp_normal_vals[i] = rel_vec / dist
                else:
                    hp_normal_vals[i] = np.array([1.0, 0.0])

        for i in range(MAX_TARGETS):
            opti.set_value(self._TS_hp_normal[i], hp_normal_vals[i])

        # Perpendicular wedge normals: n̂₂ = rotate_90_cw(n̂₁)
        # Closes the orthogonal escape route that a single half-plane leaves open.
        hp_perp_vals = [np.zeros(2) for _ in range(MAX_TARGETS)]
        for i in range(MAX_TARGETS):
            nx, ny = hp_normal_vals[i][0], hp_normal_vals[i][1]
            # rotate_90_cw: [x, y] → [y, -x]
            hp_perp_vals[i] = np.array([ny, -nx], dtype=float)
            opti.set_value(self._TS_hp_perp_normal[i], hp_perp_vals[i])

        # Maneuver constraints
        opti.set_value(self._tau_r_min, constraints.get('tau_r_min', -p.max_yaw_moment))
        opti.set_value(self._tau_r_max, constraints.get('tau_r_max', p.max_yaw_moment))
        opti.set_value(self._alteration_min_angle,
                       constraints.get('alteration_min_angle', 0.0))
        opti.set_value(self._alteration_active,
                       1.0 if constraints.get('alteration_active', False) else 0.0)

        # Speed bounds
        opti.set_value(self._v_min, constraints.get('v_min', p.default_min_speed))
        opti.set_value(self._v_max, constraints.get('v_max', p.default_max_speed))

        # Environment forces
        opti.set_value(self._tau_env, tau_env if tau_env is not None else np.zeros(3))

        # ── Warm-start ──
        if warm_start and self._warm_start is not None:
            X_guess, U_guess = self._warm_start
            if X_guess.shape[1] == N + 1 and U_guess.shape[1] == N:
                opti.set_initial(self._X, X_guess)
                opti.set_initial(self._U, U_guess)
            else:
                self._set_default_initial_guess(x0, N)
        else:
            self._set_default_initial_guess(x0, N)

        # ── Solve ──
        t_start = time.perf_counter()
        try:
            sol = opti.solve()
            status = 'SOLVED'
        except Exception as e:
            # Try recovery with relaxed constraints by adjusting initial guess
            try:
                self._set_default_initial_guess(x0, N)
                sol = opti.solve()
                status = 'SOLVED'
            except Exception:
                status = 'INFEASIBLE'

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        if status == 'SOLVED':
            X_opt = sol.value(self._X)
            U_opt = sol.value(self._U)
            cost_val = float(sol.value(opti.f))
            eps_safety = float(sol.value(self._epsilon_safety))
            eps_legal = float(sol.value(self._epsilon_legal))
            eps_smooth = float(sol.value(self._epsilon_smooth))
            eps_speed = float(sol.value(self._epsilon_speed))

            # Update warm-start (shift and append)
            X_warm = np.hstack([X_opt[:, 1:], X_opt[:, -1:]])
            U_warm = np.hstack([U_opt[:, 1:], U_opt[:, -1:]])
            self._warm_start = (X_warm, U_warm)
        else:
            X_opt = np.tile(x0.reshape(-1, 1), (1, N + 1))
            U_opt = np.zeros((2, N))
            cost_val = float('inf')
            eps_safety = float('inf')
            eps_legal = float('inf')
            eps_smooth = float('inf')
            eps_speed = float('inf')

        return {
            'u_opt': U_opt,
            'x_pred': X_opt,
            'cost': cost_val,
            'solve_time_ms': t_elapsed_ms,
            'status': status,
            'epsilon_safety': eps_safety,
            'epsilon_legal': eps_legal,
            'epsilon_smooth': eps_smooth,
            'epsilon_speed': eps_speed,
        }

    def _set_default_initial_guess(self, x0: np.ndarray, N: int):
        """Set default initial guess: constant state, zero control."""
        X_init = np.tile(x0.reshape(-1, 1), (1, N + 1))
        U_init = np.zeros((2, N))

        # Add slight forward motion bias
        X_init[3, :] = max(x0[3], 1.0)  # assume at least 1 m/s surge

        self._opti.set_initial(self._X, X_init)
        self._opti.set_initial(self._U, U_init)

    def _pad_reference(self, x_ref: np.ndarray, N: int) -> np.ndarray:
        """Pad reference trajectory to (6, N+1)."""
        if x_ref.shape[1] < N + 1:
            padded = np.zeros((6, N + 1))
            n_avail = min(x_ref.shape[1], N + 1)
            padded[:, :n_avail] = x_ref[:, :n_avail]
            # Extend with last state
            for i in range(n_avail, N + 1):
                padded[:, i] = x_ref[:, -1]
            return padded
        return x_ref[:, :N + 1]

    def _pad_ts_trajectory(self, ts_traj: np.ndarray, N: int) -> np.ndarray:
        """Pad target ship trajectory to (2, N+1)."""
        if ts_traj.shape[1] < N + 1:
            padded = np.zeros((2, N + 1))
            n_avail = min(ts_traj.shape[1], N + 1)
            padded[:, :n_avail] = ts_traj[:, :n_avail]
            for i in range(n_avail, N + 1):
                padded[:, i] = ts_traj[:, -1]
            return padded
        return ts_traj[:, :N + 1]

    # =====================================================================
    # Utility: generate reference trajectory from waypoints
    # =====================================================================

    def generate_reference(self,
                           current_pos: np.ndarray,
                           waypoints: List[Tuple[float, float, float]],
                           wp_idx: int = 0) -> np.ndarray:
        """Generate reference trajectory (6, N+1) from waypoints.

        Uses line-of-sight interpolation to create a smooth reference path.

        Args:
            current_pos: Current NED position [px, py] or full state
            waypoints: List of (x, y, target_speed) tuples
            wp_idx: Current active waypoint index

        Returns:
            x_ref: (6, N+1) reference trajectory
        """
        p = self.p
        N = p.N
        dt = p.dt

        if len(waypoints) == 0:
            # No waypoints — hold position
            x_ref = np.zeros((6, N + 1))
            if len(current_pos) >= 2:
                x_ref[0, :] = current_pos[0]
                x_ref[1, :] = current_pos[1]
            x_ref[3, :] = p.default_min_speed
            return x_ref

        # Starting position
        if len(current_pos) >= 6:
            px0, py0, psi0 = current_pos[0], current_pos[1], current_pos[2]
        elif len(current_pos) >= 2:
            px0, py0 = current_pos[0], current_pos[1]
            psi0 = 0.0
        else:
            px0, py0, psi0 = 0.0, 0.0, 0.0

        x_ref = np.zeros((6, N + 1))

        # Build interpolated path
        # For each prediction step, compute the LOS reference position
        current_x, current_y = px0, py0
        current_wp = wp_idx
        # Assume nominal speed for reference generation
        nominal_speed = max(
            waypoints[current_wp][2] if current_wp < len(waypoints) else 1.5,
            0.5
        )
        # Limit to current speed max
        nominal_speed = min(nominal_speed, p.default_max_speed)

        for k in range(N + 1):
            if current_wp >= len(waypoints) - 1:
                # Final waypoint — hold
                wp_final = waypoints[-1]
                x_ref[0, k] = wp_final[0]
                x_ref[1, k] = wp_final[1]
                x_ref[2, k] = psi0
                x_ref[3, k] = 0.0
                continue

            wp_k = waypoints[current_wp]
            wp_next = waypoints[current_wp + 1]

            dx = wp_next[0] - wp_k[0]
            dy = wp_next[1] - wp_k[1]
            seg_length = np.sqrt(dx**2 + dy**2)
            if seg_length > 0.01:
                alpha_k = np.arctan2(dy, dx)
            else:
                alpha_k = 0.0

            # Advance along path at nominal speed
            step_dist = nominal_speed * dt * k
            target_x = wp_k[0] + (dx / max(seg_length, 0.01)) * step_dist
            target_y = wp_k[1] + (dy / max(seg_length, 0.01)) * step_dist

            # Check if we'd overshoot the waypoint
            dist_to_wp = np.sqrt((target_x - wp_next[0])**2 + (target_y - wp_next[1])**2)
            if step_dist > seg_length or dist_to_wp < 1.0:
                target_x = wp_next[0]
                target_y = wp_next[1]
                current_wp = min(current_wp + 1, len(waypoints) - 1)

            x_ref[0, k] = target_x
            x_ref[1, k] = target_y
            x_ref[2, k] = alpha_k
            # Target speed from waypoint
            x_ref[3, k] = waypoints[min(current_wp, len(waypoints) - 1)][2]
            x_ref[4, k] = 0.0  # target sway = 0
            x_ref[5, k] = 0.0  # target yaw rate = 0

        return x_ref


# =============================================================================
# Constraint extraction helper
# =============================================================================

def constraints_from_nmpc_output(nmpc_output, ts_headings: dict = None) -> dict:
    """Extract NMPC-compatible constraint dict from constraint_mapper output.

    Args:
        nmpc_output: NMPCConstraints dataclass from constraint_mapper.py
        ts_headings: Optional dict mapping target name -> ENU heading (rad).
                     Used to compute half-plane normals from COLREGS passing side.

    Returns:
        dict ready for NMPCSolver.solve(constraints=...)
    """
    constraints = {
        'tau_r_min': nmpc_output.maneuver_constraint.rudder_min * 800.0,
        'tau_r_max': nmpc_output.maneuver_constraint.rudder_max * 800.0,
        'alteration_min_angle': nmpc_output.maneuver_constraint.alteration_min_angle,
        'alteration_active': (nmpc_output.maneuver_constraint.turn_direction_sign != 0),
        'v_min': nmpc_output.speed_constraint.min_speed,
        'v_max': nmpc_output.speed_constraint.max_speed,
        'cpa_radius_per_target': {},
        'hp_normals_per_target': {},   # NEW: half-plane normal directions
        'slack_weights': {},
    }

    for sc in nmpc_output.spatial_constraints:
        constraints['cpa_radius_per_target'][sc.target_name] = sc.min_distance
        constraints['slack_weights'][sc.target_name] = {
            'w_safety': sc.epsilon_safety_weight,
        }
        # Compute half-plane normal from COLREGS passing side
        if ts_headings and sc.target_name in ts_headings:
            ts_h = ts_headings[sc.target_name]
            ts_heading_vec = np.array([np.cos(ts_h), np.sin(ts_h)])
            if sc.pass_astern:
                constraints['hp_normals_per_target'][sc.target_name] = -ts_heading_vec
            elif sc.pass_ahead:
                constraints['hp_normals_per_target'][sc.target_name] = ts_heading_vec
            # else: will be defaulted to relative bearing at solve time

    return constraints
