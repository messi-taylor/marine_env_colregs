#!/usr/bin/env python3
"""
Fossen 3DOF underactuated ship dynamics model.
State: η = [x, y, ψ] (NED), ν = [u, v, r] (body-frame).
Control: τ = [τ_u, τ_r] (surge force, yaw moment) — no direct sway actuator.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ShipParams:
    """Fossen 3DOF hydrodynamic parameters for a small USV (~5m WAM-V)."""
    # Rigid-body
    m: float = 250.0       # mass [kg]
    I_z: float = 350.0     # yaw inertia [kg·m²]
    x_g: float = 0.0       # CG x-offset [m]

    # Added mass (negative of hydrodynamic derivatives)
    X_u_dot: float = -25.0
    Y_v_dot: float = -125.0
    Y_r_dot: float = 0.0
    N_v_dot: float = 0.0
    N_r_dot: float = -50.0

    # Linear damping
    X_u: float = 30.0      # surge linear drag
    Y_v: float = 80.0      # sway linear drag
    N_r: float = 60.0      # yaw linear drag

    # Nonlinear damping (quadratic)
    X_uu: float = 60.0     # surge quadratic drag
    Y_vv: float = 180.0    # sway quadratic drag
    N_rr: float = 100.0    # yaw quadratic drag


class FossenShip:
    """3DOF underactuated ship with Fossen dynamics."""

    def __init__(self, params: ShipParams = None, dt: float = 0.05):
        self.p = params or ShipParams()
        self.dt = dt

        self._build_matrices()

        # State: η = [x, y, ψ] (NED), ν = [u, v, r] (body)
        self.eta = np.zeros(3)
        self.nu = np.zeros(3)

    def _build_matrices(self):
        p = self.p
        # Mass matrix M = M_RB + M_A
        self.M = np.array([
            [p.m - p.X_u_dot, 0.0,               0.0],
            [0.0,             p.m - p.Y_v_dot,   p.m * p.x_g - p.Y_r_dot],
            [0.0,             p.m * p.x_g - p.N_v_dot, p.I_z - p.N_r_dot],
        ])
        self.M_inv = np.linalg.inv(self.M)

        # Linear damping
        self.D_lin = np.diag([p.X_u, p.Y_v, p.N_r])

    def _rotation(self, psi: float) -> np.ndarray:
        c = np.cos(psi)
        s = np.sin(psi)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _coriolis(self, nu: np.ndarray) -> np.ndarray:
        p = self.p
        u, v, r = nu

        m11 = p.m - p.X_u_dot
        m22 = p.m - p.Y_v_dot
        m23 = p.m * p.x_g - p.Y_r_dot
        m33 = p.I_z - p.N_r_dot

        c13 = -m22 * v - m23 * r
        c23 = m11 * u

        return np.array([
            [0.0,  0.0, c13],
            [0.0,  0.0, c23],
            [-c13, -c23, 0.0],
        ])

    def _damping(self, nu: np.ndarray) -> np.ndarray:
        p = self.p
        u, v, r = nu
        d_quad = np.array([
            p.X_uu * abs(u),
            p.Y_vv * abs(v),
            p.N_rr * abs(r),
        ])
        return self.D_lin @ nu + d_quad * np.sign(nu)

    def _dynamics(self, nu: np.ndarray, tau: np.ndarray,
                  tau_env: np.ndarray) -> np.ndarray:
        C = self._coriolis(nu)
        D_nu = self._damping(nu)
        nu_dot = self.M_inv @ (tau + tau_env - C @ nu - D_nu)
        return nu_dot

    def step(self, tau: np.ndarray, tau_env: np.ndarray = None,
             dt: float = None) -> tuple:
        """RK4 integration step. Returns (eta, nu) after step."""
        if tau_env is None:
            tau_env = np.zeros(3)
        dt = dt or self.dt

        def f(nu):
            return self._dynamics(nu, tau, tau_env)

        # RK4 for ν
        k1 = f(self.nu)
        k2 = f(self.nu + 0.5 * dt * k1)
        k3 = f(self.nu + 0.5 * dt * k2)
        k4 = f(self.nu + dt * k3)
        nu_new = self.nu + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Trapezoidal kinematics for η
        R0 = self._rotation(self.eta[2])
        R1 = self._rotation(self.eta[2] + 0.5 * dt * self.nu[2])
        eta_dot = 0.5 * (R0 @ self.nu + R1 @ nu_new)
        eta_new = self.eta + dt * eta_dot

        self.nu = nu_new
        self.eta = eta_new
        self.eta[2] = (self.eta[2] + np.pi) % (2 * np.pi) - np.pi

        return self.eta.copy(), self.nu.copy()

    def set_state(self, eta: np.ndarray, nu: np.ndarray):
        self.eta = eta.copy()
        self.nu = nu.copy()

    def predict(self, nu: np.ndarray, tau: np.ndarray,
                tau_env: np.ndarray = None, dt: float = None) -> np.ndarray:
        """Return ν̇ for the given state and control (used by MPC)."""
        if tau_env is None:
            tau_env = np.zeros(3)
        return self._dynamics(nu, tau, tau_env)

    def state_space_matrices(self, nu: np.ndarray) -> tuple:
        """Linearized discrete-time A, B around current ν (for EKF/MPC)."""
        dt = self.dt
        u, v, r = nu
        psi = self.eta[2]
        p = self.p

        c = np.cos(psi)
        s = np.sin(psi)

        # Linearized kinematic Jacobian ∂η̇/∂ν = R(ψ), ∂η̇/∂η = 0
        R = self._rotation(psi)

        # Build continuous A_6x6 = [[0, R], [0, ∂ν̇/∂ν]]
        # ∂ν̇/∂ν ≈ -M⁻¹ (∂C/∂ν + ∂D/∂ν)
        A_nu = np.zeros((3, 3))
        # Surge row
        A_nu[0, 0] = -p.X_u / self.M[0, 0]  # damping
        # Sway row
        A_nu[1, 0] = self.M[0, 0] / self.M[1, 1] * r  # centripetal
        A_nu[1, 1] = -p.Y_v / self.M[1, 1]
        # Yaw row
        A_nu[2, 2] = -p.N_r / self.M[2, 2]

        A_cont = np.zeros((6, 6))
        A_cont[:3, 3:] = R
        A_cont[3:, 3:] = A_nu

        B_cont = np.zeros((6, 3))
        B_cont[3:, :] = self.M_inv
        # Only surge and yaw are actuated
        B_cont = B_cont[:, [0, 2]]  # τ_u, τ_r

        # Discretize: A_d = I + A_c*dt, B_d = B_c*dt
        A_disc = np.eye(6) + A_cont * dt
        B_disc = B_cont * dt

        return A_disc, B_disc
