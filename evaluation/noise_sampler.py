#!/usr/bin/env python3
"""
Noise/disturbance random sampler for Monte Carlo evaluation.

Samples:
  - Initial position perturbations (GPS uncertainty)
  - Initial velocity perturbations
  - Environment force perturbations (wind, wave, current)
  - Sensor noise
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NoiseConfig:
    """Noise distribution parameters for Monte Carlo."""
    # Initial position noise (Gaussian, per axis)
    pos_sigma: float = 1.0        # m — GPS initial fix error
    # Initial heading noise
    heading_sigma: float = 0.015  # rad (~0.85°)
    # Initial velocity noise
    speed_sigma: float = 0.1      # m/s
    # Wind speed noise (relative fraction)
    wind_speed_sigma: float = 0.2  # 20% variation
    # Wave height noise (relative fraction)
    wave_height_sigma: float = 0.3  # 30% variation
    # Current speed noise (relative fraction)
    current_speed_sigma: float = 0.3  # 30% variation
    # Environment force noise (body-frame)
    tau_env_sigma: float = 50.0   # N — random perturbation force
    # Target ship initial position noise
    target_pos_sigma: float = 0.8  # m
    target_heading_sigma: float = 0.015  # rad
    target_speed_sigma: float = 0.08  # m/s


class NoiseSampler:
    """Generates randomized noise samples for Monte Carlo repeats."""

    def __init__(self, config: NoiseConfig = None, seed: Optional[int] = None):
        self.config = config or NoiseConfig()
        self.rng = np.random.RandomState(seed)

    def sample_own_ship_offset(self) -> dict:
        """Sample initial state perturbation for own ship."""
        cfg = self.config
        return {
            'dx': self.rng.randn() * cfg.pos_sigma,
            'dy': self.rng.randn() * cfg.pos_sigma,
            'dheading': self.rng.randn() * cfg.heading_sigma,
            'dspeed': self.rng.randn() * cfg.speed_sigma,
        }

    def sample_target_offset(self, name: str) -> dict:
        """Sample initial state perturbation for a target ship."""
        cfg = self.config
        return {
            'dx': self.rng.randn() * cfg.target_pos_sigma,
            'dy': self.rng.randn() * cfg.target_pos_sigma,
            'dheading': self.rng.randn() * cfg.target_heading_sigma,
            'dspeed': self.rng.randn() * cfg.target_speed_sigma,
        }

    def sample_environment(self, base_env: dict) -> dict:
        """Perturb environment parameters."""
        cfg = self.config
        return {
            'wind_speed': max(0, base_env.get('wind_speed', 4.5) *
                              (1.0 + self.rng.randn() * cfg.wind_speed_sigma)),
            'wind_direction': (base_env.get('wind_direction', 0.0) +
                               self.rng.randn() * 10.0) % 360.0,
            'significant_wave_height': max(0.05, base_env.get('significant_wave_height', 0.4) *
                                           (1.0 + self.rng.randn() * cfg.wave_height_sigma)),
            'peak_period': max(1.0, base_env.get('peak_period', 4.0) *
                               (1.0 + self.rng.randn() * 0.1)),
            'current_speed': max(0, base_env.get('current_speed', 0.25) *
                                 (1.0 + self.rng.randn() * cfg.current_speed_sigma)),
            'current_direction': (base_env.get('current_direction', 80.0) +
                                  self.rng.randn() * 15.0) % 360.0,
        }

    def sample_tau_env(self) -> np.ndarray:
        """Sample random environment force perturbation (body-frame)."""
        cfg = self.config
        return self.rng.randn(3) * cfg.tau_env_sigma
