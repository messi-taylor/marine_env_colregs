#!/usr/bin/env python3
"""Standalone test: Fossen 3DOF ship dynamics with step response."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/xxy/vrx_ws/src/marine_env')
from marine_env.ship_dynamics import FossenShip, ShipParams


def main():
    ship = FossenShip(dt=0.05)
    ship.set_state(eta=np.zeros(3), nu=np.zeros(3))

    # Record
    t_max = 60.0
    steps = int(t_max / ship.dt)
    history = {'t': [], 'x': [], 'y': [], 'psi': [], 'u': [], 'v': [], 'r': []}

    for i in range(steps):
        t = i * ship.dt

        # Step input: 300N thrust + small rudder moment
        tau_u = 200.0 if t > 1.0 else 0.0
        tau_r = 100.0 * np.sin(0.1 * t)  # sinusoidal yaw excitation

        tau = np.array([tau_u, 0.0, tau_r])
        eta, nu = ship.step(tau)

        history['t'].append(t)
        history['x'].append(eta[0])
        history['y'].append(eta[1])
        history['psi'].append(eta[2])
        history['u'].append(nu[0])
        history['v'].append(nu[1])
        history['r'].append(nu[2])

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Fossen 3DOF Ship Dynamics — Step Response Test', fontsize=13)

    t = history['t']

    ax = axes[0, 0]
    ax.plot(history['x'], history['y'], 'b-', linewidth=0.8)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_title('XY Trajectory (NED)')
    ax.axis('equal'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, np.rad2deg(history['psi']), 'b-', linewidth=0.8)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Yaw [deg]')
    ax.set_title('Heading ψ')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(t, history['u'], 'r-', linewidth=0.8, label='Surge u')
    ax.plot(t, history['v'], 'b-', linewidth=0.8, label='Sway v')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Velocity [m/s]')
    ax.set_title('Body-Frame Velocities')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, np.rad2deg(history['r']), 'g-', linewidth=0.8)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Yaw Rate [deg/s]')
    ax.set_title('Yaw Rate r')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    speed = np.sqrt(np.array(history['u'])**2 + np.array(history['v'])**2)
    ax.plot(t, speed, 'purple', linewidth=0.8)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Speed [m/s]')
    ax.set_title('Total Speed')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.axis('off')
    text = (
        f'Model: Fossen 3DOF\n'
        f'Mass: {ship.p.m} kg\n'
        f'Max surge: {max(history["u"]):.2f} m/s\n'
        f'Max sway:  {max(history["v"]):.2f} m/s\n'
        f'Total dist: {np.sqrt(history["x"][-1]**2 + history["y"][-1]**2):.1f} m\n'
        f'Duration: {t_max} s'
    )
    ax.text(0.1, 0.7, text, transform=ax.transAxes, fontsize=11,
            fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    out_path = '/home/xxy/vrx_ws/ekf_plots/ship_dynamics_test.png'
    import os
    os.makedirs('/home/xxy/vrx_ws/ekf_plots', exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()
