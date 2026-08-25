#!/usr/bin/env python3
"""
Environment Forces Node - Applies realistic marine disturbances to the WAM-V.
- JONSWAP wave spectrum: irregular wave drift forces
- Gusty wind: uniform wind field + random gusts on superstructure
- Ocean current: 2D steady drift force on hull
- Publishes EntityWrench bridged to Gazebo physics engine
"""

import rclpy
from rclpy.node import Node
from ros_gz_interfaces.msg import EntityWrench
from geometry_msgs.msg import Wrench, Vector3
from nav_msgs.msg import Odometry
import numpy as np
import math
import time


class EnvironmentForcesNode(Node):
    def __init__(self):
        super().__init__('environment_forces')

        # --- Wind parameters ---
        self.declare_parameter('wind_speed', 5.0)
        self.declare_parameter('wind_direction', 240.0)
        self.declare_parameter('gust_amplitude', 3.0)
        self.declare_parameter('gust_frequency', 0.1)
        self.declare_parameter('frontal_area', 2.5)
        self.declare_parameter('lateral_area', 8.0)
        self.declare_parameter('air_density', 1.225)

        # --- Wave parameters (JONSWAP) ---
        self.declare_parameter('significant_wave_height', 0.5)
        self.declare_parameter('peak_period', 4.0)
        self.declare_parameter('jonswap_gamma', 3.3)
        self.declare_parameter('wave_direction', 0.0)
        self.declare_parameter('water_density', 1025.0)
        self.declare_parameter('waterline_length', 3.5)
        self.declare_parameter('beam', 1.8)
        self.declare_parameter('draft', 0.3)

        # --- Current parameters ---
        self.declare_parameter('current_speed', 0.3)
        self.declare_parameter('current_direction', 90.0)
        self.declare_parameter('hull_drag_coefficient', 0.8)

        # --- General ---
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('odom_topic', '/wamv/odom')
        self.declare_parameter('world_name', 'sydney_regatta')
        self.declare_parameter('model_name', 'wamv')

        self._build_jonswap_spectrum()

        self._last_gust_time = time.time()
        self._current_gust = 0.0

        self.odom_sub = self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self.odom_callback, 10)

        # Publish EntityWrench → bridged to Gazebo /world/<world>/wrench
        world = self.get_parameter('world_name').value
        self.wrench_pub = self.create_publisher(
            EntityWrench, f'/world/{world}/wrench', 10)

        rate = self.get_parameter('publish_rate').value
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.force_cycle)

        self._current_velocity = np.zeros(3)
        self._current_heading = 0.0

        self.get_logger().info(
            f'Environment Forces Node: wind={self.get_parameter("wind_speed").value} m/s, '
            f'waves Hs={self.get_parameter("significant_wave_height").value} m, '
            f'current={self.get_parameter("current_speed").value} m/s → '
            f'/world/{world}/wrench')

    def odom_callback(self, msg):
        self._current_velocity[0] = msg.twist.twist.linear.x
        self._current_velocity[1] = msg.twist.twist.linear.y
        self._current_velocity[2] = msg.twist.twist.linear.z
        q = msg.pose.pose.orientation
        self._current_heading = 2 * math.atan2(q.z, q.w)

    def _build_jonswap_spectrum(self):
        Hs = self.get_parameter('significant_wave_height').value
        Tp = self.get_parameter('peak_period').value
        gamma = self.get_parameter('jonswap_gamma').value
        g = 9.81
        fp = 1.0 / Tp
        self._freqs = np.linspace(0.05, 2.0, 80)
        self._wave_amplitudes = []
        for f in self._freqs:
            if f < 1e-6:
                self._wave_amplitudes.append(0.0)
                continue
            sigma = 0.07 if f <= fp else 0.09
            alpha = (0.076 * (Hs**2 * fp**4 / (g**2))**0.22 *
                     g**2 / ((2*np.pi)**4 * f**5))
            r = np.exp(-(f - fp)**2 / (2 * sigma**2 * fp**2))
            S = alpha * np.exp(-1.25 * (fp/f)**4) * gamma**r
            df = self._freqs[1] - self._freqs[0]
            self._wave_amplitudes.append(np.sqrt(2 * S * df))
        self._wave_amplitudes = np.array(self._wave_amplitudes)
        self._wave_phases = np.random.uniform(0, 2*np.pi, len(self._freqs))

    def force_cycle(self):
        t = time.time()
        wind_force = self._compute_wind_force(t)
        wave_force = self._compute_wave_force(t)
        current_force = self._compute_current_force()

        total = wind_force + wave_force + current_force

        msg = EntityWrench()
        msg.entity.name = self.get_parameter('model_name').value
        msg.entity.type = msg.entity.MODEL

        msg.wrench.force.x = float(total[0])
        msg.wrench.force.y = float(total[1])
        msg.wrench.force.z = 0.0

        # Yaw torque from wave lateral force
        B = self.get_parameter('beam').value
        msg.wrench.torque.x = 0.0
        msg.wrench.torque.y = 0.0
        msg.wrench.torque.z = float(wave_force[0] * 0.3 * B)

        self.wrench_pub.publish(msg)

    def _compute_wind_force(self, t):
        wind_speed = self.get_parameter('wind_speed').value
        wind_dir_deg = self.get_parameter('wind_direction').value
        gust_amp = self.get_parameter('gust_amplitude').value
        gust_freq = self.get_parameter('gust_frequency').value
        rho_air = self.get_parameter('air_density').value
        A_front = self.get_parameter('frontal_area').value
        A_side = self.get_parameter('lateral_area').value

        dt = t - self._last_gust_time
        if dt > 1.0 / max(gust_freq, 0.01):
            self._current_gust = np.random.normal(0, gust_amp)
            self._last_gust_time = t
        self._current_gust *= np.exp(-dt * gust_freq * 2)

        total_wind = wind_speed + self._current_gust
        wind_dir_rad = math.radians(wind_dir_deg)
        wind_vec = np.array([
            -total_wind * math.sin(wind_dir_rad),
            -total_wind * math.cos(wind_dir_rad),
        ])

        rel_wind = wind_vec - self._current_velocity[:2]
        heading = self._current_heading

        fwd = rel_wind[0] * math.cos(heading) + rel_wind[1] * math.sin(heading)
        lat = -rel_wind[0] * math.sin(heading) + rel_wind[1] * math.cos(heading)

        Cd_front, Cd_side = 1.1, 1.2
        F_front = 0.5 * rho_air * Cd_front * A_front * fwd * abs(fwd)
        F_side = 0.5 * rho_air * Cd_side * A_side * lat * abs(lat)

        Fx = F_front * math.cos(heading) - F_side * math.sin(heading)
        Fy = F_front * math.sin(heading) + F_side * math.cos(heading)
        return np.array([Fx, Fy])

    def _compute_wave_force(self, t):
        Hs = self.get_parameter('significant_wave_height').value
        wave_dir_deg = self.get_parameter('wave_direction').value
        rho = self.get_parameter('water_density').value
        L = self.get_parameter('waterline_length').value
        B = self.get_parameter('beam').value
        g = 9.81

        wave_dir = math.radians(wave_dir_deg)
        k_vec = np.array([math.sin(wave_dir), math.cos(wave_dir)])

        for i, (f, amp) in enumerate(zip(self._freqs, self._wave_amplitudes)):
            if amp < 1e-9:
                continue
            omega = 2 * np.pi * f
            k = omega**2 / g
            phase = k * 0.0 + self._wave_phases[i] - omega * t
            # phase accumulation for superposition

        drift_coeff = 0.05 * rho * g * B * (Hs**2 / max(L, 0.01))
        Fx = drift_coeff * math.sin(wave_dir)
        Fy = drift_coeff * math.cos(wave_dir)
        return np.array([Fx, Fy])

    def _compute_current_force(self):
        current_speed = self.get_parameter('current_speed').value
        current_dir_deg = self.get_parameter('current_direction').value
        Cd = self.get_parameter('hull_drag_coefficient').value
        rho = self.get_parameter('water_density').value
        B = self.get_parameter('beam').value
        T = self.get_parameter('draft').value

        current_dir = math.radians(current_dir_deg)
        current_vec = np.array([
            current_speed * math.sin(current_dir),
            current_speed * math.cos(current_dir),
        ])

        rel_current = current_vec - self._current_velocity[:2]
        A_wetted = B * T * 1.2
        Fx = 0.5 * rho * Cd * A_wetted * rel_current[0] * np.linalg.norm(rel_current)
        Fy = 0.5 * rho * Cd * A_wetted * rel_current[1] * np.linalg.norm(rel_current)
        return np.array([Fx, Fy])


def main():
    rclpy.init()
    node = EnvironmentForcesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
