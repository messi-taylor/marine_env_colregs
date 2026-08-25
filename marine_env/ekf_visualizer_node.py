#!/usr/bin/env python3
"""
EKF Visualizer: records ground truth + EKF estimate, saves comparison plots.
Data is aligned by timestamp and interleaved per-sample for readability.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import csv
import os


class EKFVisualizer(Node):
    def __init__(self):
        super().__init__('ekf_visualizer')

        self.declare_parameter('output_dir', '/home/xxy/vrx_ws/ekf_plots')
        self.declare_parameter('max_samples', 5000)

        self.out_dir = self.get_parameter('output_dir').value
        self.max_samples = self.get_parameter('max_samples').value
        os.makedirs(self.out_dir, exist_ok=True)

        # Clear old CSV
        csv_path = os.path.join(self.out_dir, 'ekf_data.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['source', 't', 'x', 'y', 'yaw', 'vx', 'vy', 'yaw_rate'])

        # Single buffer: (t, x, y, yaw, vx, vy, yr, source)
        self.data = []
        self._gt_first_t = None
        self._ekf_first_t = None
        self._gt_ready = False
        self._ekf_ready = False

        self.gt_sub = self.create_subscription(
            Odometry, '/model/wamv/odometry', self._gt_cb, 10)
        self.ekf_sub = self.create_subscription(
            Odometry, '/wamv/state/estimated', self._ekf_cb, 10)

        self._save_timer = self.create_timer(15.0, self._periodic_save)
        self.get_logger().info(f'EKF Visualizer ready, saving to {self.out_dir}')

    def _append(self, source: str, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        # Full quaternion yaw extraction (handles non-zero roll/pitch)
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        yr = msg.twist.twist.angular.z
        self.data.append((source, t, x, y, yaw, vx, vy, yr))
        if len(self.data) > self.max_samples * 2 + 500:
            self.data = self.data[-(self.max_samples * 2):]

    def _gt_cb(self, msg: Odometry):
        if not self._gt_ready:
            self._gt_first_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._gt_ready = True
            self.get_logger().info(f'GT received, t0={self._gt_first_t:.1f}')
        self._append('gt', msg)

    def _ekf_cb(self, msg: Odometry):
        if not self._ekf_ready:
            self._ekf_first_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._ekf_ready = True
            self.get_logger().info(f'EKF received, t0={self._ekf_first_t:.1f}')
        self._append('ekf', msg)

    def _periodic_save(self):
        if len(self.data) < 20:
            return
        self._save_csv()
        try:
            self._plot()
        except Exception as e:
            self.get_logger().warn(f'Plot failed: {e}')

    def _save_csv(self):
        path = os.path.join(self.out_dir, 'ekf_data.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['source', 't', 'x', 'y', 'yaw', 'vx', 'vy', 'yaw_rate'])
            for row in self.data:
                w.writerow(list(row))

    def _plot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Separate GT and EKF — data rows: (source, t, x, y, yaw, vx, vy, yr)
        gt_idx = [i for i, r in enumerate(self.data) if r[0] == 'gt']
        ekf_idx = [i for i, r in enumerate(self.data) if r[0] == 'ekf']

        if len(gt_idx) < 5 or len(ekf_idx) < 5:
            self.get_logger().warn(
                f'Need both GT({len(gt_idx)}) and EKF({len(ekf_idx)}) data')
            return

        # Extract: cols 1-7 = t,x,y,yaw,vx,vy,yr
        gt = np.array([[self.data[i][j] for j in range(1, 8)] for i in gt_idx])
        ekf = np.array([[self.data[i][j] for j in range(1, 8)] for i in ekf_idx])

        gt_t, gt_x, gt_y, gt_yaw, gt_vx, gt_vy, gt_yr = gt.T
        ekf_t, ekf_x, ekf_y, ekf_yaw, ekf_vx, ekf_vy, ekf_yr = ekf.T

        # Use common time origin
        t0 = max(gt_t[0], ekf_t[0])
        gt_t = gt_t - t0
        ekf_t = ekf_t - t0

        # Interpolate EKF to GT timestamps for error computation
        ekf_x_i = np.interp(gt_t + t0, ekf_t + t0, ekf_x)
        ekf_y_i = np.interp(gt_t + t0, ekf_t + t0, ekf_y)
        ekf_yaw_i = np.interp(gt_t + t0, ekf_t + t0, ekf_yaw)
        ekf_vx_i = np.interp(gt_t + t0, ekf_t + t0, ekf_vx)
        ekf_vy_i = np.interp(gt_t + t0, ekf_t + t0, ekf_vy)

        pos_err = np.sqrt((gt_x - ekf_x_i)**2 + (gt_y - ekf_y_i)**2)
        yaw_err = np.abs(gt_yaw - ekf_yaw_i)
        yaw_err = np.minimum(yaw_err, 2 * np.pi - yaw_err)

        # GT velocity is body-frame (Gazebo odom). Convert EKF world-frame
        # velocity to body-frame for fair comparison.
        cos_yaw = np.cos(ekf_yaw_i)
        sin_yaw = np.sin(ekf_yaw_i)
        ekf_surge_i = ekf_vx_i * cos_yaw + ekf_vy_i * sin_yaw
        ekf_sway_i = -ekf_vx_i * sin_yaw + ekf_vy_i * cos_yaw

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('EKF State Estimation Performance', fontsize=14)

        # 1) XY Trajectory
        ax = axes[0, 0]
        ax.plot(gt_x, gt_y, 'b-', linewidth=0.8, alpha=0.5, label='Ground Truth')
        ax.plot(ekf_x, ekf_y, 'r--', linewidth=1.2, label='EKF Estimate')
        ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
        ax.set_title('XY Trajectory')
        ax.legend(); ax.axis('equal'); ax.grid(True, alpha=0.3)

        # 2) Position error
        ax = axes[0, 1]
        ax.plot(gt_t, pos_err, 'r-', linewidth=0.8)
        ax.fill_between(gt_t, 0, pos_err, alpha=0.15, color='red')
        ax.set_xlabel('Time [s]'); ax.set_ylabel('Position Error [m]')
        ax.set_title('Position Error |EKF - GT|')
        ax.grid(True, alpha=0.3)

        # 3) Yaw
        ax = axes[0, 2]
        ax.plot(gt_t, np.rad2deg(gt_yaw), 'b-', linewidth=0.6, alpha=0.5, label='GT')
        ax.plot(gt_t, np.rad2deg(ekf_yaw_i), 'r--', linewidth=0.8, label='EKF')
        ax.set_xlabel('Time [s]'); ax.set_ylabel('Yaw [deg]')
        ax.set_title('Heading (Yaw)')
        ax.legend(); ax.grid(True, alpha=0.3)

        # 4) Surge velocity (body-frame)
        ax = axes[1, 0]
        ax.plot(gt_t, gt_vx, 'b-', linewidth=0.6, alpha=0.5, label='GT')
        ax.plot(gt_t, ekf_surge_i, 'r--', linewidth=0.8, label='EKF')
        ax.set_xlabel('Time [s]'); ax.set_ylabel('Surge [m/s]')
        ax.set_title('Surge Velocity (body)')
        ax.legend(); ax.grid(True, alpha=0.3)

        # 5) Sway velocity (body-frame)
        ax = axes[1, 1]
        ax.plot(gt_t, gt_vy, 'b-', linewidth=0.6, alpha=0.5, label='GT')
        ax.plot(gt_t, ekf_sway_i, 'r--', linewidth=0.8, label='EKF')
        ax.set_xlabel('Time [s]'); ax.set_ylabel('Sway [m/s]')
        ax.set_title('Sway Velocity (body)')
        ax.legend(); ax.grid(True, alpha=0.3)

        # 6) Summary
        ax = axes[1, 2]
        ax.axis('off')
        rmse_pos = np.sqrt(np.mean(pos_err**2))
        rmse_yaw = np.sqrt(np.mean(yaw_err**2))
        rmse_vx = np.sqrt(np.mean((gt_vx - ekf_surge_i)**2))
        rmse_vy = np.sqrt(np.mean((gt_vy - ekf_sway_i)**2))

        overlap = (min(gt_t[-1], ekf_t[-1]) - max(gt_t[0], ekf_t[0]))
        text = (
            f'RMSE Position:  {rmse_pos:.3f} m\n'
            f'RMSE Yaw:       {np.rad2deg(rmse_yaw):.2f} deg\n'
            f'RMSE Surge Vel: {rmse_vx:.3f} m/s\n'
            f'RMSE Sway Vel:  {rmse_vy:.3f} m/s\n'
            f'\nGT samples:  {len(gt_t)}\n'
            f'EKF samples: {len(ekf_t)}\n'
            f'Overlap:     {overlap:.1f}s\n'
            f'Data range:  {gt_t[0]:.1f}-{gt_t[-1]:.1f}s'
        )
        ax.text(0.1, 0.7, text, transform=ax.transAxes, fontsize=11,
                fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        fig.savefig(os.path.join(self.out_dir, 'ekf_comparison.png'), dpi=150)
        plt.close(fig)
        self.get_logger().info(
            f'Plot saved: overlap={overlap:.1f}s, '
            f'RMSE pos={rmse_pos:.2f}m, yaw={np.rad2deg(rmse_yaw):.1f}deg')


def main():
    rclpy.init()
    node = EKFVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node._periodic_save()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
