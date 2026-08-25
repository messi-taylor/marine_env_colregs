#!/usr/bin/env python3
"""极简恒推力控制 — 只发固定推力, 让船直线走, 零振荡."""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class StraightThrust(Node):
    def __init__(self):
        super().__init__('straight_thrust')
        self.declare_parameter('thrust', 800.0)  # N
        thrust = self.get_parameter('thrust').value

        self.pub_l = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.pub_r = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        self.timer = self.create_timer(0.1, lambda: self._pub(thrust))
        self.get_logger().info(f'恒推力 {thrust:.0f}N → 直行')

    def _pub(self, t):
        m = Float64(data=t)
        self.pub_l.publish(m)
        self.pub_r.publish(m)


def main():
    rclpy.init()
    rclpy.spin(StraightThrust())


if __name__ == '__main__':
    main()
