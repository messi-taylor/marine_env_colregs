#!/usr/bin/env python3
"""
AIS NMEA 0183 Publisher Node.
- Tracks target ships from Gazebo pose topics
- Formats AIS VDM messages (NMEA 0183)
- Publishes at configurable intervals
- Injects configurable packet loss and delay
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import math
import time
import random


class AISPublisher(Node):
    def __init__(self):
        super().__init__('ais_publisher')

        self.declare_parameter('publish_rate', 0.1)      # Hz (every 10s per ship)
        self.declare_parameter('packet_loss_rate', 0.05)  # 5% packet loss
        self.declare_parameter('max_delay_ms', 500)       # max random delay in ms
        self.declare_parameter('target_ships', ['target_ship_1', 'target_ship_2'])

        self.targets = {}
        self.last_publish = {}
        self.mmsi_counter = 1

        # Subscribe to each target ship's pose
        for ship_name in self.get_parameter('target_ships').value:
            topic = f'/{ship_name}/pose'
            self.targets[ship_name] = {
                'lat': None, 'lon': None, 'sog': 0.0, 'cog': 0.0,
                'heading': 0.0, 'mmsi': f'{100000000 + self.mmsi_counter:09d}',
                'name': ship_name.upper()[:20]
            }
            self.mmsi_counter += 1
            self.create_subscription(PoseStamped, topic,
                                     lambda msg, s=ship_name: self.pose_callback(s, msg), 10)

        self.pub = self.create_publisher(String, '/wamv/sensors/ais/nmea', 10)

        rate = self.get_parameter('publish_rate').value
        self.timer = self.create_timer(1.0 / max(rate, 0.1), self.publish_cycle)

        # Reference position for lat/lon conversion (Sydney Regatta)
        self.ref_lat = -33.724223
        self.ref_lon = 150.679736

        self.get_logger().info(f'AIS Publisher started for {len(self.targets)} target ships')

    def pose_callback(self, ship_name, msg):
        """Track target ship positions and convert to lat/lon/COG/SOG."""
        target = self.targets[ship_name]
        x = msg.pose.position.x
        y = msg.pose.position.y

        # Convert local ENU to lat/lon (simplified flat-earth)
        lat = self.ref_lat + y / 111320.0
        lon = self.ref_lon + x / (111320.0 * math.cos(math.radians(self.ref_lat)))
        target['lat'] = lat
        target['lon'] = lon

        # Heading from quaternion
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        target['heading'] = math.degrees(2 * math.atan2(qz, qw)) % 360

    def publish_cycle(self):
        """Publish AIS NMEA sentences for active targets."""
        loss_rate = self.get_parameter('packet_loss_rate').value
        max_delay = self.get_parameter('max_delay_ms').value
        now = time.time()

        for name, target in self.targets.items():
            if target['lat'] is None:
                continue
            if now - self.last_publish.get(name, 0) < 5.0:  # at least 5s gap
                continue
            if random.random() < loss_rate:
                continue  # simulate packet loss

            delay = random.uniform(0, max_delay / 1000.0)
            self.last_publish[name] = now + delay

            sentence = self.build_ais_vdm(target)
            msg = String()
            msg.data = sentence
            self.pub.publish(msg)
            self.get_logger().debug(f'AIS {name}: {sentence[:80]}...')

    def build_ais_vdm(self, target):
        """Build NMEA 0183 !AIVDM sentence (simplified AIS message type 1/2/3)."""
        lat = target['lat']
        lon = target['lon']
        sog = target['sog'] or 0.0
        cog = target['cog'] or 0.0
        heading = target['heading']
        mmsi = target['mmsi']

        # Build payload: message type 1/2/3 (position report)
        # In real AIS this is 6-bit encoded, here we use simplified text for readability
        payload = (
            f"MSG1,{mmsi},"
            f"LAT={lat:.5f},LON={lon:.5f},"
            f"SOG={sog:.1f},COG={cog:.1f},"
            f"HDG={heading:.1f}"
        )

        # NMEA 0183 format: $TALKER,PAYLOAD*CHECKSUM
        checksum = 0
        for c in payload:
            checksum ^= ord(c)
        return f"!AIVDM,1,1,,A,{payload},{checksum:02X}"


def main():
    rclpy.init()
    node = AISPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
