#!/usr/bin/env python3
"""
Launch all marine environment simulation nodes:
- Sea clutter injector for X-band radar
- AIS NMEA publisher for target ships
- Environment forces (JONSWAP waves, wind gusts, ocean current)
- Target ship spawner for COLREGS scenarios
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # --- Sea Clutter ---
        DeclareLaunchArgument('enable_clutter', default_value='True',
                              description='Enable Weibull sea clutter'),
        DeclareLaunchArgument('enable_false_alarms', default_value='True',
                              description='Enable false alarm injection'),
        DeclareLaunchArgument('clutter_shape', default_value='1.5',
                              description='Weibull shape parameter'),
        DeclareLaunchArgument('clutter_scale', default_value='0.03',
                              description='Weibull scale parameter'),
        DeclareLaunchArgument('false_alarm_rate', default_value='0.02',
                              description='False alarm probability per point'),

        # --- AIS Publisher ---
        DeclareLaunchArgument('publish_rate', default_value='0.1',
                              description='AIS publish rate per ship (Hz)'),
        DeclareLaunchArgument('packet_loss_rate', default_value='0.05',
                              description='AIS packet loss probability'),
        DeclareLaunchArgument('max_delay_ms', default_value='500',
                              description='Max AIS random delay (ms)'),

        # --- Environment Forces ---
        DeclareLaunchArgument('wind_speed', default_value='5.0',
                              description='Mean wind speed (m/s)'),
        DeclareLaunchArgument('wind_direction', default_value='240.0',
                              description='Wind from direction (deg)'),
        DeclareLaunchArgument('significant_wave_height', default_value='0.5',
                              description='Significant wave height Hs (m)'),
        DeclareLaunchArgument('peak_period', default_value='4.0',
                              description='Wave peak period Tp (s)'),
        DeclareLaunchArgument('current_speed', default_value='0.3',
                              description='Ocean current speed (m/s)'),
        DeclareLaunchArgument('current_direction', default_value='90.0',
                              description='Ocean current direction (deg)'),

        # --- Nodes ---
        Node(
            package='marine_env',
            executable='sea_clutter',
            name='sea_clutter',
            output='screen',
            parameters=[{
                'enable_clutter': LaunchConfiguration('enable_clutter'),
                'enable_false_alarms': LaunchConfiguration('enable_false_alarms'),
                'clutter_shape': LaunchConfiguration('clutter_shape'),
                'clutter_scale': LaunchConfiguration('clutter_scale'),
                'false_alarm_rate': LaunchConfiguration('false_alarm_rate'),
            }],
            # Delay start until sensor topics are available
        ),

        Node(
            package='marine_env',
            executable='ais_publisher',
            name='ais_publisher',
            output='screen',
            parameters=[{
                'publish_rate': LaunchConfiguration('publish_rate'),
                'packet_loss_rate': LaunchConfiguration('packet_loss_rate'),
                'max_delay_ms': LaunchConfiguration('max_delay_ms'),
            }],
        ),

        Node(
            package='marine_env',
            executable='environment_forces',
            name='environment_forces',
            output='screen',
            parameters=[{
                'wind_speed': LaunchConfiguration('wind_speed'),
                'wind_direction': LaunchConfiguration('wind_direction'),
                'significant_wave_height': LaunchConfiguration('significant_wave_height'),
                'peak_period': LaunchConfiguration('peak_period'),
                'current_speed': LaunchConfiguration('current_speed'),
                'current_direction': LaunchConfiguration('current_direction'),
            }],
        ),

        Node(
            package='marine_env',
            executable='target_ship_spawner',
            name='target_ship_spawner',
            output='screen',
            parameters=[{}],
        ),

        LogInfo(msg='Marine environment nodes launched.'),
    ])
