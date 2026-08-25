from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import glob
import os
import sys

package_name = 'marine_env'
_ep_names = ['sea_clutter', 'ais_publisher', 'environment_forces',
             'target_ship_spawner', 'ekf_estimator', 'eskf_estimator',
             'jpda_tracker', 'ekf_visualizer', 'wamv_autopilot',
             'nmpc_controller', 'ship_data_subscriber',
             'referee_node', 'scene_descriptor_node',
             'trajectory_publisher', 'collision_marker_publisher']


def _create_libexec_symlinks(scripts_dir):
    libexec_dir = os.path.join(scripts_dir, '..', 'lib', package_name)
    os.makedirs(libexec_dir, exist_ok=True)
    for ep_name in _ep_names:
        src = os.path.join(scripts_dir, ep_name)
        dst = os.path.join(libexec_dir, ep_name)
        if os.path.exists(src) and not os.path.islink(dst):
            os.symlink(os.path.relpath(src, libexec_dir), dst)


class InstallWithLibexec(install):
    def run(self):
        install.run(self)
        _create_libexec_symlinks(self.install_scripts)


class DevelopWithLibexec(develop):
    def run(self):
        develop.run(self)
        scripts = getattr(self, 'script_dir', None) or os.path.join(self.install_dir, 'bin')
        _create_libexec_symlinks(scripts)


setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (f'share/{package_name}/launch', [f'launch/{f}' for f in [
            'marine_env.launch.py',
            'full_mission.launch.py',
        ]]),
        (f'share/{package_name}/config', glob.glob('config/*.yaml') + glob.glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'sea_clutter = marine_env.sea_clutter_node:main',
            'ais_publisher = marine_env.ais_publisher_node:main',
            'environment_forces = marine_env.environment_forces_node:main',
            'target_ship_spawner = marine_env.target_ship_spawner:main',
            'ekf_estimator = marine_env.ekf_estimator_node:main',
            'eskf_estimator = marine_env.eskf_estimator_node:main',
            'jpda_tracker = marine_env.jpda_tracker_node:main',
            'ekf_visualizer = marine_env.ekf_visualizer_node:main',
            'wamv_autopilot = marine_env.wamv_autopilot:main',
            'nmpc_controller = marine_env.nmpc_controller_node:main',
            'ship_data_subscriber = marine_env.ship_data_subscriber:main',
            'referee_node = marine_env.colregs_referee.referee_node:main',
            'scene_descriptor_node = marine_env.colregs_referee.scene_descriptor_node:main',
            'trajectory_publisher = marine_env.trajectory_publisher:main',
            'collision_marker_publisher = marine_env.collision_marker_publisher:main',
            'straight_thrust = marine_env.straight_thrust:main',
        ],
    },
    cmdclass={'install': InstallWithLibexec, 'develop': DevelopWithLibexec},
)
