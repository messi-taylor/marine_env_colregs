# Neuro-Symbolic COLREGS Collision Avoidance for Autonomous Ships

ROS 2 implementation of a neuro-symbolic collision-avoidance framework that combines
grammar-constrained large language models (LLMs) with safety-prioritized nonlinear model
predictive control (NMPC) and error-state extended Kalman filtering (ES-EKF) for
COLREGS-compliant autonomous surface vehicle navigation.

## Features

- **GBNF grammar-constrained LLM referee** — compiler-level format guarantees for COLREGS decisions
- **Safety-prioritized hierarchical NMPC** — lexicographic slack penalties that keep physical
  safety constraints inviolable under conflicting rule interpretations
- **Half-plane convex reformulation** — 100% solver success (from 35–80% failure) with 4–8× faster solve time
- **ES-EKF on SO(3)** — sub-GPS-noise state estimation with covariance reset
- **20-scenario COLREGS benchmark** — head-on, crossing, overtaking, narrow-channel, restricted
  visibility, and multi-ship encounters

## Package Layout

```
marine_env/
├── marine_env/            # Core ROS 2 nodes (NMPC, ES-EKF, JPDA tracker, sensors)
│   └── colregs_referee/   # COLREGS reasoning (LLM + deterministic referee, constraint mapper)
├── evaluation/            # Ablation framework, metrics, visualization, VO baseline
├── config/                # Scenario definitions (generated_scenarios_new/scenario_01..20)
├── launch/                # ROS 2 launch files
├── run_ablation.py        # Ablation experiment runner
└── ...
```

## Dependencies

### ROS 2 (Jazzy Jalisco)
https://docs.ros.org/en/jazzy/Installation.html

### VRX Simulation Environment
The simulation uses the [Virtual RobotX (VRX)](https://github.com/osrf/vrx) Gazebo environment
(WAM-V USV model, Gazebo worlds, sensor plugins). Clone it separately into `src/vrx`:

```bash
cd <workspace>/src
git clone -b jazzy https://github.com/osrf/vrx.git
```

### Python packages
```
numpy  scipy  matplotlib  casadi  pandas  pyyaml  llama-cpp-python
```

### LLM model
Experiments used `qwen2.5:7b` (Q4_K_M quantized), served via `llama-cpp-python`.

## Build & Run

```bash
colcon build --packages-select marine_env
source install/setup.bash

# Run a single scenario
ros2 launch marine_env full_mission.launch.py scenario:=scenario_01

# Run the ablation chain
bash src/marine_env/run_ablation_chain.sh
```

## Reference

X. Wang, X. Xu, and S. Xiang, "Grammar-Constrained LLMs and Safety-Prioritized NMPC for
COLREGS-Compliant Collision Avoidance of Autonomous Ships," submitted to *Ocean Engineering*.

Corresponding author: wxb@zjut.edu.cn
