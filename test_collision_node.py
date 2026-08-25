#!/usr/bin/env python3
"""Standalone test for collision_marker_publisher — no launch needed."""
import subprocess, sys, os, json, yaml

print('=' * 60)
print('Test 1: Check config file')
print('=' * 60)
cfg_path = '/home/xxy/vrx_ws/src/marine_env/config/target_ships.yaml'
if os.path.exists(cfg_path):
    print(f'  ✓ Config exists: {cfg_path}')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    sj = cfg['target_ship_spawner']['ros__parameters']['ships_json']
    ships = json.loads(sj) if isinstance(sj, str) else sj
    names = [s['name'] for s in ships]
    print(f'  ✓ Ships loaded: {names}')
else:
    print(f'  ✗ Config NOT FOUND: {cfg_path}')
    print('   Run: python3 config/load_scenario.py <N> --new')
    sys.exit(1)

print()
print('=' * 60)
print('Test 2: Check entry point')
print('=' * 60)
result = subprocess.run(
    ['ros2', 'run', 'marine_env', 'collision_marker_publisher', '--help'],
    capture_output=True, text=True, timeout=5)
print(f'  stdout: {result.stdout[:200]}')
print(f'  stderr: {result.stderr[:200]}')
# Note: --help might not work for rclpy nodes, but the import should succeed

print()
print('=' * 60)
print('Test 3: Quick import check')
print('=' * 60)
try:
    from marine_env.collision_marker_publisher import CollisionMarkerPublisher
    print('  ✓ Import successful')
except Exception as e:
    print(f'  ✗ Import FAILED: {e}')
    import traceback
    traceback.print_exc()

print()
print('=' * 60)
print('Test 4: Run node for 3 seconds (watch for errors)')
print('=' * 60)
print('  Starting collision_marker_publisher...')
print('  (If you see "INIT:" below, the node works)')
print('  (If it crashes, you will see the traceback)')
print()
sys.stdout.flush()

import subprocess, signal, time
proc = subprocess.Popen(
    ['ros2', 'run', 'marine_env', 'collision_marker_publisher'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})

start = time.time()
while time.time() - start < 4.0:
    line = proc.stdout.readline()
    if line:
        print(f'  [NODE] {line.rstrip()}')
        sys.stdout.flush()
    if proc.poll() is not None:
        print(f'  ✗ NODE EXITED with code {proc.returncode}')
        remaining = proc.stdout.read()
        if remaining:
            print(f'  [NODE] {remaining}')
        break

if proc.poll() is None:
    print(f'  ✓ Node still running after 4s — looks healthy')
    proc.send_signal(signal.SIGINT)
    time.sleep(0.5)
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=2)
