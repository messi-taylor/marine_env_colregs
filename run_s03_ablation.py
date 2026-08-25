#!/usr/bin/env python3
"""Minimal S03 ablation — runs Groups A, B, D directly."""

import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))

from evaluation.batch_runner import BatchRunner, MonteCarloConfig

GROUPS = {
    'A': {'backend': 'ollama', 'weights': None},
    'B': {'backend': 'ollama_no_cfg', 'weights': None},
    'D': {'backend': 'ollama', 'weights': {'w_legal': 0, 'w_smooth': 0, 'w_speed': 0}},
}

BASE_DIR = 'ablation_output/scenario_03'
SCENARIO = 'scenario_03'
NUM_RUNS = 20  # Match S01/S02 ablation (20 repeats for LLM groups)

for group_id, params in GROUPS.items():
    group_dir = os.path.join(BASE_DIR, f'group_{group_id}')

    # Skip if already completed
    config_path = os.path.join(group_dir, 'group_config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            if json.load(f).get('completed'):
                print(f"Group {group_id}: already completed — skipping")
                continue

    os.makedirs(group_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Group {group_id} — backend={params['backend']}, weights={params['weights']}")
    print(f"{'='*60}")

    config = MonteCarloConfig(
        scenario_id=SCENARIO,
        num_repeats=NUM_RUNS,
        sim_duration=40.0,
        output_dir=group_dir,
        parallel_workers=1,
        referee_backend=params['backend'],
        nmpc_weight_overrides=params['weights'],
    )

    t0 = time.perf_counter()
    runner = BatchRunner(config)

    try:
        collector = runner.run_batch(SCENARIO)
        wall_time = time.perf_counter() - t0

        # Save results
        collector.save_csv(os.path.join(group_dir, 'metrics.csv'))
        collector.save_json(os.path.join(group_dir, 'summary.json'))

        summary = collector.summary()
        with open(config_path, 'w') as f:
            json.dump({
                'group': group_id,
                'referee_backend': params['backend'],
                'soft_constraints_enabled': params['weights'] is None,
                'nmpc_weight_overrides': params['weights'],
                'num_runs': NUM_RUNS,
                'wall_time_s': round(wall_time, 1),
                'completed': True,
            }, f, indent=2)

        print(f"\nGroup {group_id} DONE: {NUM_RUNS} runs in {wall_time:.0f}s "
              f"({wall_time/NUM_RUNS:.1f}s/run)")
        print(f"  Collision rate: {summary['collision_rate']*100:.1f}%")
        print(f"  Median CPA: {summary['median_min_cpa']:.1f}m")
        print(f"  Turn compliance: {summary['turn_direction_compliance']*100:.1f}%")
        print(f"  LLM-NMPC disagreements: {summary.get('total_llm_nmpc_disagreements', 0)}")

    except Exception as e:
        print(f"Group {group_id} FAILED: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print("S03 Ablation Complete")
print(f"{'='*60}")
