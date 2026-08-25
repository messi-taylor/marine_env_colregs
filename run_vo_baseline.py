#!/usr/bin/env python3
"""
VO Baseline Experiment Runner
==============================

Runs Velocity Obstacle baseline on selected COLREGS scenarios for
comparison against the neuro-symbolic NMPC framework (Groups A-D).

Usage:
    # Single scenario test
    python3 run_vo_baseline.py --scenarios 1 --repeats 20

    # All 20 scenarios (matching Group C's N=100)
    python3 run_vo_baseline.py --scenarios all --repeats 100

    # Key scenarios only (quick check)
    python3 run_vo_baseline.py --scenarios 1,6,7,9,15,20 --repeats 50
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from evaluation.batch_runner import BatchRunner, MonteCarloConfig


def parse_scenarios(arg: str):
    if arg.lower() == 'all':
        return [f'scenario_{i:02d}' for i in range(1, 21)]
    scenarios = []
    for s in arg.split(','):
        s = s.strip().replace('scenario_', '').replace('S', '')
        try:
            num = int(s)
            scenarios.append(f'scenario_{num:02d}')
        except ValueError:
            print(f"Warning: cannot parse scenario '{s}' — skipping")
    return scenarios


def main():
    parser = argparse.ArgumentParser(
        description='VO Baseline for COLREGS collision avoidance')
    parser.add_argument('--scenarios', type=str, default='1',
                        help='Scenario IDs (e.g., "1,6,7,9") or "all"')
    parser.add_argument('--repeats', type=int, default=50,
                        help='Number of Monte Carlo repeats (default: 50)')
    parser.add_argument('--duration', type=float, default=40.0,
                        help='Simulation duration per run (default: 40s)')
    parser.add_argument('--output', type=str, default='vo_baseline_output',
                        help='Output directory')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Parallel workers')
    args = parser.parse_args()

    scenarios = parse_scenarios(args.scenarios)
    if not scenarios:
        print("ERROR: No valid scenarios.")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"VO Baseline Experiment")
    print(f"{'='*70}")
    print(f"Scenarios: {len(scenarios)} ({', '.join(scenarios[:5])}...)")
    print(f"Repeats:   {args.repeats}")
    print(f"Duration:  {args.duration}s/run")
    print(f"Total:     {len(scenarios) * args.repeats} runs")
    print(f"Output:    {args.output}/")
    print(f"{'='*70}")

    t_total_start = time.perf_counter()

    for sid in scenarios:
        t_start = time.perf_counter()
        group_dir = os.path.join(args.output, sid, "group_E_VO")
        os.makedirs(group_dir, exist_ok=True)

        config = MonteCarloConfig(
            scenario_id=sid,
            num_repeats=args.repeats,
            sim_duration=args.duration,
            control_backend="vo",
            output_dir=group_dir,
            parallel_workers=args.parallel,
            referee_backend="deterministic",  # not used by VO path
        )

        print(f"\n{'─'*60}")
        print(f"Scenario: {sid} ({args.repeats} runs, VO baseline)")
        print(f"{'─'*60}")

        runner = BatchRunner(config)
        collector = runner.run_batch(sid)

        # Save results
        collector.save_csv(os.path.join(group_dir, "metrics.csv"))
        collector.save_json(os.path.join(group_dir, "summary.json"))

        summary = collector.summary()
        wall_time = time.perf_counter() - t_start

        print(f"  Done: {wall_time:.0f}s ({wall_time/args.repeats:.1f}s/run)")
        print(f"  Collisions: {summary['collision_rate']*100:.1f}%")
        print(f"  Median CPA: {summary['median_min_cpa']:.1f}m")
        print(f"  Worst CPA:  {summary['worst_cpa']:.1f}m")
        print(f"  Turn compliance: {summary['turn_direction_compliance']*100:.1f}%")

    t_total = time.perf_counter() - t_total_start
    print(f"\n{'='*70}")
    print(f"VO Baseline Complete: {t_total/60:.1f} min")
    print(f"Results: {args.output}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
