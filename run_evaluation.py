#!/usr/bin/env python3
"""
Monte Carlo Evaluation Entry Point
===================================

Runs batch Monte Carlo simulation for COLREGS collision avoidance.

Usage:
    python3 run_evaluation.py                           # default: S01, 100 runs
    python3 run_evaluation.py --scenario 1 --repeats 50 # custom
    python3 run_evaluation.py --scenario all --repeats 100  # all 20 scenarios
    python3 run_evaluation.py --parallel 4              # 4-process parallel
"""

import argparse
import os
import sys
import time
import numpy as np

# Ensure package is importable
sys.path.insert(0, os.path.dirname(__file__))

from evaluation.batch_runner import BatchRunner, MonteCarloConfig
from evaluation.metrics import MetricsCollector
from evaluation.visualize import generate_report


def main():
    parser = argparse.ArgumentParser(
        description='Monte Carlo evaluation for COLREGS collision avoidance')
    parser.add_argument('--scenario', type=str, default='1',
                        help='Scenario ID (1-20) or "all"')
    parser.add_argument('--repeats', type=int, default=100,
                        help='Number of Monte Carlo repeats per scenario')
    parser.add_argument('--duration', type=float, default=40.0,
                        help='Simulation duration per run (seconds)')
    parser.add_argument('--output', type=str, default='evaluation_output',
                        help='Output directory for metrics and plots')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel processes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip visualization generation')
    args = parser.parse_args()

    # Determine scenarios to run
    if args.scenario.lower() == 'all':
        scenario_ids = [f'scenario_{i:02d}' for i in range(1, 21)]
    else:
        # Support both "1" and "scenario_01" formats
        sid = args.scenario.replace('scenario_', '').replace('S', '')
        try:
            num = int(sid)
            scenario_ids = [f'scenario_{num:02d}']
        except ValueError:
            scenario_ids = [f'scenario_{args.scenario}']

    print(f"{'='*60}")
    print(f"COLREGS Monte Carlo Evaluation")
    print(f"{'='*60}")
    print(f"Scenarios: {scenario_ids}")
    print(f"Repeats per scenario: {args.repeats}")
    print(f"Simulation duration: {args.duration}s")
    print(f"Total runs: {len(scenario_ids) * args.repeats}")
    print(f"Output: {args.output}/")
    print(f"Parallel workers: {args.parallel}")
    print(f"{'='*60}\n")

    t_total_start = time.perf_counter()

    for sid in scenario_ids:
        scenario_output = os.path.join(args.output, sid)
        os.makedirs(scenario_output, exist_ok=True)

        print(f"\n{'─'*60}")
        print(f"Running: {sid}")
        print(f"{'─'*60}")

        config = MonteCarloConfig(
            scenario_id=sid,
            num_repeats=args.repeats,
            sim_duration=args.duration,
            output_dir=scenario_output,
            parallel_workers=args.parallel,
        )

        runner = BatchRunner(config)
        collector = runner.run_batch()

        # Save raw data
        collector.save_csv(os.path.join(scenario_output, 'metrics.csv'))
        collector.save_json(os.path.join(scenario_output, 'summary.json'))

        # Generate report
        if not args.no_plot:
            print(f"\nGenerating visualizations...")
            try:
                generate_report(collector, scenario_output, sid)
            except Exception as e:
                print(f"  Warning: visualization failed: {e}")

        # Print summary
        summary = collector.summary()
        print(f"\n{sid} Summary:")
        print(f"  Collision rate: {summary['collision_rate']*100:.1f}%")
        print(f"  Near-miss rate: {summary['near_miss_rate']*100:.1f}%")
        print(f"  Median CPA:     {summary['median_min_cpa']:.1f} m")
        print(f"  Worst CPA:      {summary['worst_cpa']:.1f} m")
        print(f"  Turn compliant: {summary['turn_direction_compliance']*100:.1f}%")
        print(f"  Solve success:  {summary['solve_success_rate']*100:.1f}%")
        print(f"  Avg solve:      {summary['avg_solve_time_ms']:.0f} ms")

    t_total = time.perf_counter() - t_total_start
    print(f"\n{'='*60}")
    print(f"All evaluations complete: {t_total:.1f}s total")
    print(f"Results saved to: {args.output}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
