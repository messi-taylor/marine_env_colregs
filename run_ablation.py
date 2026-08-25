#!/usr/bin/env python3
"""
Ablation Experiment Entry Point
================================

Runs 4-group controlled-variable ablation study for COLREGS collision avoidance.

Groups:
  A (Full):  GrammarConstrainedReferee (GBNF + LLM) + soft constraints
  B (-CFG):  OllamaReferee (LLM, no GBNF grammar) + soft constraints
  C (-LLM):  DeterministicReferee (rule-based, no LLM) + soft constraints
  D (-Soft): GrammarConstrainedReferee (GBNF + LLM) + hard constraints ONLY

Usage:
    # Full ablation on all 20 scenarios
    python3 run_ablation.py

    # Single scenario, custom repeats
    python3 run_ablation.py --scenarios 1 --repeats-full 20 --repeats-fast 100

    # Specific groups only
    python3 run_ablation.py --groups A,C --scenarios 1,2,3

    # No resume (re-run completed groups)
    python3 run_ablation.py --scenarios 1 --no-resume
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from evaluation.ablation import (
    AblationGroup, AblationConfig, AblationRunner, GroupResult
)


def parse_groups(arg: str):
    """Parse comma-separated group letters into AblationGroup list."""
    groups = []
    for g in arg.split(','):
        g = g.strip().upper()
        if g == 'A':
            groups.append(AblationGroup.A_FULL)
        elif g == 'B':
            groups.append(AblationGroup.B_NO_CFG)
        elif g == 'C':
            groups.append(AblationGroup.C_NO_LLM)
        elif g == 'D':
            groups.append(AblationGroup.D_NO_SOFT)
        else:
            print(f"Warning: unknown group '{g}' — skipping")
    return groups


def parse_scenarios(arg: str):
    """Parse comma-separated scenario IDs or 'all'."""
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
        description='Ablation experiment for COLREGS collision avoidance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_ablation.py                                          # All 20 scenarios, all groups
  python3 run_ablation.py --scenarios 1 --repeats-full 10          # Quick test on S01
  python3 run_ablation.py --groups A,C --scenarios 1,2,3           # Compare A vs C on 3 scenarios
  python3 run_ablation.py --groups D --scenarios 1 --no-resume     # Re-run Group D clean
        """)
    parser.add_argument('--groups', type=str, default='A,B,C,D',
                        help='Comma-separated groups to run (default: A,B,C,D)')
    parser.add_argument('--scenarios', type=str, default='all',
                        help='Scenario IDs (e.g., "1,2,3") or "all" (default)')
    parser.add_argument('--repeats-full', type=int, default=20,
                        help='Repeats for LLM groups A,B,D (default: 20)')
    parser.add_argument('--repeats-fast', type=int, default=100,
                        help='Repeats for deterministic group C (default: 100)')
    parser.add_argument('--duration', type=float, default=40.0,
                        help='Simulation duration per run in seconds (default: 40)')
    parser.add_argument('--output', type=str, default='ablation_output',
                        help='Output root directory (default: ablation_output)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Parallel workers — forced to 1 for LLM groups (default: 1)')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Skip completed groups (default)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='Re-run all groups even if previously completed')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    args = parser.parse_args()

    # Parse groups and scenarios
    groups = parse_groups(args.groups)
    scenarios = parse_scenarios(args.scenarios)

    if not groups:
        print("ERROR: No valid groups specified.")
        sys.exit(1)
    if not scenarios:
        print("ERROR: No valid scenarios specified.")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"COLREGS Ablation Experiment")
    print(f"{'='*70}")
    print(f"Groups:    {', '.join(g.label for g in groups)}")
    print(f"Scenarios: {len(scenarios)} ({scenarios[0]} ... {scenarios[-1]})")
    print(f"LLM-group repeats: {args.repeats_full}")
    print(f"Fast-group repeats: {args.repeats_fast}")
    print(f"Duration/run: {args.duration}s")
    print(f"Total LLM runs: {len(scenarios) * sum(1 for g in groups if g != AblationGroup.C_NO_LLM) * args.repeats_full}")
    print(f"Total fast runs: {len(scenarios) * sum(1 for g in groups if g == AblationGroup.C_NO_LLM) * args.repeats_fast}")
    print(f"Output: {args.output}/")
    print(f"Resume: {'yes' if args.resume else 'no'}")
    print(f"Parallel: {args.parallel}")
    print(f"{'='*70}")

    # ── Per-scenario prerequisites check ──
    if AblationGroup.B_NO_CFG in groups:
        import requests
        import os as _os
        ollama_url = _os.environ.get("OLLAMA_HOST", "http://localhost:11435")
        try:
            resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                print(f"✓ Ollama server reachable at {ollama_url}")
            else:
                print(f"⚠ Ollama server returned {resp.status_code} at {ollama_url}")
        except Exception as e:
            print(f"⚠ Cannot reach Ollama at {ollama_url}: {e}")
            print("  Group B (-CFG) will fail. Start with:")
            print("    export OLLAMA_HOST=http://localhost:11435")
            print("    ollama serve &")

    if AblationGroup.A_FULL in groups or AblationGroup.D_NO_SOFT in groups:
        try:
            import llama_cpp
            print("✓ llama-cpp-python available (Groups A, D)")
        except ImportError:
            print("⚠ llama-cpp-python not installed. Groups A, D will fail.")
            print("  Install with: pip install llama-cpp-python")

    # ── Run ablation per scenario ──
    t_total_start = time.perf_counter()
    all_results = {}

    for sid in scenarios:
        config = AblationConfig(
            scenario_id=sid,
            repeats_full=args.repeats_full,
            repeats_fast=args.repeats_fast,
            sim_duration=args.duration,
            output_dir=args.output,
            parallel_workers=args.parallel,
            resume=args.resume,
            groups=groups,
        )
        runner = AblationRunner(config)
        results = runner.run_all_groups(sid)
        all_results[sid] = results

    # ── Save master summary ──
    if all_results and len(scenarios) > 1:
        dummy_config = AblationConfig(output_dir=args.output)
        dummy_runner = AblationRunner(dummy_config)
        dummy_runner.save_master_summary(all_results)

    t_total = time.perf_counter() - t_total_start

    # ── Print final summary ──
    print(f"\n{'='*70}")
    print(f"Ablation Complete: {t_total/60:.1f} min total")
    print(f"{'='*70}")

    # Summary table header
    print(f"\n{'Scenario':<14} ", end="")
    for g in groups:
        print(f"{'│ ' + g.label + ' ':>24s}", end="")
    print()
    print(f"{'─'*14}─", end="")
    for _ in groups:
        print(f"{'─'*26}", end="")
    print()

    for sid in scenarios:
        print(f"{sid:<14} ", end="")
        for g in groups:
            if sid in all_results and g in all_results[sid]:
                s = all_results[sid][g].summary
                c_rate = s.get('collision_rate', 0) * 100
                print(f"│ coll={c_rate:4.1f}% cpa={s.get('median_min_cpa',0):5.1f}m ", end="")
            else:
                print(f"│ {'FAILED':^24s}", end="")
        print()

    print(f"\nResults saved to: {args.output}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
