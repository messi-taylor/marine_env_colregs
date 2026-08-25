#!/usr/bin/env python3
"""
Multi-Model Comparison Runner — S06/S07/S09 x 7b vs 14b x 20 runs.
"""

import argparse, os, sys, time, json

sys.path.insert(0, os.path.dirname(__file__))
from evaluation.batch_runner import BatchRunner, MonteCarloConfig

MODELS = [
    ("qwen2.5:7b",  "7b",  "ollama",
     ""),
    ("qwen2.5:14b", "14b", "grammar_custom",
     "/home/xxy/.ollama/models/gguf/qwen2.5-14b-q4_k_m.gguf"),
]


def main():
    parser = argparse.ArgumentParser(description='Multi-model comparison')
    parser.add_argument('--scenarios', type=str, default='6,7,9')
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--duration', type=float, default=40.0)
    parser.add_argument('--output', type=str, default='multimodel_output')
    parser.add_argument('--models', type=str, default='7b,14b')
    parser.add_argument('--skip-check', action='store_true')
    args = parser.parse_args()

    scenarios = [f'scenario_{int(s.strip()):02d}' for s in args.scenarios.split(',')]

    model_variants = []
    for m in args.models.split(','):
        m = m.strip()
        for full_name, short_name, backend, gguf_path in MODELS:
            if short_name == m:
                model_variants.append((full_name, short_name, backend, gguf_path))
                break

    if not scenarios or not model_variants:
        print("ERROR: No valid scenarios or models."); sys.exit(1)

    if not args.skip_check:
        print("Checking prerequisites...")
        for _, _, backend, gguf_path in model_variants:
            if backend == "grammar_custom":
                if os.path.exists(gguf_path):
                    print(f"  ✓ GGUF {os.path.getsize(gguf_path)/1e9:.1f} GB")
                else:
                    print(f"  ✗ GGUF not found: {gguf_path}")

    total_runs = len(scenarios) * len(model_variants) * args.repeats
    print(f"\n{'='*70}")
    print(f"Multi-Model: {scenarios} x {[n for n,_,_,_ in model_variants]} x {args.repeats}")
    print(f"Total: {total_runs} runs | Output: {args.output}/")
    print(f"{'='*70}")

    t_total_start = time.perf_counter()
    all_results = {}

    for sid in scenarios:
        all_results[sid] = {}
        for model_name, model_label, backend, gguf_path in model_variants:
            group_dir = os.path.join(args.output, sid, f"model_{model_label}")
            os.makedirs(group_dir, exist_ok=True)

            config = MonteCarloConfig(
                scenario_id=sid,
                num_repeats=args.repeats,
                sim_duration=args.duration,
                control_backend="nmpc",
                referee_backend=backend,
                referee_model=model_name,
                referee_model_path=gguf_path,
                output_dir=group_dir,
                parallel_workers=1,
            )

            t_start = time.perf_counter()
            print(f"\n{'─'*60}")
            print(f"  {sid} / {model_name} ({args.repeats} runs)")
            print(f"{'─'*60}")

            runner = BatchRunner(config)
            try:
                collector = runner.run_batch(sid)
                collector.save_csv(os.path.join(group_dir, "metrics.csv"))
                collector.save_json(os.path.join(group_dir, "summary.json"))
                s = collector.summary()
                wall = time.perf_counter() - t_start
                print(f"  Done: {wall:.0f}s | coll={s['collision_rate']*100:.0f}% cpa={s['median_min_cpa']:.1f}m")
                all_results[sid][model_label] = {
                    "collision_rate": s['collision_rate'], "median_cpa": s['median_min_cpa'],
                    "worst_cpa": s['worst_cpa'], "wall_time_s": wall,
                }
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()
                all_results[sid][model_label] = {"error": str(e)}

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "multimodel_summary.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    t_total = time.perf_counter() - t_total_start
    print(f"\n{'='*70}")
    print(f"Complete: {t_total/60:.1f} min")
    print(f"{'Scenario':<14} {'Model':<16} {'Coll%':>7} {'Med CPA':>8}")
    print(f"{'─'*14} {'─'*16} {'─'*7} {'─'*8}")
    for sid in scenarios:
        for _, model_label, _, _ in model_variants:
            r = all_results.get(sid, {}).get(model_label, {})
            if "error" in r:
                print(f"{sid:<14} {model_label:<16} {'ERR':>7}")
            else:
                print(f"{sid:<14} {model_label:<16} {r['collision_rate']*100:6.1f}% {r['median_cpa']:7.1f}m")
    print(f"\nResults: {args.output}/")


if __name__ == '__main__':
    main()
