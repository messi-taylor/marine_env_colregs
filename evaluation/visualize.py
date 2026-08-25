#!/usr/bin/env python3
"""
Visualization module for Monte Carlo evaluation results.

Generates publication-quality figures (300 DPI PNG + PDF for LaTeX):
  - cpa_distribution.png/pdf   — CPA histogram + CDF
  - trajectories.png/pdf       — Trajectory overlay (all runs)
  - control_effort.png/pdf     — Control effort + solver performance
  - summary_report.txt         — Text summary table
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os, json
from typing import List, Optional, Tuple
from .metrics import MetricsCollector, RunMetrics


# ══════════════════════════════════════════════════════════════════════════════
# Publication-quality style (IEEE / Ocean Engineering compatible)
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    'safe':       '#2E86AB',   # steel blue
    'near_miss':  '#F18F01',   # orange
    'collision':  '#C73E1D',   # red
    'median':     '#236B4A',   # dark green
    'thrust':     '#2E86AB',
    'rudder':     '#8B1E3F',   # dark red
    'grid':       '#CCCCCC',
    'reference':  '#666666',
}

plt.rcParams.update({
    'figure.dpi': 150,           # screen
    'savefig.dpi': 300,          # publication
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'lines.linewidth': 1.2,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
})


def _save(fig, output_dir: str, basename: str) -> Tuple[str, str]:
    """Save figure as both PNG and PDF. Returns (png_path, pdf_path)."""
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f'{basename}.png')
    pdf_path = os.path.join(output_dir, f'{basename}.pdf')
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, format='pdf')
    plt.close(fig)
    return png_path, pdf_path


# ══════════════════════════════════════════════════════════════════════════════
# Individual plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_cpa_histogram(collector: MetricsCollector,
                       output_dir: str = "evaluation_output",
                       scenario_name: str = "Scenario",
                       ts_positions: Optional[List[Tuple[float, float]]] = None):
    """CPA histogram + CDF (single figure, two panels)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    min_cpas = np.array([r.min_cpa for r in collector.runs
                         if r.min_cpa < float('inf')])
    if len(min_cpas) == 0:
        plt.close(fig)
        return None

    n_collisions = sum(1 for r in collector.runs if r.collision)
    n_near_miss = sum(1 for r in collector.runs if r.near_miss)

    # ── Histogram ──
    cpa_max = max(np.max(min_cpas), 30)
    bins = np.linspace(0, cpa_max, 35)
    ax1.hist(min_cpas, bins=bins, color=COLORS['safe'], edgecolor='white',
             alpha=0.85, linewidth=0.3)
    ax1.axvline(x=1.0, color=COLORS['collision'], linestyle='--', linewidth=1.2,
                label='Collision (1 m)')
    ax1.axvline(x=20.0, color=COLORS['near_miss'], linestyle='--', linewidth=1.2,
                label='Near-miss (20 m)')
    ax1.axvline(x=np.median(min_cpas), color=COLORS['median'], linestyle='-',
                linewidth=1.0, label=f'Median: {np.median(min_cpas):.1f} m')
    ax1.set_xlabel('Minimum CPA (m)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'(a) CPA Histogram  [N={len(collector.runs)}, '
                  f'collisions={n_collisions}, near-miss={n_near_miss}]',
                  fontsize=9)
    ax1.legend(fontsize=7, framealpha=0.9, edgecolor='gray')
    ax1.grid(True, alpha=0.25, color=COLORS['grid'])
    ax1.set_xlim(left=0)

    # ── CDF ──
    sorted_cpas = np.sort(min_cpas)
    cdf_y = np.arange(1, len(sorted_cpas) + 1) / len(sorted_cpas)
    ax2.step(sorted_cpas, cdf_y, where='post', color=COLORS['safe'],
             linewidth=1.5)
    ax2.axvline(x=1.0, color=COLORS['collision'], linestyle='--', linewidth=1.0)
    ax2.axvline(x=20.0, color=COLORS['near_miss'], linestyle='--', linewidth=1.0)
    ax2.fill_between([0, 1], 0, 1, color=COLORS['collision'], alpha=0.08)
    ax2.fill_between([1, 20], 0, 1, color=COLORS['near_miss'], alpha=0.06)
    ax2.set_xlabel('Minimum CPA (m)')
    ax2.set_ylabel('Cumulative probability')
    ax2.set_title('(b) CPA Cumulative Distribution', fontsize=9)
    ax2.grid(True, alpha=0.25, color=COLORS['grid'])
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, 1.02)

    plt.tight_layout()
    _save(fig, output_dir, 'cpa_distribution')
    return os.path.join(output_dir, 'cpa_distribution.png')


def plot_trajectories(collector: MetricsCollector,
                      output_dir: str = "evaluation_output",
                      scenario_name: str = "Scenario",
                      max_runs: int = 30):
    """Trajectory overlay coloured by safety outcome."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Determine TS start from first run (if available)
    ts_x, ts_y = None, None

    plotted = 0
    for r in collector.runs[:max_runs]:
        if len(r.pos_x_history) < 2:
            continue
        if r.collision:
            color, alpha, z, label = COLORS['collision'], 0.55, 3, None
        elif r.near_miss:
            color, alpha, z, label = COLORS['near_miss'], 0.35, 2, None
        else:
            color, alpha, z, label = COLORS['safe'], 0.20, 1, None

        if not plotted:
            label = 'OS trajectory'
        ax.plot(r.pos_x_history, r.pos_y_history, color=color, alpha=alpha,
                linewidth=0.6, zorder=z, label=label)
        plotted += 1

    # Mark start position (first run)
    first = collector.runs[0]
    if len(first.pos_x_history) >= 1:
        ax.scatter([first.pos_x_history[0]], [first.pos_y_history[0]],
                   marker='s', s=60, color='green', edgecolors='black',
                   linewidth=0.8, zorder=5, label='OS start')

    # Mark CPA point for the worst run
    worst_run = min(collector.runs, key=lambda r: r.min_cpa)
    if len(worst_run.pos_x_history) >= 2:
        cpa_idx = np.argmin([
            np.linalg.norm(np.array([worst_run.pos_x_history[i],
                                     worst_run.pos_y_history[i]]))
            for i in range(len(worst_run.pos_x_history))
        ])
        ax.scatter([worst_run.pos_x_history[cpa_idx]],
                   [worst_run.pos_y_history[cpa_idx]],
                   marker='x', s=80, color=COLORS['collision'],
                   linewidth=2.0, zorder=6, label=f'CPA={worst_run.min_cpa:.1f}m')

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title(f'{scenario_name}: Monte Carlo Trajectories ({plotted} runs)',
                 fontsize=10)
    ax.legend(fontsize=7, framealpha=0.9, edgecolor='gray',
              loc='upper left', bbox_to_anchor=(1.01, 1.0))
    ax.grid(True, alpha=0.25, color=COLORS['grid'])
    ax.set_aspect('equal')

    plt.tight_layout()
    _save(fig, output_dir, 'trajectories')
    return os.path.join(output_dir, 'trajectories.png')


def plot_control_effort(collector: MetricsCollector,
                        output_dir: str = "evaluation_output",
                        scenario_name: str = "Scenario"):
    """Control effort + solver performance (2×2 panel)."""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

    # ── (a) Thrust envelope ──
    ax = axes[0, 0]
    for r in collector.runs[:30]:
        if len(r.thrust_history) >= 2:
            t = r.t_history[:len(r.thrust_history)]
            ax.plot(t, r.thrust_history, alpha=0.18, color=COLORS['thrust'],
                    linewidth=0.4)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Surge thrust (N)')
    ax.set_title('(a) Thrust envelope', fontsize=9)
    ax.grid(True, alpha=0.25, color=COLORS['grid'])

    # ── (b) Rudder envelope ──
    ax = axes[0, 1]
    for r in collector.runs[:30]:
        if len(r.rudder_history) >= 2:
            t = r.t_history[:len(r.rudder_history)]
            ax.plot(t, r.rudder_history, alpha=0.18, color=COLORS['rudder'],
                    linewidth=0.4)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Yaw moment $\\tau_r$ (N·m)')
    ax.set_title('(b) Rudder moment envelope', fontsize=9)
    ax.grid(True, alpha=0.25, color=COLORS['grid'])

    # ── (c) Solve time distribution ──
    ax = axes[1, 0]
    solve_times = [r.avg_solve_time_ms for r in collector.runs
                   if r.avg_solve_time_ms > 0]
    if solve_times:
        ax.hist(solve_times, bins=18, color='seagreen', edgecolor='white',
                linewidth=0.3)
        mean_st = np.mean(solve_times)
        ax.axvline(x=mean_st, color='darkgreen', linestyle='--', linewidth=1.2,
                   label=f'Mean: {mean_st:.0f} ms')
        ax.legend(fontsize=7)
    ax.set_xlabel('Average solve time (ms)')
    ax.set_ylabel('Count')
    ax.set_title('(c) NMPC solve time', fontsize=9)
    ax.grid(True, alpha=0.25, color=COLORS['grid'])

    # ── (d) CPA vs solve success rate ──
    ax = axes[1, 1]
    cpas = [r.min_cpa for r in collector.runs if r.min_cpa < float('inf')]
    rates = [r.solve_success_rate * 100 for r in collector.runs]
    if len(cpas) == len(rates) and len(cpas) > 0:
        scatter = ax.scatter(cpas, rates, c=cpas, cmap='RdYlGn_r',
                             alpha=0.55, edgecolors='gray', linewidth=0.2,
                             s=18)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.82)
        cbar.set_label('Min CPA (m)', fontsize=7)
    ax.set_xlabel('Minimum CPA (m)')
    ax.set_ylabel('Solve success rate (%)')
    ax.set_title('(d) CPA vs solver reliability', fontsize=9)
    ax.grid(True, alpha=0.25, color=COLORS['grid'])

    plt.suptitle(f'{scenario_name}: Control & Solver Performance',
                 fontsize=10, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save(fig, output_dir, 'control_effort')
    return os.path.join(output_dir, 'control_effort.png')


def plot_compliance_summary(collector: MetricsCollector,
                            output_dir: str = "evaluation_output",
                            scenario_name: str = "Scenario"):
    """Bar chart of compliance metrics."""
    s = collector.summary()
    fig, ax = plt.subplots(figsize=(4.5, 2.8))

    metrics = ['Turn\ndirection', 'Action\nmagnitude', 'Overtaking', 'Role']
    values = [
        s['turn_direction_compliance'] * 100,
        s['action_magnitude_compliance'] * 100,
        s['overtaking_compliance'] * 100,
        s['role_compliance'] * 100,
    ]
    colors = [COLORS['safe'] if v >= 80 else COLORS['near_miss'] if v >= 50
              else COLORS['collision'] for v in values]

    bars = ax.bar(range(len(metrics)), values, color=colors, edgecolor='gray',
                  linewidth=0.5, width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val:.0f}%', ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylabel('Compliance rate (%)')
    ax.set_ylim(0, 115)
    ax.set_title(f'{scenario_name}: COLREGS Compliance', fontsize=10)
    ax.axhline(y=80, color=COLORS['grid'], linestyle='--', linewidth=0.8,
               alpha=0.8, label='80% threshold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25, axis='y', color=COLORS['grid'])

    plt.tight_layout()
    _save(fig, output_dir, 'compliance_bars')
    return os.path.join(output_dir, 'compliance_bars.png')


def generate_report(collector: MetricsCollector,
                    output_dir: str = "evaluation_output",
                    scenario_name: str = "Scenario"):
    """Generate all standard figures and summary report.

    This is the single entry point for the standardised evaluation pipeline.
    Output (per scenario directory):
      cpa_distribution.png / .pdf
      trajectories.png / .pdf
      control_effort.png / .pdf
      compliance_bars.png / .pdf
      summary_report.txt
    """
    summary = collector.summary()
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Generating figures for {scenario_name}...")

    # Figures
    plot_cpa_histogram(collector, output_dir, scenario_name)
    plot_trajectories(collector, output_dir, scenario_name)
    plot_control_effort(collector, output_dir, scenario_name)
    plot_compliance_summary(collector, output_dir, scenario_name)

    # ── Text report ──
    report_path = os.path.join(output_dir, 'summary_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Monte Carlo Evaluation Report\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Scenario:   {scenario_name}\n")
        f.write(f"Runs:       {summary['num_runs']}\n\n")

        f.write(f"SAFETY METRICS\n")
        f.write(f"{'─' * 40}\n")
        f.write(f"  Collision rate:     {summary['collision_rate'] * 100:.1f}%\n")
        f.write(f"  Near-miss rate:     {summary['near_miss_rate'] * 100:.1f}%\n")
        f.write(f"  Mean min CPA:       {summary['mean_min_cpa']:.1f} m\n")
        f.write(f"  Median min CPA:     {summary['median_min_cpa']:.1f} m\n")
        f.write(f"  Worst CPA:          {summary['worst_cpa']:.1f} m\n")
        f.write(f"  Std min CPA:        {summary['std_min_cpa']:.1f} m\n\n")

        f.write(f"COMPLIANCE METRICS\n")
        f.write(f"{'─' * 40}\n")
        f.write(f"  Turn direction:     {summary['turn_direction_compliance'] * 100:.1f}%\n")
        f.write(f"  Action magnitude:   {summary['action_magnitude_compliance'] * 100:.1f}%\n")
        f.write(f"  Overtaking:         {summary['overtaking_compliance'] * 100:.1f}%\n")
        f.write(f"  Role compliance:    {summary['role_compliance'] * 100:.1f}%\n\n")

        f.write(f"CONTROL QUALITY\n")
        f.write(f"{'─' * 40}\n")
        f.write(f"  Mean rudder rate:   {summary['mean_rudder_rate']:.2f} N·m/step\n")
        f.write(f"  Thrust std:         {summary['thrust_std']:.1f} N\n")
        f.write(f"  Avg surge speed:    {summary['avg_surge']:.2f} m/s\n")
        f.write(f"  Max rudder:         {summary['max_rudder']:.1f} N·m\n\n")

        f.write(f"SOLVER PERFORMANCE\n")
        f.write(f"{'─' * 40}\n")
        f.write(f"  Total solves:       {summary['total_solves']}\n")
        f.write(f"  Success rate:       {summary['solve_success_rate'] * 100:.1f}%\n")
        f.write(f"  Avg solve time:     {summary['avg_solve_time_ms']:.0f} ms\n")
        f.write(f"  Total infeasible:   {summary['total_infeasible']}\n")
        f.write(f"  Total timeout:      {summary['total_timeout']}\n")
        f.write(f"  Retry L1 recoveries: {summary.get('total_retry1_recoveries', 0)}\n")
        f.write(f"  Retry L2 recoveries: {summary.get('total_retry2_recoveries', 0)}\n")
        f.write(f"  Retry L3 recoveries: {summary.get('total_retry3_recoveries', 0)}\n\n")

        f.write(f"DEGRADATION\n")
        f.write(f"{'─' * 40}\n")
        f.write(f"  Runs degraded:      {summary['runs_with_degradation']}\n")
        f.write(f"  Max degradation:    Level {summary['max_degradation_observed']}\n")

    print(f"  Saved: {report_path}")
    return report_path


# ══════════════════════════════════════════════════════════════════════════════
# Cross-group ablation comparison plots
# ══════════════════════════════════════════════════════════════════════════════

# Group color scheme for ablation plots
GROUP_STYLES = {
    'A (Full)':   {'color': '#2E86AB', 'linestyle': '-',   'marker': 'o', 'label': 'A (Full)'},
    'B (-CFG)':   {'color': '#F18F01', 'linestyle': '--',  'marker': 's', 'label': 'B (-CFG)'},
    'C (-LLM)':   {'color': '#236B4A', 'linestyle': ':',   'marker': '^', 'label': 'C (-LLM)'},
    'D (-Soft)':  {'color': '#C73E1D', 'linestyle': '-.',  'marker': 'D', 'label': 'D (-Soft)'},
}


def plot_cpa_cdf_overlay(group_results: dict,
                          output_dir: str = "ablation_output",
                          scenario_name: str = "Scenario"):
    """Overlay CPA CDF curves from multiple ablation groups.

    Args:
        group_results: Dict mapping group label (e.g. "A (Full)") to MetricsCollector.
        output_dir: Output directory for the figure.
        scenario_name: Name for the figure title.

    Generates: cpa_cdf_overlay.png / .pdf
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    for label, collector in group_results.items():
        style = GROUP_STYLES.get(label, {'color': 'gray', 'linestyle': '-', 'label': label})
        min_cpas = np.array([r.min_cpa for r in collector.runs
                             if r.min_cpa < float('inf')])
        if len(min_cpas) == 0:
            continue

        sorted_cpas = np.sort(min_cpas)
        cdf_y = np.arange(1, len(sorted_cpas) + 1) / len(sorted_cpas)

        collision_rate = sum(1 for r in collector.runs if r.collision) / max(len(collector.runs), 1)
        median_cpa = np.median(min_cpas)

        ax.step(sorted_cpas, cdf_y, where='post',
                color=style['color'], linestyle=style['linestyle'],
                linewidth=1.5,
                label=f"{style['label']} (coll={collision_rate*100:.1f}%, "
                      f"med={median_cpa:.1f}m)")

    ax.axvline(x=1.0, color=COLORS['collision'], linestyle='--', linewidth=1.0, alpha=0.6)
    ax.axvline(x=20.0, color=COLORS['near_miss'], linestyle='--', linewidth=1.0, alpha=0.6)
    ax.fill_between([0, 1], 0, 1, color=COLORS['collision'], alpha=0.06)
    ax.fill_between([1, 20], 0, 1, color=COLORS['near_miss'], alpha=0.04)
    ax.set_xlabel('Minimum CPA (m)')
    ax.set_ylabel('Cumulative probability')
    ax.set_title(f'{scenario_name}: CPA CDF — Ablation Comparison', fontsize=10)
    ax.legend(fontsize=7, framealpha=0.9, edgecolor='gray', loc='lower right')
    ax.grid(True, alpha=0.25, color=COLORS['grid'])
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    _save(fig, output_dir, 'cpa_cdf_overlay')
    return os.path.join(output_dir, 'cpa_cdf_overlay.png')


def plot_compliance_heatmap(group_results: dict,
                             output_dir: str = "ablation_output",
                             scenario_name: str = "Scenario"):
    """Generate compliance heatmap: groups × compliance metrics.

    Args:
        group_results: Dict mapping group label to MetricsCollector.
        output_dir: Output directory.
        scenario_name: Name for the figure title.

    Generates: compliance_heatmap.png / .pdf
    """
    group_order = ['A (Full)', 'B (-CFG)', 'C (-LLM)', 'D (-Soft)']
    metric_names = ['Turn\nDirection', 'Action\nMagnitude', 'Overtaking', 'Role']
    metric_keys = ['turn_direction_compliance', 'action_magnitude_compliance',
                   'overtaking_compliance', 'role_compliance']

    # Build data matrix
    data = np.zeros((len(group_order), len(metric_keys)))
    row_labels = []
    for i, label in enumerate(group_order):
        if label in group_results:
            s = group_results[label].summary()
            for j, key in enumerate(metric_keys):
                data[i, j] = s.get(key, 0) * 100
            row_labels.append(label)
        else:
            data[i, :] = np.nan
            row_labels.append(f"{label} (N/A)")

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')

    # Annotate cells
    for i in range(len(group_order)):
        for j in range(len(metric_keys)):
            if not np.isnan(data[i, j]):
                color = 'white' if data[i, j] < 40 or data[i, j] > 80 else 'black'
                ax.text(j, i, f'{data[i, j]:.0f}%', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)

    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels(metric_names, fontsize=8)
    ax.set_yticks(range(len(group_order)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(f'{scenario_name}: Compliance Heatmap', fontsize=10)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Compliance Rate (%)', fontsize=7)

    plt.tight_layout()
    _save(fig, output_dir, 'compliance_heatmap')
    return os.path.join(output_dir, 'compliance_heatmap.png')


def plot_ablation_radar(group_results: dict,
                         output_dir: str = "ablation_output",
                         scenario_name: str = "Scenario"):
    """Generate ablation radar chart comparing 5 dimensions across groups.

    Dimensions (normalized 0-100, higher=better):
      - Safety Score: (1 - collision_rate) * 100
      - Compliance: mean of 4 compliance rates * 100
      - Solver Reliability: solve_success_rate * 100
      - CPA Margin: min(100, median_cpa / 50 * 100)
      - Control Smoothness: 100 - min(100, mean_rudder_rate / 0.5 * 100)

    Args:
        group_results: Dict mapping group label to MetricsCollector.
        output_dir: Output directory.
        scenario_name: Name for the figure title.

    Generates: ablation_radar.png / .pdf
    """
    group_order = ['A (Full)', 'B (-CFG)', 'C (-LLM)', 'D (-Soft)']
    dimensions = ['Safety', 'Compliance', 'Solver\nReliability', 'CPA\nMargin', 'Control\nSmoothness']
    N = len(dimensions)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))

    for label in group_order:
        if label not in group_results:
            continue
        style = GROUP_STYLES.get(label, {'color': 'gray', 'linestyle': '-', 'label': label})
        s = group_results[label].summary()

        # Compute normalized scores
        safety = max(0, (1 - s.get('collision_rate', 1)) * 100)
        compliance = np.mean([
            s.get('turn_direction_compliance', 0),
            s.get('action_magnitude_compliance', 0),
            s.get('overtaking_compliance', 0),
            s.get('role_compliance', 0),
        ]) * 100
        solver = s.get('solve_success_rate', 0) * 100
        cpa_margin = min(100, s.get('median_min_cpa', 0) / 50 * 100)
        smoothness = max(0, 100 - min(100, s.get('mean_rudder_rate', 0) / 0.5 * 100))

        values = [safety, compliance, solver, cpa_margin, smoothness]
        values += values[:1]  # close

        ax.fill(angles, values, alpha=0.08, color=style['color'])
        ax.plot(angles, values, color=style['color'], linestyle=style['linestyle'],
                linewidth=1.8, label=style['label'], marker=style['marker'],
                markersize=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=6)
    ax.set_title(f'{scenario_name}: Ablation Radar', fontsize=10, pad=18)
    ax.legend(fontsize=7, framealpha=0.9, edgecolor='gray',
              loc='upper right', bbox_to_anchor=(1.15, 1.1))
    ax.grid(True, alpha=0.25, color=COLORS['grid'])

    plt.tight_layout()
    _save(fig, output_dir, 'ablation_radar')
    return os.path.join(output_dir, 'ablation_radar.png')


def generate_ablation_report(group_results: dict,
                              output_dir: str = "ablation_output",
                              scenario_name: str = "Scenario"):
    """Generate all cross-group comparison figures for ablation.

    Calls plot_cpa_cdf_overlay, plot_compliance_heatmap, plot_ablation_radar.

    Args:
        group_results: Dict mapping group label to MetricsCollector.
        output_dir: Output directory.
        scenario_name: Scenario identifier.

    Returns:
        Path to the CPA CDF overlay PNG (primary comparison plot).
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Generating cross-group comparison for {scenario_name}...")

    plot_cpa_cdf_overlay(group_results, output_dir, scenario_name)
    plot_compliance_heatmap(group_results, output_dir, scenario_name)
    plot_ablation_radar(group_results, output_dir, scenario_name)

    print(f"  Saved: {output_dir}/cpa_cdf_overlay.png")
    print(f"  Saved: {output_dir}/compliance_heatmap.png")
    print(f"  Saved: {output_dir}/ablation_radar.png")

    return os.path.join(output_dir, 'cpa_cdf_overlay.png')
