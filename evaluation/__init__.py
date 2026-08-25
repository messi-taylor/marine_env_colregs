"""Monte Carlo evaluation framework for COLREGS collision avoidance."""

from .batch_runner import BatchRunner, MonteCarloConfig
from .metrics import MetricsCollector, compute_all_metrics
from .noise_sampler import NoiseSampler
from .visualize import (plot_cpa_histogram, plot_trajectories, plot_control_effort,
                         plot_compliance_summary, generate_report,
                         plot_cpa_cdf_overlay, plot_compliance_heatmap,
                         plot_ablation_radar, generate_ablation_report)
from .ablation import (AblationGroup, AblationConfig, AblationRunner, GroupResult)

__all__ = [
    'BatchRunner', 'MonteCarloConfig',
    'MetricsCollector', 'compute_all_metrics',
    'NoiseSampler',
    'plot_cpa_histogram', 'plot_trajectories', 'plot_control_effort',
    'plot_compliance_summary', 'generate_report',
    'plot_cpa_cdf_overlay', 'plot_compliance_heatmap',
    'plot_ablation_radar', 'generate_ablation_report',
    'AblationGroup', 'AblationConfig', 'AblationRunner', 'GroupResult',
]
