"""Generate Polaris NGA plots with shared y-axis limits."""

from __future__ import annotations

import glob
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_axis_limits(metrics_paths: list[str], padding: float) -> Dict[str, Tuple[float, float]]:
    """Compute shared axis limits across all metric files."""
    cols = ["Scale", "Computation", "log_forward_density", "log_ratio"]
    frames = (pd.read_csv(path)[cols] for path in metrics_paths)
    all_metrics = pd.concat(frames, ignore_index=True)

    axis_limits: Dict[str, Tuple[float, float]] = {}
    for col in cols:
        finite_vals = all_metrics[col].replace([np.inf, -np.inf], np.nan).dropna()
        vmin, vmax = finite_vals.min(), finite_vals.max()
        pad = (vmax - vmin) * padding if vmax > vmin else 1.0
        axis_limits[col] = (vmin - pad, vmax + pad)
    return axis_limits


def replot_with_shared_axes(project_root: str, padding: float = 0.05) -> None:
    """Generate aligned plots for each NGA with global axis limits."""
    output_dir = os.path.join(project_root, "polaris_analysis_outputs")
    metrics_files = sorted(glob.glob(os.path.join(output_dir, "*_metrics.csv")))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics CSVs in {output_dir}")

    axis_limits = compute_axis_limits(metrics_files, padding)

    for metrics_path in metrics_files:
        nga = os.path.basename(metrics_path).removesuffix("_metrics.csv")
        df = pd.read_csv(metrics_path)

        years = df["Year"].to_numpy()
        scale = df["Scale"].to_numpy()
        comp = df["Computation"].to_numpy()
        log_f = df["log_forward_density"].to_numpy()
        log_r = df["log_ratio"].to_numpy()

        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        axes[0].plot(years, scale, marker="o", color="tab:blue")
        axes[0].set_ylabel("Scale")
        axes[0].set_title(f"{nga} - Scale over time")
        axes[0].set_ylim(axis_limits["Scale"])
        axes[0].grid(alpha=0.3)

        axes[1].plot(years, comp, marker="o", color="tab:orange")
        axes[1].set_ylabel("Computation")
        axes[1].set_title(f"{nga} - Computation over time")
        axes[1].set_ylim(axis_limits["Computation"])
        axes[1].grid(alpha=0.3)

        axes[2].plot(years, log_f, marker="o", color="tab:green")
        axes[2].set_ylabel("log P(x_{t+1}|x_t)")
        axes[2].set_title(f"{nga} - Forward transition log density")
        axes[2].set_ylim(axis_limits["log_forward_density"])
        axes[2].grid(alpha=0.3)

        axes[3].plot(years, log_r, marker="o", color="tab:red")
        axes[3].set_ylabel("log P(x_{t+1}|x_t) - log P(x_t|x_{t+1})")
        axes[3].set_xlabel("Year (BC/AD)")
        axes[3].set_title(f"{nga} - Irreversibility score")
        axes[3].set_ylim(axis_limits["log_ratio"])
        axes[3].grid(alpha=0.3)

        fig.tight_layout()
        plot_path = os.path.join(output_dir, f"{nga}_aligned_plots_global.png")
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

    print("Done. Shared-axis plots saved with *_aligned_plots_global.png suffix.")


if __name__ == "__main__":
    PROJECT_ROOT = "/Users/xli14/Dropbox/Hajime/PennStateWork/Stochastic_process_inference2025/yildiz_for_vch"
    replot_with_shared_axes(PROJECT_ROOT)

