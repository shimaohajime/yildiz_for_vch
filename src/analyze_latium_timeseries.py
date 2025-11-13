import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ''))
from npsde_pyro import NPSDE, transition_log_ratio


def compute_label_metrics(label, npsde, processed_df, original_df, output_dir, bandwidth=1.0, Nw=200):
    proc = processed_df[processed_df['Label'] == label].sort_values('Time').reset_index(drop=True)
    if proc.empty:
        raise ValueError(f"No processed data found for label '{label}'")

    orig = (
        original_df[original_df['NGA'] == label][['Time', 'PC1', 'PC2']]
        .drop_duplicates(subset='Time')
        .sort_values('Time')
        .reset_index(drop=True)
    )
    if len(orig) != len(proc):
        min_len = min(len(orig), len(proc))
        proc = proc.iloc[:min_len].copy()
        orig = orig.iloc[:min_len].copy()

    years = orig['Time'].to_numpy()
    pc1 = proc['x1'].to_numpy()
    pc2 = proc['x2'].to_numpy()

    log_forward = np.full_like(pc1, np.nan, dtype=float)
    log_backward = np.full_like(pc1, np.nan, dtype=float)
    log_ratio = np.full_like(pc1, np.nan, dtype=float)

    for idx in range(1, len(proc)):
        current = proc.loc[idx - 1, ['x1', 'x2']].to_numpy(dtype=np.float32).reshape(1, -1)
        nxt = proc.loc[idx, ['x1', 'x2']].to_numpy(dtype=np.float32).reshape(1, -1)

        lr, lf, lb = transition_log_ratio(
            npsde,
            current=current,
            nxt=nxt,
            bandwidth=bandwidth,
            Nw_forward=Nw,
            Nw_backward=Nw,
        )
        log_ratio[idx] = float(lr[0])
        log_forward[idx] = float(lf[0])
        log_backward[idx] = float(lb[0])

    safe_label = label.replace(' ', '_')
    metrics_df = pd.DataFrame({
        'Year': years,
        'PC1': pc1,
        'PC2': pc2,
        'log_forward_density': log_forward,
        'log_backward_density': log_backward,
        'log_ratio': log_ratio,
    })

    metrics_path = os.path.join(output_dir, f"{safe_label}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(years, pc1, marker='o', color='tab:blue')
    axes[0].set_ylabel('PC1')
    axes[0].set_title(f'{label} PC1 over time')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(years, pc2, marker='o', color='tab:orange')
    axes[1].set_ylabel('PC2')
    axes[1].set_title(f'{label} PC2 over time')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(years, log_forward, marker='o', color='tab:green')
    axes[2].set_ylabel('log P(xₜ₊₁|xₜ)')
    axes[2].set_title('Forward transition log-density (prev → current)')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(years, log_ratio, marker='o', color='tab:red')
    axes[3].set_ylabel('log-ratio')
    axes[3].set_xlabel('Year')
    axes[3].set_title('Pairwise irreversibility log-ratio (prev → current)')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{safe_label}_aligned_plots.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    summary = {
        'label': label,
        'metrics_csv': metrics_path,
        'figure_path': fig_path,
        'n_points': len(proc),
        'year_min': float(np.nanmin(years)),
        'year_max': float(np.nanmax(years)),
    }
    with open(os.path.join(output_dir, f"{safe_label}_aligned_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compute PCA trajectories, perturbation, and irreversibility metrics.")
    parser.add_argument("--label", type=str, help="Label/NGA to analyze. If omitted, all labels are processed.")
    parser.add_argument("--output-dir", type=str, default="analysis_outputs", help="Directory to store outputs.")
    parser.add_argument("--bandwidth", type=float, default=1.0, help="KDE bandwidth.")
    parser.add_argument("--samples", type=int, default=200, help="Monte Carlo samples for transition densities.")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, 'MR_replication_pyro_model.pt')
    processed_path = os.path.join(project_root, 'data', 'mr_repliciation_for_npsde_pyro.csv')
    original_path = os.path.join(project_root, 'data', 'mr_replication_dataset_with_PCs.csv')

    os.makedirs(args.output_dir, exist_ok=True)

    npsde = NPSDE.load_model(model_path)
    processed_df = pd.read_csv(processed_path)
    original_df = pd.read_csv(original_path)

    if args.label:
        labels = [args.label]
    else:
        labels = sorted(processed_df['Label'].unique())

    summaries = []
    for label in labels:
        try:
            summary = compute_label_metrics(
                label,
                npsde,
                processed_df,
                original_df,
                args.output_dir,
                bandwidth=args.bandwidth,
                Nw=args.samples,
            )
            summaries.append(summary)
            print(f"[OK] {label}: metrics saved to {summary['metrics_csv']}")
        except Exception as exc:
            print(f"[FAIL] {label}: {exc}")

    summary_path = os.path.join(args.output_dir, "analysis_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == '__main__':
    main()
