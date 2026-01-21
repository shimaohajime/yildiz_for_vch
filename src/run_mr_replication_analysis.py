"""
Run Yildiz NPSDE analysis on MR replication dataset with PC1 and PC2.
Trains the model and applies perturbation detection and irreversibility analysis.
Includes preprocessing to remove forward-filled missing values.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import from local src directory
sys.path.append(os.path.dirname(__file__))
from npsde_pyro import format_input_from_timedata, pyro_npsde_run, NPSDE, transition_log_ratio


def read_labeled_timeseries(df, reset_time=False, time_unit=1, data_dim=None):
    """
    Load a CSV file that contains timeseries data delineated by labels.
    Takes dataframe directly (not state dict).
    """
    labels = df.iloc[:, 0].to_numpy()
    indices = np.where(np.logical_not(np.equal(labels[1:], labels[:-1])))[0] + 1
    if not data_dim:
        data_dim = df.shape[1] - 2

    time_column = np.split(df.iloc[:, 1].to_numpy(dtype=np.float64) / time_unit, indices)
    data_columns = np.split(df.iloc[:, 2:data_dim+2].to_numpy(dtype=np.float64), indices, axis=0)

    if reset_time:
        time_column = [segment - segment[0] for segment in time_column]

    return (time_column, data_columns)


def remove_forward_filled_values(df, data_columns):
    """
    Detect and remove forward-filled missing values.
    A row is considered forward-filled if all data columns are identical to the previous row
    for the same label.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Label, Time, and data columns
    data_columns : list
        List of column names containing the data (e.g., ['x1', 'x2'])
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with forward-filled rows removed
    """
    df = df.copy()
    df = df.sort_values(['Label', 'Time']).reset_index(drop=True)
    
    initial_size = len(df)
    
    # Identify forward-filled rows
    # A row is forward-filled if:
    # 1. It has the same Label as the previous row
    # 2. All data columns are identical to the previous row
    is_forward_filled = pd.Series([False] * len(df))
    
    for i in range(1, len(df)):
        if df.iloc[i]['Label'] == df.iloc[i-1]['Label']:
            # Check if all data columns are identical
            all_identical = all(
                np.isclose(df.iloc[i][col], df.iloc[i-1][col], equal_nan=True)
                for col in data_columns
            )
            if all_identical:
                is_forward_filled.iloc[i] = True
    
    # Remove forward-filled rows
    df_cleaned = df[~is_forward_filled].copy().reset_index(drop=True)
    
    n_removed = initial_size - len(df_cleaned)
    if n_removed > 0:
        print(f"  Removed {n_removed} forward-filled rows ({n_removed/initial_size*100:.1f}%)")
    
    return df_cleaned


def prepare_mr_for_npsde(df_path, output_path=None):
    """
    Prepare MR replication data for NPSDE format.
    Converts NGA-year data to Label, Time, x1, x2 format with time reset per NGA.
    Removes forward-filled missing values.
    """
    df = pd.read_csv(df_path)
    
    # Select and rename columns
    df_npsde = df[['NGA', 'Time', 'PC1', 'PC2']].copy()
    df_npsde.rename(columns={
        'NGA': 'Label',
        'PC1': 'x1',
        'PC2': 'x2'
    }, inplace=True)
    
    # Drop rows with missing values
    initial_size = len(df_npsde)
    df_npsde = df_npsde.dropna()
    dropped_rows = initial_size - len(df_npsde)
    if dropped_rows > 0:
        print(f"  Dropped {dropped_rows} rows with missing values.")
    
    # Convert Time to numeric
    df_npsde['Time'] = pd.to_numeric(df_npsde['Time'], errors='coerce')
    df_npsde = df_npsde.dropna(subset=['Time'])
    
    # Remove forward-filled values
    print("  Detecting forward-filled missing values...")
    df_npsde = remove_forward_filled_values(df_npsde, ['x1', 'x2'])
    
    # Reset time to start at 0 for each NGA (in centuries)
    min_times = df_npsde.groupby('Label')['Time'].transform('min')
    df_npsde['Time'] = (df_npsde['Time'] - min_times) / 100.0
    
    # Sort by Label then Time
    df_npsde = df_npsde.sort_values(['Label', 'Time']).reset_index(drop=True)
    
    # Handle duplicates by taking mean
    df_npsde = df_npsde.groupby(['Label', 'Time']).mean().reset_index()
    
    if output_path:
        df_npsde.to_csv(output_path, index=False)
        print(f"  Prepared data saved to {output_path}")
    
    return df_npsde


def compute_nga_metrics(nga, npsde, processed_df, original_df, output_dir, bandwidth=1.0, Nw=200):
    """Compute perturbation and irreversibility metrics for a single NGA."""
    proc = processed_df[processed_df['Label'] == nga].sort_values('Time').reset_index(drop=True)
    if len(proc) < 2:
        return None
    
    # Get original years for this NGA
    orig_nga = original_df[original_df['NGA'] == nga].sort_values('Time').reset_index(drop=True)
    
    # Match processed rows to original rows by finding closest time matches
    # Since we removed forward-filled rows, we need to align carefully
    years = []
    for _, proc_row in proc.iterrows():
        # Find closest original row by time (in centuries)
        orig_time_centuries = (orig_nga['Time'] - orig_nga['Time'].min()) / 100.0
        closest_idx = (orig_time_centuries - proc_row['Time']).abs().idxmin()
        years.append(orig_nga.loc[closest_idx, 'Time'])
    years = np.array(years)
    
    x1 = proc['x1'].to_numpy()
    x2 = proc['x2'].to_numpy()
    
    log_forward = np.full_like(x1, np.nan, dtype=float)
    log_backward = np.full_like(x1, np.nan, dtype=float)
    log_ratio = np.full_like(x1, np.nan, dtype=float)
    
    # Compute transition log ratios for each consecutive pair
    for idx in range(1, len(proc)):
        current = proc.loc[idx - 1, ['x1', 'x2']].to_numpy(dtype=np.float32).reshape(1, -1)
        nxt = proc.loc[idx, ['x1', 'x2']].to_numpy(dtype=np.float32).reshape(1, -1)
        
        try:
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
        except Exception as e:
            print(f"  Warning: Failed to compute metrics for {nga} at index {idx}: {e}")
            continue
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        'Year': years,
        'PC1': x1,
        'PC2': x2,
        'log_forward_density': log_forward,
        'log_backward_density': log_backward,
        'log_ratio': log_ratio,
    })
    
    # Save metrics
    safe_nga_name = nga.replace('/', '_').replace('\\', '_')
    metrics_path = os.path.join(output_dir, f'{safe_nga_name}_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    
    # Create aligned plots
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    axes[0].plot(years, x1, marker='o', color='tab:blue')
    axes[0].set_ylabel('PC1')
    axes[0].set_title(f'{nga} - PC1 over time')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(years, x2, marker='o', color='tab:orange')
    axes[1].set_ylabel('PC2')
    axes[1].set_title(f'{nga} - PC2 over time')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(years, log_forward, marker='o', color='tab:green')
    axes[2].set_ylabel('log P(x_{t+1}|x_t)')
    axes[2].set_title(f'{nga} - Forward transition log density')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(years, log_ratio, marker='o', color='tab:red')
    axes[3].set_ylabel('log P(x_{t+1}|x_t) - log P(x_t|x_{t+1})')
    axes[3].set_xlabel('Year (BC/AD)')
    axes[3].set_title(f'{nga} - Irreversibility score')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{safe_nga_name}_aligned_plots.png')
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    
    return {
        'NGA': nga,
        'n_points': len(proc),
        'year_range': (int(years.min()), int(years.max())),
        'metrics_path': metrics_path,
        'plot_path': plot_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run Yildiz NPSDE analysis on MR replication dataset'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/mr_replication_dataset_with_PCs.csv',
        help='Path to MR replication CSV file with PCs'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='mr_replication_analysis_outputs',
        help='Directory to save analysis results'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='MR_replication_pyro_model',
        help='Name for saved model files'
    )
    parser.add_argument(
        '--train-steps',
        type=int,
        default=50,
        help='Number of training steps'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.02,
        help='Learning rate'
    )
    parser.add_argument(
        '--Nw',
        type=int,
        default=50,
        help='Number of Monte Carlo samples for training'
    )
    parser.add_argument(
        '--bandwidth',
        type=float,
        default=1.0,
        help='KDE bandwidth for perturbation/irreversibility'
    )
    parser.add_argument(
        '--metrics-samples',
        type=int,
        default=200,
        help='Number of MC samples for metrics computation'
    )
    parser.add_argument(
        '--ngas',
        nargs='+',
        help='Specific NGAs to analyze (if not provided, analyzes all)'
    )
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_path = os.path.join(project_root, args.input)
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("MR Replication Yildiz NPSDE Analysis")
    print("="*70)
    
    # Step 1: Prepare data
    print("\n[1/4] Preparing data for NPSDE format...")
    prepared_path = os.path.join(output_dir, 'mr_replication_prepared_for_npsde.csv')
    df_prepared = prepare_mr_for_npsde(input_path, prepared_path)
    print(f"  Prepared {len(df_prepared)} rows for {df_prepared['Label'].nunique()} NGAs")
    
    # Step 2: Format for NPSDE
    print("\n[2/4] Formatting data for NPSDE...")
    time_series, data_series = read_labeled_timeseries(df_prepared, reset_time=True, data_dim=2)
    X = format_input_from_timedata(time_series, data_series)
    print(f"  Formatted data shape: {X.shape}")
    
    # Step 3: Train model
    print("\n[3/4] Training NPSDE model...")
    start_time = time.time()
    npsde = pyro_npsde_run(
        X, 
        n_vars=2,
        steps=args.train_steps, 
        lr=args.lr, 
        Nw=args.Nw, 
        sf_f=1, 
        sf_g=0.2, 
        ell_f=[1.0, 1.0],
        ell_g=0.5, 
        noise=[1.0, 1.0],
        W=7, 
        fix_sf=0, 
        fix_ell=0, 
        fix_Z=0, 
        delta_t=0.1,
        save_model=os.path.join(output_dir, args.model_name),
        Z=None, 
        Zg=None, 
        U_map=None, 
        Ug_map=None
    )
    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.2f} seconds")
    
    # Generate model plots
    print("  Generating model visualization plots...")
    npsde.plot_model(X, os.path.join(output_dir, args.model_name), Nw=50)
    
    # Step 4: Compute metrics for NGAs
    print("\n[4/4] Computing perturbation and irreversibility metrics...")
    df_original = pd.read_csv(input_path)
    
    ngas_to_analyze = args.ngas if args.ngas else df_prepared['Label'].unique()
    results = []
    
    for i, nga in enumerate(ngas_to_analyze, 1):
        print(f"  [{i}/{len(ngas_to_analyze)}] Processing {nga}...")
        try:
            result = compute_nga_metrics(
                nga, npsde, df_prepared, df_original, 
                output_dir, bandwidth=args.bandwidth, Nw=args.metrics_samples
            )
            if result:
                results.append(result)
                print(f"    ✓ Saved metrics and plots")
            else:
                print(f"    ✗ Skipped (insufficient data)")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue
    
    # Save summary
    summary = {
        'model_name': args.model_name,
        'n_variables': 2,
        'variables': ['PC1', 'PC2'],
        'training_time_seconds': training_time,
        'n_ngas_analyzed': len(results),
        'total_ngas': len(ngas_to_analyze),
        'results': results,
    }
    
    summary_path = os.path.join(output_dir, 'analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print(f"  - Model: {args.model_name}.pt")
    print(f"  - Metrics for {len(results)} NGAs")
    print(f"  - Summary: analysis_summary.json")
    print("="*70)


if __name__ == '__main__':
    main()

