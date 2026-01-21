"""
Run Yildiz NPSDE analysis on SCV/Polaris datasets with Scale and Computation dims.
Trains the model and applies perturbation detection and irreversibility analysis.
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


def prepare_polaris_for_npsde(df_path, output_path=None):
    """
    Prepare Polaris data for NPSDE format.
    Uses Time column (already in centuries starting from 1) and converts to Label, Time, x1, x2 format.
    Note: Explicitly excludes 'Year' column to ensure correct column order for read_labeled_timeseries:
    - Column 0: Label
    - Column 1: Time
    - Columns 2+: Data variables (x1, x2)
    """
    df = pd.read_csv(df_path)
    
    # Select and rename columns (explicitly exclude 'Year' to ensure correct column order)
    # Expected final structure: Label, Time, x1, x2
    df_npsde = df[['NGA', 'Time', 'Scale', 'Computation']].copy()
    df_npsde.rename(columns={
        'NGA': 'Label',
        'Scale': 'x1',
        'Computation': 'x2'
    }, inplace=True)
    
    # Verify column order is correct
    expected_cols = ['Label', 'Time', 'x1', 'x2']
    if list(df_npsde.columns) != expected_cols:
        raise ValueError(f"Column order mismatch. Expected {expected_cols}, got {list(df_npsde.columns)}")
    
    # Drop rows with missing values
    initial_size = len(df_npsde)
    df_npsde = df_npsde.dropna()
    dropped_rows = initial_size - len(df_npsde)
    if dropped_rows > 0:
        print(f"  Dropped {dropped_rows} rows with missing values.")
    
    # Convert Time to numeric and cast to integer (centuries)
    df_npsde['Time'] = pd.to_numeric(df_npsde['Time'], errors='coerce')
    df_npsde = df_npsde.dropna(subset=['Time'])
    df_npsde['Time'] = df_npsde['Time'].astype(int)
    
    # Sort by Label then Time
    df_npsde = df_npsde.sort_values(['Label', 'Time']).reset_index(drop=True)
    
    # Handle duplicates (same century) by taking mean
    # This aggregates multiple observations within the same century
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
    
    # Get original years for this NGA (Time in processed data corresponds to Year in original)
    orig_nga = original_df[original_df['NGA'] == nga].sort_values('Time').reset_index(drop=True)
    
    # Match processed rows to original rows by Time (they should align)
    years = []
    for _, proc_row in proc.iterrows():
        # Find matching row by Time
        matching = orig_nga[orig_nga['Time'] == proc_row['Time']]
        if len(matching) > 0:
            years.append(matching.iloc[0]['Year'])
        else:
            # Fallback: find closest by Time
            closest_idx = (orig_nga['Time'] - proc_row['Time']).abs().idxmin()
            years.append(orig_nga.loc[closest_idx, 'Year'])
    years = np.array(years)
    
    x1 = proc['x1'].to_numpy()  # Scale
    x2 = proc['x2'].to_numpy()  # Computation
    
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
        'Scale': x1,
        'Computation': x2,
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
    axes[0].set_ylabel('Scale')
    axes[0].set_title(f'{nga} - Scale over time')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(years, x2, marker='o', color='tab:orange')
    axes[1].set_ylabel('Computation')
    axes[1].set_title(f'{nga} - Computation over time')
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
        description='Run Yildiz NPSDE analysis on SCV/Polaris dataset'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/scv_processed_for_npsde.csv',
        help='Path to processed CSV file with NGA/Time/Scale/Computation'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='scv_analysis_outputs',
        help='Directory to save analysis results'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='SCV_pyro_model',
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
        '--sf-f',
        type=float,
        default=1.0,
        help='Signal variance for drift kernel (sf_f)'
    )
    parser.add_argument(
        '--sf-g',
        type=float,
        default=0.2,
        help='Signal variance for diffusion kernel (sf_g)'
    )
    parser.add_argument(
        '--ell-f',
        type=float,
        nargs='+',
        default=[1.0, 1.0],
        help='Lengthscale(s) for drift kernel (expects 2 values for Scale/Computation)'
    )
    parser.add_argument(
        '--ell-g',
        type=float,
        default=0.5,
        help='Lengthscale for diffusion kernel'
    )
    parser.add_argument(
        '--noise',
        type=float,
        nargs='+',
        default=[1.0, 1.0],
        help='Observation noise levels (expects 2 values for Scale/Computation)'
    )
    parser.add_argument(
        '--W',
        type=int,
        default=5,
        help='Number of inducing points per dimension'
    )
    parser.add_argument(
        '--fix-sf',
        type=int,
        default=0,
        help='Fix sf parameters during training (1=True)'
    )
    parser.add_argument(
        '--fix-ell',
        type=int,
        default=0,
        help='Fix ell parameters during training (1=True)'
    )
    parser.add_argument(
        '--fix-Z',
        type=int,
        default=0,
        help='Fix Z inducing locations during training (1=True)'
    )
    parser.add_argument(
        '--delta-t',
        type=float,
        default=0.1,
        help='Discretization step size for NPSDE'
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
    if len(args.ell_f) != 2:
        raise ValueError(f"--ell-f expects 2 values, got {args.ell_f}")
    if len(args.noise) != 2:
        raise ValueError(f"--noise expects 2 values, got {args.noise}")
    
    print("SCV / Polaris Yildiz NPSDE Analysis")
    print("="*70)
    
    # Step 1: Prepare data
    print("\n[1/4] Preparing data for NPSDE format...")
    prepared_path = os.path.join(output_dir, 'prepared_for_npsde.csv')
    df_prepared = prepare_polaris_for_npsde(input_path, prepared_path)
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
        sf_f=args.sf_f, 
        sf_g=args.sf_g, 
        ell_f=args.ell_f,
        ell_g=args.ell_g, 
        noise=args.noise,
        W=args.W, 
        fix_sf=args.fix_sf, 
        fix_ell=args.fix_ell, 
        fix_Z=args.fix_Z, 
        delta_t=args.delta_t,
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
        'variables': ['Scale', 'Computation'],
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

