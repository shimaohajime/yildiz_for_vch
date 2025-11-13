"""
Script to investigate the PCA output from the MR replication dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def investigate_pca_output(data_path):
    """
    Investigate the PCA output in detail.
    """
    # Read the data with PCs
    print("Loading data with PCA results...")
    df = pd.read_csv(data_path)
    
    print(f"\n{'='*70}")
    print("DATASET OVERVIEW")
    print(f"{'='*70}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\nColumns: {list(df.columns)}")
    
    # Basic statistics for PCs
    print(f"\n{'='*70}")
    print("PRINCIPAL COMPONENTS STATISTICS")
    print(f"{'='*70}")
    print("\nPC1 Statistics:")
    print(df['PC1'].describe())
    print(f"\nPC2 Statistics:")
    print(df['PC2'].describe())
    
    # Check for unique NGAs and PolIDs
    print(f"\n{'='*70}")
    print("DATA STRUCTURE")
    print(f"{'='*70}")
    print(f"Number of unique NGAs: {df['NGA'].nunique()}")
    print(f"Number of unique PolIDs: {df['PolID'].nunique()}")
    print(f"\nTime range: {df['Time'].min()} to {df['Time'].max()}")
    
    # Show some example time series
    print(f"\n{'='*70}")
    print("SAMPLE TIME SERIES (first 5 NGAs)")
    print(f"{'='*70}")
    unique_ngas = df['NGA'].unique()[:5]
    for nga in unique_ngas:
        nga_data = df[df['NGA'] == nga].sort_values('Time')
        print(f"\n{nga}:")
        print(f"  Time points: {len(nga_data)}")
        print(f"  Time range: {nga_data['Time'].min()} - {nga_data['Time'].max()}")
        print(f"  PC1 range: [{nga_data['PC1'].min():.3f}, {nga_data['PC1'].max():.3f}]")
        print(f"  PC2 range: [{nga_data['PC2'].min():.3f}, {nga_data['PC2'].max():.3f}]")
        print(f"  PC1 mean: {nga_data['PC1'].mean():.3f}, std: {nga_data['PC1'].std():.3f}")
        print(f"  PC2 mean: {nga_data['PC2'].mean():.3f}, std: {nga_data['PC2'].std():.3f}")
    
    # Correlation between original variables and PCs
    print(f"\n{'='*70}")
    print("CORRELATION: Original Variables vs Principal Components")
    print(f"{'='*70}")
    variables = ['PolPop', 'PolTerr', 'CapPop', 'levels', 'government', 
                 'infrastr', 'writing', 'texts', 'money']
    
    print(f"\n{'Variable':<15} {'Corr with PC1':>15} {'Corr with PC2':>15}")
    print("-" * 50)
    for var in variables:
        corr_pc1 = df[var].corr(df['PC1'])
        corr_pc2 = df[var].corr(df['PC2'])
        print(f"{var:<15} {corr_pc1:>15.4f} {corr_pc2:>15.4f}")
    
    # Time evolution analysis
    print(f"\n{'='*70}")
    print("TIME EVOLUTION ANALYSIS")
    print(f"{'='*70}")
    # Group by NGA and check if PCs change over time
    nga_changes = []
    for nga in df['NGA'].unique():
        nga_data = df[df['NGA'] == nga].sort_values('Time')
        if len(nga_data) > 1:
            pc1_change = nga_data['PC1'].iloc[-1] - nga_data['PC1'].iloc[0]
            pc2_change = nga_data['PC2'].iloc[-1] - nga_data['PC2'].iloc[0]
            nga_changes.append({
                'NGA': nga,
                'time_span': nga_data['Time'].max() - nga_data['Time'].min(),
                'n_points': len(nga_data),
                'PC1_change': pc1_change,
                'PC2_change': pc2_change,
                'PC1_std': nga_data['PC1'].std(),
                'PC2_std': nga_data['PC2'].std()
            })
    
    changes_df = pd.DataFrame(nga_changes)
    changes_df['abs_PC1_change'] = changes_df['PC1_change'].abs()
    print(f"\nNGAs with largest PC1 changes over time:")
    print(changes_df.nlargest(10, 'abs_PC1_change')[['NGA', 'time_span', 'n_points', 'PC1_change', 'PC2_change']].to_string(index=False))
    
    # Check for missing values
    print(f"\n{'='*70}")
    print("DATA QUALITY")
    print(f"{'='*70}")
    print(f"Missing values in PC1: {df['PC1'].isna().sum()}")
    print(f"Missing values in PC2: {df['PC2'].isna().sum()}")
    print(f"Missing values in original variables:")
    for var in variables:
        missing = df[var].isna().sum()
        if missing > 0:
            print(f"  {var}: {missing} ({missing/len(df)*100:.2f}%)")
    
    # Distribution of PCs
    print(f"\n{'='*70}")
    print("PC DISTRIBUTION QUANTILES")
    print(f"{'='*70}")
    print(f"\nPC1 quantiles:")
    print(df['PC1'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
    print(f"\nPC2 quantiles:")
    print(df['PC2'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
    
    return df, changes_df


if __name__ == "__main__":
    # Set paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "mr_replication_dataset_with_PCs.csv")
    
    # Investigate
    df, changes_df = investigate_pca_output(data_path)
    
    print(f"\n{'='*70}")
    print("Investigation complete!")
    print(f"{'='*70}")

