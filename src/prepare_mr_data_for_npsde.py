"""
Script to prepare MR replication data for NPSDE model training.
Transforms data to match the format expected by the model:
- Label: NGA (Natural Geographic Area)
- Time: in centuries, starting from zero
- x1: PC1
- x2: PC2
"""

import pandas as pd
import numpy as np
import os

def prepare_mr_data_for_npsde(input_path, output_path):
    """
    Prepare MR replication data for NPSDE format.
    
    Parameters:
    -----------
    input_path : str
        Path to the CSV file with PCA results
    output_path : str
        Path to save the formatted data
    """
    # Read the data with PCs
    print(f"Reading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Check required columns
    required_cols = ['NGA', 'Time', 'PC1', 'PC2']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create a copy with only needed columns
    df_npsde = df[['NGA', 'Time', 'PC1', 'PC2']].copy()
    
    # Drop rows with missing values
    initial_size = df_npsde.shape[0]
    df_npsde = df_npsde.dropna()
    dropped_rows = initial_size - df_npsde.shape[0]
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with missing values.")
    
    # Convert Time to numeric (in case it's not)
    df_npsde = df_npsde[pd.to_numeric(df_npsde["Time"], errors='coerce').notnull()]
    df_npsde["Time"] = pd.to_numeric(df_npsde["Time"], errors='coerce')
    
    # Find the global time range (for reporting)
    min_time = df_npsde["Time"].min()
    max_time = df_npsde["Time"].max()
    print(f"Original time range (across all NGAs): {min_time} to {max_time} (years)")
    
    # Convert time to centuries starting from zero FOR EACH NGA INDIVIDUALLY
    # Group by NGA and reset time to start at 0 for each group
    # Use transform to subtract the minimum time for each NGA
    min_times = df_npsde.groupby("NGA")["Time"].transform("min")
    df_npsde["Time"] = (df_npsde["Time"] - min_times) / 100.0
    
    print(f"Converted time: each NGA now starts at 0.00 centuries")
    print(f"Overall time range: {df_npsde['Time'].min():.2f} to {df_npsde['Time'].max():.2f} centuries")
    
    # Rename columns to match NPSDE format
    df_npsde.rename(columns={
        "NGA": "Label",
        "PC1": "x1",
        "PC2": "x2"
    }, inplace=True)
    
    # Select and reorder columns: Label, Time, x1, x2
    df_npsde = df_npsde[["Label", "Time", "x1", "x2"]].copy()
    
    # Sort by Label then by Time
    df_npsde.sort_values(by=["Label", "Time"], inplace=True)
    
    # Reset index
    df_npsde.reset_index(drop=True, inplace=True)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Data Summary")
    print(f"{'='*60}")
    print(f"Total rows: {len(df_npsde)}")
    print(f"Number of unique Labels (NGAs): {df_npsde['Label'].nunique()}")
    print(f"Time range: {df_npsde['Time'].min():.2f} to {df_npsde['Time'].max():.2f} centuries")
    print(f"\nColumns: {list(df_npsde.columns)}")
    print(f"\nFirst few rows:")
    print(df_npsde.head(10))
    print(f"\nLast few rows:")
    print(df_npsde.tail(10))
    
    # Check for each NGA
    print(f"\n{'='*60}")
    print("NGAs (Labels) in dataset:")
    print(f"{'='*60}")
    nga_counts = df_npsde['Label'].value_counts().sort_index()
    print(f"Total NGAs: {len(nga_counts)}")
    print(f"\nSample NGAs (first 10):")
    for nga, count in list(nga_counts.items())[:10]:
        nga_data = df_npsde[df_npsde['Label'] == nga]
        time_range = f"{nga_data['Time'].min():.2f} - {nga_data['Time'].max():.2f}"
        print(f"  {nga}: {count} time points, time range: {time_range}")
    
    # Save the data
    df_npsde.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Data saved to: {output_path}")
    print(f"{'='*60}")
    
    return df_npsde


if __name__ == "__main__":
    # Set paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Input: data with PCs
    input_path = os.path.join(project_root, "data", "mr_replication_dataset_with_PCs.csv")
    
    # Output: formatted for NPSDE
    output_path = os.path.join(project_root, "data", "mr_repliciation_for_npsde_pyro.csv")
    
    # Prepare the data
    df_npsde = prepare_mr_data_for_npsde(input_path, output_path)

