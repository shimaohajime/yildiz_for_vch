"""
Process SCV/Polaris data to produce NGA-level timeseries with Scale/Computation dims.
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def process_polaris_data(
    input_path,
    output_path,
    sheet_name='SCV_Clean',
    scale_column='Scale_1',
    comp_column='Comp'
):
    """
    Process SCV/Polaris data:
    1. Load CSV/Excel input.
    2. Drop rows with blank NGA.
    3. Drop rows with missing values in Scale/Computation dimensions.
    4. Convert Year to time scale starting from 1 for each NGA (1 unit = 1 century).
    5. Aggregate duplicate NGA-Time pairs by mean.
    6. Rename dimensions to Scale/Computation for downstream compatibility.
    """
    # Read the data
    ext = os.path.splitext(input_path)[1].lower()
    print(f"Reading data from {input_path}...")
    if ext == '.csv':
        df = pd.read_csv(input_path)
    else:
        print(f"  Detected non-CSV input, reading Excel sheet '{sheet_name}'.")
        df = pd.read_excel(input_path, sheet_name=sheet_name)
    
    print(f"Original data shape: {df.shape}")
    
    # Validate columns
    value_cols = [scale_column, comp_column]
    required_cols = ['NGA', 'Year'] + value_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Select relevant columns
    df_work = df[required_cols].copy()
    
    # Drop rows with blank NGA
    initial_size = len(df_work)
    df_work = df_work.dropna(subset=['NGA'])
    df_work = df_work[df_work['NGA'].astype(str).str.strip() != '']
    nga_dropped = initial_size - len(df_work)
    if nga_dropped > 0:
        print(f"Dropped {nga_dropped} rows with blank NGA ({nga_dropped/initial_size*100:.1f}%)")
    
    # Drop rows with missing values in any of the variables
    initial_size = len(df_work)
    df_work = df_work.dropna(subset=value_cols)
    dropped_rows = initial_size - len(df_work)
    print(f"Dropped {dropped_rows} rows with missing values ({dropped_rows/initial_size*100:.1f}%)")
    print(f"Remaining data shape: {df_work.shape}")
    
    # Convert Year to time scale starting from 1 for each NGA (1 unit = 1 century)
    print("\nConverting Year to time scale (starting from 1, 1 unit = 1 century)...")
    df_work['Year'] = pd.to_numeric(df_work['Year'], errors='coerce')
    df_work = df_work.dropna(subset=['Year'])
    
    # Group by NGA and convert Year to centuries starting from 1
    min_years = df_work.groupby('NGA')['Year'].transform('min')
    df_work['Time'] = ((df_work['Year'] - min_years) / 100.0) + 1.0
    
    print(f"  Time range: {df_work['Time'].min():.1f} to {df_work['Time'].max():.1f} centuries")

    # Check for duplicates and aggregate
    print("\nChecking for duplicate NGA-Time pairs...")
    duplicates = df_work[df_work.duplicated(subset=['NGA', 'Time'], keep=False)]
    if not duplicates.empty:
        n_duplicates = len(duplicates)
        n_unique_duplicates = duplicates.duplicated(subset=['NGA', 'Time']).sum()
        affected_ngas = duplicates['NGA'].nunique()
        print(f"  Found {n_duplicates} duplicate rows (representing {n_duplicates - n_unique_duplicates} unique time points) across {affected_ngas} NGAs.")
        
        # Aggregate by mean
        print("  Aggregating duplicates by mean...")
        # Group by NGA, Year, Time and take mean of other columns
        # Note: Year and Time are 1-to-1, so grouping by both is safe
        df_work = df_work.groupby(['NGA', 'Year', 'Time'])[value_cols].mean().reset_index()
        print(f"  New shape after aggregation: {df_work.shape}")
    else:
        print("  No duplicates found.")
    
    # Create output dataframe (keep both Year and Time for reference)
    df_output = df_work[['NGA', 'Year', 'Time']].copy()
    df_output['Scale'] = pd.to_numeric(df_work[scale_column], errors='coerce')
    df_output['Computation'] = pd.to_numeric(df_work[comp_column], errors='coerce')
    df_output = df_output.dropna(subset=['Scale', 'Computation'])
    
    # Print summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    print(f"Final rows: {len(df_output)}")
    print(f"Unique NGAs: {df_output['NGA'].nunique()}")
    print(f"Year range: {df_output['Year'].min():.0f} to {df_output['Year'].max():.0f}")
    print(f"Time range: {df_output['Time'].min():.1f} to {df_output['Time'].max():.1f} centuries")
    print(f"Source columns -> outputs:")
    print(f"  {scale_column} -> Scale")
    print(f"  {comp_column} -> Computation")
    scale_desc = df_output['Scale'].describe()
    comp_desc = df_output['Computation'].describe()
    print("\nScale stats:")
    print(scale_desc[['mean', 'std', 'min', 'max']])
    print("\nComputation stats:")
    print(comp_desc[['mean', 'std', 'min', 'max']])
    print("="*60)
    
    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")
    
    return df_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process SCV/Polaris data for SDE analysis')
    parser.add_argument(
        '--input',
        type=str,
        default='data/scv_clean_all_fixed.csv',
        help='Path to SCV CSV or Polaris Excel file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/scv_processed_for_npsde.csv',
        help='Path to save processed CSV'
    )
    parser.add_argument(
        '--sheet',
        type=str,
        default='SCV_Clean',
        help='Sheet name to read (Excel input only)'
    )
    parser.add_argument(
        '--scale-col',
        type=str,
        default='Scale_1',
        help='Column name for Scale dimension'
    )
    parser.add_argument(
        '--comp-col',
        type=str,
        default='Comp',
        help='Column name for Computation dimension'
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[1]
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    if not output_path.is_absolute():
        output_path = project_root / output_path
    
    process_polaris_data(
        input_path,
        output_path,
        sheet_name=args.sheet,
        scale_column=args.scale_col,
        comp_column=args.comp_col
    )
