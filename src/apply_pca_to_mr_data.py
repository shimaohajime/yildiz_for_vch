"""
Script to read mr_replication_dataset and apply PCA on nine variables.
Extracts first two principal components.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def apply_pca_to_mr_data(data_path, output_path=None):
    """
    Read the MR replication dataset and apply PCA on nine specified variables.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file
    output_path : str, optional
        Path to save the results. If None, results are not saved.
    
    Returns:
    --------
    tuple : (original_df, pca_df, pca_model, scaler)
        - original_df: Original dataframe
        - pca_df: Dataframe with first two PCs added
        - pca_model: Fitted PCA model
        - scaler: Fitted StandardScaler
    """
    # Read the data
    print(f"Reading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Select the nine variables for PCA
    variables = ['PolPop', 'PolTerr', 'CapPop', 'levels', 'government', 
                 'infrastr', 'writing', 'texts', 'money']
    
    # Check if all variables exist
    missing_vars = [var for var in variables if var not in df.columns]
    if missing_vars:
        raise ValueError(f"Missing variables in dataset: {missing_vars}")
    
    # Extract the variables
    X = df[variables].values
    
    # Check for missing values
    if np.isnan(X).any():
        print("Warning: Missing values detected. Filling with column means.")
        X = pd.DataFrame(X, columns=variables).fillna(
            pd.DataFrame(X, columns=variables).mean()
        ).values
    
    # Standardize the data (important for PCA)
    print("Standardizing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    print("Applying PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create a copy of the dataframe and add PC columns
    df_pca = df.copy()
    df_pca['PC1'] = X_pca[:, 0]
    df_pca['PC2'] = X_pca[:, 1]
    
    # Print summary
    print("\n" + "="*60)
    print("PCA Summary")
    print("="*60)
    print(f"Number of samples: {len(df)}")
    print(f"Number of variables: {len(variables)}")
    print(f"\nExplained variance ratio:")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.4f} ({pca.explained_variance_ratio_[0]*100:.2f}%)")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.4f} ({pca.explained_variance_ratio_[1]*100:.2f}%)")
    print(f"  Total: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")
    print(f"\nPrincipal Component Loadings:")
    print(f"{'Variable':<15} {'PC1':>10} {'PC2':>10}")
    print("-" * 40)
    for i, var in enumerate(variables):
        print(f"{var:<15} {pca.components_[0, i]:>10.4f} {pca.components_[1, i]:>10.4f}")
    print("="*60)
    
    # Save if output path is provided
    if output_path:
        df_pca.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    
    return df, df_pca, pca, scaler


if __name__ == "__main__":
    # Set paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "mr_replication_dataset.02.2020.csv")
    output_path = os.path.join(project_root, "data", "mr_replication_dataset_with_PCs.csv")
    
    # Apply PCA
    original_df, pca_df, pca_model, scaler = apply_pca_to_mr_data(
        data_path, 
        output_path=output_path
    )
    
    print(f"\nOriginal dataframe shape: {original_df.shape}")
    print(f"PCA dataframe shape: {pca_df.shape}")
    print(f"\nFirst few rows with PCs:")
    print(pca_df[['NGA', 'PolID', 'Time', 'PC1', 'PC2']].head(10))

