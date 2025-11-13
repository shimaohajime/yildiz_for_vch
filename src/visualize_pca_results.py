"""
Script to visualize the PCA results from the MR replication dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_pca_results(data_path, output_dir=None):
    """
    Create visualizations of the PCA results.
    """
    # Read the data
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Set up output directory
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, "data")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Scatter plot: PC1 vs PC2
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(df['PC1'], df['PC2'], alpha=0.5, s=20, c=df['Time'], cmap='viridis')
    ax1.set_xlabel('PC1 (82.47% variance)', fontsize=10)
    ax1.set_ylabel('PC2 (5.09% variance)', fontsize=10)
    ax1.set_title('PC1 vs PC2 (colored by Time)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Time')
    
    # 2. Distribution of PC1
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(df['PC1'], bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('PC1', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Distribution of PC1', fontsize=12)
    ax2.axvline(df['PC1'].mean(), color='red', linestyle='--', label=f'Mean: {df["PC1"].mean():.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution of PC2
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(df['PC2'], bins=50, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('PC2', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Distribution of PC2', fontsize=12)
    ax3.axvline(df['PC2'].mean(), color='red', linestyle='--', label=f'Mean: {df["PC2"].mean():.3f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. PC1 over time (sample of NGAs)
    ax4 = plt.subplot(2, 3, 4)
    unique_ngas = df['NGA'].unique()[:10]  # First 10 NGAs
    for nga in unique_ngas:
        nga_data = df[df['NGA'] == nga].sort_values('Time')
        ax4.plot(nga_data['Time'], nga_data['PC1'], marker='o', markersize=3, label=nga, alpha=0.7)
    ax4.set_xlabel('Time', fontsize=10)
    ax4.set_ylabel('PC1', fontsize=10)
    ax4.set_title('PC1 Over Time (Sample NGAs)', fontsize=12)
    ax4.legend(fontsize=6, loc='best')
    ax4.grid(True, alpha=0.3)
    
    # 5. PC2 over time (sample of NGAs)
    ax5 = plt.subplot(2, 3, 5)
    for nga in unique_ngas:
        nga_data = df[df['NGA'] == nga].sort_values('Time')
        ax5.plot(nga_data['Time'], nga_data['PC2'], marker='o', markersize=3, label=nga, alpha=0.7)
    ax5.set_xlabel('Time', fontsize=10)
    ax5.set_ylabel('PC2', fontsize=10)
    ax5.set_title('PC2 Over Time (Sample NGAs)', fontsize=12)
    ax5.legend(fontsize=6, loc='best')
    ax5.grid(True, alpha=0.3)
    
    # 6. Correlation heatmap: Original variables vs PCs
    ax6 = plt.subplot(2, 3, 6)
    variables = ['PolPop', 'PolTerr', 'CapPop', 'levels', 'government', 
                 'infrastr', 'writing', 'texts', 'money']
    corr_data = []
    for var in variables:
        corr_pc1 = df[var].corr(df['PC1'])
        corr_pc2 = df[var].corr(df['PC2'])
        corr_data.append([corr_pc1, corr_pc2])
    
    corr_df = pd.DataFrame(corr_data, index=variables, columns=['PC1', 'PC2'])
    im = ax6.imshow(corr_df.values, aspect='auto', cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax6.set_xticks(range(len(corr_df.columns)))
    ax6.set_xticklabels(corr_df.columns)
    ax6.set_yticks(range(len(corr_df.index)))
    ax6.set_yticklabels(corr_df.index, fontsize=8)
    ax6.set_title('Correlation: Variables vs PCs', fontsize=12)
    plt.colorbar(im, ax=ax6, label='Correlation')
    
    # Add correlation values as text
    for i in range(len(variables)):
        for j in range(2):
            text = ax6.text(j, i, f'{corr_df.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'pca_visualizations.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualizations saved to: {output_path}")
    
    # Create a second figure for PC trajectories
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # PC1 trajectories for all NGAs
    for nga in df['NGA'].unique():
        nga_data = df[df['NGA'] == nga].sort_values('Time')
        if len(nga_data) > 1:
            ax1.plot(nga_data['Time'], nga_data['PC1'], alpha=0.5, linewidth=1)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('PC1', fontsize=12)
    ax1.set_title('PC1 Trajectories for All NGAs', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # PC2 trajectories for all NGAs
    for nga in df['NGA'].unique():
        nga_data = df[df['NGA'] == nga].sort_values('Time')
        if len(nga_data) > 1:
            ax2.plot(nga_data['Time'], nga_data['PC2'], alpha=0.5, linewidth=1)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('PC2', fontsize=12)
    ax2.set_title('PC2 Trajectories for All NGAs', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path2 = os.path.join(output_dir, 'pca_trajectories.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Trajectory plots saved to: {output_path2}")
    
    plt.close('all')
    print("\nVisualization complete!")


if __name__ == "__main__":
    # Set paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "mr_replication_dataset_with_PCs.csv")
    
    # Create visualizations
    visualize_pca_results(data_path)


