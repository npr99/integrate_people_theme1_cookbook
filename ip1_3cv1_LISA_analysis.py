"""
LISA Analysis Functions for Spatial Autocorrelation
====================================================

Functions for conducting Local Indicators of Spatial Association (LISA) analysis
on grid-based spatial data.

Author: Nathanael Rosenheim
Project: Southeast Texas Urban Integrated Field Lab
Funding: DOE
Version: 2026-02-04
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from libpysal.weights import lat2W, KNN
from esda.moran import Moran, Moran_Local


def create_spatial_weights(df, row_col='ROW', col_col='COL'):
    """
    Create spatial weights matrix for a grid dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with grid data
    row_col : str
        Column name for row index (default: 'ROW')
    col_col : str
        Column name for column index (default: 'COL')
    
    Returns:
    --------
    w : libpysal.weights.W
        Spatial weights matrix
    df_sorted : pandas.DataFrame
        Sorted dataframe by ROW and COL
    """
    # Get grid dimensions
    n_rows = df[row_col].max() - df[row_col].min() + 1
    n_cols = df[col_col].max() - df[col_col].min() + 1
    
    print(f"Grid dimensions: {n_rows} rows x {n_cols} columns")
    
    # Sort dataframe by ROW and COL
    df_sorted = df.sort_values([row_col, col_col]).reset_index(drop=True)
    
    # Create spatial weights using queen contiguity
    w = lat2W(nrows=int(n_rows), ncols=int(n_cols), rook=False)
    
    print(f"Number of observations: {w.n}")
    print(f"Average number of neighbors: {w.mean_neighbors:.2f}")
    
    return w, df_sorted


def create_spatial_weights_knn(df, x_col='x', y_col='y', k=8):
    """
    Create spatial weights matrix using K-nearest neighbors for irregular/filtered data.
    Use this when you've filtered out cells and no longer have a regular grid.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with spatial data
    x_col : str
        Column name for x coordinates (default: 'x')
    y_col : str
        Column name for y coordinates (default: 'y')
    k : int
        Number of nearest neighbors (default: 8 for queen contiguity equivalent)
    
    Returns:
    --------
    w : libpysal.weights.W
        Spatial weights matrix based on KNN
    df_reset : pandas.DataFrame
        Dataframe with reset index
    """
    # Reset index to ensure proper alignment
    df_reset = df.reset_index(drop=True)
    
    # Create coordinate array
    coords = df_reset[[x_col, y_col]].values
    
    # Create KNN weights
    w = KNN.from_array(coords, k=k)
    
    print(f"Number of observations: {w.n}")
    print(f"K-nearest neighbors: {k}")
    print(f"Average number of neighbors: {w.mean_neighbors:.2f}")
    
    return w, df_reset


def calculate_global_morans_i(df_sorted, var_name, w):
    """
    Calculate Global Moran's I statistic.
    
    Parameters:
    -----------
    df_sorted : pandas.DataFrame
        Sorted dataframe
    var_name : str
        Name of variable to analyze
    w : libpysal.weights.W
        Spatial weights matrix
    
    Returns:
    --------
    moran_global : esda.moran.Moran
        Global Moran's I object
    """
    moran_global = Moran(df_sorted[var_name].values, w)
    
    print(f"Global Moran's I: {moran_global.I:.4f}")
    print(f"Expected I: {moran_global.EI:.4f}")
    print(f"p-value: {moran_global.p_sim:.4f}")
    print(f"z-score: {moran_global.z_sim:.4f}")
    
    if moran_global.p_sim < 0.05:
        if moran_global.I > 0:
            print("\nResult: Significant positive spatial autocorrelation (clustering)")
        else:
            print("\nResult: Significant negative spatial autocorrelation (dispersion)")
    else:
        print("\nResult: No significant spatial autocorrelation")
    
    return moran_global


def plot_morans_i_scatterplot(df_sorted, var_name, w, moran_global, figsize=(10, 8)):
    """
    Create Moran's I scatterplot.
    
    Parameters:
    -----------
    df_sorted : pandas.DataFrame
        Sorted dataframe
    var_name : str
        Name of variable to analyze
    w : libpysal.weights.W
        Spatial weights matrix
    moran_global : esda.moran.Moran
        Global Moran's I object
    figsize : tuple
        Figure size (default: (10, 8))
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate standardized values
    var_standardized = (df_sorted[var_name] - df_sorted[var_name].mean()) / df_sorted[var_name].std()
    
    # Calculate spatial lag
    var_lag = np.array([np.mean(df_sorted[var_name].iloc[w.neighbors[i]]) for i in range(len(df_sorted))])
    var_lag_standardized = (var_lag - var_lag.mean()) / var_lag.std()
    
    # Create scatterplot
    ax.scatter(var_standardized, var_lag_standardized, alpha=0.5, s=20)
    
    # Add regression line
    z = np.polyfit(var_standardized, var_lag_standardized, 1)
    p = np.poly1d(z)
    ax.plot(var_standardized, p(var_standardized), "r-", linewidth=2, label=f"Slope = {z[0]:.3f}")
    
    # Add quadrant lines
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    
    # Add labels and title
    ax.set_xlabel(f'Standardized {var_name}', fontsize=12)
    ax.set_ylabel('Spatial Lag (Standardized)', fontsize=12)
    ax.set_title(f"Moran's I Scatterplot\nI = {moran_global.I:.4f}, p-value = {moran_global.p_sim:.4f}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def calculate_local_morans_i(df_sorted, var_name, w, significance_level=0.05):
    """
    Calculate Local Moran's I (LISA) and add results to dataframe.
    
    Parameters:
    -----------
    df_sorted : pandas.DataFrame
        Sorted dataframe
    var_name : str
        Name of variable to analyze
    w : libpysal.weights.W
        Spatial weights matrix
    significance_level : float
        P-value threshold for significance (default: 0.05)
    
    Returns:
    --------
    df_sorted : pandas.DataFrame
        Dataframe with LISA results added
    moran_local : esda.moran.Moran_Local
        Local Moran's I object
    """
    # Calculate Local Moran's I
    moran_local = Moran_Local(df_sorted[var_name].values, w)
    
    # Add LISA results to dataframe
    df_sorted['moran_local_i'] = moran_local.Is
    df_sorted['moran_p_value'] = moran_local.p_sim
    df_sorted['moran_quadrant'] = moran_local.q
    
    # Create significance mask
    df_sorted['significant'] = df_sorted['moran_p_value'] < significance_level
    
    # Label the quadrants
    quadrant_labels = {1: 'HH (Hot spot)', 2: 'LH', 3: 'LL (Cold spot)', 4: 'HL'}
    df_sorted['cluster_label'] = df_sorted['moran_quadrant'].map(quadrant_labels)
    df_sorted.loc[~df_sorted['significant'], 'cluster_label'] = 'Not significant'
    
    # Summary of clusters
    print("LISA Cluster Summary:")
    print(df_sorted['cluster_label'].value_counts())
    print(f"\nTotal significant clusters: {df_sorted['significant'].sum()} ({100*df_sorted['significant'].mean():.1f}%)")
    
    return df_sorted, moran_local


def plot_lisa_cluster_map(df_sorted, var_name, x_col='x', y_col='y', figsize=(12, 10)):
    """
    Create LISA cluster map showing hot spots and cold spots.
    
    Parameters:
    -----------
    df_sorted : pandas.DataFrame
        Sorted dataframe with LISA results
    var_name : str
        Name of variable analyzed
    x_col : str
        Column name for x coordinates (default: 'x')
    y_col : str
        Column name for y coordinates (default: 'y')
    figsize : tuple
        Figure size (default: (12, 10))
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors for each cluster type
    colors = {
        'HH (Hot spot)': 'red',
        'LL (Cold spot)': 'blue',
        'LH': 'lightblue',
        'HL': 'pink',
        'Not significant': 'lightgray'
    }
    
    # Plot each cluster type
    for cluster_type, color in colors.items():
        mask = df_sorted['cluster_label'] == cluster_type
        subset = df_sorted[mask]
        ax.scatter(subset[x_col], subset[y_col], 
                  c=color, label=f"{cluster_type} (n={len(subset)})", 
                  s=15, alpha=0.7, edgecolors='none')
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(f'LISA Cluster Map - {var_name}\n(Hot spots and Cold spots)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
