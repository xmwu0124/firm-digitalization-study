"""
Spatial Analysis of Digital Transformation
==========================================

Analyzes geographic clustering and spatial patterns in digital adoption.

Methods:
- Moran's I: Global spatial autocorrelation
- Getis-Ord G*: Local hotspot detection
- Spatial visualization: Choropleth maps

Author: Research Team
Date: October 2025
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config_loader import CONFIG, PATHS, setup_logger

# Setup logging
logger = setup_logger('spatial_analysis')

# Check for spatial packages
try:
    from libpysal.weights import DistanceBand, KNN
    from esda.moran import Moran
    from esda.getisord import G_Local
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False
    logger.warning("Spatial analysis packages not installed. Install with: pip install libpysal esda")


def load_panel() -> pd.DataFrame:
    """Load firm panel data"""
    panel_path = PATHS['data_processed'] / 'firm_panel.csv'
    logger.info(f"Loading panel from: {panel_path}")
    
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel data not found: {panel_path}")
    
    return pd.read_csv(panel_path)


def create_spatial_weights(df: pd.DataFrame, method: str = 'knn', k: int = 5) -> object:
    """
    Create spatial weights matrix
    
    Args:
        df: DataFrame with lat/lon coordinates
        method: 'knn' or 'distance'
        k: Number of nearest neighbors (if knn)
    
    Returns:
        Spatial weights object
    """
    if not HAS_SPATIAL:
        logger.error("Spatial packages not available")
        return None
    
    # Get coordinates
    coords = df[['longitude', 'latitude']].values
    
    logger.info(f"Creating spatial weights matrix ({method})...")
    
    if method == 'knn':
        w = KNN.from_array(coords, k=k)
        logger.info(f"  K-nearest neighbors: k={k}")
    else:
        # Distance-based weights (within threshold)
        threshold = 5.0  # degrees (roughly 500km)
        w = DistanceBand.from_array(coords, threshold=threshold)
        logger.info(f"  Distance band: {threshold} degrees")
    
    w.transform = 'r'  # Row-standardized weights
    logger.info(f"  Created weights matrix: {w.n} observations, {w.s0:.0f} links")
    
    return w


def compute_morans_i(df: pd.DataFrame, variable: str, w: object) -> Dict:
    """
    Compute Moran's I statistic for spatial autocorrelation
    
    Args:
        df: DataFrame with variable
        variable: Column name to analyze
        w: Spatial weights matrix
    
    Returns:
        Dictionary with test results
    """
    if not HAS_SPATIAL or w is None:
        return {}
    
    logger.info(f"\nComputing Moran's I for: {variable}")
    
    # Remove missing values
    valid = df[variable].notna()
    y = df.loc[valid, variable].values
    
    # Compute Moran's I
    moran = Moran(y, w)
    
    results = {
        'variable': variable,
        'moran_i': moran.I,
        'expected_i': moran.EI,
        'variance': moran.VI_norm,
        'z_score': moran.z_norm,
        'p_value': moran.p_norm,
        'interpretation': 'Significant positive' if moran.p_norm < 0.05 and moran.I > 0 
                         else 'Significant negative' if moran.p_norm < 0.05 and moran.I < 0
                         else 'Not significant'
    }
    
    logger.info(f"  Moran's I: {moran.I:.4f}")
    logger.info(f"  Expected I: {moran.EI:.4f}")
    logger.info(f"  Z-score: {moran.z_norm:.4f}")
    logger.info(f"  P-value: {moran.p_norm:.4f}")
    logger.info(f"  Result: {results['interpretation']}")
    
    return results


def detect_hotspots(df: pd.DataFrame, variable: str, w: object) -> pd.DataFrame:
    """
    Detect spatial hotspots using Getis-Ord G*
    
    Args:
        df: DataFrame with variable
        variable: Column name to analyze
        w: Spatial weights matrix
    
    Returns:
        DataFrame with hotspot indicators
    """
    if not HAS_SPATIAL or w is None:
        return df
    
    logger.info(f"\nDetecting hotspots for: {variable}")
    
    # Remove missing values
    valid = df[variable].notna()
    y = df.loc[valid, variable].values
    
    # Compute G* statistic
    g_star = G_Local(y, w, star=True)
    
    # Add to dataframe
    df_result = df.copy()
    df_result['g_star'] = np.nan
    df_result['hotspot'] = 'Not significant'
    
    df_result.loc[valid, 'g_star'] = g_star.Zs
    
    # Classify hotspots (95% confidence)
    df_result.loc[valid & (g_star.Zs > 1.96), 'hotspot'] = 'Hot spot (high-high)'
    df_result.loc[valid & (g_star.Zs < -1.96), 'hotspot'] = 'Cold spot (low-low)'
    
    # Count hotspots
    n_hot = (df_result['hotspot'] == 'Hot spot (high-high)').sum()
    n_cold = (df_result['hotspot'] == 'Cold spot (low-low)').sum()
    
    logger.info(f"  Hot spots detected: {n_hot}")
    logger.info(f"  Cold spots detected: {n_cold}")
    
    return df_result


def plot_spatial_distribution(df: pd.DataFrame, variable: str, output_name: str):
    """
    Create choropleth map of variable
    
    Args:
        df: DataFrame with coordinates and variable
        variable: Column to plot
        output_name: Output filename
    """
    logger.info(f"\nCreating spatial distribution plot for: {variable}")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot with color representing variable
    scatter = ax.scatter(
        df['longitude'], 
        df['latitude'],
        c=df[variable],
        cmap='RdYlBu_r',
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    
    plt.colorbar(scatter, ax=ax, label=variable.replace('_', ' ').title())
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Geographic Distribution: {variable.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = PATHS['figures'] / output_name
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved plot to: {output_path}")
    plt.close()


def plot_hotspots(df: pd.DataFrame, output_name: str):
    """
    Create hotspot map
    
    Args:
        df: DataFrame with hotspot classifications
        output_name: Output filename
    """
    logger.info("\nCreating hotspot map...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color mapping
    colors = {
        'Hot spot (high-high)': 'red',
        'Cold spot (low-low)': 'blue',
        'Not significant': 'lightgray'
    }
    
    for category, color in colors.items():
        subset = df[df['hotspot'] == category]
        ax.scatter(
            subset['longitude'],
            subset['latitude'],
            c=color,
            label=category,
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Spatial Hotspot Analysis: Digital Adoption', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = PATHS['figures'] / output_name
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved plot to: {output_path}")
    plt.close()


def run_spatial_analysis():
    """Main spatial analysis workflow"""
    logger.info("=" * 70)
    logger.info("SPATIAL ANALYSIS - DIGITAL TRANSFORMATION")
    logger.info("=" * 70)
    
    if not HAS_SPATIAL:
        logger.error("Spatial packages not installed!")
        logger.error("Install with: pip install libpysal esda --break-system-packages")
        return None
    
    # Load data
    panel = load_panel()
    
    # Get firm-level data (aggregate to firm)
    firm_data = panel.groupby('gvkey').agg({
        'digital_year': 'first',
        'longitude': 'first',
        'latitude': 'first',
        'log_revenue': 'mean',
        'tech_industry': 'first'
    }).reset_index()
    
    # Create adoption indicator
    firm_data['adopted'] = firm_data['digital_year'].notna().astype(int)
    
    logger.info(f"\nFirm-level data: {len(firm_data)} firms")
    logger.info(f"  Adopters: {firm_data['adopted'].sum()}")
    logger.info(f"  Non-adopters: {(1 - firm_data['adopted']).sum()}")
    
    # Create spatial weights
    w = create_spatial_weights(firm_data, method='knn', k=5)
    
    if w is None:
        return None
    
    # Analyze spatial patterns
    results = {}
    
    # 1. Digital adoption
    results['adoption'] = compute_morans_i(firm_data, 'adopted', w)
    
    # 2. Revenue
    results['revenue'] = compute_morans_i(firm_data, 'log_revenue', w)
    
    # 3. Tech industry concentration
    results['tech'] = compute_morans_i(firm_data, 'tech_industry', w)
    
    # Hotspot detection
    firm_hotspots = detect_hotspots(firm_data, 'adopted', w)
    
    # Visualizations
    plot_spatial_distribution(firm_data, 'adopted', 'spatial_adoption.png')
    plot_spatial_distribution(firm_data, 'log_revenue', 'spatial_revenue.png')
    plot_hotspots(firm_hotspots, 'spatial_hotspots.png')
    
    # Save results
    results_df = pd.DataFrame([
        {
            'Variable': r['variable'],
            'Morans_I': r['moran_i'],
            'Expected_I': r['expected_i'],
            'Z_Score': r['z_score'],
            'P_Value': r['p_value'],
            'Interpretation': r['interpretation']
        }
        for r in results.values()
    ])
    
    output_path = PATHS['tables'] / 'spatial_analysis_results.csv'
    results_df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Saved results to: {output_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("SPATIAL ANALYSIS COMPLETE")
    logger.info("=" * 70)
    
    return results


if __name__ == '__main__':
    results = run_spatial_analysis()
