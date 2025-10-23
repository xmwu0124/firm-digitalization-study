"""
Spatial Analysis with Enhanced Visualizations
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config_loader import CONFIG, PATHS, setup_logger

logger = setup_logger('spatial_analysis')


def compute_distance_matrix(firm_data: pd.DataFrame) -> np.ndarray:
    """Compute pairwise distances between firms"""
    coords = firm_data[['hq_lat', 'hq_lon']].values
    n = len(coords)
    
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            # Euclidean distance (rough approximation)
            dist = np.sqrt((coords[i, 0] - coords[j, 0])**2 + 
                          (coords[i, 1] - coords[j, 1])**2)
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances


def analyze_spatial_clustering(firm_data: pd.DataFrame):
    """Analyze whether adopters are spatially clustered"""
    logger.info("\nAnalyzing spatial clustering...")
    
    # Compute distance matrix
    distances = compute_distance_matrix(firm_data)
    
    # For each adopter, find average distance to other adopters vs non-adopters
    adopters_idx = firm_data[firm_data['adopted'] == 1].index.values
    non_adopters_idx = firm_data[firm_data['adopted'] == 0].index.values
    
    if len(adopters_idx) > 1 and len(non_adopters_idx) > 0:
        # Average distance between adopters
        adopter_distances = []
        for i in adopters_idx:
            for j in adopters_idx:
                if i < j:
                    adopter_distances.append(distances[i, j])
        
        # Average distance from adopters to non-adopters
        mixed_distances = []
        for i in adopters_idx:
            for j in non_adopters_idx:
                mixed_distances.append(distances[i, j])
        
        avg_adopter_dist = np.mean(adopter_distances) if adopter_distances else 0
        avg_mixed_dist = np.mean(mixed_distances) if mixed_distances else 0
        
        clustering_ratio = avg_adopter_dist / avg_mixed_dist if avg_mixed_dist > 0 else 0
        
        logger.info(f"  Avg distance between adopters: {avg_adopter_dist:.3f}")
        logger.info(f"  Avg distance adopters-to-non: {avg_mixed_dist:.3f}")
        logger.info(f"  Clustering ratio: {clustering_ratio:.3f}")
        
        if clustering_ratio < 0.8:
            logger.info("  → Adopters are spatially CLUSTERED (closer to each other)")
        elif clustering_ratio > 1.2:
            logger.info("  → Adopters are spatially DISPERSED (farther apart)")
        else:
            logger.info("  → Adopters show NO significant spatial pattern")
        
        return {
            'avg_adopter_distance': avg_adopter_dist,
            'avg_mixed_distance': avg_mixed_dist,
            'clustering_ratio': clustering_ratio
        }
    
    return None


def create_enhanced_maps(firm_data: pd.DataFrame):
    """Create multiple spatial visualizations"""
    logger.info("\nCreating enhanced spatial maps...")
    
    # Create 2x2 subplot
    fig = plt.figure(figsize=(18, 14))
    
    # Plot 1: Basic adoption map with larger points
    ax1 = plt.subplot(2, 2, 1)
    colors = firm_data['adopted'].map({1: '#d62728', 0: '#1f77b4'})
    sizes = firm_data['log_revenue'] * 20  # Size proportional to revenue
    
    ax1.scatter(firm_data['hq_lon'], firm_data['hq_lat'], 
               c=colors, s=sizes, alpha=0.6, 
               edgecolors='black', linewidth=0.8)
    ax1.set_xlabel('Longitude', fontsize=11)
    ax1.set_ylabel('Latitude', fontsize=11)
    ax1.set_title('Digital Adoption (size = revenue)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor='#d62728', label=f'Adopted ({firm_data["adopted"].sum()})'),
        Patch(facecolor='#1f77b4', label=f'Not Adopted ({(~firm_data["adopted"].astype(bool)).sum()})')
    ]
    ax1.legend(handles=legend, loc='best', fontsize=10)
    
    # Plot 2: Density heatmap of adopters
    ax2 = plt.subplot(2, 2, 2)
    adopters = firm_data[firm_data['adopted'] == 1]
    
    if len(adopters) > 5:
        # Create 2D histogram
        h = ax2.hexbin(adopters['hq_lon'], adopters['hq_lat'], 
                      gridsize=15, cmap='Reds', alpha=0.7)
        plt.colorbar(h, ax=ax2, label='Adopter Count')
    
    # Overlay all firms
    ax2.scatter(firm_data['hq_lon'], firm_data['hq_lat'], 
               c='lightgray', s=30, alpha=0.4, edgecolors='black', linewidth=0.3)
    ax2.scatter(adopters['hq_lon'], adopters['hq_lat'], 
               c='red', s=80, alpha=0.8, edgecolors='black', linewidth=0.5, 
               marker='*', label='Adopters')
    
    ax2.set_xlabel('Longitude', fontsize=11)
    ax2.set_ylabel('Latitude', fontsize=11)
    ax2.set_title('Adopter Density Heatmap', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=10)
    
    # Plot 3: Revenue by location
    ax3 = plt.subplot(2, 2, 3)
    scatter = ax3.scatter(firm_data['hq_lon'], firm_data['hq_lat'], 
                         c=firm_data['log_revenue'], cmap='viridis', 
                         s=150, alpha=0.7, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, ax=ax3, label='Log Revenue')
    ax3.set_xlabel('Longitude', fontsize=11)
    ax3.set_ylabel('Latitude', fontsize=11)
    ax3.set_title('Firm Size Distribution', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 4: Industry distribution
    ax4 = plt.subplot(2, 2, 4)
    
    # Get unique industries and assign colors
    industries = firm_data['industry'].unique()
    industry_colors = plt.cm.Set3(np.linspace(0, 1, len(industries)))
    color_map = dict(zip(industries, industry_colors))
    
    for industry in industries:
        subset = firm_data[firm_data['industry'] == industry]
        ax4.scatter(subset['hq_lon'], subset['hq_lat'], 
                   c=[color_map[industry]], s=100, alpha=0.6,
                   edgecolors='black', linewidth=0.5,
                   label=f'{industry} ({len(subset)})')
    
    ax4.set_xlabel('Longitude', fontsize=11)
    ax4.set_ylabel('Latitude', fontsize=11)
    ax4.set_title('Industry Distribution', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(loc='best', fontsize=9, ncol=2)
    
    plt.tight_layout()
    
    output_path = PATHS['figures'] / 'spatial_analysis_comprehensive.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved comprehensive map to: {output_path}")
    plt.close()


def run_spatial_analysis():
    """Main analysis"""
    logger.info("=" * 70)
    logger.info("SPATIAL ANALYSIS - DIGITAL TRANSFORMATION")
    logger.info("=" * 70)
    
    # Load data
    panel_path = PATHS['data_processed'] / 'firm_panel.csv'
    panel = pd.read_csv(panel_path)
    
    logger.info(f"\nLoaded panel: {len(panel)} observations")
    
    # Aggregate to firm level
    firm_data = panel.groupby('gvkey').agg({
        'digital_year': 'first',
        'hq_lat': 'first',
        'hq_lon': 'first',
        'log_revenue': 'mean',
        'firm_name': 'first',
        'industry': 'first'
    }).reset_index()
    
    firm_data['adopted'] = firm_data['digital_year'].notna().astype(int)
    
    logger.info(f"\nFirm-level data: {len(firm_data)} firms")
    logger.info(f"  Adopters: {firm_data['adopted'].sum()} ({100*firm_data['adopted'].mean():.1f}%)")
    logger.info(f"  Non-adopters: {(~firm_data['adopted'].astype(bool)).sum()}")
    
    # Geographic ranges
    logger.info("\nGeographic coverage:")
    logger.info(f"  Latitude: [{firm_data['hq_lat'].min():.2f}, {firm_data['hq_lat'].max():.2f}]")
    logger.info(f"  Longitude: [{firm_data['hq_lon'].min():.2f}, {firm_data['hq_lon'].max():.2f}]")
    
    # Spatial clustering analysis
    clustering_stats = analyze_spatial_clustering(firm_data)
    
    # Create visualizations
    create_enhanced_maps(firm_data)
    
    # Compute summary statistics
    cv_lat = firm_data['hq_lat'].std() / abs(firm_data['hq_lat'].mean())
    cv_lon = firm_data['hq_lon'].std() / abs(firm_data['hq_lon'].mean())
    
    adopters = firm_data[firm_data['adopted'] == 1]
    non_adopters = firm_data[firm_data['adopted'] == 0]
    
    results = {
        'total_firms': len(firm_data),
        'adopters': firm_data['adopted'].sum(),
        'adoption_rate': firm_data['adopted'].mean(),
        'geographic_spread_lat': firm_data['hq_lat'].std(),
        'geographic_spread_lon': firm_data['hq_lon'].std(),
        'cv_latitude': cv_lat,
        'cv_longitude': cv_lon,
        'adopter_lat_mean': adopters['hq_lat'].mean(),
        'adopter_lon_mean': adopters['hq_lon'].mean(),
        'non_adopter_lat_mean': non_adopters['hq_lat'].mean(),
        'non_adopter_lon_mean': non_adopters['hq_lon'].mean()
    }
    
    if clustering_stats:
        results.update(clustering_stats)
    
    # Save results
    results_df = pd.DataFrame([results])
    output_path = PATHS['tables'] / 'spatial_analysis_results.csv'
    results_df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Saved results to: {output_path}")
    
    # Industry breakdown
    logger.info("\nIndustry distribution:")
    for ind in firm_data['industry'].unique():
        subset = firm_data[firm_data['industry'] == ind]
        adopted_pct = subset['adopted'].mean() * 100
        logger.info(f"  {ind}: {len(subset)} firms, {adopted_pct:.1f}% adoption")
    
    logger.info("\n" + "=" * 70)
    logger.info("SPATIAL ANALYSIS COMPLETE")
    logger.info("=" * 70)
    
    return results


if __name__ == '__main__':
    run_spatial_analysis()
