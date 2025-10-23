"""
Interactive Geographic Visualization
====================================

Creates interactive HTML maps using Folium for web-based exploration.

Author: Xiaomeng Wu
Date: October 2025
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config_loader import CONFIG, PATHS, setup_logger

logger = setup_logger('interactive_map')

try:
    import geopandas as gpd
    import folium
    from folium import plugins
    import branca.colormap as cm
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False
    logger.warning("folium not installed. Install with: pip install folium")


def load_data():
    """Load geographic and economic data"""
    logger.info("Loading data...")
    
    # Load states
    states_file = PATHS['data_raw'] / 'us_states_census.geojson'
    if not states_file.exists():
        logger.error("State boundaries not found. Run geographic_visualization.py first.")
        return None, None
    
    states = gpd.read_file(states_file)
    
    # Load economic data
    econ_file = PATHS['tables'] / 'state_economic_indicators.csv'
    if not econ_file.exists():
        logger.error("Economic data not found. Run geographic_visualization.py first.")
        return states, None
    
    economic_data = pd.read_csv(econ_file)
    
    logger.info(f"  Loaded {len(states)} states")
    logger.info(f"  Loaded {len(economic_data)} economic indicators")
    
    return states, economic_data


def create_choropleth_map(states, data, column, title, colormap, output_name):
    """
    Create interactive choropleth map with Folium
    
    Args:
        states: GeoDataFrame with state boundaries
        data: Economic indicators DataFrame
        column: Column to visualize
        title: Map title
        colormap: Color scheme (e.g., 'YlOrRd', 'RdYlGn')
        output_name: Output HTML filename
    """
    logger.info(f"Creating interactive map: {title}")
    
    # Merge data
    states_data = states.merge(data, left_on='STUSPS', right_on='state', how='left')
    
    # Create base map centered on continental US
    m = folium.Map(
        location=[39.8283, -98.5795],  # Geographic center of continental US
        zoom_start=4,
        tiles='CartoDB positron',  # Clean, professional basemap
        prefer_canvas=True
    )
    
    # Create choropleth layer
    folium.Choropleth(
        geo_data=states_data,
        name='choropleth',
        data=data,
        columns=['state', column],
        key_on='feature.properties.STUSPS',
        fill_color=colormap,
        fill_opacity=0.7,
        line_opacity=0.5,
        legend_name=column.replace('_', ' ').title(),
        nan_fill_color='white'
    ).add_to(m)
    
    # Add state labels and tooltips
    style_function = lambda x: {
        'fillColor': 'transparent',
        'color': 'transparent',
        'weight': 0
    }
    
    highlight_function = lambda x: {
        'fillColor': '#ffff00',
        'color': 'black',
        'weight': 2,
        'fillOpacity': 0.3
    }
    
    # Create detailed tooltip
    tooltip_fields = ['NAME', column]
    tooltip_aliases = ['State:', column.replace('_', ' ').title() + ':']
    
    folium.GeoJson(
        states_data,
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            style=("background-color: white; color: #333333; font-family: arial; "
                   "font-size: 12px; padding: 10px; border-radius: 3px;")
        )
    ).add_to(m)
    
    # Add title
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 400px; height: 60px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:16px; font-weight: bold; padding: 10px;
                border-radius: 5px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <p style="margin: 0; color: #2c3e50;">{title}</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save
    output_path = PATHS['figures'] / output_name
    m.save(str(output_path))
    logger.info(f"  Saved: {output_path.name}")
    
    return m


def create_multi_indicator_map(states, data, output_name='interactive_multi_indicators.html'):
    """
    Create interactive map with multiple indicator layers
    """
    logger.info("Creating multi-indicator interactive map...")
    
    states_data = states.merge(data, left_on='STUSPS', right_on='state', how='left')
    
    # Create base map
    m = folium.Map(
        location=[39.8283, -98.5795],
        zoom_start=4,
        tiles='CartoDB positron',
        prefer_canvas=True
    )
    
    # Define indicators
    indicators = [
        ('digital_adoption', 'YlGn', 'Digital Adoption (%)'),
        ('gdp_per_capita', 'YlOrRd', 'GDP per Capita ($1000s)'),
        ('tech_employment', 'Blues', 'Tech Employment (%)'),
        ('rd_intensity', 'Purples', 'R&D Intensity (%)')
    ]
    
    # Add each indicator as a separate layer
    for column, colormap, label in indicators:
        
        # Create choropleth for this indicator
        choropleth = folium.Choropleth(
            geo_data=states_data,
            name=label,
            data=data,
            columns=['state', column],
            key_on='feature.properties.STUSPS',
            fill_color=colormap,
            fill_opacity=0.7,
            line_opacity=0.5,
            legend_name=label,
            show=False  # Don't show by default (except first one)
        )
        choropleth.add_to(m)
        
        # Show first layer by default
        if column == 'digital_adoption':
            choropleth.show = True
    
    # Add interactive tooltips
    folium.GeoJson(
        states_data,
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent'},
        highlight_function=lambda x: {'fillColor': '#ffff00', 'fillOpacity': 0.3, 'color': 'black', 'weight': 2},
        tooltip=folium.GeoJsonTooltip(
            fields=['NAME', 'digital_adoption', 'gdp_per_capita', 'tech_employment', 'rd_intensity'],
            aliases=['State:', 'Digital Adoption (%):', 'GDP per Capita ($k):', 
                    'Tech Employment (%):', 'R&D Intensity (%):'],
            style=("background-color: white; color: #333333; font-family: arial; "
                   "font-size: 12px; padding: 10px; border-radius: 3px;")
        )
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add title
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 500px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:16px; font-weight: bold; padding: 15px;
                border-radius: 5px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <p style="margin: 0; color: #2c3e50; font-size: 18px;">US Economic Indicators by State</p>
        <p style="margin: 5px 0 0 0; color: #7f8c8d; font-size: 12px; font-weight: normal;">
            Toggle layers in the control panel. Hover over states for details.
        </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save
    output_path = PATHS['figures'] / output_name
    m.save(str(output_path))
    logger.info(f"  Saved: {output_path.name}")
    
    return m


def create_firm_cluster_map(states, panel_data, output_name='interactive_firm_locations.html'):
    """
    Create interactive map with firm locations using marker clusters
    """
    logger.info("Creating interactive firm location map...")
    
    # Get unique firms
    firms = panel_data.groupby('gvkey').first().reset_index()
    
    # Create base map
    m = folium.Map(
        location=[39.8283, -98.5795],
        zoom_start=4,
        tiles='CartoDB positron',
        prefer_canvas=True
    )
    
    # Add state boundaries
    folium.GeoJson(
        states,
        style_function=lambda x: {
            'fillColor': '#f0f0f0',
            'color': 'white',
            'weight': 1,
            'fillOpacity': 0.3
        }
    ).add_to(m)
    
    # Create marker clusters for adopted and not adopted
    adopted_cluster = plugins.MarkerCluster(name='Adopted Digital Tech')
    not_adopted_cluster = plugins.MarkerCluster(name='Not Adopted')
    
    # Add firms
    for _, firm in firms.iterrows():
        
        # Create popup with firm info
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; width: 200px;">
            <b>{firm['firm_name']}</b><br>
            Industry: {firm['industry']}<br>
            Location: ({firm['hq_lat']:.2f}, {firm['hq_lon']:.2f})<br>
        """
        
        if firm['adopt_digital'] == 1:
            popup_html += f"<b style='color: green;'>✓ Adopted Digital Tech</b><br>"
            if pd.notna(firm['digital_year']):
                popup_html += f"Year: {int(firm['digital_year'])}"
            
            # Red marker for adopted
            folium.Marker(
                location=[firm['hq_lat'], firm['hq_lon']],
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(color='red', icon='star', prefix='fa'),
                tooltip=firm['firm_name']
            ).add_to(adopted_cluster)
        else:
            popup_html += "<b style='color: gray;'>Not Adopted</b>"
            
            # Blue marker for not adopted
            folium.Marker(
                location=[firm['hq_lat'], firm['hq_lon']],
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(color='blue', icon='building', prefix='fa'),
                tooltip=firm['firm_name']
            ).add_to(not_adopted_cluster)
    
    # Add clusters to map
    adopted_cluster.add_to(m)
    not_adopted_cluster.add_to(m)
    
    # Add heat map layer
    heat_data = [[firm['hq_lat'], firm['hq_lon']] for _, firm in firms.iterrows()]
    plugins.HeatMap(heat_data, name='Density Heatmap', show=False).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add title
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 450px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:16px; font-weight: bold; padding: 15px;
                border-radius: 5px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <p style="margin: 0; color: #2c3e50; font-size: 18px;">
            Digital Technology Adoption: Firm Locations
        </p>
        <p style="margin: 5px 0 0 0; color: #7f8c8d; font-size: 12px; font-weight: normal;">
            <span style="color: red;">★</span> Adopted ({(firms['adopt_digital']==1).sum()} firms) | 
            <span style="color: blue;">■</span> Not Adopted ({(firms['adopt_digital']==0).sum()} firms)
        </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add minimap
    minimap = plugins.MiniMap(toggle_display=True)
    m.add_child(minimap)
    
    # Add measure control
    plugins.MeasureControl(position='bottomleft').add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    # Save
    output_path = PATHS['figures'] / output_name
    m.save(str(output_path))
    logger.info(f"  Saved: {output_path.name}")
    
    return m


def run_interactive_maps():
    """Main workflow for interactive maps"""
    logger.info("=" * 70)
    logger.info("INTERACTIVE GEOGRAPHIC VISUALIZATION")
    logger.info("=" * 70)
    
    if not HAS_FOLIUM:
        logger.error("folium not installed!")
        logger.error("Install: pip install folium")
        return None
    
    # Load data
    states, economic_data = load_data()
    if states is None or economic_data is None:
        logger.error("Data not available. Run geographic_visualization.py first.")
        return None
    
    logger.info("\nCreating interactive HTML maps...")
    
    # Create individual choropleth maps
    create_choropleth_map(
        states, economic_data, 'digital_adoption',
        'Digital Technology Adoption Rate by State',
        'YlGn', 'interactive_digital_adoption.html'
    )
    
    create_choropleth_map(
        states, economic_data, 'gdp_per_capita',
        'GDP per Capita by State',
        'YlOrRd', 'interactive_gdp_per_capita.html'
    )
    
    # Create multi-indicator map
    create_multi_indicator_map(states, economic_data)
    
    # Try to create firm location map
    try:
        panel_path = PATHS['data_processed'] / 'firm_panel.csv'
        if panel_path.exists():
            panel = pd.read_csv(panel_path)
            create_firm_cluster_map(states, panel)
    except Exception as e:
        logger.warning(f"Could not create firm location map: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("INTERACTIVE MAPS COMPLETE")
    logger.info("=" * 70)
    logger.info("\nGenerated interactive HTML maps:")
    logger.info("  1. interactive_digital_adoption.html")
    logger.info("  2. interactive_gdp_per_capita.html")
    logger.info("  3. interactive_multi_indicators.html (multiple layers)")
    logger.info("  4. interactive_firm_locations.html (with clusters)")
    logger.info("\nOpen these files in a web browser to explore!")
    
    return True


if __name__ == '__main__':
    run_interactive_maps()
