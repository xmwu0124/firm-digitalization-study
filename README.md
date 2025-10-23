# Firm Digitalization Study

Applied Econometrics Code Sample for Predoctoral Research Position Applications

## Overview

This repository demonstrates end-to-end empirical research capabilities, combining causal inference, structural estimation, and geospatial analysis. The project analyzes corporate digital technology adoption using synthetic panel data.

## Key Results

### Difference-in-Differences Estimates

| Event Time | Coefficient | Std. Error | P-value |
|------------|-------------|------------|---------|
| -3 | -0.052 | 0.021 | 0.012 |
| -2 | 0.015 | 0.023 | 0.509 |
| 0 | 0.086 | 0.023 | 0.000 |
| +1 | 0.140 | 0.024 | 0.000 |
| +2 | 0.219 | 0.031 | 0.000 |
| +3 | 0.306 | 0.045 | 0.000 |
| +4 | 0.382 | 0.057 | 0.000 |

![Event Study Results](output/figures/event_study_log_revenue.png)

### Geographic Distribution of Economic Indicators

#### Digital Technology Adoption Rates
![Digital Adoption Map](output/figures/map_digital_adoption.png)

#### GDP per Capita by State
![GDP per Capita Map](output/figures/map_gdp_per_capita.png)

#### Technology Employment Share
![Tech Employment Map](output/figures/map_tech_employment.png)

#### R&D Intensity
![R&D Intensity Map](output/figures/map_rd_intensity.png)

#### Multi-Panel Economic Indicators
![Economic Indicators Panel](output/figures/economic_indicators_panel.png)

#### Firm Headquarters and Adoption Status
![Firm Locations](output/figures/map_firm_locations.png)

### Interactive Maps

Explore the data through interactive web-based visualizations:

- [Digital Adoption Map (Interactive)](https://github.com/xmwu0124/firm-digitalization-study/blob/main/output/figures/interactive_digital_adoption.html)
- [GDP per Capita (Interactive)](https://github.com/xmwu0124/firm-digitalization-study/blob/main/output/figures/interactive_gdp_per_capita.html)
- [Multiple Indicators (Interactive)](https://github.com/xmwu0124/firm-digitalization-study/blob/main/output/figures/interactive_multi_indicators.html)
- [Firm Locations with Clustering (Interactive)](https://github.com/xmwu0124/firm-digitalization-study/blob/main/output/figures/interactive_firm_locations.html)
- [Maps Index Page](https://github.com/xmwu0124/firm-digitalization-study/blob/main/output/figures/maps_index.html)

## Methodology

### Causal Inference: Difference-in-Differences
- Event study specification with two-way fixed effects
- Dynamic treatment effect estimation
- Pre-trend testing for identification
- Cluster-robust standard errors

### Structural Estimation: Dynamic Discrete Choice
- Industrial Organization framework
- Bellman equation solution via value function iteration
- Maximum likelihood parameter estimation
- Counterfactual policy simulations

### Geospatial Analysis
- Real US Census Bureau state boundaries (2023)
- Choropleth maps of economic indicators
- Interactive HTML maps with Folium
- Firm location clustering and spatial patterns

## Technical Stack

- **Python 3.9+**: pandas, numpy, scipy
- **Econometrics**: pyfixest, statsmodels
- **Optimization**: JAX (JIT compilation)
- **Geospatial**: geopandas, shapely, folium
- **Visualization**: matplotlib, seaborn

## Installation
```bash
git clone https://github.com/xmwu0124/firm-digitalization-study.git
cd firm-digitalization-study
pip install -r requirements.txt
```

## Usage
```bash
# Complete pipeline
python src/run_all.py

# Individual components
python src/02_data_generation/generate_panel.py
python src/03_analysis/did_analysis.py
python src/03_analysis/geographic_visualization.py
python src/03_analysis/interactive_map.py
```

## Author

**Xiaomeng Wu**  
GitHub: [@xmwu0124](https://github.com/xmwu0124)

## License

MIT License
