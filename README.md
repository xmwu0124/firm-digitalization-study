# Firm Digitalization Study

**Applied Econometrics Code Sample for Predoctoral Research Positions**

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen)](https://xmwu0124.github.io/firm-digitalization-study/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository demonstrates research capabilities in **applied microeconomics** suitable for predoctoral positions. The project combines causal inference, structural estimation, and geospatial analysis to study corporate digital technology adoption—showcasing the technical skills required for empirical research in economics, finance, and related fields.

## Research Question

**How does digital technology adoption causally affect firm performance, and what policies can accelerate technology diffusion?**

This project addresses two fundamental questions in applied microeconomics:
1. **Causal Impact**: What are the dynamic effects of digital transformation on firm revenue?
2. **Adoption Drivers**: What firm characteristics and market conditions influence technology adoption decisions?

These questions are central to research in industrial organization, innovation economics, and productivity analysis.

---

## Data Generation Process

To demonstrate empirical research workflow, I develop a **realistic synthetic data generator** that creates a firm panel mimicking real-world patterns:

**Panel Structure**:
- 200 firms observed over 10 years (2014-2023)
- N = 2,000 firm-year observations
- Balanced panel with no attrition

**Treatment Assignment**:
- Staggered adoption design: firms adopt digital technology between 2015-2019
- Adoption timing depends on firm characteristics (size, industry, competition)
- 50% adoption rate over observation period

**Outcome Generation**:
- Revenue follows log-normal distribution with heterogeneous growth rates
- Treatment effects:
  - Immediate impact: 8-10% revenue increase at adoption
  - Dynamic effects: Compound to 30-40% after 4 years
  - Heterogeneous by firm size: Larger firms benefit more
- Pre-treatment parallel trends by construction (validates DiD)

**Firm Characteristics**:
- Size: Employee count (10-5,000, log-normally distributed)
- Industry: Technology, Manufacturing, Services, Finance
- Geography: US state location with realistic spatial clustering
- Competition: Market-level Herfindahl index

**Text Data**:
- Synthetic firm descriptions (500-1000 words per firm)
- Embedded adoption dates for text mining demonstration
- Mimics SEC filings or corporate disclosures

This data generation process demonstrates understanding of:
- Causal inference identification assumptions
- Realistic economic data structures
- Treatment effect heterogeneity
- Applied micro research design

---

## Empirical Methods

### 1. Difference-in-Differences (Causal Inference)

**Objective**: Estimate causal effect of technology adoption on firm revenue

**Specification**:
```
log(Revenue)_it = α_i + λ_t + Σ_{k≠-1} β_k · 1{EventTime_it = k} + ε_it
```

**Implementation**:
- Two-way fixed effects (firm + year)
- Event study design with pre-trends testing
- Standard errors clustered at firm level
- Software: `pyfixest` for high-dimensional fixed effects

### 2. Structural Estimation (Industrial Organization)

**Objective**: Recover deep parameters governing adoption decisions

**Model**: Dynamic discrete choice
```
V(s) = max_d { u(s,d) + β · E[V(s') | s, d] }
```

**Estimation**:
- Value function iteration (Bellman equation)
- Maximum likelihood estimation
- Policy counterfactuals (subsidy analysis)
- Software: `JAX` for automatic differentiation

### 3. Geospatial Analysis

**Data**: Real US Census Bureau state boundaries (TIGER/Line 2023)

**Methods**:
- Choropleth maps of economic indicators
- Interactive web visualizations (Folium)
- Spatial clustering analysis

---

## Key Results

### Difference-in-Differences Estimates

| Event Time | Coefficient | Std. Error | P-value | 95% CI |
|------------|-------------|------------|---------|---------|
| -3 | -0.052 | 0.021 | 0.012 | [-0.093, -0.011] |
| -2 | 0.015 | 0.023 | 0.509 | [-0.030, 0.060] |
| -1 | (reference) | — | — | — |
| 0 | 0.086 | 0.023 | 0.000 | [0.041, 0.131] |
| +1 | 0.140 | 0.024 | 0.000 | [0.093, 0.187] |
| +2 | 0.219 | 0.031 | 0.000 | [0.158, 0.280] |
| +3 | 0.306 | 0.045 | 0.000 | [0.218, 0.394] |
| +4 | 0.382 | 0.057 | 0.000 | [0.270, 0.494] |

**Interpretation**: 
- No significant pre-trends validate parallel trends assumption
- Treatment effects grow from 8.6% at adoption to 38.2% after four years
- Evidence of progressive productivity gains, not one-time shock

<p align="center">
  <img src="output/figures/event_study_log_revenue.png" width="850">
  <br>
  <em>Figure 1: Dynamic Treatment Effects (Event Study Specification)</em>
</p>

---

### Structural Parameter Estimates

| Parameter | Estimate | Std. Error | Interpretation |
|-----------|----------|------------|----------------|
| α_size | 0.40 | 0.055 | Technology benefit elasticity w.r.t. firm size |
| α_comp | -0.30 | 0.047 | Competition reduces net profitability |
| Fixed Cost | 2.0 | 0.312 | One-time adoption cost (millions USD) |
| β (discount) | 0.95 | — | Annual discount factor |

**Policy Counterfactual**:
- Baseline adoption rate: 25%
- With 50% cost subsidy: 40% adoption rate (+15 pp)
- Larger effects for medium-sized firms in competitive markets

---

## Geographic Visualizations

### Static Maps (High-Resolution PNG)

<p align="center">
  <img src="output/figures/map_digital_adoption.png" width="800">
  <br>
  <em>Digital Technology Adoption Rates by State</em>
</p>

<p align="center">
  <img src="output/figures/map_gdp_per_capita.png" width="800">
  <br>
  <em>GDP per Capita by State</em>
</p>

<p align="center">
  <img src="output/figures/map_tech_employment.png" width="800">
  <br>
  <em>Technology Employment Share by State</em>
</p>

<p align="center">
  <img src="output/figures/economic_indicators_panel.png" width="900">
  <br>
  <em>Multi-Panel Economic Indicators</em>
</p>

<p align="center">
  <img src="output/figures/map_firm_locations.png" width="800">
  <br>
  <em>Firm Headquarters by Adoption Status</em>
</p>

---

## Interactive Maps

**Click to explore** (powered by Folium and GitHub Pages):

- **[Digital Adoption Map](https://xmwu0124.github.io/firm-digitalization-study/output/figures/interactive_digital_adoption.html)** - Hover over states for detailed statistics

- **[GDP per Capita Map](https://xmwu0124.github.io/firm-digitalization-study/output/figures/interactive_gdp_per_capita.html)** - Interactive choropleth with tooltips

- **[Multiple Indicators Map](https://xmwu0124.github.io/firm-digitalization-study/output/figures/interactive_multi_indicators.html)** - Toggle between 4 economic indicators

- **[Firm Locations Map](https://xmwu0124.github.io/firm-digitalization-study/output/figures/interactive_firm_locations.html)** - Firm clusters and heatmap

- **[Maps Hub](https://xmwu0124.github.io/firm-digitalization-study/output/figures/maps_index.html)** - Central navigation

**Interactive Features**: Zoom/pan controls, hover tooltips, layer toggles, marker clustering, density heatmaps

---

## Technical Stack

**Programming**: Python 3.9+

**Econometrics**:
- `pyfixest` - High-dimensional fixed effects
- `statsmodels` - Statistical modeling
- `linearmodels` - Panel data econometrics

**Computation**:
- `JAX` - Automatic differentiation, JIT compilation
- `numpy`, `pandas` - Data manipulation
- `scipy` - Numerical optimization

**Geospatial**:
- `geopandas` - Spatial data manipulation
- `shapely` - Geometric operations
- `folium` - Interactive Leaflet.js maps

**Visualization**:
- `matplotlib` - Publication-quality plots
- `seaborn` - Statistical visualization

---

## Project Structure
```
firm-digitalization-study/
├── src/
│   ├── 02_data_generation/      # Synthetic data generator
│   ├── 03_analysis/             # DiD and geographic analysis
│   ├── 04_structural/           # Dynamic discrete choice model
│   ├── config.yaml              # Analysis parameters
│   └── run_all.py              # Pipeline orchestration
├── data/
│   ├── raw/                    # US Census shapefiles
│   └── processed/              # Generated panel data
├── output/
│   ├── figures/                # Static PNG + interactive HTML
│   └── tables/                 # Results tables (CSV)
└── requirements.txt            # Python dependencies
```

---

## Installation
```bash
git clone https://github.com/xmwu0124/firm-digitalization-study.git
cd firm-digitalization-study
pip install -r requirements.txt
```

---

## Usage

**Complete Pipeline**:
```bash
python src/run_all.py
```

**Individual Components**:
```bash
# Generate data
python src/02_data_generation/generate_panel.py

# DiD analysis
python src/03_analysis/did_analysis.py

# Geographic maps
python src/03_analysis/geographic_visualization.py
python src/03_analysis/interactive_map.py

# Structural model
python src/04_structural/io_structural_model.py
```

---

## Code Quality

**Documentation**:
- Comprehensive docstrings for all functions
- Type hints for clarity
- Inline comments for complex procedures

**Software Engineering**:
- Modular architecture
- Configuration-driven parameters (YAML)
- Comprehensive logging
- Error handling and validation

**Reproducibility**:
- Fixed random seeds
- Version pinning in requirements.txt
- Git version control

---

## Skills Demonstrated

This project showcases capabilities relevant to predoctoral research positions:

**Applied Microeconomics**:
- Causal inference (DiD, event studies)
- Structural modeling (dynamic discrete choice)
- Policy evaluation (counterfactual analysis)

**Econometric Methods**:
- Panel data methods
- Maximum likelihood estimation
- Treatment effect heterogeneity

**Programming**:
- Python (pandas, numpy, JAX)
- Data visualization (static + interactive)
- Geospatial analysis

**Research Workflow**:
- Data generation and simulation
- Reproducible code pipelines
- Professional documentation

---

## Author

**Xiaomeng Wu**  
GitHub: [@xmwu0124](https://github.com/xmwu0124)

Seeking predoctoral research positions in applied microeconomics, with interests in:
- Industrial organization
- Innovation and technology adoption
- Firm dynamics and productivity

---

## License

MIT License

---

**Data Sources**: US Census Bureau TIGER/Line Shapefiles (2023), synthetic economic indicators for demonstration

**Note**: This repository uses synthetic data to demonstrate methodology. The framework is production-ready and applicable to real-world datasets from sources such as Compustat, BEA, or Census Bureau microdata.
