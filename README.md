# Firm Digitalization Study

**Empirical analysis of corporate digital transformation using causal inference and structural methods**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

## Overview

Complete empirical pipeline for analyzing corporate digital adoption, combining:
- **Causal Inference**: Difference-in-Differences event study analysis
- **Structural Estimation**: Dynamic discrete choice IO model
- **Spatial Analysis**: Geographic clustering and diffusion patterns
- **Policy Simulation**: Counterfactual adoption under various regimes

## Project Structure
```
firm-digitalization-study/
├── src/
│   ├── 01_data_construction/    # Text mining & extraction
│   ├── 02_data_generation/      # Synthetic panel creation
│   ├── 03_analysis/             # DiD, EDA, spatial analysis
│   ├── 04_structural/           # IO structural estimation
│   └── run_all.py              # Master execution script
├── data/
│   ├── raw/                    # Input data
│   └── processed/              # Analysis-ready datasets
├── output/
│   ├── figures/                # Visualizations
│   └── tables/                 # Results tables
└── docs/                       # Documentation
```

## Key Features

### 1. Data Generation
- Synthetic firm panel: 200 firms × 10 years
- Realistic adoption timing with heterogeneous treatment effects
- Geographic, industry, and firm-level characteristics

### 2. Causal Analysis (Difference-in-Differences)
- Event study design with staggered adoption
- Two-way fixed effects (firm + year)
- Robust standard errors clustered at firm level
- Pre-trend testing and dynamic treatment effects

### 3. Structural Estimation (IO Framework)
- Dynamic discrete choice model of technology adoption
- State space: firm size, competition, tech sector, adoption status
- Bellman equation solved via value function iteration
- Maximum likelihood estimation with JAX optimization
- Policy counterfactuals: subsidy simulations

### 4. Spatial Analysis
- Geographic distribution visualization
- Clustering metrics and hotspot detection
- Industry-location interaction patterns

## Installation
```bash
# Clone repository
git clone https://github.com/xmwu0124/firm-digitalization-study.git
cd firm-digitalization-study

# Install dependencies
pip install -r requirements.txt
```

## Quick Start
```bash
# Run complete analysis pipeline
python src/run_all.py

# Or run individual components:
python src/02_data_generation/generate_panel.py    # Data generation
python src/03_analysis/did_analysis.py             # DiD analysis
python src/04_structural/io_structural_model.py    # Structural model
python src/03_analysis/spatial_analysis_simple.py  # Spatial analysis
```

## Key Results

### Difference-in-Differences
- **Pre-trends**: No significant effects in periods t-3 to t-1
- **Treatment effects**: Progressive increase from 8.6% (t=0) to 38.2% (t+4) in log revenue
- **Heterogeneity**: Larger effects for tech firms and competitive industries

### Structural Parameters
| Parameter | Estimate | Interpretation |
|-----------|----------|----------------|
| α_size | 0.40 | Technology benefit scales with firm size |
| Fixed cost | 2.0 | Adoption cost (millions) |
| β (discount) | 0.95 | Forward-looking behavior |

### Policy Counterfactual
- **Baseline adoption rate**: 25%
- **With 50% subsidy**: Simulated increase to ~40%
- **Most responsive**: Medium-sized firms in competitive markets

## Methods

### Econometric Approach
- **Identification**: Conditional parallel trends assumption
- **Specification**: `Y_it = α_i + λ_t + Σ β_k·D_it^k + ε_it`
- **Inference**: Cluster-robust standard errors (firm level)

### Structural Model
- **Framework**: Discrete choice with forward-looking firms
- **Solution**: Value function iteration (Bellman equation)
- **Estimation**: Maximum likelihood with gradient-based optimization
- **Validation**: Predicted vs. actual adoption patterns

## Technologies

- **Core**: Python 3.9+, pandas, numpy
- **Econometrics**: pyfixest, statsmodels
- **Optimization**: scipy, JAX
- **Visualization**: matplotlib, seaborn
- **Spatial** (optional): libpysal, esda

## Output Files

### Key Tables
- `event_study_results.csv` - DiD coefficient estimates
- `io_structural_estimates.csv` - Structural parameters
- `io_counterfactual_subsidy.csv` - Policy simulation results
- `spatial_analysis_results.csv` - Geographic clustering metrics

### Key Figures
- `event_study_log_revenue.png` - Dynamic treatment effects
- `spatial_analysis_comprehensive.png` - Geographic distribution maps

## Documentation

- See `docs/methodology.md` for detailed methods
- See `docs/data_dictionary.md` for variable definitions
- Configuration: Edit `src/config.yaml` for parameters

## License

MIT License

## Author

Xiaomeng Wu  
GitHub: [@xmwu0124](https://github.com/xmwu0124)

## Acknowledgments

This project demonstrates applied econometric methods for empirical analysis, suitable as a code sample for research positions.
