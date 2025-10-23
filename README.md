# Firm Digitalization Study

**Applied Econometrics Code Sample - Research Position Application**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Methods](https://img.shields.io/badge/Methods-DiD%20%7C%20Structural%20IO-green.svg)]()

## Purpose

This repository serves as a **code sample** demonstrating proficiency in applied econometric methods for **empirical research positions**. The project showcases end-to-end analytical capabilities from data construction through causal inference and structural estimation.

## What This Project Does

### Research Question
How does digital technology adoption affect firm performance, and what policies can accelerate adoption?

### Approach
1. **Data Construction**: Synthesize realistic firm panel data (200 firms × 10 years) with:
   - Staggered digital adoption timing
   - Heterogeneous treatment effects by firm characteristics
   - Geographic and industry variation
   - Simulated 10-K text excerpts for extraction exercises

2. **Causal Analysis** (Difference-in-Differences):
   - Event study design with two-way fixed effects
   - Dynamic treatment effect estimation
   - Robustness checks and pre-trend testing
   - Cluster-robust inference

3. **Structural Estimation** (IO Framework):
   - Dynamic discrete choice model of technology adoption
   - Bellman equation solution via value function iteration
   - Maximum likelihood estimation with JAX optimization
   - Counterfactual policy simulations (e.g., adoption subsidies)

4. **Spatial Analysis**:
   - Geographic clustering patterns
   - Industry-location interactions
   - Diffusion dynamics visualization

## Key Technical Skills Demonstrated

### Econometric Methods
- **Causal Inference**: Difference-in-Differences, event studies, parallel trends
- **Structural Modeling**: Dynamic programming, discrete choice, forward-looking behavior
- **Estimation**: Maximum likelihood, gradient-based optimization
- **Inference**: Cluster-robust standard errors, bootstrapping

### Programming & Tools
- **Python**: pandas, numpy, scipy, JAX
- **Econometrics**: pyfixest (high-dimensional fixed effects), statsmodels
- **Optimization**: Constrained optimization, numerical methods
- **Visualization**: matplotlib, seaborn
- **Workflow**: Modular code structure, configuration management, reproducible pipeline

### Software Engineering
- Clean, documented, and reproducible code
- Modular design with separation of concerns
- Configuration-driven analysis (YAML)
- Version control (Git/GitHub)
- Comprehensive logging and error handling

## Project Structure
```
firm-digitalization-study/
├── src/
│   ├── 01_data_construction/     # Text mining & data extraction
│   │   └── extract_digital_dates.py
│   ├── 02_data_generation/       # Synthetic panel creation
│   │   ├── generate_panel.py    # Firm panel with adoption timing
│   │   └── generate_text.py     # Simulated 10-K excerpts
│   ├── 03_analysis/              # Econometric analysis
│   │   ├── did_analysis.py      # Event study estimation
│   │   └── spatial_analysis_simple.py
│   ├── 04_structural/            # Structural estimation
│   │   └── io_structural_model.py
│   ├── config.yaml               # Analysis parameters
│   └── run_all.py               # Master execution script
├── data/
│   ├── raw/                     # Simulated input data
│   └── processed/               # Analysis-ready datasets
├── output/
│   ├── figures/                 # Publication-quality plots
│   └── tables/                  # Results tables (CSV)
└── docs/                        # Documentation
```

## Key Results

### 1. Causal Effects (Difference-in-Differences)
```
Event Time    Coefficient    SE      P-value
-3            -0.052        0.021    0.012
-2             0.015        0.023    0.509
0              0.086        0.023    0.000
+1             0.140        0.024    0.000
+2             0.219        0.031    0.000
+3             0.306        0.045    0.000
+4             0.382        0.057    0.000
```

**Interpretation**: 
- No significant pre-trends (supports parallel trends assumption)
- Treatment effects grow over time: 8.6% → 38.2% revenue increase
- Largest effects 3-4 years post-adoption

### 2. Structural Parameters
| Parameter | Estimate | Interpretation |
|-----------|----------|----------------|
| α_size | 0.40 | Technology benefit increases with firm size |
| α_comp | -0.30 | Competition reduces standalone profitability |
| Fixed Cost | 2.0M | One-time adoption investment |
| β (discount) | 0.95 | Firms are forward-looking |

### 3. Policy Counterfactual
- **Baseline**: 25% adoption rate
- **50% Subsidy**: Simulated increase to 40% adoption
- **Mechanism**: Subsidy reduces effective fixed costs, making adoption optimal for marginal firms
- **Heterogeneity**: Largest effects for medium-sized firms in competitive industries

## Installation & Usage
```bash
# Clone repository
git clone https://github.com/xmwu0124/firm-digitalization-study.git
cd firm-digitalization-study

# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python src/run_all.py

# Or run individual components:
python src/02_data_generation/generate_panel.py    # Generate data
python src/03_analysis/did_analysis.py             # DiD analysis
python src/04_structural/io_structural_model.py    # Structural model
```

## Methods Overview

### Difference-in-Differences Specification
```
Y_it = α_i + λ_t + Σ_{k≠-1} β_k · 1{EventTime_it = k} + ε_it
```
- `α_i`: Firm fixed effects
- `λ_t`: Year fixed effects  
- `β_k`: Dynamic treatment effects
- Standard errors clustered at firm level

### Structural Model
**State Space**: (firm size, competition, tech sector, adoption status)

**Bellman Equation**:
```
V(s) = max_d { u(s,d) + β · E[V(s') | s, d] }
```
where `d ∈ {0,1}` is adoption decision

**Estimation**: Maximum likelihood over observed adoption decisions

**Identification**: Forward-looking behavior, fixed costs, and continuation values identified from variation in adoption timing and firm characteristics

## Output Files

### Tables
- `event_study_results.csv` - DiD coefficients with standard errors
- `io_structural_estimates.csv` - Structural parameters
- `io_counterfactual_subsidy.csv` - Policy simulation results
- `spatial_analysis_results.csv` - Geographic clustering metrics

### Figures
- `event_study_log_revenue.png` - Dynamic treatment effects plot
- `spatial_analysis_comprehensive.png` - Geographic distribution maps

## Technologies

**Core Stack**:
- Python 3.9+
- pandas, numpy (data manipulation)
- pyfixest (high-dimensional fixed effects)
- JAX (automatic differentiation, GPU acceleration)
- scipy (optimization)
- matplotlib, seaborn (visualization)

**Econometric Tools**:
- Two-way fixed effects regression
- Clustered standard errors
- Value function iteration
- Maximum likelihood estimation
- Bootstrap inference

## Why This Approach?

### Synthetic Data Benefits
1. **Full Ground Truth**: Known treatment effects enable validation
2. **Reproducibility**: No data access restrictions
3. **Flexibility**: Easy to modify parameters for robustness checks
4. **Pedagogical**: Clearly demonstrates methodological competencies

### Real-World Extensions
This pipeline can be adapted for:
- Actual firm-level data (Compustat, SEC filings)
- Patent adoption studies
- Environmental regulation compliance
- Technology diffusion in developing countries

## Code Quality Features

✓ **Modular Design**: Separate data generation, analysis, and visualization  
✓ **Configuration-Driven**: YAML-based parameter management  
✓ **Comprehensive Logging**: Detailed execution logs for debugging  
✓ **Error Handling**: Graceful failures with informative messages  
✓ **Reproducibility**: Fixed random seeds, version control  
✓ **Documentation**: Inline comments, docstrings, README  

## Author

**Xiaomeng Wu**  
GitHub: [@xmwu0124](https://github.com/xmwu0124)  

*This project demonstrates applied econometric skills for empirical research positions in economics, finance, and data science.*

## License

MIT License - See LICENSE file for details

---

**Note**: This is a demonstration project using synthetic data. The methods and code structure are production-ready and can be adapted for real-world research applications.
