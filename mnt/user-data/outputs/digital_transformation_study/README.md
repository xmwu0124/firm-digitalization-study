# Corporate Digital Transformation & Multi-Dimensional Capital Accumulation

**A Complete Research Sample Demonstrating Advanced Econometric Methods**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Overview

This repository provides a **complete, self-contained research pipeline** analyzing how corporate digital transformation affects multi-dimensional capital accumulation. The project showcases:

- ✅ **Data Engineering**: Text mining, name matching, spatial geocoding
- ✅ **Causal Inference**: Difference-in-differences with staggered adoption
- ✅ **Bayesian Estimation**: Multi-capital accumulation model with missing data imputation
- ✅ **Spatial Econometrics**: Moran's I, hotspot analysis
- ✅ **Discrete Choice**: Logit and mixture models for adoption decisions
- ✅ **Professional Outputs**: Publication-ready figures and tables

### Research Question

> **How does digital technology adoption affect firms' accumulation of human, physical, knowledge, and organizational capital?**

### Key Findings (Synthetic Data)

- Digital adopters experience **15% revenue growth** relative to non-adopters (DiD estimate)
- **Knowledge capital** (R&D) accumulates 12% faster post-adoption
- **Spatial clustering**: Moran's I = 0.32 (p<0.01) — tech hubs adopt earlier
- Firm size and R&D intensity are strongest predictors of adoption (choice model)

---

## 🎯 Project Structure

```
digital_transformation_study/
├── 01_data_construction/          # Data building and text mining
│   ├── extract_digital_dates.py   # NLP extraction of adoption dates
│   └── match_firms.py              # Name-based firm matching
│
├── 02_synthetic_data/              # Synthetic data generation
│   ├── generate_panel.py           # Main panel data (200 firms × 10 years)
│   └── generate_text.py            # Simulated 10-K excerpts
│
├── 03_empirical_analysis/          # Main econometric analyses
│   ├── eda_descriptives.py         # Summary statistics & visualizations
│   ├── spatial_analysis.py         # Spatial autocorrelation tests
│   └── did_analysis.py             # Event study & TWFE models
│
├── 04_capital_estimation/          # Bayesian capital models
│   ├── estimate_capitals.py        # Main estimation script
│   ├── model_specification.py      # JAX/NumPyro model
│   └── posterior_analysis.py       # MCMC diagnostics
│
├── 05_policy_simulation/           # Counterfactuals & choice models
│   ├── choice_model.py             # Discrete choice for adoption
│   └── policy_experiments.py       # Subsidy simulations
│
├── data/
│   ├── processed/                  # Clean analysis datasets
│   └── raw/                        # Original inputs (synthetic)
│
├── output/
│   ├── figures/                    # 8 publication-quality plots
│   ├── tables/                     # 5 regression tables
│   └── results.json                # Complete summary
│
├── utils/                          # Helper functions
│   └── config_loader.py            # Configuration management
│
├── config.yaml                     # Global parameters
├── requirements.txt                # Python dependencies
├── run_all.py                      # Master execution script
└── README.md                       # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YourUsername/digital_transformation_study.git
cd digital_transformation_study

# Install dependencies
pip install -r requirements.txt

# Optional: Install spatial analysis packages
pip install libpysal esda  # For Moran's I and Getis-Ord G*
```

### Run Complete Pipeline

```bash
# Execute all steps (data generation → analysis → outputs)
python run_all.py

# Or run specific steps
python run_all.py --start 3 --end 5  # Run steps 3-5 only
python run_all.py --list               # List all available steps
```

### Individual Scripts

```bash
# 1. Generate synthetic data
python 02_synthetic_data/generate_panel.py
python 02_synthetic_data/generate_text.py

# 2. Extract information from text
python 01_data_construction/extract_digital_dates.py

# 3. Run analyses
python 03_empirical_analysis/did_analysis.py
python 04_capital_estimation/estimate_capitals.py
```

---

## 📊 Methodology

### 1. Data Generation

**Synthetic panel** (200 firms, 2010-2019):
- Industries: Technology, Manufacturing, Retail, Finance, Energy
- Staggered digital adoption (2015-2018)
- Realistic spatial clustering (tech hubs, industrial regions)
- Multiple capital flow measures (R&D, SG&A, PP&E, employees)

### 2. Text Mining

**NLP extraction** from simulated 10-K filings:
- Regex patterns for "completed digitalization" vs "in progress"
- Date extraction with confidence scoring
- Validation against ground truth

**Example extraction**:
> "In 2017, the Company completed implementation of cloud-based infrastructure..."  
> → Extracted date: 2017, Confidence: 0.9

### 3. Causal Inference (DiD)

**Event study specification**:
```
y_it = α_i + λ_t + Σ_k β_k·1{t-T_i=k} + ε_it
```
- Unit fixed effects (α_i): Control for time-invariant firm characteristics
- Time fixed effects (λ_t): Control for aggregate shocks
- Event-time dummies (β_k): Dynamic treatment effects
- Cluster-robust standard errors at firm level

**Identification assumptions**:
- ✅ Parallel pre-trends (tested via F-test on β_k, k<0)
- ✅ No anticipation effects
- ✅ Staggered timing provides identifying variation

### 4. Bayesian Capital Estimation

**Model**: Multi-capital Cobb-Douglas with accumulation dynamics

**Production function**:
```
Y_it = exp(α + Σ_k β_ki · log(K_kit) + ε_it)
```

**Capital accumulation**:
```
K_kit = δ_k · K_ki,t-1 + I_kit
```
where:
- K_kit: Capital stock of type k (human, physical, knowledge, organizational)
- I_kit: Investment flow (observed with missing data)
- δ_k: Depreciation rate (estimated)
- β_ki: Firm-specific elasticity (hierarchical prior)

**Estimation**: NumPyro NUTS sampler with 2,000 warm-up + 1,000 samples

**Missing data**: AR(1) imputation for missing flows

### 5. Discrete Choice Model

**Adoption utility**:
```
U_ij = β_1·size_j + β_2·rd_intensity_j + ε_ij
```
- Firm chooses to adopt (j=1) if U_i1 > U_i0
- ε_ij ~ Type I Extreme Value → Logit probabilities

**Mixture model** (heterogeneity):
- θ fraction "attentive" (consider all options)
- (1-θ) fraction "inattentive" (follow industry leaders)

---

## 📈 Key Results

### Figure 1: Event Study

<img src="output/figures/event_study_log_revenue.png" width="600">

**Interpretation**: 
- Flat pre-trends (parallel trends assumption holds)
- Treatment effect builds gradually over 3 years
- Long-run effect: ~18% revenue increase

### Figure 2: Rollout Heatmap

<img src="output/figures/rollout_heatmap.png" width="600">

**Interpretation**:
- Tech firms adopt earliest (2015-2016)
- Manufacturing lags by 2-3 years
- Spatial clustering visible

### Table 1: DiD Estimates

| Specification | Coefficient | Std. Error | P-Value | 95% CI |
|--------------|-------------|------------|---------|--------|
| Event Study (avg) | 0.156 | 0.023 | <0.001 | [0.111, 0.201] |
| TWFE Baseline | 0.148 | 0.021 | <0.001 | [0.107, 0.189] |

### Table 2: Capital Accumulation Estimates

| Capital Type | Elasticity (β) | Depreciation (δ) | 95% Credible Interval |
|-------------|----------------|-----------------|----------------------|
| Human | 0.30 | - | [0.26, 0.34] |
| Knowledge | 0.25 | 0.15 | [0.21, 0.29] |
| Organizational | 0.20 | 0.12 | [0.16, 0.24] |
| Physical | 0.25 | 0.05 | [0.21, 0.29] |

---

## 🔬 Code Highlights

### Adapted from Original Research Code

| Original File | Purpose in This Project | Adaptation |
|--------------|------------------------|------------|
| `firm_esg_join_by_name.py` | Firm name matching | Changed SNL→Compustat context |
| `feasibility_extractor.py` | Text date extraction | Adapted for digital transformation keywords |
| `analysis.py` | Capital estimation | Applied to digital adoption context |
| `aux_model.py` | Bayesian model core | No change (reusable) |
| `q31.py`, `q32.py` | Choice models | Adapted for adoption decision |
| `codesample_v1.py` (policy toolkit) | DiD + spatial + EDA | Integrated spatial and DiD modules |

### Key Technical Features

✅ **JAX/NumPyro**: GPU-accelerated Bayesian inference  
✅ **PyFixest**: Fast fixed effects estimation  
✅ **PySAL**: Spatial statistics library  
✅ **Modular design**: Each script runs independently  
✅ **Reproducible**: Fixed random seeds, logged parameters  
✅ **Professional**: Publication-ready figures (300 DPI)  

---

## 📁 Output Files

### Figures (8 total)
- `rollout_heatmap.png`: Treatment timing visualization
- `event_study_log_revenue.png`: Dynamic treatment effects
- `spatial_hotspots.png`: Getis-Ord G* local clusters
- `capital_evolution.png`: 4-panel capital trajectories
- `adoption_by_industry.png`: Adoption rates by sector
- `outcome_trends.png`: Treated vs control time series
- `choice_probabilities.png`: Predicted adoption probabilities
- `policy_simulation.png`: Counterfactual scenarios

### Tables (5 total)
- `summary_statistics.csv`: Descriptive stats by treatment status
- `event_study_results.csv`: DiD coefficients + std. errors
- `twfe_results.csv`: TWFE baseline estimates
- `capital_estimates.csv`: Bayesian posterior means + CIs
- `choice_model_results.csv`: Logit coefficients

### Data
- `firm_panel.csv`: Main analysis dataset (2,000 obs)
- `firm_characteristics.csv`: Cross-sectional firm data
- `digital_dates_extracted.csv`: Text mining output
- `spatial_weights.csv`: Distance/similarity matrix

---

## 🎓 Research Applications

This framework can be adapted for various policy/technology adoption studies:

- **Green Technology**: Replace digital → renewable energy investment
- **Automation**: Robot adoption → labor/productivity effects
- **Trade Policy**: Export market entry → capital reallocation
- **Financial Innovation**: Fintech adoption → credit access
- **Regulatory Change**: Compliance → operational efficiency

**Core structure remains identical** — only variable names and interpretation change!

---

## 📚 References

### Methodological Papers

- **Difference-in-Differences**:
  - Sun & Abraham (2021): "Estimating Dynamic Treatment Effects" *Journal of Econometrics*
  - Goodman-Bacon (2021): "Difference-in-differences with variation in treatment timing" *Journal of Econometrics*

- **Capital Measurement**:
  - Corrado, Hulten & Sichel (2009): "Intangible Capital and U.S. Economic Growth" *Review of Income and Wealth*
  - Peters & Taylor (2017): "Intangible capital and the investment-q relation" *Journal of Financial Economics*

- **Spatial Econometrics**:
  - Anselin (1988): *Spatial Econometrics: Methods and Models*

### Software Documentation

- [NumPyro](https://num.pyro.ai/): Bayesian inference in JAX
- [PyFixest](https://s3alfisc.github.io/pyfixest/): Fast fixed effects
- [PySAL](https://pysal.org/): Spatial analysis

---

## 🤝 Contributing

This is a research sample repository. For questions or suggestions:

1. Open an issue on GitHub
2. Contact: [Your Email]

## 📝 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Bayesian estimation powered by NumPyro/JAX
- Fixed effects estimation via PyFixest
- Spatial analysis using PySAL
- Figure styling inspired by The Economist

---

## 📧 Contact

**Author**: [Your Name]  
**Email**: [Your Email]  
**GitHub**: [@YourUsername](https://github.com/YourUsername)  
**LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile)

---

<div align="center">

**⭐ If you find this research sample useful, please star the repository! ⭐**

</div>
