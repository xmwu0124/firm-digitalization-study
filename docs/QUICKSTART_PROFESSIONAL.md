# Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 500MB available disk space
- Recommended: 8GB RAM for structural estimation

## Installation

### Step 1: Clone or Download Repository

```bash
cd ~/Dropbox
# Repository should be in CODE_SAMPLE directory
cd CODE_SAMPLE
```

### Step 2: Install Core Dependencies

```bash
pip install pandas numpy scipy matplotlib seaborn pyyaml statsmodels
```

### Step 3: Install Optional Components

For fast fixed effects estimation:
```bash
pip install pyfixest
```

For structural estimation with JAX:
```bash
# CPU version
pip install jax jaxlib

# GPU version (CUDA 12)
pip install jax[cuda12]
```

For spatial analysis:
```bash
pip install libpysal esda
```

### Step 4: Verify Installation

```bash
python -c "import pandas, numpy, scipy, matplotlib; print('Core dependencies installed')"
```

## Five-Minute Walkthrough

### Generate Synthetic Data

```bash
cd digital_transformation_study
python 02_synthetic_data/generate_panel.py
```

Expected output:
- `data/processed/firm_panel.csv` (2,000 observations)
- `data/processed/firm_characteristics.csv` (200 firms)

Runtime: Approximately 30 seconds

### Run Difference-in-Differences Analysis

```bash
python 03_empirical_analysis/did_analysis.py
```

Outputs:
- `output/figures/event_study_log_revenue.png`
- `output/tables/event_study_results.csv`
- `output/tables/twfe_results.csv`

Runtime: Approximately 15 seconds

Expected results:
- Average treatment effect: 0.156 (SE: 0.023)
- Pre-trend test p-value: >0.05 (parallel trends supported)

### Run IO Structural Estimation (Optional)

```bash
python 04_capital_estimation/io_structural_model.py
```

Outputs:
- `output/tables/io_structural_estimates.csv`
- `output/tables/io_counterfactual_subsidy.csv`

Runtime: 10-20 minutes depending on hardware

Expected results:
- Fixed adoption cost estimate: approximately 2.0-2.5
- Benefit coefficient estimates: positive and significant
- Counterfactual subsidy effect: 20-25% adoption increase

## Running Complete Pipeline

Execute all analyses sequentially:

```bash
python run_all.py
```

This runs:
1. Data generation (panel and text)
2. Text mining extraction
3. Exploratory data analysis
4. Spatial analysis (if libpysal installed)
5. Difference-in-differences estimation
6. Structural estimation (if JAX installed)
7. Output compilation

Total runtime: Approximately 25 minutes

To run specific steps only:

```bash
# List available steps
python run_all.py --list

# Run steps 1-3 only
python run_all.py --start 1 --end 3
```

## Customizing Parameters

Edit `config.yaml` to adjust analysis parameters:

```yaml
data:
  n_firms: 200          # Number of firms
  n_years: 10           # Time periods
  start_year: 2010      # Panel start year

analysis:
  event_window_pre: -3  # Pre-treatment periods
  event_window_post: 5  # Post-treatment periods
  mcmc_samples: 1000    # Bayesian estimation samples
  
outputs:
  figure_dpi: 300       # Plot resolution
  decimal_places: 4     # Table precision
```

After modification, re-run analyses:

```bash
python run_all.py
```

## Verifying Outputs

Check that outputs were generated:

```bash
# List figures
ls output/figures/

# List tables
ls output/tables/

# View summary statistics
cat output/tables/summary_statistics.csv
```

Expected files:
- Figures: event_study_log_revenue.png, rollout_heatmap.png, outcome_trends.png
- Tables: summary_statistics.csv, event_study_results.csv, twfe_results.csv

## Troubleshooting

### Import Error: No module named 'pandas'

Install missing dependency:
```bash
pip install pandas
```

### FileNotFoundError: firm_panel.csv

Generate data first:
```bash
python 02_synthetic_data/generate_panel.py
```

### JAX Not Available

Structural estimation requires JAX. Either:
1. Install JAX: `pip install jax jaxlib`
2. Skip structural estimation and run other analyses only

### Memory Error

Reduce state space size in `io_structural_model.py`:
```python
state_space = create_state_space(
    n_size=6,  # Reduced from 8
    n_comp=3   # Reduced from 5
)
```

### Convergence Issues

Increase iteration limits in relevant scripts:
```python
# In did_analysis.py or io_structural_model.py
max_iter=2000  # Increased from 1000
tol=1e-4       # Relaxed from 1e-6
```

## Next Steps

After completing the quick start:

1. Review methodology in main README.md
2. Examine detailed results in output directory
3. Modify parameters for sensitivity analysis
4. Adapt framework to alternative research questions

## Support

For detailed documentation:
- Main README: Project overview and methodology
- Technical specs: docs/IO_MODEL_TECHNICAL.md
- Configuration: See inline comments in config.yaml

For implementation questions, consult inline documentation in source files.

## Performance Notes

Typical runtimes on standard hardware (Intel i7, 16GB RAM):

| Module | Runtime |
|--------|---------|
| Data generation | 30s |
| Text mining | 10s |
| DiD analysis | 15s |
| Spatial analysis | 45s |
| IO estimation | 10-20m |
| Complete pipeline | 25m |

GPU acceleration reduces IO estimation to 3-5 minutes.

## Validation Checklist

Verify successful execution:

- [ ] Data files created in data/processed/
- [ ] Figures generated in output/figures/
- [ ] Tables generated in output/tables/
- [ ] Log files created in logs/
- [ ] No error messages in console output
- [ ] Event study plot shows clear treatment effects
- [ ] Parameter estimates have reasonable magnitudes

If all items checked, the installation and execution were successful.
