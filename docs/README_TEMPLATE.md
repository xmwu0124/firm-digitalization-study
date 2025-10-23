# Policy Evaluation Study Toolkit

A comprehensive Python framework for causal inference analysis of policy interventions using panel data. This toolkit integrates exploratory data analysis, spatial statistics, text processing, and difference-in-differences (DiD) estimation methods.

## 🎯 Overview

This toolkit is designed for researchers and analysts conducting policy evaluation studies. It provides an end-to-end pipeline that handles:

- **Data Quality Assessment**: Automated checks for missingness, outliers, and data distribution
- **Spatial Analysis**: Raster processing and spatial autocorrelation tests
- **Text Mining**: Policy document parsing and classification using NLP
- **Causal Inference**: Multiple DiD estimators including event studies and Sun & Abraham (2021)
- **Visualization**: Publication-ready plots for results presentation

## ✨ Key Features

### 📊 Exploratory Data Analysis (EDA)
- Comprehensive missingness reports with flagging thresholds
- Distribution visualizations with automatic outlier clipping
- Correlation heatmaps to detect multicollinearity
- Categorical variable frequency analysis
- Duplicate and constant column detection

### 🗺️ Spatial Analysis
- Multi-resolution raster resampling (bilinear, cubic, nearest, lanczos)
- Moran's I global autocorrelation test
- Getis-Ord G* local hotspot analysis
- Integration with PySAL for advanced spatial statistics

### 📝 Text Processing
- Regex-based date extraction from policy documents
- Confidence scoring for extracted dates
- TF-IDF vectorization with customizable n-grams
- Logistic regression classifier for policy categorization
- Support for multi-class classification

### 📈 Difference-in-Differences Estimation
- **Event Study**: Dynamic treatment effects with flexible event windows
- **Sun & Abraham (2021)**: Heterogeneity-robust estimator for staggered adoption
- **Two-Way Fixed Effects (TWFE)**: Baseline specification with clustered standard errors
- Automatic coefficient extraction and confidence intervals

### 🎨 Visualization
- Event study plots with confidence bands
- Policy rollout heatmaps showing treatment timing
- High-resolution output (customizable DPI)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/XiaomengWu0124/codesample.git
cd codesample

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for text processing)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### Basic Usage

```python
from codesample_v1 import PolicyStudy, StudyConfig

# Create configuration
config = StudyConfig(
    project_name="my_policy_study",
    main_data_file="panel.csv",
    unit_id="state_id",
    time="year",
    treat_flag="treated",
    treat_start="treatment_year",
    outcomes=["outcome1", "outcome2"],
    controls=["control1", "control2"],
    do_eda=True,
    do_spatial=True,
    do_text=True,
    do_did=True
)

# Run full analysis pipeline
study = PolicyStudy(config)
results = study.run_all()

# Results are saved to: my_policy_study/output/results.json
```

## 📁 Project Structure

```
your_project/
├── data/
│   ├── panel.csv              # Main panel dataset
│   ├── raster.tif             # Optional: spatial raster data
│   └── text_train.csv         # Optional: labeled text for training
├── output/
│   ├── figures/
│   │   ├── event_*.png        # Event study plots
│   │   ├── rollout.png        # Treatment rollout heatmap
│   │   └── eda_*.png          # EDA visualizations
│   ├── tables/
│   │   ├── missing_*.csv      # Missingness reports
│   │   ├── text_extract.csv   # Extracted policy dates
│   │   └── spatial_*.csv      # Spatial statistics
│   ├── did_ready.csv          # Preprocessed DiD data
│   └── results.json           # Complete analysis summary
├── logs/
│   └── run_*.log              # Execution logs
└── config.yaml                # Saved configuration
```

## 📋 Data Requirements

### Panel Data Format (`panel.csv`)

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `unit_id` | int/str | Unit identifier (e.g., state FIPS) | ✅ |
| `year` | int | Time period | ✅ |
| `treated` | 0/1 | Treatment indicator | ✅ |
| `treatment_year` | int | First treatment year (NA if never treated) | ✅ |
| `outcome1`, `outcome2`, ... | float | Outcome variables | ✅ |
| `control1`, `control2`, ... | float | Control variables | Optional |
| `policy_text` | str | Policy document text | Optional |
| `cluster_id` | int/str | Clustering variable | Optional |

### Example Data

```csv
unit_id,year,treated,treatment_year,log_gdp,renewables,policy_text
1,2010,0,,8.5,10.2,"Energy policy document..."
1,2011,0,,8.6,11.5,"Updated regulations..."
1,2012,1,2012,8.7,15.3,"Implemented in 2012..."
2,2010,0,,9.1,8.7,"No major changes..."
```

## ⚙️ Configuration Options

### Core Settings

```python
config = StudyConfig(
    # Project paths
    project_name="policy_study",
    data_dir="data",
    output_dir="output",
    
    # Variable mappings
    unit_id="unit_id",
    time="year",
    treat_flag="treated",
    treat_start="treat_start_year",
    cluster="cluster_id",
    
    # Analysis variables
    outcomes=["y1", "y2"],
    controls=["x1", "x2"],
    
    # Analysis toggles
    do_eda=True,
    do_spatial=True,
    do_text=True,
    do_did=True,
    
    # DiD settings
    event_window=(-3, 5),  # 3 pre-periods, 5 post-periods
    conf_level=0.95,
    
    # Reproducibility
    seed=42
)
```

### Advanced Options

<details>
<summary>Click to expand full configuration options</summary>

```python
config = StudyConfig(
    # Spatial analysis
    target_res_deg=0.01,
    resampling="bilinear",  # Options: bilinear, nearest, cubic, lanczos
    resample_scales=[0.1, 0.05, 0.01],
    
    # Text processing
    text_col="policy_text",
    text_label="policy_category",
    text_year_min=1990,
    text_year_max=2035,
    text_low_conf_threshold=0.6,
    
    # TF-IDF parameters
    tfidf_max_features=5000,
    tfidf_ngram_min=1,
    tfidf_ngram_max=3,
    tfidf_min_df=2,
    tfidf_max_df=0.95,
    
    # Logistic regression
    logreg_C=1.0,
    logreg_max_iter=1000,
    logreg_class_weight="balanced",
    
    # EDA settings
    eda_bins=30,
    eda_clip_quantiles=(0.01, 0.99),
    eda_sample_rows=100_000,
    eda_top_categories=20,
    eda_corr_method="pearson",  # Options: pearson, spearman
    eda_missing_flag=0.20,
    eda_fig_dpi=200
)
```
</details>

## 📊 Example Output

### Event Study Results

```json
{
  "event_study": {
    "log_renewables": {
      "pre_trend_pvalue": 0.234,
      "avg_effect": 0.156,
      "coefficients": {
        "-3": -0.012,
        "-2": 0.008,
        "-1": 0.003,
        "0": 0.089,
        "1": 0.134,
        "2": 0.178
      },
      "model_summary": "..."
    }
  }
}
```

### Visualization Examples

<div align="center">

**Event Study Plot**  
*Dynamic treatment effects with 95% confidence intervals*

**Rollout Heatmap**  
*Spatial and temporal variation in policy adoption*

</div>

## 🔬 Methodology

### Difference-in-Differences Estimators

#### 1. Event Study Specification

```
y_it = α_i + λ_t + Σ_k β_k D_it^k + γ X_it + ε_it
```

Where:
- `D_it^k` = 1 if unit `i` is `k` periods from treatment at time `t`
- Omitted period: `k = -1` (period before treatment)
- Tails binned for `k < event_min` and `k > event_max`

#### 2. Sun & Abraham (2021)

Addresses heterogeneous treatment effects in staggered adoption designs using cohort-specific estimators.

#### 3. Two-Way Fixed Effects

```
y_it = α_i + λ_t + β D_it + γ X_it + ε_it
```

Standard errors clustered at specified level (e.g., state or state×year).

## 🛠️ Dependencies

```txt
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
rasterio>=1.3.0
pyfixest>=0.18.0
scikit-learn>=1.3.0
nltk>=3.8.0
PyYAML>=6.0
scipy>=1.11.0
libpysal>=4.9.0  # Optional
esda>=2.5.0      # Optional
```

## 📚 Usage Examples

### Example 1: State-Level Renewable Energy Policy

```python
config = StudyConfig(
    project_name="renewable_energy_study",
    main_data_file="state_panel.csv",
    unit_id="state_fips",
    time="year",
    treat_flag="rps_active",
    treat_start="rps_adoption_year",
    cluster="census_region",
    outcomes=["log_renewables", "log_co2_emissions"],
    controls=["log_gdp_pc", "pop_density", "manufacturing_share"],
    event_window=(-5, 7),
    seed=2024
)

study = PolicyStudy(config)
results = study.run_all()
```

### Example 2: With Text Classification

```python
# Prepare training data (text_train.csv)
# Columns: policy_text, policy_category

config = StudyConfig(
    project_name="text_classification_study",
    training_text_data="text_train.csv",
    text_col="policy_text",
    text_label="policy_category",
    do_text=True,
    tfidf_max_features=3000,
    tfidf_ngram_max=2
)

study = PolicyStudy(config)
results = study.run_all()

# Access classification metrics
print(f"Test Accuracy: {results['text_clf']['test_accuracy']:.3f}")
```

### Example 3: Custom Configuration from YAML

```yaml
# config.yaml
project_name: my_analysis
main_data_file: data.csv
unit_id: county_fips
time: year
outcomes:
  - outcome_1
  - outcome_2
controls:
  - control_1
event_window: [-4, 6]
```

```python
config = StudyConfig.from_yaml("config.yaml")
study = PolicyStudy(config)
results = study.run_all()
```

## 🔍 Interpreting Results

### Pre-Trend Testing

Check `pre_trend_pvalue` in event study results:
- **p > 0.05**: Parallel trends assumption plausible ✅
- **p < 0.05**: Evidence of differential pre-trends ⚠️

### Effect Magnitude

- Coefficients represent percentage changes when outcomes are log-transformed
- Example: β = 0.15 → ~16% increase (exp(0.15) - 1)

### Confidence Intervals

All estimators report 95% confidence intervals by default. Adjust with `conf_level` parameter.

## 🐛 Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: panel.csv not found`
- **Solution**: Ensure data file is in `<project_name>/data/` directory

**Issue**: `KeyError: 'treated'`
- **Solution**: Check that column names match config settings

**Issue**: `No variation in treatment variable`
- **Solution**: Verify treatment timing varies across units

**Issue**: PySAL import error
- **Solution**: Install optional dependency: `pip install libpysal esda`

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📖 References

### Methodological Papers

- **Sun & Abraham (2021)**: "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects", *Journal of Econometrics*
- **Goodman-Bacon (2021)**: "Difference-in-differences with variation in treatment timing", *Journal of Econometrics*
- **Callaway & Sant'Anna (2021)**: "Difference-in-Differences with multiple time periods", *Journal of Econometrics*

### Software Documentation

- [PyFixest](https://s3alfisc.github.io/pyfixest/): Fast fixed effects estimation
- [Rasterio](https://rasterio.readthedocs.io/): Geospatial raster processing
- [PySAL](https://pysal.org/): Spatial analysis library

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Author**: Xiaomeng Wu  
**GitHub**: [@XiaomengWu0124](https://github.com/XiaomengWu0124)

For questions or collaboration inquiries, please open an issue on GitHub.

## 🙏 Acknowledgments

- Built with [PyFixest](https://github.com/s3alfisc/pyfixest) for fast fixed effects estimation
- Spatial analysis powered by [PySAL](https://github.com/pysal)
- Text processing using [scikit-learn](https://github.com/scikit-learn/scikit-learn) and [NLTK](https://github.com/nltk/nltk)

## 📊 Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{wu2025policy,
  author = {Wu, Xiaomeng},
  title = {Policy Evaluation Study Toolkit},
  year = {2025},
  url = {https://github.com/XiaomengWu0124/codesample}
}
```

---

<div align="center">

**⭐ Star this repository if you find it useful!**

[Report Bug](https://github.com/XiaomengWu0124/codesample/issues) · [Request Feature](https://github.com/XiaomengWu0124/codesample/issues)

</div>
