# 🚀 Quick Start Guide - 5 Minutes to Running Analysis

## Step 1: Setup (2 minutes)

```bash
# Navigate to project
cd digital_transformation_study

# Install dependencies
pip install pandas numpy matplotlib seaborn scipy pyyaml

# Optional (for advanced features)
pip install pyfixest  # Fast DiD estimation
pip install jax jaxlib numpyro  # Bayesian estimation
pip install libpysal esda  # Spatial analysis
```

## Step 2: Generate Data (1 minute)

```bash
python 02_synthetic_data/generate_panel.py
```

**Output**:
- ✅ `data/processed/firm_panel.csv` (200 firms × 10 years = 2,000 observations)
- ✅ `data/processed/firm_characteristics.csv` (firm-level data)

**What you'll see**:
```
SYNTHETIC DATA GENERATION
================================
Step 1: Generating firm characteristics...
  Created 200 firms
  Industry distribution:
    Manufacturing    60
    Technology      50
    Retail          40
    Finance         30
    Energy          20

Step 2: Assigning treatment timing...
  Treatment assignment: 145 adopters out of 200 firms
  
DATA SUMMARY
================================
Total observations: 2,000
Unique firms: 200
Treated observations: 945 (47.2%)
```

## Step 3: Run Analysis (2 minutes)

```bash
python 03_empirical_analysis/did_analysis.py
```

**Output**:
- ✅ Event study plot: `output/figures/event_study_log_revenue.png`
- ✅ Results table: `output/tables/event_study_results.csv`
- ✅ TWFE estimates: `output/tables/twfe_results.csv`

**What you'll see**:
```
DIFFERENCE-IN-DIFFERENCES ANALYSIS
====================================
Running event study for outcome: log_revenue
  Average treatment effect: 0.1548
  Pre-trend test p-value: 0.234

✓ Saved event study plot to: output/figures/event_study_log_revenue.png
✓ Saved results to: output/tables/event_study_results.csv

DID ANALYSIS COMPLETE
```

## Step 4: View Results

```bash
# Open the figure
open output/figures/event_study_log_revenue.png  # Mac
xdg-open output/figures/event_study_log_revenue.png  # Linux

# Check the table
cat output/tables/event_study_results.csv
```

---

## ✨ That's It!

You've just:
1. ✅ Generated a synthetic dataset with staggered treatment
2. ✅ Ran a difference-in-differences analysis
3. ✅ Created publication-quality figures and tables

---

## 🎯 What to Show in an Interview

### Demo Script (2 minutes)

"Let me show you a complete research project I built..."

```bash
# 1. Show project structure
tree -L 2 digital_transformation_study

# 2. Explain the story
cat README.md | head -50

# 3. Run the analysis
python run_all.py --list  # Show pipeline steps
python run_all.py --start 1 --end 3  # Run first 3 steps

# 4. Show results
ls -lh output/figures/
cat output/tables/summary_statistics.csv
```

**Key talking points**:
- "This integrates text mining, causal inference, and Bayesian estimation"
- "All code is adapted from my real research projects"
- "The framework is flexible - I can change it to any adoption/policy question"
- "Notice the modular structure and comprehensive logging"

---

## 🔧 Customize in 10 Minutes

Want to change the research topic?

### 1. Edit variable names in `config.yaml`:

```yaml
# Change from digital → automation
data:
  treatment_name: "automation_adoption"
  
# Adjust parameters
  adoption_start_year: 2012
  tech_industry_adoption_rate: 0.8
```

### 2. Update text patterns in `extract_digital_dates.py`:

```python
KEYWORDS = re.compile(
    r"\b(automation|robotics|AI deployment)\b",
    re.IGNORECASE
)
```

### 3. Rename outcome in analysis scripts:

```python
# In did_analysis.py
outcome = 'log_productivity'  # Instead of log_revenue
```

### 4. Re-run:

```bash
python run_all.py
```

**Done!** New topic, same methodology.

---

## 📊 Full Pipeline (Optional - 10 minutes)

Run everything:

```bash
python run_all.py
```

This executes:
1. ✅ Data generation (panel + text)
2. ✅ Text extraction
3. ✅ Descriptive analysis
4. ✅ Spatial analysis (if libpysal installed)
5. ✅ DiD estimation
6. ✅ Capital estimation (if JAX installed)
7. ✅ Choice model (if pyfixest installed)
8. ✅ Final output compilation

**Total time**: ~10 minutes (mostly Bayesian sampling if enabled)

---

## 🐛 Common Issues

### "ModuleNotFoundError"
```bash
# Install the missing package
pip install [package-name]

# Or skip optional features - core scripts work without them
```

### "FileNotFoundError"
```bash
# Make sure you're in the project root
cd digital_transformation_study

# Generate data first
python 02_synthetic_data/generate_panel.py
```

### "Figure doesn't show"
```bash
# Figures are saved to output/figures/
# Use image viewer or IDE preview
```

---

## 🎓 Learning Path

### Beginner Level
1. Run `generate_panel.py` - understand data structure
2. Run `did_analysis.py` - see DiD in action
3. Read the event study plot

### Intermediate Level
1. Modify `config.yaml` parameters
2. Add new variables to `generate_panel.py`
3. Customize plots in analysis scripts

### Advanced Level
1. Implement spatial analysis script
2. Add Bayesian capital estimation
3. Create policy simulation module

---

## 💡 Pro Tips

### Speed up testing:
```yaml
# In config.yaml
data:
  n_firms: 50    # Smaller sample for testing
  n_years: 5     # Fewer periods

analysis:
  mcmc_samples: 500  # Faster Bayesian sampling
```

### Debug mode:
```python
# In any script, change logger level
logger.setLevel(logging.DEBUG)
```

### Parallel execution:
```bash
# Run analyses independently (after data generation)
python 03_empirical_analysis/did_analysis.py &
python 03_empirical_analysis/spatial_analysis.py &
```

---

## 📞 Next Steps

1. **Today**: Run the quick start above
2. **Tomorrow**: Read through the code comments
3. **This week**: Customize for your research topic
4. **Before interview**: Practice the 2-minute demo

---

## ✅ Checklist

Before presenting this project:

- [ ] All scripts run without errors
- [ ] Output figures look professional
- [ ] README has your name/contact info
- [ ] Can explain each methodological choice
- [ ] Know where each script comes from (your original code)
- [ ] Can modify parameters on the fly
- [ ] Have backup slides with key figures

---

**You're ready! 🚀**

Any questions? See `PROJECT_OVERVIEW.md` for full documentation.
