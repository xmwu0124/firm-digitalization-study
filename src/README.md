# Source Code

This directory contains all Python scripts for the digital transformation study.

## Directory Structure

- `01_data_construction/` - Data building and text mining modules
- `02_data_generation/` - Synthetic data generation scripts
- `03_analysis/` - Main econometric analyses
- `04_structural/` - IO structural estimation modules

## Configuration

Edit `config.yaml` to adjust analysis parameters including:
- Sample size and time periods
- Estimation windows and convergence tolerances
- Output formatting specifications

## Execution

Run the complete pipeline:
```bash
python run_all.py
```

Run individual modules:
```bash
python 02_data_generation/generate_panel.py
python 03_analysis/did_analysis.py
python 04_structural/io_structural_model.py
```

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

See main README.md for detailed installation instructions.
