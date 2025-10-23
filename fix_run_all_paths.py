from pathlib import Path

# Read run_all.py
run_all = Path('src/run_all.py')
content = run_all.read_text()

# Update paths to match new structure
replacements = {
    "02_synthetic_data/generate_panel.py": "02_data_generation/generate_panel.py",
    "02_synthetic_data/generate_text.py": "02_data_generation/generate_text.py",
    "03_empirical_analysis/eda_descriptives.py": "03_analysis/eda_descriptives.py",
    "03_empirical_analysis/spatial_analysis.py": "03_analysis/spatial_analysis.py",
    "03_empirical_analysis/did_analysis.py": "03_analysis/did_analysis.py",
    "04_capital_estimation/estimate_capitals.py": "04_structural/estimate_capitals.py",
    "04_capital_estimation/io_structural_model.py": "04_structural/io_structural_model.py",
    "05_policy_simulation/": "04_structural/",
}

for old, new in replacements.items():
    content = content.replace(old, new)

# Write back
run_all.write_text(content)
print("✓ Updated run_all.py paths")
