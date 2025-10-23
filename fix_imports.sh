#!/bin/bash

files=(
    "src/run_all.py"
    "src/02_data_generation/generate_panel.py"
    "src/02_data_generation/generate_text.py"
    "src/01_data_construction/extract_digital_dates.py"
    "src/03_analysis/did_analysis.py"
    "src/04_structural/io_structural_model.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        sed -i '' 's/from utils\.config_loader import/from config_loader import/g' "$file"
        echo "Fixed: $file"
    fi
done

echo "All imports fixed!"
