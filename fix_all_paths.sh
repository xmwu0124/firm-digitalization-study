#!/bin/bash
# Complete path fix after reorganization

echo "======================================"
echo "Fixing all paths after reorganization"
echo "======================================"

# 1. Fix config_loader.py - config.yaml path
echo ""
echo "[1/4] Fixing config_loader.py..."
if [ -f "src/config_loader.py" ]; then
    cp src/config_loader.py src/config_loader.py.backup
    
    # Change config.yaml path from parent.parent to parent
    sed -i '' "s|config_path = Path(__file__).resolve().parent.parent / 'config.yaml'|config_path = Path(__file__).resolve().parent / 'config.yaml'|g" src/config_loader.py
    
    echo "✓ Fixed config_loader.py"
else
    echo "✗ config_loader.py not found"
fi

# 2. Fix all Python files - import statements
echo ""
echo "[2/4] Fixing import statements in all Python files..."

python_files=(
    "src/run_all.py"
    "src/01_data_construction/extract_digital_dates.py"
    "src/02_data_generation/generate_panel.py"
    "src/02_data_generation/generate_text.py"
    "src/03_analysis/did_analysis.py"
    "src/04_structural/io_structural_model.py"
)

for file in "${python_files[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "${file}.backup"
        
        # Fix: from utils.config_loader import -> from config_loader import
        sed -i '' 's/from utils\.config_loader import/from config_loader import/g' "$file"
        
        echo "✓ Fixed $file"
    else
        echo "✗ Not found: $file"
    fi
done

# 3. Fix sys.path in run_all.py specifically
echo ""
echo "[3/4] Fixing sys.path in run_all.py..."
if [ -f "src/run_all.py" ]; then
    # Make sure sys.path points to src directory
    sed -i '' 's|sys.path.append(str(Path(__file__).resolve().parent.parent))|sys.path.append(str(Path(__file__).resolve().parent))|g' src/run_all.py
    echo "✓ Fixed sys.path in run_all.py"
fi

# 4. Fix subdirectory scripts sys.path (they need to go up 2 levels)
echo ""
echo "[4/4] Fixing sys.path in subdirectory scripts..."

subdir_files=(
    "src/01_data_construction/extract_digital_dates.py"
    "src/02_data_generation/generate_panel.py"
    "src/02_data_generation/generate_text.py"
    "src/03_analysis/did_analysis.py"
    "src/04_structural/io_structural_model.py"
)

for file in "${subdir_files[@]}"; do
    if [ -f "$file" ]; then
        # These files need to go up 2 levels (from 0X_xxx/ to src/ to code_sample/)
        # Then add src to path
        # Replace the sys.path line
        sed -i '' '/sys.path.append.*parent.parent/d' "$file"
        
        # Add correct sys.path after imports
        # Find the line with "from pathlib import Path" and add sys.path after it
        awk '/from pathlib import Path/ && !done {print; print "sys.path.append(str(Path(__file__).resolve().parent.parent))"; done=1; next} 1' "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"
        
        echo "✓ Updated sys.path in $file"
    fi
done

echo ""
echo "======================================"
echo "All fixes complete!"
echo "======================================"
echo ""
echo "Verifying fixes..."
echo ""

# Verify config_loader
if grep -q "config_path = Path(__file__).resolve().parent / 'config.yaml'" src/config_loader.py; then
    echo "✓ config_loader.py: config path correct"
else
    echo "✗ config_loader.py: config path may need manual check"
fi

# Verify imports
if grep -q "from config_loader import" src/run_all.py; then
    echo "✓ run_all.py: import statement correct"
else
    echo "✗ run_all.py: import statement may need manual check"
fi

echo ""
echo "Testing configuration load..."
python3 << PYEOF
import sys
from pathlib import Path
sys.path.insert(0, 'src')
try:
    from config_loader import CONFIG, PATHS
    print("✓ Configuration loads successfully")
    print(f"  Project root: {PATHS['project_root']}")
except Exception as e:
    print(f"✗ Error loading config: {e}")
PYEOF

echo ""
echo "Backup files created with .backup extension"
echo "If everything works, you can remove them with: find src -name '*.backup' -delete"
