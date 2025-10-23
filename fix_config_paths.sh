#!/bin/bash

echo "Fixing config_loader.py paths..."

# Backup
cp src/config_loader.py src/config_loader.py.original

# Fix config.yaml path
sed -i '' "s|config_path = Path(__file__).resolve().parent.parent / 'config.yaml'|config_path = Path(__file__).resolve().parent / 'config.yaml'|g" src/config_loader.py

echo "✓ Fixed config.yaml path"

# Verify
grep "config_path = Path" src/config_loader.py

echo ""
echo "Testing..."
python3 -c "import sys; sys.path.insert(0, 'src'); from config_loader import CONFIG; print('✓ Config loaded successfully')"
