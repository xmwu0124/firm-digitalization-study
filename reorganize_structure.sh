#!/bin/bash
# reorganize_structure.sh
# 
# Project reorganization script for digital_transformation_study
# Author: Research Team
# Date: October 2025
#
# Purpose: Restructure flat directory into organized hierarchy with
#          separate documentation and source code directories
#
# Usage: bash reorganize_structure.sh

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
PROJECT_ROOT="${HOME}/Dropbox/CODE_SAMPLE"
BACKUP_DIR="${PROJECT_ROOT}_backup_$(date +%Y%m%d_%H%M%S)"

# Color codes for output (optional, can be removed if terminal doesn't support)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo "[INFO] $1"
}

log_success() {
    echo "[SUCCESS] $1"
}

log_warning() {
    echo "[WARNING] $1"
}

log_error() {
    echo "[ERROR] $1" >&2
}

# Main reorganization function
main() {
    log_info "Starting project reorganization"
    log_info "Project root: ${PROJECT_ROOT}"
    
    # Verify project directory exists
    if [ ! -d "${PROJECT_ROOT}" ]; then
        log_error "Project directory not found: ${PROJECT_ROOT}"
        log_error "Please update PROJECT_ROOT variable in script"
        exit 1
    fi
    
    # Create backup
    log_info "Creating backup at ${BACKUP_DIR}"
    cp -r "${PROJECT_ROOT}" "${BACKUP_DIR}"
    log_success "Backup created"
    
    # Change to project directory
    cd "${PROJECT_ROOT}"
    
    # Step 1: Create directory structure
    create_directory_structure
    
    # Step 2: Move documentation files
    move_documentation_files
    
    # Step 3: Move source code files
    move_source_files
    
    # Step 4: Create subdirectory README files
    create_subdirectory_readme_files
    
    # Step 5: Update import paths
    update_import_paths
    
    # Step 6: Clean up
    cleanup_empty_directories
    
    # Step 7: Display final structure
    display_final_structure
    
    log_success "Reorganization complete"
    log_info "Backup saved at: ${BACKUP_DIR}"
}

# Create directory structure
create_directory_structure() {
    log_info "Creating directory structure"
    
    mkdir -p docs/io_model
    mkdir -p src/01_data_construction
    mkdir -p src/02_data_generation
    mkdir -p src/03_analysis
    mkdir -p src/04_structural
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p output/figures
    mkdir -p output/tables
    mkdir -p output/logs
    
    log_success "Directory structure created"
}

# Move documentation files
move_documentation_files() {
    log_info "Moving documentation files"
    
    # Main documentation files
    local doc_files=(
        "INDEX.md"
        "QUICK_START.md"
        "FINAL_DELIVERY.md"
        "CHECKLIST.md"
        "PROJECT_OVERVIEW.md"
        "implementation_guide.md"
        "project_structure.md"
        "sample_structure.md"
        "README_TEMPLATE.md"
    )
    
    for file in "${doc_files[@]}"; do
        if [ -f "${file}" ]; then
            mv "${file}" docs/
            log_success "Moved ${file} to docs/"
        else
            log_warning "File not found: ${file}"
        fi
    done
    
    # IO model documentation
    for file in IO_MODEL_*.md; do
        if [ -f "${file}" ]; then
            mv "${file}" docs/io_model/
            log_success "Moved ${file} to docs/io_model/"
        fi
    done
    
    log_success "Documentation files moved"
}

# Move source code files
move_source_files() {
    log_info "Moving source code files"
    
    # Data construction scripts
    if [ -f "extract_digital_dates.py" ]; then
        mv extract_digital_dates.py src/01_data_construction/
        log_success "Moved extract_digital_dates.py"
    fi
    
    # Data generation scripts
    local datagen_files=("generate_panel.py" "generate_text.py")
    for file in "${datagen_files[@]}"; do
        if [ -f "${file}" ]; then
            mv "${file}" src/02_data_generation/
            log_success "Moved ${file}"
        fi
    done
    
    # Analysis scripts
    if [ -f "did_analysis.py" ]; then
        mv did_analysis.py src/03_analysis/
        log_success "Moved did_analysis.py"
    fi
    
    # Structural estimation scripts
    if [ -f "io_structural_model.py" ]; then
        mv io_structural_model.py src/04_structural/
        log_success "Moved io_structural_model.py"
    fi
    
    # Configuration and utility files
    local config_files=("config.yaml" "config_loader.py" "run_all.py" "requirements.txt")
    for file in "${config_files[@]}"; do
        if [ -f "${file}" ]; then
            mv "${file}" src/
            log_success "Moved ${file} to src/"
        fi
    done
    
    log_success "Source code files moved"
}

# Create README files for subdirectories
create_subdirectory_readme_files() {
    log_info "Creating subdirectory README files"
    
    # src/README.md
    cat > src/README.md << 'EOF'
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
EOF
    
    # docs/README.md
    cat > docs/README.md << 'EOF'
# Documentation

Complete documentation for the digital transformation research project.

## Quick Start Guide

1. **INDEX.md** - Complete navigation guide and project overview
2. **QUICK_START.md** - Five-minute tutorial for running analyses
3. **FINAL_DELIVERY.md** - Project summary and deliverables
4. **CHECKLIST.md** - Pre-submission verification checklist

## Specialized Documentation

- `io_model/` - Industrial organization structural model documentation
  - IO_MODEL_QUICKSTART.md - Quick start guide for IO estimation
  - IO_MODEL_EXPLAINED.md - Detailed technical specification

## Additional Resources

- implementation_guide.md - Implementation details
- project_structure.md - Directory structure reference
- README_TEMPLATE.md - Template for new projects

## Methodological References

Consult the main README.md for citations to relevant econometric papers and software documentation.
EOF
    
    log_success "Subdirectory README files created"
}

# Update import paths in Python files
update_import_paths() {
    log_info "Updating import paths in Python files"
    
    local python_files=(
        "src/01_data_construction/extract_digital_dates.py"
        "src/02_data_generation/generate_panel.py"
        "src/02_data_generation/generate_text.py"
        "src/03_analysis/did_analysis.py"
        "src/04_structural/io_structural_model.py"
    )
    
    for file in "${python_files[@]}"; do
        if [ -f "${file}" ]; then
            # Backup original file
            cp "${file}" "${file}.bak"
            
            # Update import statement
            # From: from utils.config_loader import
            # To: from config_loader import
            sed -i.tmp 's/from utils\.config_loader import/from config_loader import/g' "${file}"
            
            # Remove temporary file
            rm -f "${file}.tmp"
            
            log_success "Updated imports in ${file}"
        else
            log_warning "File not found: ${file}"
        fi
    done
    
    log_success "Import paths updated"
}

# Clean up empty directories
cleanup_empty_directories() {
    log_info "Cleaning up empty directories"
    
    # Remove utils directory if empty
    if [ -d "utils" ]; then
        rmdir utils 2>/dev/null && log_success "Removed empty utils directory" || log_info "utils directory not empty or not found"
    fi
    
    # Remove mnt directory if exists and empty (optional)
    if [ -d "mnt" ]; then
        log_info "mnt directory found - preserved (manual deletion if desired)"
    fi
    
    log_success "Cleanup complete"
}

# Display final directory structure
display_final_structure() {
    log_info "Final directory structure:"
    echo ""
    
    if command -v tree &> /dev/null; then
        tree -L 2 -I 'mnt|*.pyc|__pycache__|*.bak' -C
    else
        log_info "tree command not available, using ls"
        echo "Project root:"
        ls -1
        echo ""
        echo "docs/:"
        ls -1 docs/
        echo ""
        echo "src/:"
        ls -1 src/
    fi
    
    echo ""
}

# Error handling
trap 'log_error "Script failed at line $LINENO"' ERR

# Run main function
main

# Exit successfully
exit 0
