#%%
"""
Main Runner Script
Executes complete research pipeline from data generation to final outputs
"""

import sys
from pathlib import Path
import subprocess
import time

sys.path.append(str(Path(__file__).resolve().parent))
from config_loader import CONFIG, PATHS, setup_logger

logger = setup_logger("main_runner")

SCRIPTS = [
    ("02_data_generation/generate_panel.py", "Generate Synthetic Panel Data"),
    ("02_data_generation/generate_text.py", "Generate Text Data"),
    ("01_data_construction/extract_digital_dates.py", "Extract Digital Dates from Text"),
    ("03_analysis/eda_descriptives.py", "Exploratory Data Analysis"),
    ("03_analysis/spatial_analysis.py", "Spatial Analysis"),
    ("03_analysis/did_analysis.py", "Difference-in-Differences Analysis"),
    ("04_structural/estimate_capitals.py", "Bayesian Capital Estimation"),
    ("04_structural/choice_model.py", "Discrete Choice Model"),
    ("04_structural/generate_all_outputs.py", "Generate Final Outputs"),
]

def run_script(script_path: str, description: str) -> bool:
    """Run a single Python script"""
    full_path = Path(__file__).parent / script_path
    
    if not full_path.exists():
        logger.warning(f"⚠️  Script not found: {script_path}")
        return False
    
    logger.info("="*70)
    logger.info(f"RUNNING: {description}")
    logger.info(f"Script: {script_path}")
    logger.info("="*70)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(full_path)],
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✓ {description} completed in {elapsed:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"✗ {description} failed with error: {str(e)}")
        return False

def run_all(skip_existing: bool = False):
    """Run complete pipeline"""
    logger.info("\n" + "="*70)
    logger.info("DIGITAL TRANSFORMATION STUDY - COMPLETE PIPELINE")
    logger.info("="*70)
    logger.info(f"\nProject: {CONFIG['project']['name']}")
    logger.info(f"Version: {CONFIG['project']['version']}")
    logger.info(f"Author: {CONFIG['project']['author']}")
    logger.info(f"\nTotal scripts: {len(SCRIPTS)}")
    
    overall_start = time.time()
    results = []
    
    for i, (script_path, description) in enumerate(SCRIPTS, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"STEP {i}/{len(SCRIPTS)}")
        logger.info(f"{'='*70}")
        
        success = run_script(script_path, description)
        results.append((description, success))
        
        if not success:
            logger.warning(f"\nScript {i} failed. Continuing with next step...")
            # Don't stop pipeline - some scripts may be optional
    
    # Summary
    overall_elapsed = time.time() - overall_start
    
    logger.info("\n" + "="*70)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("="*70)
    
    successes = sum(1 for _, s in results if s)
    failures = len(results) - successes
    
    logger.info(f"\nTotal time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
    logger.info(f"Successful: {successes}/{len(results)}")
    logger.info(f"Failed: {failures}/{len(results)}")
    
    logger.info("\nDetailed Results:")
    for desc, success in results:
        status = "✓" if success else "✗"
        logger.info(f"  {status} {desc}")
    
    if failures == 0:
        logger.info("\n🎉 All scripts completed successfully!")
        logger.info(f"\nOutputs available in:")
        logger.info(f"  - Figures: {PATHS['figures']}")
        logger.info(f"  - Tables: {PATHS['tables']}")
        logger.info(f"  - Data: {PATHS['data_processed']}")
    else:
        logger.warning(f"\n⚠️  {failures} script(s) encountered errors")
        logger.warning("Check individual logs for details")
    
    logger.info("\n" + "="*70)

def run_subset(start_step: int = 1, end_step: int = None):
    """Run a subset of the pipeline"""
    if end_step is None:
        end_step = len(SCRIPTS)
    
    subset = SCRIPTS[start_step-1:end_step]
    
    logger.info(f"\nRunning steps {start_step} to {end_step}:")
    for i, (script_path, description) in enumerate(subset, start_step):
        logger.info(f"  {i}. {description}")
    
    for i, (script_path, description) in enumerate(subset, start_step):
        run_script(script_path, description)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Digital Transformation Study Pipeline")
    parser.add_argument('--start', type=int, default=1, help='Start step number')
    parser.add_argument('--end', type=int, default=None, help='End step number')
    parser.add_argument('--list', action='store_true', help='List all steps')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable pipeline steps:")
        for i, (script_path, description) in enumerate(SCRIPTS, 1):
            print(f"  {i}. {description}")
            print(f"     {script_path}")
        sys.exit(0)
    
    if args.start > 1 or args.end is not None:
        run_subset(args.start, args.end)
    else:
        run_all()
