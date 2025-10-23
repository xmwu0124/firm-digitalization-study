"""
Utility functions for loading configuration and setting up paths
"""
from pathlib import Path
import yaml
import logging
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_config(config_path: str = None) -> dict:
    """Load YAML configuration file"""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def setup_paths(config: dict) -> dict:
    """Convert relative paths to absolute Path objects"""
    paths = {}
    for key, val in config['paths'].items():
        paths[key] = PROJECT_ROOT / val
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths

def setup_logger(name: str = "digital_study", log_dir: Path = None) -> logging.Logger:
    """Setup logger with file and console handlers"""
    if log_dir is None:
        log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(log_dir / f"run_{timestamp}.log")
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def set_random_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import jax
        import jax.numpy as jnp
        # JAX uses a different random system
        pass
    except ImportError:
        pass

# Load config on import
CONFIG = load_config()
PATHS = setup_paths(CONFIG)
