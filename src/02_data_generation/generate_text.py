"""
Generate Synthetic Text Data for Digital Transformation Study

Creates simulated 10-K excerpts mentioning digital technology adoption,
with realistic patterns for text mining demonstration.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config_loader import CONFIG, PATHS, setup_logger, set_random_seed

logger = setup_logger("text_generation")

# Text templates for digital transformation mentions
COMPLETION_TEMPLATES = [
    "In {year}, the Company completed implementation of cloud-based infrastructure. The digital transformation initiative was finalized in Q{quarter} {year}.",
    "The Company concluded its digital transformation program in {year}, successfully migrating to advanced analytics platforms.",
    "Digital infrastructure upgrade was completed during fiscal year {year}, enhancing operational efficiency.",
    "{year} marked the completion of our enterprise-wide digitalization project, implemented across all business units.",
    "The Board approved and the Company finished deployment of AI-driven systems in {month} {year}.",
    "Cloud migration was finalized in {year}, replacing legacy IT infrastructure.",
    "In {year}, we completed the rollout of digital tools company-wide, effective from {month} {year}.",
]

INPROGRESS_TEMPLATES = [
    "The Company is currently advancing its digital transformation initiative, commenced in {year}.",
    "Digital infrastructure modernization is underway as of {year}, with ongoing deployment across divisions.",
    "We are progressing on cloud adoption, started in {year} and expected to complete by {year_end}.",
    "The digitalization program is in progress, having begun in {quarter} {year}.",
    "AI and analytics integration is being conducted throughout {year}.",
]

VAGUE_TEMPLATES = [
    "The Company continues to evaluate opportunities in digital technology and artificial intelligence.",
    "We recognize the importance of digital transformation for future competitiveness.",
    "Management is exploring various technology solutions to enhance operations.",
    "The Company remains committed to technological innovation and modernization.",
    "Digital strategy development is a priority for the organization.",
]

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']

def generate_text_sample(gvkey: int, year: int, digital_year: float, 
                         text_type: str, seed: int) -> str:
    """Generate a single text excerpt"""
    np.random.seed(seed + gvkey + year)
    
    if text_type == 'completion':
        template = np.random.choice(COMPLETION_TEMPLATES)
        quarter = f"Q{np.random.randint(1, 5)}"
        month = np.random.choice(MONTHS)
        text = template.format(
            year=int(digital_year),
            quarter=quarter,
            month=month
        )
    elif text_type == 'inprogress':
        template = np.random.choice(INPROGRESS_TEMPLATES)
        quarter = f"Q{np.random.randint(1, 5)}"
        year_end = int(digital_year) + 1
        text = template.format(
            year=int(digital_year),
            quarter=quarter,
            year_end=year_end
        )
    else:  # vague
        template = np.random.choice(VAGUE_TEMPLATES)
        text = template
    
    # Add some context sentences before/after
    prefix = np.random.choice([
        "In our annual business review: ",
        "Management's discussion: ",
        "Strategic update: ",
        ""
    ])
    
    suffix = np.random.choice([
        " This represents a significant milestone for the Company.",
        " Additional details are provided in the technology section.",
        " We believe this positions us well for future growth.",
        ""
    ])
    
    return prefix + text + suffix

def generate_text_dataset(panel: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Generate text excerpts for firms with digital transformation mentions
    """
    logger.info("="*70)
    logger.info("TEXT DATA GENERATION")
    logger.info("="*70)
    
    seed = CONFIG['analysis']['seed']
    set_random_seed(seed)
    
    # Load panel if not provided
    if panel is None:
        panel_path = PATHS['data_processed'] / 'firm_panel.csv'
        panel = pd.read_csv(panel_path)
        logger.info(f"Loaded panel from {panel_path}")
    
    # Filter to firm-years with digital mentions
    text_data = panel[panel['has_digital_mention'] == 1].copy()
    logger.info(f"Creating text excerpts for {len(text_data)} firm-year observations")
    
    texts = []
    
    for idx, row in text_data.iterrows():
        gvkey = row['gvkey']
        year = row['year']
        digital_year = row['digital_year']
        
        # Determine text type based on timing
        rel_year = year - digital_year
        
        if rel_year < -0.5:
            # Before adoption: vague or in-progress
            text_type = np.random.choice(['vague', 'inprogress'], p=[0.7, 0.3])
        elif -0.5 <= rel_year <= 0.5:
            # Around adoption year: completion
            text_type = 'completion'
        else:
            # After adoption: completion or vague
            text_type = np.random.choice(['completion', 'vague'], p=[0.6, 0.4])
        
        text = generate_text_sample(gvkey, year, digital_year, text_type, seed)
        
        texts.append({
            'gvkey': gvkey,
            'fiscal_year': year,
            'text': text,
            'text_type': text_type,
            'true_digital_year': digital_year if not np.isnan(digital_year) else None,
            'has_explicit_date': text_type == 'completion'
        })
    
    text_df = pd.DataFrame(texts)
    
    # Add some random text excerpts for non-adopters
    non_adopters = panel[
        (panel['adopt_digital'] == 0) & 
        (panel['year'].isin([2015, 2016, 2017]))
    ].sample(n=min(50, len(panel[panel['adopt_digital']==0])), random_state=seed)
    
    for idx, row in non_adopters.iterrows():
        text = generate_text_sample(row['gvkey'], row['year'], np.nan, 'vague', seed)
        texts.append({
            'gvkey': row['gvkey'],
            'fiscal_year': row['year'],
            'text': text,
            'text_type': 'vague',
            'true_digital_year': None,
            'has_explicit_date': False
        })
    
    text_df = pd.DataFrame(texts)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEXT DATA SUMMARY")
    logger.info("="*70)
    logger.info(f"Total text excerpts: {len(text_df):,}")
    logger.info(f"Unique firms: {text_df['gvkey'].nunique():,}")
    logger.info(f"\nText type distribution:")
    logger.info(text_df['text_type'].value_counts())
    logger.info(f"\nExcerpts with explicit dates: {text_df['has_explicit_date'].sum()} ({text_df['has_explicit_date'].mean():.1%})")
    
    # Save
    if save:
        output_path = PATHS['data_processed'] / 'digital_text_sample.xlsx'
        text_df.to_excel(output_path, index=False)
        logger.info(f"\n✓ Saved text data to: {output_path}")
        
        # Also save as CSV for easier inspection
        csv_path = PATHS['data_processed'] / 'digital_text_sample.csv'
        text_df.to_csv(csv_path, index=False)
        logger.info(f"✓ Saved text data (CSV) to: {csv_path}")
    
    # Print examples
    logger.info("\n" + "="*70)
    logger.info("SAMPLE TEXT EXCERPTS")
    logger.info("="*70)
    
    for text_type in ['completion', 'inprogress', 'vague']:
        logger.info(f"\n--- Example: {text_type.upper()} ---")
        example = text_df[text_df['text_type'] == text_type].iloc[0] if len(text_df[text_df['text_type'] == text_type]) > 0 else None
        if example is not None:
            logger.info(f"Firm: {example['gvkey']}, Year: {example['fiscal_year']}")
            logger.info(f"Text: {example['text']}")
    
    return text_df

if __name__ == "__main__":
    # Load panel
    panel_path = PATHS['data_processed'] / 'firm_panel.csv'
    if panel_path.exists():
        panel = pd.read_csv(panel_path)
        text_df = generate_text_dataset(panel, save=True)
        
        print("\n" + "="*70)
        print("TEXT GENERATION COMPLETE")
        print("="*70)
    else:
        logger.error(f"Panel data not found at {panel_path}")
        logger.error("Run 02_synthetic_data/generate_panel.py first")
