"""
Extract Digital Transformation Dates from Text
Adapted from feasibility_extractor.py

This module extracts digitalization adoption dates from firm text disclosures
using regex patterns and natural language processing.
"""

import re
from typing import Optional, Tuple, List, Dict
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config_loader import CONFIG, PATHS, setup_logger

logger = setup_logger("text_extraction")

# ========== REGEX PATTERNS ==========

YEAR_RE = re.compile(r"\b(20[0-2]\d)\b")  # 2000-2029

# Digital transformation keywords
DIGITAL_KEYWORDS = re.compile(
    r"\b(digital transformation|digitalization|digitization|"
    r"cloud computing|cloud migration|cloud adoption|"
    r"artificial intelligence|AI implementation|AI deployment|"
    r"machine learning|data analytics|big data|"
    r"automation|robotic process automation|RPA)\b",
    re.IGNORECASE
)

# Completion verbs
COMPLETE_VERBS = r"(?:completed?|finali[sz]ed|concluded|finished|implemented|deployed|launched)"
COMPLETE_PATTERN = re.compile(
    rf"{COMPLETE_VERBS}\b(?:\W+\w+){{0,8}}\W+(?:digital|cloud|AI|automation)",
    re.IGNORECASE | re.DOTALL
)

# In-progress indicators
INPROG_TERMS = re.compile(
    r"\b(in progress|underway|ongoing|being|conducted|commenced|started|"
    r"advancing|proceeding|rolling out)\b",
    re.IGNORECASE
)

# Future/planning terms to exclude
EXCLUDE_FUTURE = re.compile(
    r"\b(planned|plans|planning|expected|expects|will|would|"
    r"targeted|targets|aims|aimed|proposed|intends?)\b",
    re.IGNORECASE
)

# Date stamp patterns (e.g., "(January 15, 2018)" at start of sentence)
DATE_STAMP = re.compile(
    r"^\s*\((?:[A-Za-z]{3,9}\s+\d{1,2},\s+)?(20[0-2]\d)\)"
)

# Leading year pattern (e.g., "2018: We completed...")
LEADING_YEAR = re.compile(r"^\s*(20[0-2]\d)\s*[:\-]")

# ========== HELPER FUNCTIONS ==========

def split_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    if not isinstance(text, str) or not text.strip():
        return []
    return re.split(r"(?<=[.!?;])\s+", text)

def year_nearest_to(text: str, anchor_span: Tuple[int, int]) -> Optional[int]:
    """Find year closest to anchor position in text"""
    years = list(YEAR_RE.finditer(text))
    if not years:
        return None
    
    center = (anchor_span[0] + anchor_span[1]) // 2
    nearest = min(years, key=lambda m: abs(((m.start() + m.end()) // 2) - center))
    return int(nearest.group(1))

def detect_completed_from_sentence(s: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Detect if sentence mentions completed digital transformation
    Returns: (year, evidence_sentence)
    """
    # Exclude future/planning language
    if EXCLUDE_FUTURE.search(s):
        return None, None
    
    # Check for completion pattern
    m = COMPLETE_PATTERN.search(s)
    if not m:
        return None, None
    
    # Check for digital keywords nearby
    digital_m = DIGITAL_KEYWORDS.search(s)
    if not digital_m:
        return None, None
    
    # Priority 1: Date stamp at start
    ds = DATE_STAMP.match(s)
    if ds:
        return int(ds.group(1)), s.strip()
    
    # Priority 2: Leading year
    ly = LEADING_YEAR.match(s)
    if ly:
        return int(ly.group(1)), s.strip()
    
    # Priority 3: Nearest year to digital keyword
    y = year_nearest_to(s, (digital_m.start(), digital_m.end()))
    return (y, s.strip()) if y else (None, None)

def detect_inprogress_from_sentence(s: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Detect if sentence mentions in-progress digital transformation
    Returns: (year, evidence_sentence)
    """
    # Must have in-progress indicator
    if not INPROG_TERMS.search(s):
        return None, None
    
    # Must have digital keywords
    digital_m = DIGITAL_KEYWORDS.search(s)
    if not digital_m:
        return None, None
    
    # Priority 1: Date stamp
    ds = DATE_STAMP.match(s)
    if ds:
        return int(ds.group(1)), s.strip()
    
    # Priority 2: Leading year
    ly = LEADING_YEAR.match(s)
    if ly:
        return int(ly.group(1)), s.strip()
    
    # Priority 3: Nearest year
    y = year_nearest_to(s, (digital_m.start(), digital_m.end()))
    return (y, s.strip()) if y else (None, None)

def analyze_text(text: str) -> Dict[str, Optional[any]]:
    """
    Analyze text and extract digital transformation dates
    
    Returns dictionary with:
    - Digital_Completed (detected): Year of completion
    - Digital_InProgress (detected): Year of initiation
    - Evidence_Completed: Supporting sentence
    - Evidence_InProgress: Supporting sentence
    """
    if not isinstance(text, str) or not text.strip():
        return {
            'Digital_Completed': None,
            'Digital_InProgress': None,
            'Evidence_Completed': None,
            'Evidence_InProgress': None,
            'Confidence': 0.0
        }
    
    completed_candidates: List[Tuple[int, str]] = []
    inprog_candidates: List[Tuple[int, str]] = []
    
    for sentence in split_sentences(text):
        # Check for completion
        cy, cev = detect_completed_from_sentence(sentence)
        if cy is not None:
            completed_candidates.append((cy, sentence.strip()))
        
        # Check for in-progress
        py, pev = detect_inprogress_from_sentence(sentence)
        if py is not None:
            inprog_candidates.append((py, sentence.strip()))
    
    # Take earliest year for each category
    cy = cev = None
    if completed_candidates:
        completed_candidates.sort(key=lambda x: x[0])
        cy, cev = completed_candidates[0]
    
    py = pev = None
    if inprog_candidates:
        inprog_candidates.sort(key=lambda x: x[0])
        py, pev = inprog_candidates[0]
    
    # Confidence score
    confidence = 0.0
    if cy is not None:
        confidence = 0.9  # High confidence for completion
    elif py is not None:
        confidence = 0.6  # Medium confidence for in-progress
    
    return {
        'Digital_Completed': cy,
        'Digital_InProgress': py,
        'Evidence_Completed': cev,
        'Evidence_InProgress': pev,
        'Confidence': confidence
    }

def extract_digital_dates(
    data_path: Path = None,
    text_col: str = 'text',
    id_cols: List[str] = ['gvkey', 'fiscal_year'],
    save: bool = True
) -> pd.DataFrame:
    """
    Main extraction function
    
    Args:
        data_path: Path to text data (Excel or CSV)
        text_col: Column name containing text
        id_cols: Columns to keep as identifiers
        save: Whether to save results
    
    Returns:
        DataFrame with extracted dates and evidence
    """
    logger.info("="*70)
    logger.info("DIGITAL TRANSFORMATION DATE EXTRACTION")
    logger.info("="*70)
    
    # Load data
    if data_path is None:
        data_path = PATHS['data_processed'] / 'digital_text_sample.xlsx'
    
    logger.info(f"Loading text data from: {data_path}")
    
    if data_path.suffix == '.xlsx':
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)
    
    logger.info(f"  Loaded {len(df):,} text excerpts")
    
    # Extract dates
    logger.info("\nExtracting digital transformation dates...")
    
    results = []
    for idx, row in df.iterrows():
        text = row.get(text_col, '')
        
        # Extract dates
        extraction = analyze_text(text)
        
        # Build result row
        result = {col: row.get(col) for col in id_cols}
        result.update(extraction)
        
        # Add original text (truncated for display)
        result['Text_Preview'] = text[:200] + '...' if len(text) > 200 else text
        
        results.append(result)
        
        if (idx + 1) % 50 == 0:
            logger.info(f"  Processed {idx+1:,} / {len(df):,} excerpts")
    
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    logger.info("\n" + "="*70)
    logger.info("EXTRACTION SUMMARY")
    logger.info("="*70)
    
    n_completed = results_df['Digital_Completed'].notna().sum()
    n_inprogress = results_df['Digital_InProgress'].notna().sum()
    n_any = ((results_df['Digital_Completed'].notna()) | 
             (results_df['Digital_InProgress'].notna())).sum()
    
    logger.info(f"Total excerpts: {len(results_df):,}")
    logger.info(f"Excerpts with completed date: {n_completed} ({n_completed/len(results_df):.1%})")
    logger.info(f"Excerpts with in-progress date: {n_inprogress} ({n_inprogress/len(results_df):.1%})")
    logger.info(f"Excerpts with any date: {n_any} ({n_any/len(results_df):.1%})")
    
    logger.info(f"\nAverage confidence: {results_df['Confidence'].mean():.2f}")
    
    # Compare with ground truth if available
    if 'true_digital_year' in results_df.columns:
        # Create predicted year (prefer completed over in-progress)
        results_df['Predicted_Year'] = results_df['Digital_Completed'].fillna(
            results_df['Digital_InProgress']
        )
        
        # Accuracy calculation
        valid_truth = results_df['true_digital_year'].notna()
        if valid_truth.sum() > 0:
            correct = (
                (results_df.loc[valid_truth, 'Predicted_Year'] == 
                 results_df.loc[valid_truth, 'true_digital_year'])
            ).sum()
            
            accuracy = correct / valid_truth.sum()
            logger.info(f"\nAccuracy (vs ground truth): {accuracy:.1%}")
            logger.info(f"  Correct predictions: {correct} / {valid_truth.sum()}")
    
    # Save results
    if save:
        output_path = PATHS['data_processed'] / 'digital_dates_extracted.csv'
        results_df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Saved extraction results to: {output_path}")
    
    # Display examples
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE EXTRACTIONS")
    logger.info("="*70)
    
    # Show a few successful extractions
    successful = results_df[results_df['Digital_Completed'].notna()].head(3)
    for idx, row in successful.iterrows():
        logger.info(f"\n--- Example {idx+1} ---")
        logger.info(f"Firm: {row.get('gvkey')}, Year: {row.get('fiscal_year')}")
        logger.info(f"Extracted Date: {int(row['Digital_Completed'])}")
        logger.info(f"Evidence: {row['Evidence_Completed']}")
    
    return results_df

if __name__ == "__main__":
    results = extract_digital_dates(save=True)
    
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"\nExtracted dates from {len(results):,} text excerpts")
    print(f"Found {results['Digital_Completed'].notna().sum()} completion dates")
