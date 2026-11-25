"""
pipeline/clean_and_split.py
---------------------------
Data cleaning, clinical risk factor normalization, antibiotic resistance target encoding,
and prospective temporal train/test splitting for the AMR Surveillance System.

Author: AI-Native Capstone Project
Framework Stage: Stage 1 (Problem) -> Stage 2 (Bound) -> Stage 3 (Model Prep)
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import dateparser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("clean_and_split")

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "Bacteria_dataset_Multiresictance.csv"
DATA_DIR = BASE_DIR / "data"

# Clinical Antibiotic Classes Taxonomy (EUCAST / CDC Antimicrobial Classification)
ANTIBIOTIC_CLASSES = {
    "Penicillins": ["AMX/AMP"],
    "Beta-lactamase_inhibitors": ["AMC"],
    "Cephalosporins_1st_2nd": ["CZ", "FOX"],
    "Cephalosporins_3rd": ["CTX/CRO"],
    "Carbapenems": ["IPM"],
    "Aminoglycosides": ["GEN", "AN"],
    "Quinolones_Fluoroquinolones": ["Acide nalidixique", "ofx", "CIP"],
    "Phenicols": ["C"],
    "Folate_pathway": ["Co-trimoxazole"],
    "Nitrofurans": ["Furanes"],
    "Polymyxins_Reserve": ["colistine"],
}

ALL_ANTIBIOTICS = [abx for class_list in ANTIBIOTIC_CLASSES.values() for abx in class_list]

# Standardized Species Canonical Mappings
SPECIES_NORMALIZATION_MAP = {
    "escherichia coli": "Escherichia coli",
    "e. coli": "Escherichia coli",
    "e.coi": "Escherichia coli",
    "e.cli": "Escherichia coli",
    "klebsiella pneumoniae": "Klebsiella pneumoniae",
    "klbsiella pneumoniae": "Klebsiella pneumoniae",
    "klebsie.lla pneumoniae": "Klebsiella pneumoniae",
    "proteus mirabilis": "Proteus mirabilis",
    "prot.eus mirabilis": "Proteus mirabilis",
    "proeus mirabilis": "Proteus mirabilis",
    "protus mirabilis": "Proteus mirabilis",
    "enterobacteria spp.": "Enterobacteria spp.",
    "enter.bacteria spp.": "Enterobacteria spp.",
    "enteobacteria spp.": "Enterobacteria spp.",
    "citrobacter spp.": "Citrobacter spp.",
    "morganella morganii": "Morganella morganii",
    "serratia marcescens": "Serratia marcescens",
    "pseudomonas aeruginosa": "Pseudomonas aeruginosa",
    "acinetobacter baumannii": "Acinetobacter baumannii",
}


def normalize_species_name(raw_val: str) -> str:
    """Normalize bacterial species names, removing isolate tokens and correcting OCR/clerical typos."""
    if pd.isna(raw_val):
        return "Unknown"
    
    val = str(raw_val).strip()
    if val.lower() in ["?", "missing", "nan", "none", "unknown", ""]:
        return "Unknown"
    
    # Strip isolate prefix (e.g. S3413, S290)
    cleaned = re.sub(r"^[sS]\d+\s*", "", val).strip()
    
    # Handle known OCR punctuation glitches
    cleaned_lower = cleaned.lower()
    cleaned_lower = cleaned_lower.replace("prot.eus", "proteus")
    cleaned_lower = cleaned_lower.replace("enter.bacteria", "enterobacteria")
    cleaned_lower = cleaned_lower.replace("klebsie.lla", "klebsiella")
    
    if cleaned_lower in SPECIES_NORMALIZATION_MAP:
        return SPECIES_NORMALIZATION_MAP[cleaned_lower]
    
    # Fallback pattern matches
    if "coli" in cleaned_lower:
        return "Escherichia coli"
    elif "klebsiella" in cleaned_lower:
        return "Klebsiella pneumoniae"
    elif "proteus" in cleaned_lower:
        return "Proteus mirabilis"
    elif "enterobacter" in cleaned_lower:
        return "Enterobacteria spp."
    elif "citrobacter" in cleaned_lower:
        return "Citrobacter spp."
    elif "morganella" in cleaned_lower:
        return "Morganella morganii"
    elif "serratia" in cleaned_lower:
        return "Serratia marcescens"
    elif "pseudomonas" in cleaned_lower:
        return "Pseudomonas aeruginosa"
    elif "acinetobacter" in cleaned_lower:
        return "Acinetobacter baumannii"
    
    return cleaned


def parse_clinical_date(raw_val: str) -> Optional[pd.Timestamp]:
    """
    Robust clinical date parser handling ISO dates, French month abbreviations,
    and historical year typos (e.g., '1019' -> '2019').
    """
    if pd.isna(raw_val):
        return None
    val_str = str(raw_val).strip()
    if val_str.lower() in ["?", "missing", "error", "nan", "none", ""]:
        return None
    
    # Correct typographical year error 1019 -> 2019
    if "1019" in val_str:
        val_str = val_str.replace("1019", "2019")
        
    # Translate French month names commonly encountered in microbiology lab records
    french_months = {
        "janv": "jan", "fév": "feb", "fev": "feb", "mars": "mar",
        "avr": "apr", "mai": "may", "juin": "jun", "juil": "jul",
        "août": "aug", "aout": "aug", "sept": "sep", "oct": "oct",
        "nov": "nov", "déc": "dec", "dec": "dec"
    }
    for fr, en in french_months.items():
        val_str = re.sub(rf"\b{fr}\b", en, val_str, flags=re.IGNORECASE)
    
    try:
        if "-" in val_str and len(val_str.split("-")[0]) == 4:
            return pd.to_datetime(val_str)
        return pd.to_datetime(val_str, dayfirst=True)
    except Exception:
        try:
            parsed = dateparser.parse(val_str)
            if parsed:
                return pd.Timestamp(parsed)
        except Exception:
            pass
    return None


def extract_age_and_gender(val: str) -> Tuple[Optional[float], str]:
    """Extract age (float) and gender ('M', 'F', 'Unknown') from 'age/gender' column."""
    if pd.isna(val):
        return None, "Unknown"
    
    val_str = str(val).strip()
    match = re.search(r"(\d+)\s*/\s*([A-Za-z]+)", val_str)
    if match:
        age = float(match.group(1))
        gender_char = match.group(2).strip().upper()
        if gender_char.startswith("M"):
            gender = "M"
        elif gender_char.startswith("F"):
            gender = "F"
        else:
            gender = "Unknown"
        return age, gender
    
    # Check if only age or only gender is present
    if val_str.isdigit():
        return float(val_str), "Unknown"
    if val_str.upper() in ["M", "MALE"]:
        return None, "M"
    if val_str.upper() in ["F", "FEMALE"]:
        return None, "F"
        
    return None, "Unknown"


def encode_antibiotic_response(val: str, treat_intermediate_as_resistant: bool = True) -> float:
    """
    Encode phenotypic antibiotic AST results to standard binary target:
    1 = Resistant (or Non-Susceptible)
    0 = Susceptible
    NaN = Missing / untested / corrupt
    """
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip().lower()
    
    if val_str in ["r"]:
        return 1.0
    elif val_str in ["s"]:
        return 0.0
    elif val_str in ["i", "intermediate"]:
        return 1.0 if treat_intermediate_as_resistant else 0.0
    elif val_str in ["?", "missing", "error", "nan", "none"]:
        return np.nan
    return np.nan


def compute_multidrug_resistance(row: pd.Series) -> Tuple[int, int]:
    """
    Compute:
    - resistant_class_count: number of distinct antimicrobial classes with at least one resistant marker.
    - is_mdr: binary 1/0 indicator (resistant to >= 3 antimicrobial classes).
    """
    resistant_classes = 0
    total_evaluable_classes = 0
    
    for class_name, abx_list in ANTIBIOTIC_CLASSES.items():
        class_targets = [f"Target_{abx.replace('/', '_').replace(' ', '_')}" for abx in abx_list]
        class_res = [row[t] for t in class_targets if t in row and pd.notna(row[t])]
        if class_res:
            total_evaluable_classes += 1
            if any(r == 1.0 for r in class_res):
                resistant_classes += 1
                
    is_mdr = 1 if resistant_classes >= 3 else 0
    return resistant_classes, is_mdr


def run_data_cleaning_pipeline(
    input_path: Path = RAW_DATA_PATH,
    output_dir: Path = DATA_DIR,
    train_ratio: float = 0.80,
    treat_intermediate_as_resistant: bool = True,
) -> Dict:
    """
    Executes full end-to-end data cleaning, standardization, target encoding,
    and prospective temporal train/test split.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading raw microbiology dataset from: {input_path}")
    df_raw = pd.read_csv(input_path)
    logger.info(f"Loaded raw records: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")

    df = df_raw.copy()

    # 1. Anonymization & De-identification (HIPAA/GDPR Data Governance)
    # Retain clinical identifier 'ID', drop direct PII names, emails, physical addresses
    pii_columns = ["Name", "Email", "Address"]
    df = df.drop(columns=[c for c in pii_columns if c in df.columns])

    # 2. Species Normalization
    logger.info("Normalizing bacterial species taxonomy...")
    df["Species"] = df["Souches"].apply(normalize_species_name)

    # 3. Demographics Extraction
    logger.info("Parsing demographic risk factors (age/gender)...")
    age_gender_parsed = df["age/gender"].apply(extract_age_and_gender)
    df["Age"] = age_gender_parsed.apply(lambda x: x[0])
    df["Gender"] = age_gender_parsed.apply(lambda x: x[1])

    # Impute missing Age with clinical median
    median_age = df["Age"].median()
    df["Age_Imputed"] = df["Age"].fillna(median_age)

    # 4. Clinical Comorbidities & Risk Factor Normalization
    logger.info("Standardizing clinical risk factors (Diabetes, Hypertension, Hospital_before, Infection_Freq)...")
    
    def parse_binary_risk(val):
        if pd.isna(val):
            return 0
        v = str(val).strip().lower()
        return 1 if v in ["yes", "true", "1", "1.0", "y"] else 0

    df["Diabetes_Flag"] = df["Diabetes"].apply(parse_binary_risk)
    df["Hypertension_Flag"] = df["Hypertension"].apply(parse_binary_risk)
    df["Prior_Hospitalization"] = df["Hospital_before"].apply(parse_binary_risk)

    def parse_infection_freq(val):
        try:
            v_float = float(val)
            return max(0.0, v_float)
        except (ValueError, TypeError):
            return np.nan

    df["Infection_Frequency"] = df["Infection_Freq"].apply(parse_infection_freq)
    median_freq = df["Infection_Frequency"].median()
    df["Infection_Frequency"] = df["Infection_Frequency"].fillna(median_freq if not pd.isna(median_freq) else 1.0)

    # 5. Target Antibiotic Outcomes Binary Encoding
    logger.info("Encoding 15 antibiotic phenotypic targets...")
    for abx in ALL_ANTIBIOTICS:
        target_col = f"Target_{abx.replace('/', '_').replace(' ', '_')}"
        df[target_col] = df[abx].apply(
            lambda x: encode_antibiotic_response(x, treat_intermediate_as_resistant)
        )

    # 6. Multi-Drug Resistance (MDR) & Phenotype Complexity Metrics
    logger.info("Computing Multi-Drug Resistance (MDR) phenotypes...")
    mdr_results = df.apply(compute_multidrug_resistance, axis=1)
    df["Resistant_Class_Count"] = [res[0] for res in mdr_results]
    df["Is_MDR"] = [res[1] for res in mdr_results]

    # 7. Date Parsing & Prospective Temporal Partitioning
    logger.info("Normalizing collection dates and constructing prospective split...")
    df["Parsed_Date"] = df["Collection_Date"].apply(parse_clinical_date)
    
    valid_dates_mask = df["Parsed_Date"].notna()
    df_dated = df[valid_dates_mask].copy()
    df_undated = df[~valid_dates_mask].copy()

    # Sort strictly chronologically to prevent prospective future leakage
    df_dated = df_dated.sort_values(by="Parsed_Date").reset_index(drop=True)

    n_dated = len(df_dated)
    n_train = int(n_dated * train_ratio)
    
    train_df = df_dated.iloc[:n_train].copy()
    test_df = df_dated.iloc[n_train:].copy()

    train_min_date = train_df["Parsed_Date"].min().strftime("%Y-%m-%d")
    train_max_date = train_df["Parsed_Date"].max().strftime("%Y-%m-%d")
    test_min_date = test_df["Parsed_Date"].min().strftime("%Y-%m-%d")
    test_max_date = test_df["Parsed_Date"].max().strftime("%Y-%m-%d")

    logger.info(f"Temporal Train Split: {len(train_df)} isolates ({train_min_date} to {train_max_date})")
    logger.info(f"Temporal Test Split: {len(test_df)} isolates ({test_min_date} to {test_max_date})")
    logger.info(f"Undated Cohort: {len(df_undated)} isolates reserved without temporal leak.")

    # 8. Save Artifacts to Parquet and CSV
    logger.info("Exporting cleaned splits...")
    
    # Save full cleaned, train, test, and undated
    full_csv = output_dir / "cleaned_full.csv"
    train_csv = output_dir / "train.csv"
    test_csv = output_dir / "test.csv"
    undated_csv = output_dir / "undated_cohort.csv"

    df.to_csv(full_csv, index=False)
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    df_undated.to_csv(undated_csv, index=False)

    # Parquet export if engine available
    has_parquet = False
    try:
        df.to_parquet(output_dir / "cleaned_full.parquet", index=False)
        train_df.to_parquet(output_dir / "train.parquet", index=False)
        test_df.to_parquet(output_dir / "test.parquet", index=False)
        has_parquet = True
    except Exception as e:
        logger.warning(f"Parquet export skipped ({e}). CSV formats saved successfully.")

    # 9. Summary Metadata for Tracking & Seven-Stage Validation
    metadata = {
        "total_raw_records": len(df_raw),
        "total_cleaned_records": len(df),
        "dated_records": n_dated,
        "undated_records": len(df_undated),
        "train_records": len(train_df),
        "test_records": len(test_df),
        "train_date_range": [train_min_date, train_max_date],
        "test_date_range": [test_min_date, test_max_date],
        "species_distribution": df["Species"].value_counts().to_dict(),
        "mdr_prevalence_train": float(train_df["Is_MDR"].mean()),
        "mdr_prevalence_test": float(test_df["Is_MDR"].mean()),
        "antibiotics_evaluated": ALL_ANTIBIOTICS,
        "has_parquet": has_parquet,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Data cleaning pipeline successfully completed! Metadata written to {metadata_path}")
    return metadata


if __name__ == "__main__":
    run_data_cleaning_pipeline()
