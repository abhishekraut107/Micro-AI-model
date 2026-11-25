"""
models/anomaly_detector.py
--------------------------
Phenotypic Outlier & Emerging Anomaly Detector using Isolation Forest
and Latent Feature Space Outlier Scoring for AMR Surveillance.

Framework Stage: Stage 3 (Model) & Stage 5 (Decide)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("anomaly_detector")

ALL_ANTIBIOTICS = [
    "AMX/AMP", "AMC", "CZ", "FOX", "CTX/CRO", "IPM", "GEN", "AN",
    "Acide nalidixique", "ofx", "CIP", "C", "Co-trimoxazole", "Furanes", "colistine"
]
TARGET_COLS = [f"Target_{abx.replace('/', '_').replace(' ', '_')}" for abx in ALL_ANTIBIOTICS]


class AMRPhenotypicAnomalyDetector:
    """
    Detects unprecedented, rare, or outlier resistance combinations (potential emerging clones/superbugs).
    Combines Isolation Forest with species-conditional baseline deviation scores.
    """

    def __init__(self, contamination: float = 0.02, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.iso_forest = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.species_baselines: Dict[str, Dict[str, float]] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, preprocessor=None) -> "AMRPhenotypicAnomalyDetector":
        """Fits the anomaly detector on historical isolate antibiograms and features."""
        # 1. Target resistance signature matrix (fill unperformed tests with median 0.0)
        abx_matrix = df[TARGET_COLS].fillna(0.0).values

        # 2. Extract clinical features if preprocessor provided
        if preprocessor is not None:
            feat_matrix = preprocessor.transform(df)
            combined_matrix = np.hstack([abx_matrix, feat_matrix])
        else:
            combined_matrix = abx_matrix

        logger.info(f"Fitting Isolation Forest on {combined_matrix.shape[0]} historical isolates...")
        self.iso_forest.fit(combined_matrix)

        # 3. Calculate species conditional empirical resistance baselines for explanations
        for sp in df["Species"].dropna().unique():
            sp_df = df[df["Species"] == sp]
            self.species_baselines[sp] = {}
            for t_col, abx in zip(TARGET_COLS, ALL_ANTIBIOTICS):
                val_mean = float(sp_df[t_col].dropna().mean()) if t_col in sp_df else 0.0
                self.species_baselines[sp][abx] = val_mean

        self.is_fitted = True
        return self

    def score_isolate(
        self,
        df_row_or_dict: Union[pd.Series, Dict, pd.DataFrame],
        preprocessor=None,
    ) -> Dict:
        """
        Calculates the anomaly score and anomaly explanation for a single isolate.
        Returns:
        - anomaly_score: float in [0.0, 1.0] (higher means more anomalous)
        - is_anomaly: bool flag
        - rare_resistance_traits: list of unexpected resistance attributes
        """
        if isinstance(df_row_or_dict, dict):
            row_df = pd.DataFrame([df_row_or_dict])
        elif isinstance(df_row_or_dict, pd.Series):
            row_df = pd.DataFrame([df_row_or_dict.to_dict()])
        else:
            row_df = df_row_or_dict

        # Extract target matrix
        abx_vals = []
        for t_col in TARGET_COLS:
            v = row_df[t_col].iloc[0] if t_col in row_df.columns else 0.0
            abx_vals.append(0.0 if pd.isna(v) else float(v))
        abx_matrix = np.array([abx_vals], dtype=np.float32)

        if preprocessor is not None:
            feat_matrix = preprocessor.transform(row_df)
            combined = np.hstack([abx_matrix, feat_matrix])
        else:
            combined = abx_matrix

        # Raw score from IsolationForest: negative values mean anomalies
        raw_score = float(self.iso_forest.score_samples(combined)[0])
        # Normalize into an intuitive [0, 1] severity scale
        # Typically raw_score is between -0.7 (very anomalous) and -0.3 (very normal)
        normalized_score = float(np.clip(1.0 - (raw_score + 0.7) / 0.5, 0.0, 1.0))
        is_anomaly = bool(self.iso_forest.predict(combined)[0] == -1)

        # Generate Explainability Traits: which resistance traits are unprecedented for this species?
        sp = str(row_df["Species"].iloc[0]) if "Species" in row_df.columns else "Unknown"
        sp_base = self.species_baselines.get(sp, {})
        rare_traits = []

        for t_col, abx in zip(TARGET_COLS, ALL_ANTIBIOTICS):
            is_res = (row_df[t_col].iloc[0] == 1.0) if t_col in row_df.columns else False
            if is_res:
                base_prob = sp_base.get(abx, 0.5)
                # If observed resistant but historical baseline prevalence is < 15%, flag as rare
                if base_prob < 0.15:
                    rare_traits.append({
                        "antibiotic": abx,
                        "species": sp,
                        "species_baseline_prevalence": round(base_prob * 100, 1),
                        "description": f"Uncommon resistance to {abx} (typical species rate: {base_prob*100:.1f}%)"
                    })

        # Check for critical reserve antibiotic resistance
        has_colistin_res = bool(row_df.get("Target_colistine", [0]).iloc[0] == 1.0 if "Target_colistine" in row_df else False)
        has_carbapenem_res = bool(row_df.get("Target_IPM", [0]).iloc[0] == 1.0 if "Target_IPM" in row_df else False)

        critical_pathogen_flag = False
        if has_colistin_res and has_carbapenem_res:
            critical_pathogen_flag = True
            is_anomaly = True
            normalized_score = max(normalized_score, 0.95)
            rare_traits.append({
                "antibiotic": "IPM + colistine",
                "species": sp,
                "species_baseline_prevalence": 0.0,
                "description": "CRITICAL PAN-DRUG RESISTANCE: Dual Carbapenem & Colistin Non-Susceptibility!"
            })

        return {
            "anomaly_score": round(normalized_score, 4),
            "is_anomaly": is_anomaly,
            "critical_pathogen_flag": critical_pathogen_flag,
            "rare_resistance_traits": rare_traits,
            "recommended_action": "IMMEDIATE_LAB_CONFIRMATION" if is_anomaly else "ROUTINE_MONITORING",
        }

    def save(self, filepath: Union[str, Path]):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "AMRPhenotypicAnomalyDetector":
        return joblib.load(filepath)
