"""
models/dataset.py
-----------------
Feature engineering, categorical encoding, target masking, and PyTorch Dataset
for multi-label AMR resistance modeling.

Framework Stage: Stage 3 (Model)
"""

import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ALL_ANTIBIOTICS = [
    "AMX/AMP", "AMC", "CZ", "FOX", "CTX/CRO", "IPM", "GEN", "AN",
    "Acide nalidixique", "ofx", "CIP", "C", "Co-trimoxazole", "Furanes", "colistine"
]

TARGET_COLS = [f"Target_{abx.replace('/', '_').replace(' ', '_')}" for abx in ALL_ANTIBIOTICS]

CATEGORICAL_FEATURES = ["Species", "Gender"]
NUMERICAL_FEATURES = ["Age", "Infection_Frequency"]
BINARY_FEATURES = ["Diabetes_Flag", "Hypertension_Flag", "Prior_Hospitalization"]


class AMRFeaturePreprocessor:
    """Preprocessor for tabular clinical features and bacterial species taxonomy."""

    def __init__(self):
        self.cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.num_scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "AMRFeaturePreprocessor":
        # Numerical feature imputation & scaling
        df_num = df[NUMERICAL_FEATURES].copy()
        df_num["Age"] = df_num["Age"].fillna(df_num["Age"].median())
        df_num["Infection_Frequency"] = df_num["Infection_Frequency"].fillna(1.0)
        self.num_scaler.fit(df_num)

        # Categorical feature encoding
        df_cat = df[CATEGORICAL_FEATURES].fillna("Unknown")
        self.cat_encoder.fit(df_cat)

        # Record full feature names
        cat_names = self.cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        self.feature_names = NUMERICAL_FEATURES + BINARY_FEATURES + cat_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("AMRFeaturePreprocessor must be fitted before transform.")

        df_num = df[NUMERICAL_FEATURES].copy()
        df_num["Age"] = df_num["Age"].fillna(self.num_scaler.mean_[0])
        df_num["Infection_Frequency"] = df_num["Infection_Frequency"].fillna(self.num_scaler.mean_[1])
        scaled_num = self.num_scaler.transform(df_num)

        bin_feats = df[BINARY_FEATURES].fillna(0).values.astype(np.float32)

        df_cat = df[CATEGORICAL_FEATURES].fillna("Unknown")
        encoded_cat = self.cat_encoder.transform(df_cat).astype(np.float32)

        X = np.hstack([scaled_num, bin_feats, encoded_cat]).astype(np.float32)
        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def save(self, filepath: Union[str, Path]):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "AMRFeaturePreprocessor":
        return joblib.load(filepath)


class AMRDataset(Dataset):
    """
    PyTorch Dataset for multi-label resistance classification.
    Returns:
    - X: feature vector tensor (float32)
    - y: 15-dim target tensor (float32, 0 or 1, NaNs filled with 0)
    - mask: 15-dim boolean mask tensor (True if target was observed/tested, False if missing)
    """

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None, masks: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
            self.masks = torch.tensor(masks, dtype=torch.bool)
        else:
            self.y = None
            self.masks = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if self.y is not None:
            return self.X[idx], self.y[idx], self.masks[idx]
        return (self.X[idx],)


def extract_targets_and_masks(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts binary target values and observation masks across the 15 antibiotic columns.
    NaNs in targets are replaced with 0.0 (masked out during loss computation).
    """
    y_raw = df[TARGET_COLS].values
    masks = ~np.isnan(y_raw)
    y_clean = np.where(masks, y_raw, 0.0).astype(np.float32)
    return y_clean, masks
