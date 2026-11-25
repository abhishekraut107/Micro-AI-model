"""
models/classifier.py
--------------------
Deep Residual Multi-Label Resistance Classifier with Masked BCE Loss,
Temperature-Scaled Probability Calibration, and Monte Carlo Dropout Uncertainty.

Framework Stage: Stage 3 (Model) & Stage 4 (Understand)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("classifier")


class ResidualBlock(nn.Module):
    """Residual dense block with LayerNorm, GELU, and Dropout."""

    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dropout1(self.act1(self.ln1(self.fc1(x))))
        out = self.dropout2(self.act2(self.ln2(self.fc2(out))))
        return out + residual


class AMRMultiLabelClassifier(nn.Module):
    """
    Deep Multi-Task Neural Network for 15-class Antimicrobial Resistance Prediction.
    """

    def __init__(
        self,
        input_dim: int,
        num_targets: int = 15,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_targets = num_targets

        # Initial Projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Residual Feature Extraction Backbone
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
            for _ in range(num_blocks)
        ])

        # Latent Representation Layer
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        # Multi-Label Classification Output Head
        self.head = nn.Linear(hidden_dim // 2, num_targets)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        features = self.bottleneck(h)
        logits = self.head(features)
        if return_features:
            return logits, features
        return logits


class MaskedBCELoss(nn.Module):
    """
    Masked Binary Cross-Entropy Loss with Logits.
    Computes loss strictly over clinically evaluated antibiotic targets, ignoring unperformed tests.
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        # Base BCE element-wise without reduction
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        # Apply boolean observation mask
        masked_bce = bce * masks.float()
        total_valid = masks.float().sum()
        if total_valid > 0:
            return masked_bce.sum() / total_valid
        return masked_bce.sum()


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling calibration module for multi-label logits.
    Learns an optimal temperature parameter T > 0 such that confidence matches accuracy.
    """

    def __init__(self, num_targets: int = 15):
        super().__init__()
        # Single global temperature or per-class temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Scaled logits: z / T
        return logits / torch.clamp(self.temperature, min=0.1, max=10.0)

    def fit(self, val_logits: torch.Tensor, val_targets: torch.Tensor, val_masks: torch.Tensor, max_iter: int = 50):
        """Finds optimal temperature using Adam to minimize masked NLL on validation set without PyTorch warnings."""
        optimizer = torch.optim.Adam([self.temperature], lr=0.05)
        loss_fn = MaskedBCELoss()

        for _ in range(max_iter):
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = loss_fn(scaled_logits, val_targets, val_masks)
            loss.backward()
            optimizer.step()

        logger.info(f"Optimal calibration temperature calibrated: T = {self.temperature.item():.3f}")


class CalibratedAMRPipeline:
    """
    High-level clinical inference pipeline combining the preprocessor,
    deep neural classifier, temperature scaler, and Monte Carlo dropout uncertainty estimator.
    """

    def __init__(
        self,
        model: AMRMultiLabelClassifier,
        scaler: TemperatureScaler,
        preprocessor,
        antibiotic_names: List[str],
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model.to(device)
        self.scaler = scaler.to(device)
        self.preprocessor = preprocessor
        self.antibiotic_names = antibiotic_names
        self.device = device

    def predict(
        self,
        sample_dict_or_df: Union[Dict, pd.DataFrame],
        mc_samples: int = 20,
        threshold: float = 0.50,
    ) -> Dict:
        """
        Runs calibrated inference with epistemic uncertainty estimation via Monte Carlo Dropout.
        """
        if isinstance(sample_dict_or_df, dict):
            df = pd.DataFrame([sample_dict_or_df])
        else:
            df = sample_dict_or_df

        X = self.preprocessor.transform(df)
        x_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

        # Enable Dropout for Monte Carlo Uncertainty Estimation
        self.model.train()
        probs_samples = []

        with torch.no_grad():
            for _ in range(mc_samples):
                raw_logits = self.model(x_tensor)
                scaled_logits = self.scaler(raw_logits)
                probs = torch.sigmoid(scaled_logits).cpu().numpy()
                probs_samples.append(probs)

        self.model.eval()

        # Compute empirical mean and epistemic uncertainty across samples
        probs_stacked = np.stack(probs_samples, axis=0)  # Shape: (mc_samples, batch_size, 15)
        mean_probs = np.mean(probs_stacked, axis=0)[0]    # Shape: (15,)
        epistemic_uncertainty = np.std(probs_stacked, axis=0)[0]  # Shape: (15,)

        results = {}
        for idx, abx in enumerate(self.antibiotic_names):
            p = float(mean_probs[idx])
            u = float(epistemic_uncertainty[idx])
            results[abx] = {
                "predicted_resistance_prob": round(p, 4),
                "epistemic_uncertainty": round(u, 4),
                "is_resistant_prediction": bool(p >= threshold),
                "confidence_interval_95": [
                    round(max(0.0, p - 1.96 * u), 4),
                    round(min(1.0, p + 1.96 * u), 4),
                ],
                "confidence_level": "HIGH" if u < 0.05 else "MODERATE" if u < 0.12 else "LOW (HIGH UNCERTAINTY)",
            }

        # Compute aggregate isolate metrics
        mean_isolate_uncertainty = float(np.mean(epistemic_uncertainty))
        predicted_resistant_count = sum(1 for v in results.values() if v["is_resistant_prediction"])
        
        return {
            "antibiotic_predictions": results,
            "mean_isolate_uncertainty": round(mean_isolate_uncertainty, 4),
            "predicted_resistant_classes_count": predicted_resistant_count,
            "predicted_mdr_flag": bool(predicted_resistant_count >= 3),
            "requires_clinical_escalation": bool(mean_isolate_uncertainty > 0.12 or results.get("IPM", {}).get("is_resistant_prediction", False)),
        }
