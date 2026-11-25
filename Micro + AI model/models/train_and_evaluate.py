"""
models/train_and_evaluate.py
----------------------------
End-to-end training, temperature calibration, anomaly detector fitting,
and prospective temporal test evaluation for the AMR Multi-Label Resistance System.

Framework Stage: Stage 3 (Model) & Stage 7 (Measure & Learn)
"""

import os
import json
import logging
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss

try:
    from models.dataset import (
        AMRFeaturePreprocessor,
        AMRDataset,
        extract_targets_and_masks,
        ALL_ANTIBIOTICS,
        TARGET_COLS,
    )
    from models.classifier import (
        AMRMultiLabelClassifier,
        MaskedBCELoss,
        TemperatureScaler,
        CalibratedAMRPipeline,
    )
    from models.anomaly_detector import AMRPhenotypicAnomalyDetector
except (ImportError, ModuleNotFoundError):
    from dataset import (
        AMRFeaturePreprocessor,
        AMRDataset,
        extract_targets_and_masks,
        ALL_ANTIBIOTICS,
        TARGET_COLS,
    )
    from classifier import (
        AMRMultiLabelClassifier,
        MaskedBCELoss,
        TemperatureScaler,
        CalibratedAMRPipeline,
    )
    from anomaly_detector import AMRPhenotypicAnomalyDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("train_and_eval")

DATA_DIR = BASE_DIR / "data"
SAVED_DIR = BASE_DIR / "models" / "saved"


def compute_expected_calibration_error(probs: np.ndarray, targets: np.ndarray, masks: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) across all observed antibiotic predictions.
    ECE measures the absolute difference between predicted confidence and true positive rate.
    """
    valid_probs = probs[masks]
    valid_targets = targets[masks]
    if len(valid_probs) == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_assignments = np.digitize(valid_probs, bins) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    ece = 0.0
    total_samples = len(valid_probs)
    for b in range(n_bins):
        in_bin = bin_assignments == b
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(valid_targets[in_bin])
            bin_conf = np.mean(valid_probs[in_bin])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)
    return float(ece)


def train_model(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_blocks: int = 2,
    device_name: str = "cpu",
) -> Dict:
    """Trains the multi-label neural classifier and fits calibration and anomaly modules."""
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)

    # 1. Load Data
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing splits in {DATA_DIR}. Run pipeline/clean_and_split.py first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    logger.info(f"Loaded train cohort ({len(train_df)} isolates) and test cohort ({len(test_df)} isolates).")

    # 2. Partition Train into Train/Val for calibration (Temporal split: 85% train, 15% val)
    n_train = int(len(train_df) * 0.85)
    tr_df = train_df.iloc[:n_train].reset_index(drop=True)
    val_df = train_df.iloc[n_train:].reset_index(drop=True)
    logger.info(f"Training split: {len(tr_df)} isolates | Calibration Val split: {len(val_df)} isolates.")

    # 3. Fit Feature Preprocessor
    preprocessor = AMRFeaturePreprocessor()
    X_tr = preprocessor.fit_transform(tr_df)
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)

    y_tr, m_tr = extract_targets_and_masks(tr_df)
    y_val, m_val = extract_targets_and_masks(val_df)
    y_test, m_test = extract_targets_and_masks(test_df)

    input_dim = X_tr.shape[1]
    num_targets = len(ALL_ANTIBIOTICS)
    logger.info(f"Feature vector dimensionality: {input_dim} | Target channels: {num_targets}")

    # 4. PyTorch DataLoaders
    train_ds = AMRDataset(X_tr, y_tr, m_tr)
    val_ds = AMRDataset(X_val, y_val, m_val)
    test_ds = AMRDataset(X_test, y_test, m_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 5. Initialize Model, Loss, Optimizer
    model = AMRMultiLabelClassifier(
        input_dim=input_dim,
        num_targets=num_targets,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=0.25,
    ).to(device)

    loss_fn = MaskedBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 6. Training Loop with Early Stopping
    best_val_loss = float("inf")
    best_model_state = None

    logger.info(f"Starting neural training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y_b, m_b in train_loader:
            x_b, y_b, m_b = x_b.to(device), y_b.to(device), m_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = loss_fn(logits, y_b, m_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_b)

        scheduler.step()
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b, m_b in val_loader:
                x_b, y_b, m_b = x_b.to(device), y_b.to(device), m_b.to(device)
                logits = model(x_b)
                loss = loss_fn(logits, y_b, m_b)
                val_loss += loss.item() * len(x_b)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"Epoch [{epoch:02d}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Load best weights
    model.load_state_dict(best_model_state)
    logger.info(f"Restored best model with validation loss: {best_val_loss:.4f}")

    # 7. Post-Hoc Temperature Calibration
    logger.info("Fitting post-hoc Temperature Scaler on calibration set...")
    model.eval()
    val_x_all = torch.tensor(X_val, dtype=torch.float32, device=device)
    val_y_all = torch.tensor(y_val, dtype=torch.float32, device=device)
    val_m_all = torch.tensor(m_val, dtype=torch.bool, device=device)

    with torch.no_grad():
        val_logits_all = model(val_x_all)

    scaler = TemperatureScaler(num_targets=num_targets).to(device)
    scaler.fit(val_logits_all, val_y_all, val_m_all)

    # 8. Fit AMRPhenotypicAnomalyDetector on Training Data
    logger.info("Fitting Phenotypic Anomaly Detector on historical cohort...")
    anomaly_detector = AMRPhenotypicAnomalyDetector(contamination=0.02)
    anomaly_detector.fit(train_df, preprocessor=preprocessor)

    # 9. Prospective Temporal Evaluation on Test Split
    logger.info("Evaluating calibrated model on prospective test cohort (2024-2025)...")
    test_x_all = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        raw_test_logits = model(test_x_all)
        calibrated_test_logits = scaler(raw_test_logits)
        test_probs = torch.sigmoid(calibrated_test_logits).cpu().numpy()

    # Per-antibiotic evaluation metrics
    per_abx_metrics = {}
    aurocs, pr_aucs, briers, f1s = [], [], [], []

    for idx, abx in enumerate(ALL_ANTIBIOTICS):
        mask_col = m_test[:, idx]
        y_true = y_test[mask_col, idx]
        y_pred_prob = test_probs[mask_col, idx]

        if len(y_true) >= 30 and len(np.unique(y_true)) > 1:
            try:
                auc = float(roc_auc_score(y_true, y_pred_prob))
                pr_auc = float(average_precision_score(y_true, y_pred_prob))
                brier = float(brier_score_loss(y_true, y_pred_prob))
                f1 = float(f1_score(y_true, (y_pred_prob >= 0.5).astype(int), zero_division=0))

                per_abx_metrics[abx] = {
                    "test_evaluated_count": int(len(y_true)),
                    "resistance_rate": float(np.mean(y_true)),
                    "auroc": round(auc, 4),
                    "pr_auc": round(pr_auc, 4),
                    "brier_score": round(brier, 4),
                    "f1_score": round(f1, 4),
                }
                aurocs.append(auc)
                pr_aucs.append(pr_auc)
                briers.append(brier)
                f1s.append(f1)
            except Exception as e:
                logger.warning(f"Metric calculation failed for {abx}: {e}")

    # Expected Calibration Error (ECE)
    overall_ece = compute_expected_calibration_error(test_probs, y_test, m_test)

    summary_metrics = {
        "macro_auroc": round(float(np.mean(aurocs)), 4) if aurocs else 0.0,
        "macro_pr_auc": round(float(np.mean(pr_aucs)), 4) if pr_aucs else 0.0,
        "macro_brier_score": round(float(np.mean(briers)), 4) if briers else 0.0,
        "macro_f1_score": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "expected_calibration_error_ece": round(overall_ece, 4),
        "optimal_temperature": round(float(scaler.temperature.item()), 4),
        "per_antibiotic": per_abx_metrics,
    }

    # 10. Save All Serialized Artifacts
    logger.info(f"Saving serialized artifacts to {SAVED_DIR}...")
    torch.save(model.state_dict(), SAVED_DIR / "classifier.pt")
    torch.save(scaler.state_dict(), SAVED_DIR / "scaler.pt")
    preprocessor.save(SAVED_DIR / "preprocessor.joblib")
    anomaly_detector.save(SAVED_DIR / "anomaly_detector.joblib")

    with open(SAVED_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    logger.info("Training and evaluation successfully completed!")
    logger.info(f"==> Macro AUROC: {summary_metrics['macro_auroc']:.4f}")
    logger.info(f"==> Macro PR-AUC: {summary_metrics['macro_pr_auc']:.4f}")
    logger.info(f"==> Macro Brier Score: {summary_metrics['macro_brier_score']:.4f}")
    logger.info(f"==> Expected Calibration Error (ECE): {summary_metrics['expected_calibration_error_ece']:.4f}")

    return summary_metrics


def load_inference_pipeline(saved_dir: Union[str, Path] = SAVED_DIR) -> CalibratedAMRPipeline:
    """Helper to instantiate a fully configured inference pipeline from saved artifacts."""
    saved_dir = Path(saved_dir)
    preprocessor = AMRFeaturePreprocessor.load(saved_dir / "preprocessor.joblib")
    
    input_dim = len(preprocessor.feature_names)
    num_targets = len(ALL_ANTIBIOTICS)

    model = AMRMultiLabelClassifier(input_dim=input_dim, num_targets=num_targets)
    model.load_state_dict(torch.load(saved_dir / "classifier.pt", map_location="cpu", weights_only=True))
    model.eval()

    scaler = TemperatureScaler(num_targets=num_targets)
    scaler.load_state_dict(torch.load(saved_dir / "scaler.pt", map_location="cpu", weights_only=True))
    scaler.eval()

    return CalibratedAMRPipeline(model, scaler, preprocessor, ALL_ANTIBIOTICS)


if __name__ == "__main__":
    train_model(epochs=25, batch_size=64, lr=1e-3)
