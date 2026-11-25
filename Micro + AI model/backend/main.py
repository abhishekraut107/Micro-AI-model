"""
backend/main.py
---------------
FastAPI REST Service for AI-Native AMR Surveillance, Digital Twin World Model,
and Human-in-the-Loop Clinical Escalation.

Framework Stages: Stage 4 (Understand), Stage 5 (Decide), Stage 6 (Act), Stage 7 (Measure)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.schemas import (
    IsolatePredictionRequest,
    IsolatePredictionResponse,
    CounterfactualRequest,
    CounterfactualResponse,
    EscalationActionRequest,
    CreateEscalationRequest,
    EscalationItem,
)
from backend.decision_engine import ClinicalDecisionEngine
from engine.digital_twin import AMRDigitalTwin
from models.train_and_evaluate import load_inference_pipeline
from models.anomaly_detector import AMRPhenotypicAnomalyDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("amr_backend")

# Global instances loaded at startup
state: Dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes models, digital twin world model, and decision engine on startup."""
    logger.info("Initializing BioTwin AMR Surveillance Service...")

    models_dir = BASE_DIR / "models" / "saved"
    train_data_path = BASE_DIR / "data" / "train.csv"
    test_data_path = BASE_DIR / "data" / "test.csv"

    # 1. Load ML Pipeline & Anomaly Detector
    try:
        pipeline = load_inference_pipeline(models_dir)
        anomaly_detector = AMRPhenotypicAnomalyDetector.load(models_dir / "anomaly_detector.joblib")
        logger.info("Loaded Calibrated Neural Pipeline and Anomaly Detector successfully.")
    except Exception as e:
        logger.error(f"Failed to load ML models from {models_dir}: {e}")
        pipeline = None
        anomaly_detector = None

    # 2. Load Digital Twin & Prospective Cohort
    try:
        twin = AMRDigitalTwin.from_csv(train_data_path)
        test_df = pd.read_csv(test_data_path) if test_data_path.exists() else None
        logger.info("Loaded Digital Twin World Model and Prospective Test cohort.")
    except Exception as e:
        logger.error(f"Failed to load Digital Twin from {train_data_path}: {e}")
        twin = None
        test_df = None

    # 3. Decision Engine
    decision_engine = ClinicalDecisionEngine()

    state["pipeline"] = pipeline
    state["anomaly_detector"] = anomaly_detector
    state["twin"] = twin
    state["test_df"] = test_df
    state["decision_engine"] = decision_engine

    logger.info("AMR Surveillance REST service initialization complete.")
    yield
    logger.info("Shutting down BioTwin AMR Surveillance Service...")


app = FastAPI(
    title="BioTwin AMR Surveillance & Decision API",
    description="AI-Native Antimicrobial Resistance Surveillance, Digital Twin Counterfactual Simulation & Clinical Escalation Platform.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Frontend Dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", summary="Service Health & Diagnostic Status")
async def health_check():
    """Returns service operational status, model availability, and active escalation count."""
    return {
        "status": "healthy",
        "service": "BioTwin AMR Surveillance",
        "models_loaded": bool(state.get("pipeline") is not None),
        "twin_loaded": bool(state.get("twin") is not None),
        "pending_escalations": len(state["decision_engine"].get_pending_escalations()) if state.get("decision_engine") else 0,
        "framework": "Seven-Stage AI Lifecycle",
    }


@app.get("/api/twin/baselines", summary="Digital Twin Rolling Baseline Rates")
async def get_twin_baselines(species: Optional[str] = None, antibiotic: Optional[str] = None):
    """Retrieves rolling baseline resistance rates by bacterial species or antibiotic class."""
    twin: AMRDigitalTwin = state.get("twin")
    if not twin:
        raise HTTPException(status_code=503, detail="Digital Twin is not loaded.")
    return twin.get_baseline_resistance(species=species, antibiotic=antibiotic)


@app.get("/api/twin/coresistance", summary="Phenotypic Cross-Resistance Correlation Matrix")
async def get_coresistance_matrix():
    """Returns the pairwise co-resistance correlation matrix across all 15 evaluated antibiotics."""
    twin: AMRDigitalTwin = state.get("twin")
    if not twin:
        raise HTTPException(status_code=503, detail="Digital Twin is not loaded.")
    matrix_df = twin.get_co_resistance_matrix()
    return matrix_df.to_dict()


@app.post("/api/twin/simulate", response_model=CounterfactualResponse, summary="Counterfactual Policy Simulator")
async def simulate_counterfactual(request: CounterfactualRequest):
    """
    Stage 6: Simulates the downstream impact of hospital antimicrobial stewardship interventions:
    - Fluoroquinolone restriction
    - Cephalosporin restriction
    - Infection control transmission reduction
    """
    twin: AMRDigitalTwin = state.get("twin")
    if not twin:
        raise HTTPException(status_code=503, detail="Digital Twin is not loaded.")

    interventions = request.model_dump()
    sim_result = twin.simulate_counterfactual(interventions)
    return sim_result


@app.get("/api/surveillance/signals", summary="Early-Warning Emerging Resistance Signals")
async def get_emerging_signals(z_threshold: float = 2.0):
    """
    Stage 3/7: Detects statistically significant emerging resistance signals in the prospective cohort
    relative to the historical baseline.
    """
    twin: AMRDigitalTwin = state.get("twin")
    test_df = state.get("test_df")
    if not twin or test_df is None:
        raise HTTPException(status_code=503, detail="Surveillance engine or prospective data unavailable.")

    signals = twin.detect_emerging_signals(test_df, z_threshold=z_threshold)
    return {
        "count": len(signals),
        "z_threshold": z_threshold,
        "cohort_tested": len(test_df),
        "signals": signals,
    }


@app.post("/api/predict", response_model=IsolatePredictionResponse, summary="Isolate Resistance & Triage Scoring")
async def predict_isolate(request: IsolatePredictionRequest):
    """
    Stage 3, 4 & 5: Predicts multi-label resistance probabilities, calibrated epistemic uncertainty,
    phenotypic anomaly score, and executes clinical triage logic.
    """
    pipeline = state.get("pipeline")
    anomaly_detector = state.get("anomaly_detector")
    decision_engine: ClinicalDecisionEngine = state.get("decision_engine")

    if not pipeline or not anomaly_detector or not decision_engine:
        raise HTTPException(status_code=503, detail="Prediction models or decision engine are not ready.")

    patient_data = request.model_dump(exclude_unset=True)
    isolate_id = request.Isolate_ID or f"ISO-{str(pd.Timestamp.now().value)[-6:]}"

    # 1. Calibrated Neural Inference & Monte Carlo Dropout Epistemic Uncertainty
    pred_result = pipeline.predict(patient_data, mc_samples=15)

    # 2. Phenotypic Anomaly Detection
    anomaly_result = anomaly_detector.score_isolate(patient_data, preprocessor=pipeline.preprocessor)

    # 3. Clinical Triage Evaluation
    tier, title, guidance, requires_esc, critical_drugs = decision_engine.evaluate_triage(
        prediction_result=pred_result,
        anomaly_result=anomaly_result,
        patient_input=patient_data,
        isolate_id=isolate_id,
    )

    calibrated_probs = {
        abx: details["predicted_resistance_prob"]
        for abx, details in pred_result["antibiotic_predictions"].items()
    }

    return {
        "isolate_id": isolate_id,
        "species": request.Species,
        "triage_tier": tier,
        "triage_title": title,
        "clinical_guidance": guidance,
        "requires_escalation": requires_esc,
        "predicted_mdr_flag": pred_result["predicted_mdr_flag"],
        "predicted_resistant_classes_count": pred_result["predicted_resistant_classes_count"],
        "mean_isolate_uncertainty": pred_result["mean_isolate_uncertainty"],
        "anomaly_score": anomaly_result["anomaly_score"],
        "is_anomaly": anomaly_result["is_anomaly"],
        "rare_resistance_traits": anomaly_result["rare_resistance_traits"],
        "antibiotic_predictions": pred_result["antibiotic_predictions"],
        "calibrated_probabilities": calibrated_probs,
        "triage": {
            "tier": tier,
            "action_summary": title,
            "clinical_guidance": guidance,
            "requires_escalation": requires_esc,
            "epistemic_uncertainty": pred_result["mean_isolate_uncertainty"],
        },
        "is_mdr": pred_result["predicted_mdr_flag"],
        "resistant_classes_count": pred_result["predicted_resistant_classes_count"],
    }


@app.get("/api/escalations", response_model=List[EscalationItem], summary="Human-in-the-Loop Escalation Queue")
async def get_escalations():
    """Stage 5: Lists all pending isolates requiring clinical review by an ID pharmacist or microbiologist."""
    decision_engine: ClinicalDecisionEngine = state.get("decision_engine")
    if not decision_engine:
        raise HTTPException(status_code=503, detail="Decision engine is not initialized.")
    return decision_engine.get_pending_escalations()


@app.post("/api/escalations", response_model=EscalationItem, status_code=status.HTTP_201_CREATED, summary="Create Manual Clinical Escalation")
async def create_manual_escalation(request: CreateEscalationRequest):
    """Stage 5: Allows clinical specialists to enqueue an isolate or ward outbreak case manually."""
    decision_engine: ClinicalDecisionEngine = state.get("decision_engine")
    if not decision_engine:
        raise HTTPException(status_code=503, detail="Decision engine is not initialized.")

    new_item = decision_engine.add_manual_escalation(
        species=request.species,
        tier=request.triage_tier,
        reason=request.reason,
        critical_drugs=request.critical_drugs_at_risk,
        isolate_id=request.isolate_id,
        epistemic_uncertainty=request.epistemic_uncertainty,
        anomaly_score=request.anomaly_score,
        patient_data={"Department": request.department, "Source": "Manual Enqueue"},
    )
    return new_item


@app.post("/api/escalations/{escalation_id}/action", response_model=EscalationItem, summary="Review & Triage Escalation")
async def resolve_escalation(escalation_id: str, request: EscalationActionRequest):
    """Stage 5/6: Records human-in-the-loop review action (Approve, Order Reflex Test, Override)."""
    decision_engine: ClinicalDecisionEngine = state.get("decision_engine")
    if not decision_engine:
        raise HTTPException(status_code=503, detail="Decision engine is not initialized.")

    resolved_item = decision_engine.resolve_escalation(
        escalation_id=escalation_id,
        action=request.action,
        reviewer_name=request.reviewer_name,
        clinical_notes=request.clinical_notes,
    )
    if not resolved_item:
        raise HTTPException(status_code=404, detail=f"Escalation {escalation_id} not found.")

    return resolved_item


# Mount static frontend files for unified single-port dashboard delivery
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
