"""
backend/schemas.py
------------------
Pydantic data schemas for API requests, responses, and clinical decision objects.

Framework Stage: Stage 4 (Understand) & Stage 5 (Decide)
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, model_validator


class IsolatePredictionRequest(BaseModel):
    """Input payload for scoring a clinical isolate."""
    Species: str = Field(..., example="Klebsiella pneumoniae", description="Standardized bacterial species name")
    Age: float = Field(..., ge=0.0, le=120.0, example=68.0, description="Patient age in years")
    Gender: str = Field(..., example="M", description="'M', 'F', or 'Unknown'")
    Diabetes_Flag: int = Field(0, ge=0, le=1, example=1, description="Comorbidity flag: 1 = Diabetic, 0 = Non-diabetic")
    Hypertension_Flag: int = Field(0, ge=0, le=1, example=1, description="Comorbidity flag: 1 = Hypertensive, 0 = Normotensive")
    Prior_Hospitalization: int = Field(0, ge=0, le=1, example=1, description="1 = Hospitalized within past 90 days, 0 = Community-acquired")
    Infection_Frequency: float = Field(1.0, ge=0.0, example=2.0, description="Prior recorded infection episodes")
    Isolate_ID: Optional[str] = Field(None, example="ISO-2025-0842", description="Optional clinical isolate tracking identifier")

    @model_validator(mode="before")
    @classmethod
    def normalize_input_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            species = data.get("Species") or data.get("species") or "Klebsiella pneumoniae"
            age = data.get("Age") if data.get("Age") is not None else data.get("age", 45.0)
            gender = data.get("Gender") or data.get("gender") or "M"
            
            comorbs = data.get("comorbidities", {}) if isinstance(data.get("comorbidities"), dict) else {}
            diabetes = data.get("Diabetes_Flag") if data.get("Diabetes_Flag") is not None else data.get("diabetes_flag", comorbs.get("diabetes", data.get("diabetes", 0)))
            hypertension = data.get("Hypertension_Flag") if data.get("Hypertension_Flag") is not None else data.get("hypertension_flag", comorbs.get("hypertension", data.get("hypertension", 0)))
            
            hospital = data.get("Prior_Hospitalization") if data.get("Prior_Hospitalization") is not None else data.get("prior_hospitalization", data.get("hospital", 0))
            freq = data.get("Infection_Frequency") if data.get("Infection_Frequency") is not None else data.get("infection_frequency", data.get("frequency", 1.0))
            isolate_id = data.get("Isolate_ID") or data.get("isolate_id")
            
            return {
                "Species": str(species),
                "Age": float(age),
                "Gender": str(gender).upper()[:1] if str(gender) else "M",
                "Diabetes_Flag": int(diabetes),
                "Hypertension_Flag": int(hypertension),
                "Prior_Hospitalization": int(hospital),
                "Infection_Frequency": float(freq),
                "Isolate_ID": isolate_id,
            }
        return data


class AntibioticPredictionDetail(BaseModel):
    predicted_resistance_prob: float
    epistemic_uncertainty: float
    is_resistant_prediction: bool
    confidence_interval_95: List[float]
    confidence_level: str


class IsolatePredictionResponse(BaseModel):
    """Output payload returned by the calibrated inference and decision pipeline."""
    isolate_id: str
    species: str
    triage_tier: str
    triage_title: str
    clinical_guidance: str
    requires_escalation: bool
    predicted_mdr_flag: bool
    predicted_resistant_classes_count: int
    mean_isolate_uncertainty: float
    anomaly_score: float
    is_anomaly: bool
    rare_resistance_traits: List[Dict[str, Any]]
    antibiotic_predictions: Dict[str, AntibioticPredictionDetail]
    calibrated_probabilities: Optional[Dict[str, float]] = None
    triage: Optional[Dict[str, Any]] = None
    is_mdr: Optional[bool] = None
    resistant_classes_count: Optional[int] = None


class CounterfactualRequest(BaseModel):
    """Parameters for simulated clinical stewardship policies."""
    fluoroquinolone_restriction: float = Field(0.0, ge=0.0, le=1.0, example=0.30, description="Fractional reduction in empirical FQ prescription")
    cephalosporin_stewardship: float = Field(0.0, ge=0.0, le=1.0, example=0.20, description="Fractional reduction in 3rd-gen cephalosporin exposure")
    infection_control_improvement: float = Field(0.0, ge=0.0, le=1.0, example=0.15, description="Fractional reduction in nosocomial transmission")


class CounterfactualResponse(BaseModel):
    applied_interventions: Dict[str, float]
    cohort_size: int
    baseline_metrics: Dict[str, float]
    counterfactual_projected_metrics: Dict[str, float]
    projected_impact: Dict[str, Any]


class EscalationActionRequest(BaseModel):
    """Payload for human-in-the-loop clinical review and sign-off."""
    action: str = Field(..., example="APPROVE_STEWARDSHIP", description="'APPROVE_STEWARDSHIP', 'REQUEST_REFLEX_TEST', or 'OVERRIDE'")
    reviewer_name: str = Field(..., example="Dr. H. Vance, PharmD", description="Clinical microbiologist or infectious disease specialist name")
    clinical_notes: str = Field(..., example="Confirmed carbapenem resistance; reflex PCR ordered for bla_KPC.", description="Clinical justification notes")


class CreateEscalationRequest(BaseModel):
    """Payload for manually enqueuing a clinical escalation task."""
    species: str = Field(..., example="Klebsiella pneumoniae")
    isolate_id: Optional[str] = Field(None, example="ISO-994102")
    triage_tier: str = Field("TIER_1_CRITICAL_ESCALATION", example="TIER_1_CRITICAL_ESCALATION")
    reason: str = Field(..., example="Phenotypic carbapenem resistance surge in surgical ward")
    critical_drugs_at_risk: List[str] = Field(default_factory=list, example=["Carbapenems (Imipenem)"])
    epistemic_uncertainty: float = Field(0.035, ge=0.0, le=1.0)
    anomaly_score: float = Field(0.82, ge=0.0, le=1.0)
    department: Optional[str] = Field("Inpatient Ward", example="Inpatient Ward")


class EscalationItem(BaseModel):
    escalation_id: str
    isolate_id: str
    timestamp: str
    species: str
    triage_tier: str
    status: str  # 'PENDING_REVIEW', 'RESOLVED_APPROVED', 'RESOLVED_REFLEX_ORDERED', 'RESOLVED_OVERRIDDEN'
    reason: str
    anomaly_score: float
    epistemic_uncertainty: float
    critical_drugs_at_risk: List[str]
    input_data: Dict[str, Any]
    review_history: List[Dict[str, Any]] = []
