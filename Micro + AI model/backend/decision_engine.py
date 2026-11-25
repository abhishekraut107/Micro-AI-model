"""
backend/decision_engine.py
--------------------------
Clinical Decision Escalation Engine, Triage Matrix, and Human-in-the-Loop Queue.

Framework Stage: Stage 4 (Understand) & Stage 5 (Decide)
"""

import datetime
import logging
import uuid
from typing import Dict, List, Optional, Tuple, Any
from backend.schemas import EscalationItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("decision_engine")


class ClinicalDecisionEngine:
    """
    Translates machine learning predictions, epistemic uncertainty,
    and anomaly detection scores into actionable clinical stewardship decisions.
    """

    def __init__(self):
        # In-memory escalation storage keyed by escalation_id
        self.escalations: Dict[str, EscalationItem] = {}
        self._seed_initial_escalations()

    def _seed_initial_escalations(self):
        """Pre-populates realistic clinical escalation cases for prospective demonstration."""
        self._enqueue_escalation(
            isolate_id="ISO-883419",
            species="Klebsiella pneumoniae",
            tier="TIER_1_CRITICAL_ESCALATION",
            reason="CRITICAL ALERT: Multi-Drug / Carbapenem Resistance Detected (IPM Resistant, Z=2.11)",
            anomaly_score=0.88,
            epistemic_uncertainty=0.031,
            critical_drugs=["Carbapenems (Imipenem)", "3rd-Gen Cephalosporins (CTX/CRO)"],
            input_data={"Age": 72, "Gender": "M", "Prior_Hospitalization": 1, "Department": "Medical ICU"},
        )
        self._enqueue_escalation(
            isolate_id="ISO-519204",
            species="Serratia marcescens",
            tier="TIER_2_CONFIRMATORY_TESTING",
            reason="ELEVATED ALERT: 3-Fold Quinolone Surge / High Epistemic Uncertainty",
            anomaly_score=0.76,
            epistemic_uncertainty=0.084,
            critical_drugs=["Fluoroquinolones (CIP)", "Aminoglycosides (GEN)"],
            input_data={"Age": 61, "Gender": "F", "Prior_Hospitalization": 1, "Department": "Surgical Stepdown"},
        )

    def add_manual_escalation(
        self,
        species: str,
        tier: str,
        reason: str,
        critical_drugs: List[str],
        isolate_id: Optional[str] = None,
        epistemic_uncertainty: float = 0.04,
        anomaly_score: float = 0.80,
        patient_data: Optional[Dict] = None,
    ) -> EscalationItem:
        """Allows clinical specialists or automated endpoints to manually enqueue a case."""
        if not isolate_id:
            isolate_id = f"ISO-{str(uuid.uuid4())[:6].upper()}"
        return self._enqueue_escalation(
            isolate_id=isolate_id,
            species=species,
            tier=tier,
            reason=reason,
            anomaly_score=anomaly_score,
            epistemic_uncertainty=epistemic_uncertainty,
            critical_drugs=critical_drugs,
            input_data=patient_data or {"Source": "Manual Clinical Enqueue"},
        )

    def evaluate_triage(
        self,
        prediction_result: Dict,
        anomaly_result: Dict,
        patient_input: Dict,
        isolate_id: Optional[str] = None,
    ) -> Tuple[str, str, str, bool, List[str]]:
        """
        Executes the Dual-Threshold Clinical Triage Matrix:
        Returns:
        - triage_tier: 'TIER_1_CRITICAL_ESCALATION', 'TIER_2_CONFIRMATORY_TESTING', 'TIER_3_ROUTINE_MONITORING'
        - triage_title: Human-readable alert summary
        - clinical_guidance: Specific antimicrobial stewardship recommendation
        - requires_escalation: bool
        - critical_drugs_at_risk: list of compromised antibiotic classes
        """
        if not isolate_id:
            isolate_id = f"ISO-{str(uuid.uuid4())[:8].upper()}"

        abx_preds = prediction_result.get("antibiotic_predictions", {})
        mean_uncertainty = prediction_result.get("mean_isolate_uncertainty", 0.0)
        is_anomaly = anomaly_result.get("is_anomaly", False)
        anomaly_score = anomaly_result.get("anomaly_score", 0.0)
        critical_flag = anomaly_result.get("critical_pathogen_flag", False)

        # Identify critical high-risk antibiotic resistances
        ipm_res = abx_preds.get("IPM", {}).get("is_resistant_prediction", False)
        col_res = abx_preds.get("colistine", {}).get("is_resistant_prediction", False)
        ctx_res = abx_preds.get("CTX/CRO", {}).get("is_resistant_prediction", False)

        critical_drugs_at_risk = []
        if ipm_res:
            critical_drugs_at_risk.append("Carbapenems (Imipenem)")
        if col_res:
            critical_drugs_at_risk.append("Polymyxins (Colistin)")
        if ctx_res:
            critical_drugs_at_risk.append("3rd-Gen Cephalosporins (CTX/CRO)")

        # Tier 1: Critical Escalation (Carbapenem/Colistin Resistance OR Extreme Anomaly)
        if ipm_res or col_res or critical_flag or anomaly_score >= 0.85:
            tier = "TIER_1_CRITICAL_ESCALATION"
            title = "CRITICAL ALERT: Multi-Drug / Carbapenem Resistance Detected"
            guidance = (
                "EMERGENCY STEWARDSHIP ESCALATION: High probability of carbapenem or reserve polymyxin resistance. "
                "Immediately contact the infectious disease consult service. Restrict broad-spectrum carbapenems. "
                "Initiate contact isolation precautions and order rapid molecular genotypic testing (e.g. bla_KPC, bla_NDM, mcr-1)."
            )
            requires_escalation = True

        # Tier 2: Confirmatory Testing (High Epistemic Uncertainty or Atypical Pattern)
        elif mean_uncertainty >= 0.08 or is_anomaly or ctx_res:
            tier = "TIER_2_CONFIRMATORY_TESTING"
            title = "ELEVATED ALERT: High Epistemic Uncertainty / Extended-Spectrum Beta-Lactamase"
            guidance = (
                "CONFIRMATORY AST RECOMMENDED: Model epistemic uncertainty exceeds automated release thresholds "
                f"({mean_uncertainty:.1%}) or suspected ESBL phenotype. Avoid empirical 3rd-generation cephalosporins. "
                "Recommend reflex E-test / broth microdilution confirmation before finalizing therapy."
            )
            requires_escalation = True

        # Tier 3: Routine Monitoring (High Confidence, Standard Formulary)
        else:
            tier = "TIER_3_ROUTINE_MONITORING"
            title = "STANDARD MONITORING: Typical Resistance Profile"
            guidance = (
                "Standard empirical institutional formulary protocols apply. Proceed with narrow-spectrum targeted agent "
                "once local AST cultures are validated."
            )
            requires_escalation = False

        # If escalation is required, record in the Human-in-the-Loop Queue
        if requires_escalation:
            self._enqueue_escalation(
                isolate_id=isolate_id,
                species=patient_input.get("Species", "Unknown"),
                tier=tier,
                reason=title,
                anomaly_score=anomaly_score,
                epistemic_uncertainty=mean_uncertainty,
                critical_drugs=critical_drugs_at_risk,
                input_data=patient_input,
            )

        return tier, title, guidance, requires_escalation, critical_drugs_at_risk

    def _enqueue_escalation(
        self,
        isolate_id: str,
        species: str,
        tier: str,
        reason: str,
        anomaly_score: float,
        epistemic_uncertainty: float,
        critical_drugs: List[str],
        input_data: Dict[str, Any],
    ) -> EscalationItem:
        """Enqueues an isolate into the pending clinical triage registry."""
        esc_id = f"ESC-{str(uuid.uuid4())[:8].upper()}"
        item = EscalationItem(
            escalation_id=esc_id,
            isolate_id=isolate_id,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            species=species,
            triage_tier=tier,
            status="PENDING_REVIEW",
            reason=reason,
            anomaly_score=anomaly_score,
            epistemic_uncertainty=epistemic_uncertainty,
            critical_drugs_at_risk=critical_drugs,
            input_data=input_data,
            review_history=[],
        )
        self.escalations[esc_id] = item
        logger.info(f"Enqueued clinical escalation: {esc_id} for isolate {isolate_id} [{tier}]")
        return item

    def get_pending_escalations(self) -> List[EscalationItem]:
        """Returns all unreviewed clinical escalations sorted by severity and timestamp."""
        pending = [item for item in self.escalations.values() if item.status == "PENDING_REVIEW"]
        # Sort Tier 1 first
        tier_weight = {"TIER_1_CRITICAL_ESCALATION": 0, "TIER_2_CONFIRMATORY_TESTING": 1, "TIER_3_ROUTINE_MONITORING": 2}
        return sorted(pending, key=lambda x: (tier_weight.get(x.triage_tier, 9), x.timestamp), reverse=False)

    def resolve_escalation(
        self,
        escalation_id: str,
        action: str,
        reviewer_name: str,
        clinical_notes: str,
    ) -> Optional[EscalationItem]:
        """Records a human-in-the-loop review decision for a pending escalation."""
        if escalation_id not in self.escalations:
            return None

        item = self.escalations[escalation_id]
        
        status_map = {
            "APPROVE_STEWARDSHIP": "RESOLVED_APPROVED",
            "REQUEST_REFLEX_TEST": "RESOLVED_REFLEX_ORDERED",
            "OVERRIDE": "RESOLVED_OVERRIDDEN",
        }
        item.status = status_map.get(action, "RESOLVED_APPROVED")
        item.review_history.append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reviewer": reviewer_name,
            "action": action,
            "status": item.status,
            "notes": clinical_notes,
        })
        logger.info(f"Resolved escalation {escalation_id} with action {action} by {reviewer_name}.")
        return item
