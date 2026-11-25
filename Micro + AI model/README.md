# BioTwin AMR: AI-Native Antimicrobial Resistance Surveillance & Clinical Decision Twin

An AI-Native Antimicrobial Resistance (AMR) Surveillance and Decision-Support System engineered under the **Seven-Stage AI Engineering Lifecycle**:
> **Problem $\rightarrow$ Bound $\rightarrow$ Model $\rightarrow$ Understand $\rightarrow$ Decide $\rightarrow$ Act $\rightarrow$ Measure $\rightarrow$ Learn**

---

## 🎯 Research Challenge
> **"Can AI identify meaningful emerging AMR patterns earlier than conventional analysis?"**

### Empirical Answer: **YES**
Across rigorous prospective temporal holdout testing (2024–2025 cohort), BioTwin AMR identified critical emerging multi-drug resistance patterns **14 to 28 days (2 to 4 weeks) earlier** than standard CLSI M39 / NHSN retrospective monthly infection control surveillance:
- **Carbapenem-Resistant *Klebsiella pneumoniae* (CRKP)**: Flagged on **2024-12-31** ($Z = 2.03, p < 0.05$) vs. conventional committee detection on **2025-01-14** (**+14 days lead-time**).
- **Fluoroquinolone-Resistant *Serratia marcescens***: Flagged on **2025-01-05** ($Z = 2.78, p < 0.005$) vs. conventional committee detection on **2025-01-19** (**+14 days lead-time**).

---

## 🏛️ The Seven-Stage System Architecture

```
                                  [ CLINICAL AST DATA ]
                                            │
                                            ▼
                    ┌────────────────────────────────────────────────┐
STAGE 1: PROBLEM    │  Early AMR Pattern Recognition & Outbreak Alarms│
                    └───────────────────────┬────────────────────────┘
                                            │
                                            ▼
                    ┌────────────────────────────────────────────────┐
STAGE 2: BOUND      │  Data Cleaning, EUCAST/CLSI Standardization,   │
                    │  15 Antibiotics, Prospective 80/20 Temporal    │
                    └───────────────────────┬────────────────────────┘
                                            │
                                            ▼
                    ┌────────────────────────────────────────────────┐
STAGE 3: MODEL      │  Deep Residual Multi-Task MLP + Masked BCE     │
                    │  Temperature Scaling (ECE = 1.58%) + Isolation │
                    └───────────────────────┬────────────────────────┘
                                            │
                                            ▼
                    ┌────────────────────────────────────────────────┐
STAGE 4: UNDERSTAND │  15x15 Phenotypic Co-Resistance Matrix +       │
                    │  Monte Carlo Dropout Epistemic Uncertainty     │
                    └───────────────────────┬────────────────────────┘
                                            │
                                            ▼
                    ┌────────────────────────────────────────────────┐
STAGE 5: DECIDE     │  Dual-Threshold Clinical Triage Matrix +       │
                    │  Human-in-the-Loop Clinical Escalation Queue   │
                    └───────────────────────┬────────────────────────┘
                                            │
                                            ▼
                    ┌────────────────────────────────────────────────┐
FastAPI REST API + Counterfactual Lab +       │
                    │  Interactive Clinical Surveillance Dashboard   │
                    └───────────────────────┬────────────────────────┘
                                            │
                                            ▼
                    ┌────────────────────────────────────────────────┐
STAGE 7: MEASURE    │  Lead-Time Advantage Benchmarking (+14-28 Days)│
         & LEARN    │  Continual Learning & Calibration Drift Tracking│
                    └────────────────────────────────────────────────┘
```

---

## 📦 Project Structure

```
.
├── Bacteria_dataset_Multiresictance.csv   # Raw microbiology dataset (10,710 isolates)
├── data/                                  # Processed data splits and clinical benchmarks
│   ├── cleaned_full.csv / .parquet        # Sanitized cohort (10,710 isolates)
│   ├── train.csv / .parquet               # Prospective training set (6,716 isolates: 2019-2024)
│   ├── test.csv / .parquet                # Prospective holdout test set (1,679 isolates: 2024-2025)
│   ├── undated_cohort.csv                 # Un-timestamped isolates (2,315 isolates)
│   ├── metadata.json                      # Clinical distribution and MDR class statistics
│   ├── leadtime_benchmark.json            # Empirical lead-time advantage metrics
│   └── continuous_learning_report.json    # PSI drift and calibration report
├── pipeline/                              # Data engineering & learning
│   ├── __init__.py
│   ├── clean_and_split.py                 # Normalization, target encoding & prospective splitting
│   └── continuous_learning.py             # Feature drift (PSI) & online model fine-tuning
├── engine/                                # Digital Twin world model
│   ├── __init__.py
│   ├── digital_twin.py                    # Rolling baselines, early warning signals, counterfactuals
│   └── benchmark_leadtime.py              # Quantitative comparison vs. conventional surveillance
├── models/                                # Stage 3 ML & Calibration Engine
│   ├── __init__.py
│   ├── dataset.py                         # Preprocessor & PyTorch Dataset with masked AST targets
│   ├── classifier.py                      # Deep Residual MLP, Masked BCE Loss & Temperature Scaler
│   ├── anomaly_detector.py                # Isolation Forest phenotypic anomaly detection
│   ├── train_and_evaluate.py              # End-to-end training & prospective temporal evaluation
│   └── saved/                             # Serialized weights, scalers, and encoders
│       ├── classifier.pt
│       ├── scaler.pt                      # Calibrated T = 1.066
│       ├── preprocessor.joblib
│       ├── anomaly_detector.joblib
│       └── metrics.json
├── backend/                               # Stage 4 & 5 FastAPI REST Service
│   ├── __init__.py
│   ├── schemas.py                         # Pydantic request/response validation
│   ├── decision_engine.py                 # Clinical triage matrix & escalation queue
│   └── main.py                            # Unified REST API & static frontend hosting
├── frontend/                              # Stage 6 Clinical Surveillance Dashboard
│   ├── index.html                         # Modern dark-theme glassmorphic interface
│   ├── styles.css                         # Medical UI design system (Inter + JetBrains Mono)
│   └── app.js                             # Client controller & real-time REST integration
└── README.md                              # Complete capstone documentation
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Prerequisites
```bash
# Python 3.11+
pip install pandas numpy torch scikit-learn pyarrow fastapi uvicorn dateparser
```

### 2. Execute Data Pipeline (Stages 1 & 2)
```bash
python pipeline/clean_and_split.py
```
- Standardizes species taxonomy (resolving `E.coi`, `Prot.eus`, `Klbsiella`).
- Encodes 15 antibiotics into binary non-susceptibility targets ($1 = \text{Resistant}, 0 = \text{Susceptible}$).
- Executes strict chronological 80/20 train/test split.

### 3. Train & Calibrate Models (Stage 3)
```bash
python models/train_and_evaluate.py
```
- Trains Deep Residual MLP with `MaskedBCELoss`.
- Optimizes post-hoc temperature parameter ($T = 1.066$, $ECE = 0.0158$).
- Fits Isolation Forest on historical multi-drug antibiograms.

### 4. Run Lead-Time Advantage Benchmark (Stage 7A)
```bash
python engine/benchmark_leadtime.py
```
- Quantifies exact lead-time gained against CLSI M39 monthly infection control reviews.

### 5. Launch Unified Surveillance API & Clinical Dashboard (Stage 6)
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
- Open your browser to: **`http://127.0.0.1:8000/`**
- Interactive Swagger API Documentation: **`http://127.0.0.1:8000/docs`**

---

## 🔬 Clinical Triage Matrix & Decision Gates

Every clinical isolate evaluated by BioTwin AMR is gated through standardized clinical decision rules:

| Triage Tier | Trigger Condition | Recommended Clinical Action |
| :--- | :--- | :--- |
| **TIER 1: CRITICAL ESCALATION** | $P(\text{IPM}) \ge 0.50$ OR $P(\text{colistine}) \ge 0.50$ OR Anomaly Score $\ge 0.85$ | **EMERGENCY ESCALATION**: Immediate contact isolation. Restrict carbapenems. Order reflex PCR (*bla*<sub>KPC</sub>/*bla*<sub>NDM</sub>). Requires mandatory ID pharmacist sign-off. |
| **TIER 2: CONFIRMATORY TESTING** | Epistemic Uncertainty $\sigma_U \ge 0.08$ OR $P(\text{CTX/CRO}) \ge 0.50$ (ESBL suspected) | **CONFIRMATORY AST**: High model uncertainty. Avoid empirical 3rd-gen cephalosporins. Order reflex E-test / broth microdilution before broad-spectrum release. |
| **TIER 3: ROUTINE FORMULARY** | Low uncertainty ($\sigma_U < 0.08$), standard wild-type or routine resistance | **STANDARD FORMULARY**: Institutional empiric guidance valid. De-escalate once local cultures return. |

---

## 📊 Summary of Prospective Validation Metrics

Evaluated on the prospective holdout test cohort (1,679 isolates, Dec 2024 to Feb 2025):
- **Expected Calibration Error (ECE)**: **$0.0158$** ($1.58\%$)
- **Macro Brier Score**: **$0.1685$**
- **Optimal Calibration Temperature**: **$T = 1.0659$**
- **Top Class Performance**:
  - Carbapenems (`IPM`): AUROC **0.6728**, PR-AUC **0.7092**, F1 **0.7518**
  - Cephalosporins (`FOX`): AUROC **0.6600**, PR-AUC **0.6954**, F1 **0.7438**
  - Beta-lactamase Inhibitors (`AMC`): AUROC **0.6545**, PR-AUC **0.6999**, F1 **0.7498**
  - Cephalosporins (`CTX/CRO`): AUROC **0.6543**, PR-AUC **0.6654**, F1 **0.7416**

---

## ⚖️ Clinical Data Governance & Ethical Considerations
- **De-Identification**: All personal identifying information (`Name`, `Email`, `Address`) is stripped during preprocessing to comply with HIPAA and GDPR guidelines.
- **Human-in-the-Loop Accountability**: The system operates strictly as a Clinical Decision Support System (CDSS). Critical antimicrobial restriction orders cannot be executed autonomously; they require review and electronic signature in the escalation registry.
