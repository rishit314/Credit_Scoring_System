# 🏦 Explainable Credit Scoring System

> XGBoost + SHAP explainability | 0.86 ROC-AUC | FastAPI + Streamlit | AI-narrated decisions with guardrails | Docker-deployable

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red?logo=streamlit)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)
![Docker](https://img.shields.io/badge/Docker-Deployable-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 What This Project Does

A binary classifier that predicts whether a loan applicant will default — and **explains every decision in plain English**. Built to production standards with a REST API, an interactive dashboard, a guardrailed AI narration layer, and regulation-ready explanations powered by SHAP.

Every prediction returns:
- A **decision** (APPROVED / REVIEW / REJECTED)
- A **risk score** (300–850, like a real credit score)
- **Top 3 plain-English reasons**, satisfying adverse-action transparency requirements
- **SHAP impact values** for technical audit trails
- Optionally, an **AI-narrated applicant message** and underwriter checklist — see below

---

## 🎯 Key Results

| Model | ROC-AUC | PR-AUC | KS Statistic | Gini |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.8372 | 0.3382 | 0.5248 | 0.6743 |
| XGBoost + SMOTE | 0.8370 | 0.3475 | 0.5226 | 0.6740 |
| XGBoost + Optuna (SMOTE) | 0.7956 | 0.2882 | 0.4506 | 0.5913 |
| **XGBoost + scale_pos_weight ✅** | **0.8618** | **0.4050** | **0.5727** | **0.7236** |

**Why XGBoost beat Logistic Regression — and why SMOTE didn't help**

Starting with Logistic Regression as a baseline gave 0.837 ROC-AUC using `class_weight='balanced'`. The first XGBoost upgrade with SMOTE oversampling performed identically — synthetic minority samples introduced noise that cancelled out the model's capacity gains. Switching to XGBoost's native `scale_pos_weight` parameter (ratio of negatives to positives = ~13.9) eliminated synthetic noise entirely and pushed ROC-AUC to **0.862** — a 2.5 point improvement. PR-AUC improved by 20%, which matters far more than ROC-AUC for a dataset with 6.7% positive rate.

---

## 🔍 The Explainability Layer

Most credit models are black boxes. This one isn't.

**Example output for a high-risk applicant (95.3% default probability):**

```json
{
  "decision": "REJECTED",
  "default_probability": 0.9529,
  "risk_score": 325,
  "top_reasons": [
    "9 missed payment(s) recorded across all accounts",
    "Credit utilization is 107% — most available credit is being used (threshold: 30%)",
    "3 instance(s) of being 90+ days past due — serious delinquency history"
  ],
  "shap_impacts": {
    "total_missed_payments": 1.3780,
    "missed_90_plus": 0.3838,
    "credit_utilization": 0.3472
  },
  "model_version": "xgboost-v2-1.0",
  "confidence": "HIGH"
}
```

**Global feature importance** (what drives defaults across all applicants):

1. `total_missed_payments` — chronic payment behaviour is the strongest signal
2. `credit_utilization` — spending above credit limits is a leading indicator
3. `age` — younger applicants show statistically higher default rates in this dataset

---

## 🤖 AI Decision-Narration Layer

On top of the deterministic model, `POST /predict/agent` adds an AI-narrated layer — built around one non-negotiable rule: **the AI never makes or changes a risk decision, it only narrates the model's own output.**

```
XGBoost model → decision + probability + SHAP reasons   (deterministic, unchanged)
        │
        ▼
   AI model    → applicant-facing message + (if REVIEW) underwriter checklist
        │
        ▼
  Guardrail    → deterministic check (not another AI call) — rejects any
                  output that contradicts the decision, invents numbers,
                  or breaks the checklist rule
        │
   ┌────┴────┐
   ▼         ▼
 passed    failed / AI errors out
   │         │
   ▼         ▼
returned   safe template fallback (still returned — never a crash)
   │         │
   └────┬────┘
        ▼
  Audit log (every call, pass or fail, gets a unique ID + timestamp)
```

**Why this split matters:** a risk decision needs to be deterministic and defensible — that's not somewhere an AI model should be guessing. But turning that decision into a clear, human-readable message is exactly where an AI model adds real value. Keeping those two responsibilities separate, and gating the AI's output behind a guardrail it can't talk its way around, is what keeps this bounded instead of just "an AI wrapped around a model."

**Example `/predict/agent` response:**
```json
{
  "decision": "REJECTED",
  "default_probability": 0.9529,
  "risk_score": 325,
  "top_reasons": ["...same as /predict..."],
  "applicant_message": "Your application was not approved at this time, primarily due to a history of missed payments and high credit utilization...",
  "underwriter_checklist": [],
  "narrative_source": "ai",
  "guardrail_status": "passed",
  "audit_id": "1a82f2ca-7c05-430a-9c78-f8abe4d169ec"
}
```

Every call is written to an append-only audit log, viewable at `GET /audit-log`. Full design notes and setup/testing steps are in [`AGENT_LAYER.md`](./AGENT_LAYER.md).

---

## 🏗️ Architecture

```
Credit_Scoring_System/
│
├── data/                          # Raw + processed datasets
│
├── notebooks/
│   ├── 01_EDA.ipynb               # Class imbalance, distributions, correlations
│   ├── 02_feature_engineering.ipynb  # Feature pipeline, imputation, encoding
│   ├── 03_modelling.ipynb         # 4 models, Optuna tuning, MLflow tracking
│   └── 04_explainability.ipynb   # SHAP global + local, reason generator
│
├── models/
│   ├── credit_model.pkl           # Champion XGBoost model
│   ├── shap_explainer.pkl         # SHAP TreeExplainer
│   └── feature_names.pkl          # Feature registry (ensures consistent ordering)
│
├── src/
│   ├── api/
│   │   ├── main.py                # FastAPI routes (/predict, /predict/agent, /predict/batch, /audit-log, /model-info)
│   │   ├── predictor.py           # ML pipeline (feature engineering + inference)
│   │   ├── schemas.py             # Pydantic request/response validation
│   │   └── llm_agent.py           # AI narration layer — prompt, guardrail, fallback, audit logging
│   └── dashboard.py                # Streamlit UI
│
├── Dockerfile                     # Builds the API service for deployment
├── render.yaml                    # One-click Render deploy config
├── requirements.txt                # Full environment (notebooks + API)
├── requirements-api.txt            # Lean, deploy-only requirements
├── .env.example                    # Template for GROQ_API_KEY etc. — copy to .env
├── .gitignore                      # Keeps .env, audit logs, caches out of git
└── AGENT_LAYER.md                  # Full design notes for the AI narration layer
```

---

## 🚀 Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/rishit314/Credit_Scoring_System.git
cd Credit_Scoring_System

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt        # full environment (notebooks + API)
# or: pip install -r requirements-api.txt   # if you only need to run the API
```

### 2. Get the data

```bash
# Option A — Kaggle API
kaggle competitions download -c GiveMeSomeCredit -p data/

# Option B — Manual download
# Go to kaggle.com/c/GiveMeSomeCredit/data
# Download cs-training.csv → place in data/
```

### 3. Run the notebooks in order

```
notebooks/01_EDA.ipynb
notebooks/02_feature_engineering.ipynb
notebooks/03_modelling.ipynb
notebooks/04_explainability.ipynb
```

This trains the model and saves everything to `models/`.

### 4. (Optional) Set up the AI narration layer

```bash
cp .env.example .env
# edit .env — paste your Groq API key (free tier at console.groq.com/keys)
```

The API boots and `/predict` works fine even without this step — the AI layer is only needed for `/predict/agent`.

### 5. Start the API

```bash
cd src/api
uvicorn main:app --reload
# API running at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### 6. Start the dashboard (new terminal)

```bash
cd src
streamlit run dashboard.py
# Dashboard at http://localhost:8501
```

---

## 📡 API Reference

### `POST /predict`
Score a single loan applicant using the deterministic model only.

**Request:**
```json
{
  "credit_utilization": 0.45,
  "monthly_income": 5000,
  "debt_ratio": 0.35,
  "age": 42,
  "missed_30_59": 0,
  "missed_60_89": 0,
  "missed_90_plus": 0,
  "open_credit_lines": 8,
  "real_estate_loans": 1,
  "dependents": 2
}
```

**Response:** Decision, probability, risk score, plain-English reasons, SHAP impacts.

### `POST /predict/agent`
Same request format and same underlying decision as `/predict`, plus an AI-narrated applicant message and (for REVIEW cases) an underwriter checklist — guardrail-checked, with an automatic safe fallback if the AI call fails. Requires `GROQ_API_KEY` to be set for the AI path; falls back gracefully to a template otherwise.

### `POST /predict/batch`
Score up to 100 applicants in a single request.

### `GET /audit-log?limit=20`
Returns recent `/predict/agent` audit records — every call, pass or fail, with a unique ID, timestamp, and outcome.

### `GET /model-info`
Returns model version, ROC-AUC, training date, feature count.

### `GET /health`
Health check for monitoring systems.

**Full interactive docs:** `http://localhost:8000/docs`

---

## ☁️ Deployment

The API is Docker-ready and deploys to Render with the included config:

1. Push this repo to GitHub (`.env` is gitignored — never commit your real key).
2. On [Render](https://render.com), New → Web Service → connect the repo. `render.yaml` is auto-detected.
3. In the Render dashboard, add `GROQ_API_KEY` as an environment variable (kept out of the repo intentionally).
4. Deploy. Render builds the Docker image and gives you a public URL.
5. Verify with `curl https://<your-url>/health`.

Full steps and troubleshooting in [`AGENT_LAYER.md`](./AGENT_LAYER.md).

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data processing | Pandas, NumPy, SQLAlchemy |
| ML modelling | Scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Hyperparameter tuning | Optuna (Bayesian optimisation) |
| Explainability | SHAP (TreeExplainer), LIME |
| Experiment tracking | MLflow |
| AI narration layer | Groq-hosted LLM, custom deterministic guardrail |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Deployment | Docker, Render |

---

## 📐 Feature Engineering

20 features fed to the model, including 6 engineered features:

| Feature | Formula | Why it matters |
|---|---|---|
| `debt_to_income` | `debt_ratio × monthly_income` | Absolute debt load, not just ratio |
| `total_missed_payments` | Sum of all late payment counts | Single strongest default signal |
| `income_per_dependent` | `income / (dependents + 1)` | Financial pressure per person supported |
| `high_credit_utilization` | `credit_utilization > 0.3` | Binary risk flag at industry threshold |
| `ever_seriously_late` | `missed_90_plus > 0` | Any 90+ day delinquency is a red flag |
| `income_was_missing` | Flag for imputed income | Lets model learn that missing income behaves differently |

---

## ⚖️ Regulatory Compliance

**RBI Fair Practices Code & Digital Lending Guidelines**
The plain-English reason generator produces specific, itemized reasons for every rejection, in line with expectations around transparency and clear communication in digital lending decisions.

**ECOA (Equal Credit Opportunity Act, US)**
The same reason generator satisfies the US requirement to inform applicants of the specific reasons credit was denied — included since the underlying design principle (specific, itemized reasons, not a generic rejection) generalizes across jurisdictions.

**EU AI Act (Article 13)**
SHAP-based explanations provide the algorithmic transparency required for high-risk AI systems operating in the financial domain.

**Fairness**
A Fairlearn audit was conducted across age groups. Any detected performance disparity is documented in the model card. Model decisions are based solely on financial behaviour features.

---

## 💡 Key Technical Decisions

**Why not use accuracy as the metric?**
With a 6.7% default rate, a model that always predicts "no default" would be 93.3% accurate — and completely useless. ROC-AUC and PR-AUC measure the model's ability to rank defaulters above non-defaulters regardless of class balance. KS Statistic and Gini Coefficient are standard in the credit industry for the same reason.

**Why SHAP over a simpler method?**
SHAP values are mathematically grounded in cooperative game theory (Shapley values). Unlike feature importance scores, they are consistent, locally accurate, and additive — meaning they can be summed to exactly reproduce any individual prediction. This is required for audit trails.

**Why FastAPI over Flask?**
Automatic input validation via Pydantic, automatic OpenAPI documentation at `/docs`, native async support, and significantly better performance. The interactive docs page alone makes demos substantially easier.

**Why keep the AI model out of the decision path?**
An LLM can be persuaded, can hallucinate a number, and can't be held to the same statistical standard as a model validated on a held-out test set. So the risk decision is made once, deterministically, by XGBoost — the AI model is only ever allowed to narrate that fixed output, and a separate deterministic guardrail (not another AI call) checks its narration before anything is returned. That's what keeps the system bounded and auditable instead of just "an LLM bolted onto a model."

---

## 📬 Contact

Built by **Rishit Mishra**
📧 [rishit.mishra314@gmail.com](mailto:rishit.mishra314@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/rishit-mishra-915676275/) · 🐙 [GitHub](https://github.com/rishit314)