"""
main.py — The FastAPI application

FastAPI works like this:
  1. You write a normal Python function
  2. You add a decorator (@app.get or @app.post) to say what URL it handles
  3. FastAPI handles all the HTTP stuff automatically

That's it. Your ML model becomes a web service.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # reads .env in this directory (or cwd) if present — no-op if missing

# Import our own files
from schemas import ApplicantFeatures, PredictionResponse, ModelInfoResponse, AgentPredictionResponse
from predictor import predict
import llm_agent


# ── CREATE THE APP ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Scoring API",
    description="""
    ## Explainable Credit Scoring System
    
    Predicts loan default risk and explains every decision in plain English.
    
    ### Endpoints
    - **POST /predict** — Score a loan applicant (deterministic model only)
    - **POST /predict/agent** — Same model, plus an LLM-narrated applicant message
      and underwriter checklist (Groq-backed, guardrailed, audit-logged)
    - **POST /predict/batch** — Score up to 100 applicants at once
    - **GET /audit-log** — View recent LLM-agent audit records
    - **GET /model-info** — Get model metadata
    - **GET /health** — Check if API is running
    - **GET /docs** — Interactive documentation (you are here!)
    
    ### About the Model
    XGBoost classifier trained on 120k loan applications.
    Achieves 0.86 ROC-AUC with full SHAP explainability.
    Adverse action reasons comply with regulatory transparency requirements.

    ### About the Agent Layer
    The LLM (via Groq) never makes or changes a risk decision — it only
    narrates the model's own output. Every LLM call is guardrail-checked
    and audit-logged; failures fall back to a safe deterministic template.
    """,
    version="1.1.0",
)

# CORS = allows other websites/apps to call this API
# Without this, a browser would block requests from other origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # in production, list specific allowed domains
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track prediction count (resets when server restarts — use DB for persistence)
prediction_count = 0


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Simple health check — tells you the API is running.
    Used by monitoring systems to check if the service is alive.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "predictions_served": prediction_count
    }


@app.get("/audit-log")
def get_audit_log(limit: int = 20):
    """
    Returns the most recent LLM-agent audit records (newest first).
    Every /predict/agent call — whether the LLM narrative passed guardrails
    or fell back to a template — is logged here for compliance review.
    """
    import json

    if not llm_agent.AUDIT_LOG_PATH.exists():
        return {"total": 0, "records": []}

    with open(llm_agent.AUDIT_LOG_PATH) as f:
        lines = [json.loads(line) for line in f if line.strip()]

    lines = lines[-limit:][::-1]
    return {"total": len(lines), "records": lines}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """
    Returns metadata about the current model.
    Useful for audits, compliance, and model versioning.
    """
    return ModelInfoResponse(
        model_type="XGBoost (scale_pos_weight)",
        roc_auc=0.8618,
        training_date="2025-01-01",
        feature_count=20,
        default_threshold=0.35,
        version="xgboost-v2-1.0"
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_default(applicant: ApplicantFeatures):
    """
    ## Score a Loan Applicant
    
    Send applicant features, get back:
    - **Decision**: APPROVED / REVIEW / REJECTED
    - **Default probability**: 0.0 (safe) to 1.0 (certain default)
    - **Risk score**: 300 (worst) to 850 (best)
    - **Top 3 reasons**: Plain English explanation of the decision
    - **SHAP impacts**: Raw explainability values for technical use
    
    ### Example Request
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
    """
    global prediction_count

    try:
        # Convert Pydantic model to plain dict for predictor
        applicant_dict = applicant.model_dump()

        # Run prediction pipeline
        result = predict(applicant_dict)

        # Increment counter
        prediction_count += 1

        return PredictionResponse(**result)

    except Exception as e:
        # If anything goes wrong, return a proper HTTP error
        # Never expose raw Python errors to API callers
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/agent", response_model=AgentPredictionResponse)
def predict_default_with_agent(applicant: ApplicantFeatures):
    """
    ## Score an Applicant AND Generate an LLM-Narrated Decision

    Runs the exact same deterministic model as **POST /predict** — the
    decision, probability, risk score, and SHAP reasons are identical and
    the LLM cannot change them. On top of that, a Groq-hosted LLM writes:

    - An **applicant-facing message** in plain, professional language
    - An **underwriter checklist** (only for REVIEW decisions)

    Every LLM output is checked against a guardrail before being returned
    (it must not contradict the model's decision, invent numbers, or ignore
    length limits). If the guardrail fails, or the LLM call errors out, the
    endpoint automatically falls back to a deterministic template — the
    caller still gets a valid, safe response, never a raw error.

    Every call, pass or fail, is written to an append-only audit log with a
    unique `audit_id` for compliance traceability.
    """
    global prediction_count

    try:
        applicant_dict = applicant.model_dump()
        result = predict(applicant_dict)
        prediction_count += 1

        narrative = llm_agent.generate_narrative(result)

        return AgentPredictionResponse(**result, **narrative)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch")
def predict_batch(applicants: list[ApplicantFeatures]):
    """
    ## Score Multiple Applicants at Once
    
    Send a list of applicants, get back a list of decisions.
    More efficient than calling /predict individually for each one.
    Maximum 100 applicants per batch.
    """
    if len(applicants) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum batch size is 100 applicants"
        )

    results = []
    for i, applicant in enumerate(applicants):
        try:
            result = predict(applicant.model_dump())
            results.append({"applicant_index": i, "success": True, **result})
        except Exception as e:
            results.append({"applicant_index": i, "success": False, "error": str(e)})

    return {
        "total": len(applicants),
        "successful": sum(1 for r in results if r["success"]),
        "results": results
    }