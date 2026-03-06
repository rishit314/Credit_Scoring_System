"""
predictor.py — Model loading and prediction logic

This file is the brain of the API.
It loads the model ONCE when the server starts (not on every request)
and exposes a clean predict() function that main.py calls.

Separation of concerns:
  main.py     → handles HTTP (routes, requests, responses)
  predictor.py → handles ML (loading, predicting, explaining)
  schemas.py  → handles data shapes (validation)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────────
# Path(__file__) = this file's location
# .parent.parent.parent = go up 3 levels to project root
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


# ── LOAD MODEL ONCE AT STARTUP ────────────────────────────────────────────────
# We load these globally so they stay in memory
# Loading from disk on every request would be ~2 seconds per call — too slow

print("Loading model and explainer...")

model = joblib.load(MODELS_DIR / "credit_model.pkl")
explainer = joblib.load(MODELS_DIR / "shap_explainer.pkl")
feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")

print(f"✅ Model loaded: {type(model).__name__}")
print(f"✅ Features: {len(feature_names)}")


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
# We must apply the SAME transformations we did in Phase 3
# Otherwise the model gets different features than it was trained on

def engineer_features(raw: dict) -> pd.DataFrame:
    """
    Takes raw applicant data (from the API request).
    Applies the same feature engineering as Phase 3.
    Returns a DataFrame ready for the model.
    """

    # Derived features — same formulas as Phase 3
    raw['debt_to_income']           = raw['debt_ratio'] * raw['monthly_income']
    raw['total_missed_payments']    = (raw['missed_30_59'] +
                                       raw['missed_60_89'] +
                                       raw['missed_90_plus'])
    raw['income_per_dependent']     = raw['monthly_income'] / (raw['dependents'] + 1)
    raw['high_credit_utilization']  = int(raw['credit_utilization'] > 0.3)
    raw['ever_seriously_late']      = int(raw['missed_90_plus'] > 0)
    raw['income_was_missing']       = 0  # API requires income, so never missing

    # Age group one-hot encoding — same bins as Phase 3
    age = raw['age']
    raw['age_group_26-35'] = int(26 <= age <= 35)
    raw['age_group_36-50'] = int(36 <= age <= 50)
    raw['age_group_51-65'] = int(51 <= age <= 65)
    raw['age_group_65+']   = int(age > 65)
    # Note: 18-25 is the reference category (all zeros = age 18-25)

    # Build DataFrame with columns in exact training order
    df = pd.DataFrame([raw])

    # Keep only features the model was trained on, in the right order
    df = df.reindex(columns=feature_names, fill_value=0)

    return df


# ── PLAIN ENGLISH REASON GENERATOR ───────────────────────────────────────────
# Same logic as Phase 5 — packaged as a reusable function

REASON_TEMPLATES = {
    'credit_utilization': {
        'high': "Credit utilization is {val:.0%} — most available credit is being used (threshold: 30%)",
        'low':  "Credit utilization is {val:.0%} — well within healthy limits"
    },
    'total_missed_payments': {
        'high': "{val:.0f} missed payment(s) recorded across all accounts",
        'low':  "No missed payments — strong payment history"
    },
    'monthly_income': {
        'high': "Monthly income of ${val:,.0f} supports this loan",
        'low':  "Monthly income of ${val:,.0f} is below the typical approved applicant"
    },
    'debt_to_income': {
        'high': "Debt-to-income of {val:.2f} — significant existing obligations",
        'low':  "Debt-to-income of {val:.2f} — within acceptable range"
    },
    'debt_ratio': {
        'high': "Debt ratio of {val:.2f} — high existing financial obligations",
        'low':  "Debt ratio of {val:.2f} — manageable debt load"
    },
    'age': {
        'high': "Applicant age of {val:.0f} is within the lower-risk range",
        'low':  "Applicant age of {val:.0f} — younger applicants show statistically higher default rates"
    },
    'ever_seriously_late': {
        'high': "Previous serious delinquency (90+ days late) on record",
        'low':  "No serious delinquency history"
    },
    'missed_90_plus': {
        'high': "{val:.0f} instance(s) of being 90+ days past due",
        'low':  "No 90+ day delinquencies"
    },
    'income_per_dependent': {
        'high': "Income per dependent of ${val:,.0f} — adequate buffer",
        'low':  "Income per dependent of ${val:,.0f} — financial pressure from dependents"
    },
    'open_credit_lines': {
        'high': "{val:.0f} open credit lines — higher exposure",
        'low':  "{val:.0f} open credit lines — manageable"
    },
}

def generate_reasons(feature_row: pd.DataFrame, shap_vals: np.ndarray,
                     top_n: int = 3) -> tuple[list[str], dict[str, float]]:
    """
    Returns:
        reasons      — list of plain English strings
        shap_impacts — dict of {feature: shap_value} for top features
    """
    pairs = sorted(
        zip(feature_names, shap_vals, feature_row.values[0]),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]

    reasons = []
    shap_impacts = {}

    for feature, shap_val, value in pairs:
        direction = 'high' if shap_val > 0 else 'low'
        template = REASON_TEMPLATES.get(feature, {}).get(
            direction,
            f"{feature.replace('_', ' ').title()}: {value:.3f}"
        )
        reasons.append(template.format(val=value))
        shap_impacts[feature] = round(float(shap_val), 4)

    return reasons, shap_impacts


# ── PROBABILITY → CREDIT SCORE ────────────────────────────────────────────────
def prob_to_score(probability: float) -> int:
    """
    Converts default probability to a 300-850 credit score scale.
    Higher score = lower risk (same as real credit scores).

    Formula: linear mapping from [0,1] probability to [850,300] score
    prob=0.0 → score=850 (perfect)
    prob=1.0 → score=300 (worst)
    """
    return int(850 - (probability * 550))


# ── DECISION LOGIC ────────────────────────────────────────────────────────────
def make_decision(probability: float) -> tuple[str, str]:
    """
    Returns (decision, confidence) based on predicted probability.
    Thresholds are tunable — this is a business decision, not a model decision.
    """
    if probability >= 0.6:
        return "REJECTED", "HIGH"
    elif probability >= 0.35:
        return "REVIEW", "MEDIUM"
    else:
        return "APPROVED", "HIGH" if probability < 0.15 else "MEDIUM"


# ── MAIN PREDICT FUNCTION ─────────────────────────────────────────────────────
def predict(applicant_dict: dict) -> dict:
    """
    Full prediction pipeline.
    Input:  raw applicant dict (from API request)
    Output: dict with decision, probability, score, reasons, metadata
    """

    # Step 1 — Engineer features
    feature_row = engineer_features(applicant_dict)

    # Step 2 — Predict probability
    probability = float(model.predict_proba(feature_row)[0, 1])

    # Step 3 — Calculate SHAP values for this applicant
    shap_vals = explainer.shap_values(feature_row)[0]

    # Step 4 — Generate plain English reasons
    reasons, shap_impacts = generate_reasons(feature_row, shap_vals)

    # Step 5 — Decision + score
    decision, confidence = make_decision(probability)
    risk_score = prob_to_score(probability)

    return {
        "decision":           decision,
        "default_probability": round(probability, 4),
        "risk_score":         risk_score,
        "top_reasons":        reasons,
        "shap_impacts":       shap_impacts,
        "model_version":      "xgboost-v2-1.0",
        "confidence":         confidence,
    }