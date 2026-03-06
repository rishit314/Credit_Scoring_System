"""
schemas.py — Data shapes for our API

Pydantic BaseModel does two things automatically:
1. Validates incoming data (wrong type = instant error with clear message)
2. Generates the /docs page so anyone can test your API in a browser
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


# ── INPUT SCHEMA ──────────────────────────────────────────────────────────────
# This is the data a caller must send to /predict
# Every field maps to a feature our model was trained on

class ApplicantFeatures(BaseModel):
    """
    One loan applicant's features.
    These are the same columns we engineered in Phase 3.
    """

    # Core financial features
    credit_utilization: float = Field(
        ...,                        # "..." means required — no default
        ge=0,                       # must be >= 0
        le=10,                      # must be <= 10
        description="Revolving credit utilization (0 to 1 = normal, >1 = over limit)",
        example=0.45
    )
    monthly_income: float = Field(
        ...,
        ge=0,
        description="Monthly income in dollars",
        example=5000.0
    )
    debt_ratio: float = Field(
        ...,
        ge=0,
        description="Monthly debt payments divided by monthly gross income",
        example=0.35
    )
    age: int = Field(
        ...,
        ge=18,
        le=110,
        description="Applicant age in years",
        example=42
    )

    # Payment history features
    missed_30_59: int = Field(
        default=0,
        ge=0,
        description="Times 30-59 days past due in last 2 years",
        example=0
    )
    missed_60_89: int = Field(
        default=0,
        ge=0,
        description="Times 60-89 days past due in last 2 years",
        example=0
    )
    missed_90_plus: int = Field(
        default=0,
        ge=0,
        description="Times 90+ days past due in last 2 years",
        example=0
    )

    # Credit profile features
    open_credit_lines: int = Field(
        default=5,
        ge=0,
        description="Number of open credit lines and loans",
        example=8
    )
    real_estate_loans: int = Field(
        default=0,
        ge=0,
        description="Number of real estate loans or lines",
        example=1
    )
    dependents: int = Field(
        default=0,
        ge=0,
        description="Number of dependents",
        example=2
    )

    # Validator example — shows Pydantic's power
    @field_validator('credit_utilization')
    @classmethod
    def check_utilization(cls, v):
        if v > 5:
            raise ValueError(
                f"credit_utilization of {v} seems like a data error. "
                f"Values above 5 are extremely rare."
            )
        return v


# ── OUTPUT SCHEMA ─────────────────────────────────────────────────────────────
# This is exactly what your API sends BACK after making a prediction

class PredictionResponse(BaseModel):
    """
    What the API returns after scoring an applicant.
    """

    # Core decision
    decision: str = Field(
        description="Final decision: APPROVED, REVIEW, or REJECTED"
    )
    default_probability: float = Field(
        description="Model's predicted probability of default (0.0 to 1.0)"
    )
    risk_score: int = Field(
        description="Risk score from 300 (worst) to 850 (best) — like a credit score"
    )

    # Explanation
    top_reasons: list[str] = Field(
        description="Top 3 plain-English reasons for this decision"
    )
    shap_impacts: dict[str, float] = Field(
        description="Raw SHAP values for top features (for technical consumers)"
    )

    # Metadata
    model_version: str = Field(description="Which model version made this prediction")
    confidence: str = Field(description="HIGH, MEDIUM, or LOW confidence in prediction")


class ModelInfoResponse(BaseModel):
    """
    What /model-info returns — useful for monitoring and audits.
    """
    model_type: str
    roc_auc: float
    training_date: str
    feature_count: int
    default_threshold: float
    version: str