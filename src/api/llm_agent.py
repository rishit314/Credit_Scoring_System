"""
llm_agent.py — LLM agent layer on top of the deterministic credit model

DESIGN PRINCIPLE (important — read this before changing anything):
The XGBoost model + SHAP explainer make the actual risk DECISION.
The LLM is NEVER allowed to change a decision, invent a number, or override
the model. Its only job is to communicate the model's own output — as a
natural, well-written adverse-action notice or approval message, and (for
REVIEW cases) as a structured checklist of what a human underwriter should
verify next.

This keeps the system "bounded and gated": the LLM is a narrator, not a
decision-maker. Every LLM call is validated against the model's own output
before being returned, and every call — pass or fail — is written to an
append-only audit log.

Why Groq: Groq's LPU inference is fast enough (typically <1s) to sit inline
in a synchronous API request without materially hurting latency, which
matters for a "batch of 100 applicants" workflow.
"""

import os
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from groq import Groq

# ── CONFIG ────────────────────────────────────────────────────────────────────
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
BASE_DIR = Path(__file__).parent.parent.parent
AUDIT_LOG_PATH = Path(os.environ.get("AUDIT_LOG_PATH", BASE_DIR / "audit_log.jsonl"))

_client = None


def get_client() -> Groq:
    """Lazily create the Groq client so the API can still boot (and /health
    can still respond) even if GROQ_API_KEY isn't set — it will only fail
    when someone actually calls the LLM-backed endpoint."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Set it as an environment variable "
                "before calling /predict/agent."
            )
        _client = Groq(api_key=api_key)
    return _client


# ── PROMPT ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a compliance-safe communications assistant for a lending risk platform.

You will be given a FINAL, ALREADY-MADE decision from a risk model, along with its
probability, risk score, and the top SHAP-driven reasons behind it. Your only job is to
write the applicant-facing message and (if the decision is REVIEW) an underwriter
checklist. You must NOT change the decision, invent numbers, or add reasons that were
not given to you.

Rules:
- Never state a probability, score, or reason that was not provided to you.
- Never imply a different decision than the one given.
- Keep the applicant-facing message under 120 words, professional, and free of jargon.
- If decision is REVIEW, add 2-4 concrete, specific checklist items an underwriter should
  verify, based only on the reasons provided.
- Respond ONLY with valid JSON, no markdown fences, no preamble, matching this schema:
{
  "applicant_message": "string",
  "underwriter_checklist": ["string", ...]   // empty list unless decision is REVIEW
}
"""


def _build_user_prompt(prediction: dict) -> str:
    return json.dumps({
        "decision": prediction["decision"],
        "default_probability": prediction["default_probability"],
        "risk_score": prediction["risk_score"],
        "top_reasons": prediction["top_reasons"],
    })


# ── GUARDRAIL ─────────────────────────────────────────────────────────────────
def _guardrail_ok(llm_output: dict, prediction: dict) -> tuple[bool, str]:
    """Cheap, deterministic checks — not another LLM call — so the guardrail
    itself can't be talked into approving a bad output."""
    if "applicant_message" not in llm_output or not isinstance(llm_output["applicant_message"], str):
        return False, "missing or malformed applicant_message"

    if len(llm_output["applicant_message"].split()) > 160:
        return False, "applicant_message exceeds length bound"

    decision = prediction["decision"]
    msg_lower = llm_output["applicant_message"].lower()

    # The message must not claim a contradictory outcome.
    contradictions = {
        "REJECTED": ["approved", "congratulations, your loan"],
        "APPROVED": ["rejected", "denied", "unfortunately"],
    }
    for bad_phrase in contradictions.get(decision, []):
        if bad_phrase in msg_lower:
            return False, f"message contradicts decision ({bad_phrase!r} found)"

    checklist = llm_output.get("underwriter_checklist", [])
    if decision != "REVIEW" and checklist:
        return False, "checklist present for a non-REVIEW decision"
    if decision == "REVIEW" and not checklist:
        return False, "REVIEW decision missing an underwriter checklist"

    return True, "ok"


# ── AUDIT LOG ─────────────────────────────────────────────────────────────────
def _write_audit_record(record: dict) -> None:
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────
def generate_narrative(prediction: dict) -> dict:
    """
    Takes the model's own prediction dict (decision, probability, risk_score,
    top_reasons, shap_impacts — already computed by predictor.predict()) and
    returns an LLM-narrated version, with a guardrail check and an audit
    record for every call, pass or fail.
    """
    audit_id = str(uuid.uuid4())
    started = time.monotonic()
    timestamp = datetime.now(timezone.utc).isoformat()

    record = {
        "audit_id": audit_id,
        "timestamp": timestamp,
        "model_version": prediction.get("model_version"),
        "decision": prediction["decision"],
        "default_probability": prediction["default_probability"],
        "llm_model": GROQ_MODEL,
    }

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(prediction)},
            ],
            temperature=0.2,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        raw_text = response.choices[0].message.content
        llm_output = json.loads(raw_text)

    except Exception as e:
        record.update({
            "status": "llm_call_failed",
            "error": str(e),
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "fallback_used": True,
        })
        _write_audit_record(record)
        return _fallback_output(prediction, audit_id, reason=f"llm_call_failed: {e}")

    ok, reason = _guardrail_ok(llm_output, prediction)
    latency_ms = round((time.monotonic() - started) * 1000, 1)

    if not ok:
        record.update({
            "status": "guardrail_rejected",
            "reject_reason": reason,
            "raw_llm_output": llm_output,
            "latency_ms": latency_ms,
            "fallback_used": True,
        })
        _write_audit_record(record)
        return _fallback_output(prediction, audit_id, reason=f"guardrail_rejected: {reason}")

    record.update({
        "status": "ok",
        "latency_ms": latency_ms,
        "fallback_used": False,
    })
    _write_audit_record(record)

    return {
        "audit_id": audit_id,
        "applicant_message": llm_output["applicant_message"],
        "underwriter_checklist": llm_output.get("underwriter_checklist", []),
        "narrative_source": "llm",
        "guardrail_status": "passed",
    }


def _fallback_output(prediction: dict, audit_id: str, reason: str) -> dict:
    """
    Deterministic, template-based fallback — this is what makes the system
    'handle one failure gracefully' instead of just erroring out to the
    caller when Groq is slow, down, or returns something malformed.
    """
    decision = prediction["decision"]
    reasons_text = "; ".join(prediction["top_reasons"])

    templates = {
        "APPROVED": f"Your application has been approved. Key factors: {reasons_text}.",
        "REJECTED": f"Your application was not approved at this time. Key factors: {reasons_text}.",
        "REVIEW": f"Your application requires manual review. Key factors: {reasons_text}.",
    }

    checklist = (
        [f"Verify: {r}" for r in prediction["top_reasons"]]
        if decision == "REVIEW" else []
    )

    return {
        "audit_id": audit_id,
        "applicant_message": templates.get(decision, reasons_text),
        "underwriter_checklist": checklist,
        "narrative_source": "template_fallback",
        "guardrail_status": f"fallback ({reason})",
    }
