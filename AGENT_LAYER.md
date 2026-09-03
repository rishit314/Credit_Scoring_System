# LLM Agent Layer — Setup, Testing, and Deployment

This adds a Groq-powered narration layer on top of the existing deterministic
XGBoost model. **The LLM never makes or changes a decision** — it only writes
the applicant-facing message and (for REVIEW cases) an underwriter checklist,
based strictly on the model's own output. Every call is guardrail-checked and
audit-logged, with an automatic fallback to a safe template if Groq fails or
the guardrail rejects the output.

## What was added

| File | What it does |
|---|---|
| `src/api/llm_agent.py` | Groq client, prompt, guardrail checks, audit logging, fallback |
| `src/api/schemas.py` | + `AgentPredictionResponse` (extends the existing response) |
| `src/api/main.py` | + `POST /predict/agent`, + `GET /audit-log` |
| `requirements-api.txt` | Lean deploy-only requirements (includes `groq`) |
| `Dockerfile` | Builds just the API service |
| `render.yaml` | One-click Render deploy config |
| `.env.example` | Template — copy to `.env`, fill in your real key |
| `.gitignore` | Keeps `.env` and audit logs out of git |

**Nothing in the existing `/predict`, `/predict/batch`, `/model-info`, or
`/health` endpoints changed.** They behave exactly as before.

## 1. Get a Groq API key

Free tier is enough for this. Go to https://console.groq.com/keys, sign up,
create a key.

## 2. Test locally

```bash
cd Credit_Scoring_System
cp .env.example .env
# edit .env and paste your real key in place of GROQ_API_KEY=your_groq_api_key_here

pip install -r requirements-api.txt   # or your existing requirements.txt, both work

cd src/api
uvicorn main:app --reload
```

Then hit the new endpoint:

```bash
curl -X POST http://localhost:8000/predict/agent \
  -H "Content-Type: application/json" \
  -d '{
    "credit_utilization": 1.07,
    "monthly_income": 3000,
    "debt_ratio": 0.6,
    "age": 29,
    "missed_30_59": 3,
    "missed_60_89": 2,
    "missed_90_plus": 3,
    "open_credit_lines": 10,
    "real_estate_loans": 0,
    "dependents": 1
  }'
```

You should see the normal decision/probability/reasons fields, **plus**
`applicant_message`, `underwriter_checklist`, `narrative_source` (should say
`"llm"` if your key works), `guardrail_status` (`"passed"`), and `audit_id`.

Check the audit trail:

```bash
curl "http://localhost:8000/audit-log?limit=10"
```

**If you ever see `narrative_source: "template_fallback"`** — that's not a
bug, that's the safety net working. Check the `guardrail_status` field for
why (bad/missing API key, Groq timeout, or the LLM output failed a guardrail
check). The response is still valid and safe to use either way.

## 3. What to actually demo in your pitch video

- Call `/predict/agent` once for a clearly REJECTED applicant (like the
  example above) and once for a borderline REVIEW applicant — show the
  underwriter checklist only appears for REVIEW.
- Show `/audit-log` right after, to prove every call is traceable.
- Optionally, temporarily break `GROQ_API_KEY` on camera and show the
  fallback kicking in instead of the API just erroring out — this is a
  genuinely strong "one failure handled gracefully" moment for the judges.

## 4. Deploy (Render)

1. Push this repo (including the new files, but **not** your `.env`) to
   GitHub.
2. Go to https://render.com → New → Web Service → connect your repo.
3. Render should auto-detect `render.yaml`. If not, set manually:
   - Environment: **Docker**
   - Dockerfile path: `./Dockerfile`
4. In the Render dashboard, under **Environment**, add `GROQ_API_KEY` with
   your real key (this is why `render.yaml` marks it `sync: false` — it's
   never stored in the repo).
5. Deploy. Render builds the Docker image and gives you a public URL like
   `https://credit-scoring-api-xxxx.onrender.com`.
6. Verify: `curl https://<your-url>/health`

**Note on the free tier:** Render's free web services spin down after
inactivity and take ~30-60s to wake up on the next request. For a live demo,
hit `/health` a minute before you go on camera to warm it up, or upgrade to
a paid instance for the actual interview slot.

## 5. Deploy the Streamlit dashboard too (optional)

`src/dashboard.py` isn't touched by any of this — it still works as-is,
pointed at whatever `API_URL` it's configured to call. If you want it public
too, Streamlit Community Cloud (share.streamlit.io) is the fastest free
option and doesn't need a Dockerfile.
