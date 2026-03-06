"""
dashboard.py — Streamlit Credit Scoring Dashboard

Streamlit works like this:
  - You write a normal Python script top to bottom
  - Every widget (slider, button, text box) is one line of code
  - When a user interacts with anything, the whole script reruns
  - Streamlit figures out what changed and updates only that part

No HTML. No CSS. No JavaScript. Just Python.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
# Must be the first Streamlit command in the file
st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="🏦",
    layout="wide",                    # use full browser width
    initial_sidebar_state="expanded"
)

# ── STYLING ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .approved-card {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        padding: 25px; border-radius: 15px; text-align: center;
        color: white; margin: 10px 0;
    }
    .rejected-card {
        background: linear-gradient(135deg, #7b1c1c, #c0392b);
        padding: 25px; border-radius: 15px; text-align: center;
        color: white; margin: 10px 0;
    }
    .review-card {
        background: linear-gradient(135deg, #7d5a00, #d4930a);
        padding: 25px; border-radius: 15px; text-align: center;
        color: white; margin: 10px 0;
    }
    .metric-card {
        background: #1e1e2e; padding: 20px; border-radius: 10px;
        text-align: center; margin: 5px;
    }
    .reason-box {
        background: #2a2a3e; padding: 15px; border-radius: 8px;
        border-left: 4px solid #e74c3c; margin: 8px 0; color: white;
    }
    .reason-box-good {
        background: #1a2e1a; padding: 15px; border-radius: 8px;
        border-left: 4px solid #2ecc71; margin: 8px 0; color: white;
    }
    .big-number { font-size: 2.5em; font-weight: bold; }
    .subtitle { font-size: 0.9em; opacity: 0.8; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
# Streamlit reruns the whole script on every interaction
# session_state persists data between reruns — like a memory
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🏦 Credit Scoring System")
st.markdown("*Explainable AI-powered loan default prediction*")
st.divider()

# ── SIDEBAR — APPLICANT INPUT FORM ────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Applicant Details")
    st.caption("Enter the applicant's financial information below")

    st.subheader("💰 Financial Profile")
    monthly_income = st.number_input(
        "Monthly Income ($)",
        min_value=0, max_value=500000,
        value=5000, step=100,
        help="Gross monthly income before taxes"
    )
    debt_ratio = st.slider(
        "Debt Ratio",
        min_value=0.0, max_value=5.0,
        value=0.35, step=0.01,
        help="Monthly debt payments / monthly gross income"
    )
    credit_utilization = st.slider(
        "Credit Utilization",
        min_value=0.0, max_value=5.0,
        value=0.3, step=0.01,
        help="Revolving balance / revolving credit limit (0.3 = 30% used)"
    )

    st.subheader("👤 Personal Details")
    age = st.number_input(
        "Age",
        min_value=18, max_value=100,
        value=40, step=1
    )
    dependents = st.number_input(
        "Number of Dependents",
        min_value=0, max_value=20,
        value=0, step=1
    )

    st.subheader("📊 Credit History")
    open_credit_lines = st.number_input(
        "Open Credit Lines",
        min_value=0, max_value=50,
        value=6, step=1,
        help="Total open credit lines and loans"
    )
    real_estate_loans = st.number_input(
        "Real Estate Loans",
        min_value=0, max_value=20,
        value=0, step=1
    )

    st.subheader("⚠️ Payment History")
    missed_30_59 = st.number_input(
        "Times 30-59 Days Late",
        min_value=0, max_value=20, value=0
    )
    missed_60_89 = st.number_input(
        "Times 60-89 Days Late",
        min_value=0, max_value=20, value=0
    )
    missed_90_plus = st.number_input(
        "Times 90+ Days Late",
        min_value=0, max_value=20, value=0
    )

    st.divider()
    submit = st.button(
        "🔍 Score This Applicant",
        type="primary",
        use_container_width=True
    )

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Decision Dashboard",
    "📈 Prediction History",
    "ℹ️ Model Info"
])

# ── TAB 1 — DECISION DASHBOARD ────────────────────────────────────────────────
with tab1:

    if submit:
        # Build request payload
        payload = {
            "credit_utilization": credit_utilization,
            "monthly_income": float(monthly_income),
            "debt_ratio": debt_ratio,
            "age": int(age),
            "missed_30_59": int(missed_30_59),
            "missed_60_89": int(missed_60_89),
            "missed_90_plus": int(missed_90_plus),
            "open_credit_lines": int(open_credit_lines),
            "real_estate_loans": int(real_estate_loans),
            "dependents": int(dependents)
        }

        # Call our FastAPI endpoint
        with st.spinner("Scoring applicant..."):
            try:
                response = requests.post(
                    "http://localhost:8000/predict",
                    json=payload,
                    timeout=30
                )
                result = response.json()
                st.session_state.last_result = result

                # Save to history
                st.session_state.prediction_history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "age": age,
                    "income": monthly_income,
                    "decision": result["decision"],
                    "probability": result["default_probability"],
                    "score": result["risk_score"]
                })

            except requests.exceptions.ConnectionError:
                st.error("""
                ❌ Cannot connect to the API.

                Make sure your FastAPI server is running:
                ```
                cd src/api
                uvicorn main:app --reload
                ```
                """)
                st.stop()

    # Display results
    if st.session_state.last_result:
        result = st.session_state.last_result
        decision = result["decision"]
        prob = result["default_probability"]
        score = result["risk_score"]
        reasons = result["top_reasons"]
        shap_impacts = result["shap_impacts"]

        # ── DECISION CARD ──
        card_class = {
            "APPROVED": "approved-card",
            "REJECTED": "rejected-card",
            "REVIEW":   "review-card"
        }[decision]

        decision_emoji = {"APPROVED": "✅", "REJECTED": "❌", "REVIEW": "⚠️"}[decision]
        decision_text  = {
            "APPROVED": "Application Approved",
            "REJECTED": "Application Rejected",
            "REVIEW":   "Manual Review Required"
        }[decision]

        st.markdown(f"""
        <div class="{card_class}">
            <div style="font-size: 3em">{decision_emoji}</div>
            <div style="font-size: 1.8em; font-weight: bold; margin: 10px 0">
                {decision_text}
            </div>
            <div style="opacity: 0.9">
                Confidence: {result['confidence']} &nbsp;|&nbsp; 
                Model: {result['model_version']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # ── KEY METRICS ROW ──
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Default Probability",
                value=f"{prob:.1%}",
                delta=f"{'High risk' if prob > 0.5 else 'Low risk'}",
                delta_color="inverse"
            )

        with col2:
            st.metric(
                label="Risk Score",
                value=f"{score}",
                delta=f"{'Below' if score < 580 else 'Above'} average (580)",
                delta_color="normal" if score >= 580 else "inverse"
            )

        with col3:
            total_missed = missed_30_59 + missed_60_89 + missed_90_plus
            st.metric(
                label="Total Missed Payments",
                value=total_missed,
                delta="Clean history" if total_missed == 0 else f"{total_missed} incidents",
                delta_color="normal" if total_missed == 0 else "inverse"
            )

        st.divider()

        # ── TWO COLUMN LAYOUT ──
        left_col, right_col = st.columns([1, 1])

        with left_col:
            # ── PROBABILITY GAUGE ──
            st.subheader("📊 Default Probability Gauge")

            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                number={'suffix': '%', 'font': {'size': 32}},
                delta={
                    'reference': 6.7,           # baseline default rate
                    'increasing': {'color': '#e74c3c'},
                    'decreasing': {'color': '#2ecc71'}
                },
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': '#e74c3c' if prob > 0.5 else
                                     '#f39c12' if prob > 0.35 else '#2ecc71'},
                    'steps': [
                        {'range': [0, 35],   'color': '#1a2e1a'},
                        {'range': [35, 60],  'color': '#2e2a1a'},
                        {'range': [60, 100], 'color': '#2e1a1a'},
                    ],
                    'threshold': {
                        'line': {'color': 'white', 'width': 3},
                        'thickness': 0.8,
                        'value': 35
                    }
                },
                title={'text': "Probability of Default<br><span style='font-size:0.8em;color:gray'>Threshold: 35%</span>"}
            ))
            gauge.update_layout(
                height=280,
                margin=dict(t=60, b=20, l=30, r=30),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            st.plotly_chart(gauge, use_container_width=True)

            # ── RISK SCORE BAR ──
            st.subheader("🎯 Risk Score")
            score_pct = (score - 300) / 550 * 100

            score_bar = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                gauge={
                    'axis': {'range': [300, 850]},
                    'bar': {'color': '#2ecc71' if score > 650 else
                                     '#f39c12' if score > 500 else '#e74c3c'},
                    'steps': [
                        {'range': [300, 500], 'color': '#2e1a1a'},
                        {'range': [500, 650], 'color': '#2e2a1a'},
                        {'range': [650, 850], 'color': '#1a2e1a'},
                    ]
                },
                title={'text': "Credit Risk Score<br><span style='font-size:0.8em;color:gray'>300 (worst) → 850 (best)</span>"}
            ))
            score_bar.update_layout(
                height=250,
                margin=dict(t=60, b=20, l=30, r=30),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            st.plotly_chart(score_bar, use_container_width=True)

        with right_col:
            # ── PLAIN ENGLISH REASONS ──
            st.subheader("📝 Decision Explanation")
            st.caption("Top factors that drove this decision (ECOA-compliant)")

            for i, reason in enumerate(reasons, 1):
                # Detect if reason is positive or negative
                negative_words = ['missed', 'late', 'high', 'elevated',
                                   'below', 'serious', 'delinquency', 'pressure']
                is_negative = any(w in reason.lower() for w in negative_words)

                box_class = "reason-box" if is_negative else "reason-box-good"
                icon = "🔴" if is_negative else "🟢"

                st.markdown(f"""
                <div class="{box_class}">
                    <strong>{icon} Reason {i}</strong><br>
                    {reason}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")

            # ── SHAP IMPACT CHART ──
            st.subheader("🔬 SHAP Feature Impacts")
            st.caption("How much each factor pushed the risk score up or down")

            shap_df = pd.DataFrame([
                {"Feature": k.replace("_", " ").title(), "Impact": v}
                for k, v in shap_impacts.items()
            ]).sort_values("Impact")

            colors = ['#2ecc71' if v < 0 else '#e74c3c'
                      for v in shap_df["Impact"]]

            shap_fig = go.Figure(go.Bar(
                x=shap_df["Impact"],
                y=shap_df["Feature"],
                orientation='h',
                marker_color=colors,
                text=[f"{v:+.4f}" for v in shap_df["Impact"]],
                textposition='outside'
            ))
            shap_fig.update_layout(
                height=280,
                xaxis_title="SHAP Value (+ increases risk, - decreases risk)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                margin=dict(t=20, b=40, l=20, r=60),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(shap_fig, use_container_width=True)

        # ── RAW JSON (collapsible) ──
        with st.expander("🔧 Raw API Response (for developers)"):
            st.json(result)

    else:
        # Empty state — shown before first prediction
        st.markdown("""
        <div style="text-align: center; padding: 60px; opacity: 0.5;">
            <div style="font-size: 4em">🏦</div>
            <h3>Ready to Score</h3>
            <p>Fill in the applicant details in the sidebar and click<br>
            <strong>"Score This Applicant"</strong> to see the decision.</p>
        </div>
        """, unsafe_allow_html=True)

# ── TAB 2 — PREDICTION HISTORY ────────────────────────────────────────────────
with tab2:
    st.subheader("📈 Prediction History")
    st.caption("All predictions made this session")

    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Scored", len(history_df))
        col2.metric("Approved", (history_df["decision"] == "APPROVED").sum())
        col3.metric("Rejected", (history_df["decision"] == "REJECTED").sum())
        col4.metric("For Review", (history_df["decision"] == "REVIEW").sum())

        st.divider()

        # Color-coded table
        def color_decision(val):
            colors = {
                "APPROVED": "background-color: #1a2e1a; color: #2ecc71",
                "REJECTED": "background-color: #2e1a1a; color: #e74c3c",
                "REVIEW":   "background-color: #2e2a1a; color: #f39c12"
            }
            return colors.get(val, "")

        styled_df = history_df.style.applymap(
            color_decision, subset=["decision"]
        ).format({
            "probability": "{:.1%}",
            "income": "${:,.0f}"
        })

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Risk distribution chart
        if len(history_df) > 1:
            st.subheader("Risk Score Distribution")
            fig = px.histogram(
                history_df, x="score",
                color="decision",
                color_discrete_map={
                    "APPROVED": "#2ecc71",
                    "REJECTED": "#e74c3c",
                    "REVIEW":   "#f39c12"
                },
                title="Distribution of Risk Scores This Session",
                nbins=20
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            st.plotly_chart(fig, use_container_width=True)

        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

    else:
        st.info("No predictions yet. Score some applicants in the Decision Dashboard tab!")

# ── TAB 3 — MODEL INFO ────────────────────────────────────────────────────────
with tab3:
    st.subheader("ℹ️ Model Information")

    try:
        info = requests.get("http://localhost:8000/model-info", timeout=5).json()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🤖 Model Details")
            st.markdown(f"""
            | Property | Value |
            |---|---|
            | Model Type | {info['model_type']} |
            | ROC-AUC | {info['roc_auc']} |
            | Training Date | {info['training_date']} |
            | Features | {info['feature_count']} |
            | Version | {info['version']} |
            | Decision Threshold | {info['default_threshold']} |
            """)

        with col2:
            st.markdown("### 📊 Performance Metrics")
            metrics = {
                "ROC-AUC": 0.8618,
                "PR-AUC": 0.4050,
                "KS Statistic": 0.5727,
                "Gini Coefficient": 0.7236
            }
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.4f}")

        st.divider()
        st.markdown("### 📜 Compliance Notes")
        st.info("""
        **ECOA Compliance**: This system generates adverse action notices
        with specific plain-English reasons for every rejection, satisfying
        the Equal Credit Opportunity Act's notification requirements.

        **EU AI Act**: SHAP-based explanations provide the algorithmic
        transparency required for high-risk AI systems under Article 13.

        **Model Card**: Full training methodology, dataset description,
        and fairness audit available in the project repository.
        """)

    except:
        st.warning("Cannot reach API. Make sure FastAPI is running on port 8000.")