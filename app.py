
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.inference import load_model_artefact, predict_customer


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "churn_model.joblib"
METADATA_PATH = APP_DIR / "model_metadata.json"

st.set_page_config(
    page_title="Telecom Churn Research Prototype",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {max-width: 1240px; padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stSidebar"] {border-right: 1px solid #d8e2e8;}
    .hero {
        padding: 1.45rem 1.6rem; border: 1px solid #d8e2e8; border-radius: 14px;
        background: linear-gradient(120deg, #f4f8fa 0%, #ffffff 72%);
        margin-bottom: 1.15rem;
    }
    .hero h1 {font-size: 2rem; color: #17384b; margin: 0 0 .35rem 0;}
    .hero p {color: #526773; margin: 0; max-width: 850px;}
    .eyebrow {font-size: .78rem; font-weight: 700; letter-spacing: .08em;
              text-transform: uppercase; color: #1f6f78; margin-bottom: .4rem;}
    .evidence-note {padding: .9rem 1rem; border-left: 4px solid #1f6f78;
                    background: #f1f7f7; border-radius: 0 8px 8px 0;}
    .risk-high {padding: 1rem 1.1rem; border-radius: 10px; background: #fff4f1;
                border: 1px solid #efc7bb; color: #713326;}
    .risk-low {padding: 1rem 1.1rem; border-radius: 10px; background: #eef8f3;
               border: 1px solid #bddfce; color: #20583e;}
    .method-step {padding: 1rem; border: 1px solid #d8e2e8; border-radius: 10px;
                  min-height: 142px; background: #ffffff;}
    .method-step b {color: #1f5a7a;}
    div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #dbe4e9;
                                 padding: .8rem 1rem; border-radius: 10px;}
    div[data-testid="stForm"] {border: 1px solid #d8e2e8; border-radius: 12px;
                                padding: 1rem 1rem .25rem 1rem;}
    .small-muted {font-size: .85rem; color: #627782;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_resources():
    artefact = load_model_artefact(MODEL_PATH)
    with METADATA_PATH.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if float(artefact["threshold"]) != float(metadata["model"]["selected_threshold"]):
        raise ValueError("Model and metadata thresholds do not match.")
    if len(artefact["training_columns"]) != metadata["dataset"]["model_features"]:
        raise ValueError("Model and metadata feature counts do not match.")
    return artefact, metadata


def friendly_feature(name: str) -> str:
    labels = {
        "SeniorCitizen": "Senior citizen",
        "tenure": "Tenure",
        "MonthlyCharges": "Monthly charges",
        "TotalCharges": "Total charges",
        "TotalServices": "Number of add-on services",
        "ChargeRatio": "Charge ratio",
        "AutoPayment": "Automatic payment",
    }
    if name in labels:
        return labels[name]
    return name.replace("_", ": ")


try:
    artefact, metadata = load_resources()
except Exception as exc:
    st.error("The verified model artefact could not be loaded.")
    st.code(str(exc))
    st.stop()

st.markdown(
    """
<section class="hero">
  <div class="eyebrow">MSc Artificial Intelligence Dissertation Project</div>
  <h1>Telecom churn decision-support prototype</h1>
  <p>This application demonstrates the Logistic Regression model developed for my
dissertation. It estimates customer churn probability from account, billing and
service information. The result supports human decision-making rather than
replacing it.</p>
<p class="small-muted">Ayodamola Olusanya |23109390 | University of South Wales | 2026</p>
</section>
""",
    unsafe_allow_html=True,
)

overview = metadata["dataset"]
summary_cols = st.columns(4)
summary_cols[0].metric("Dataset", f"{overview['records']:,} customers")
summary_cols[1].metric("Development set", f"{overview['development_rows']:,}")
summary_cols[2].metric("Held-out test set", f"{overview['held_out_rows']:,}")
summary_cols[3].metric("Observed churn", f"{overview['churn_rate']:.1%}")

with st.sidebar:
    st.markdown("### Prediction threshold")
    policy_values = {
        "Primary threshold, optimised for F1": 0.56,
        "Cost sensitivity, FN:FP 3:1": 0.53,
        "Cost sensitivity, FN:FP 5:1": 0.34,
        "Cost sensitivity, FN:FP 10:1": 0.26,
        "Custom exploration": None,
    }
    policy_name = st.selectbox("Operating threshold", list(policy_values))
    if policy_values[policy_name] is None:
        threshold = st.slider("Custom threshold", 0.05, 0.95, 0.56, 0.01)
    else:
        threshold = policy_values[policy_name]
    st.metric("Active threshold", f"{threshold:.0%}")
    st.caption(
        "The 0.56 primary threshold maximised out-of-fold F1 on development data. "
        "The other thresholds are sensitivity scenarios based on hypothetical "
        "relative error costs, not measured financial optima."
    )
    st.divider()
    st.markdown("### Model specification")
    st.markdown(
        "**Logistic Regression**  \n"
        "Class weight: `balanced`  \n"
        "C: `0.01`  \n"
        "Solver: `lbfgs`  \n"
        "Features: `36`  \n"
        "Random seed: `42`"
    )
    st.caption("The application loads a frozen model. It does not retrain on start-up.")

assessment_tab, evidence_tab, method_tab = st.tabs(
    ["Churn prediction", "Model performance", "Methodology and limitations"]
)

with assessment_tab:
    st.subheader("Enter customer details")
    st.caption(
        "Provide observed account characteristics. Service dependencies are enforced "
        "before feature engineering and prediction."
    )

    profile_col, account_col = st.columns(2, gap="large")
    with profile_col:
        st.markdown("#### Customer and account")
        p1, p2 = st.columns(2)
        gender = p1.selectbox("Gender", ["Female", "Male"])
        senior = p2.selectbox("Senior citizen", ["No", "Yes"])
        partner = p1.selectbox("Partner", ["No", "Yes"])
        dependents = p2.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure in months", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with account_col:
        st.markdown("#### Billing")
        b1, b2 = st.columns(2)
        paperless = b1.selectbox("Paperless billing", ["No", "Yes"])
        payment = b2.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly = st.number_input(
            "Monthly charges",
            min_value=18.0,
            max_value=120.0,
            value=70.0,
            step=0.5,
        )
        default_total = 0.0 if tenure == 0 else float(round(monthly * tenure, 2))
        total = st.number_input(
            "Total charges",
            min_value=0.0,
            max_value=9000.0,
            value=default_total,
            step=1.0,
            disabled=tenure == 0,
            help="Set to zero for a new customer with zero months of tenure.",
        )
        st.caption("The source dataset does not specify a currency.")
        if tenure == 0:
            total = 0.0

    st.markdown("#### Services")
    phone_col, internet_col = st.columns([1, 2], gap="large")
    with phone_col:
        phone = st.selectbox("Phone service", ["No", "Yes"], index=1)
        multiple_lines = st.selectbox(
            "Multiple lines",
            ["No", "Yes"],
            disabled=phone == "No",
            help="Recorded as 'No phone service' when phone service is absent.",
        )

    with internet_col:
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        has_internet = internet != "No"
        service_cols = st.columns(3)
        online_security = service_cols[0].selectbox(
            "Online security", ["No", "Yes"], disabled=not has_internet
        )
        online_backup = service_cols[1].selectbox(
            "Online backup", ["No", "Yes"], disabled=not has_internet
        )
        device_protection = service_cols[2].selectbox(
            "Device protection", ["No", "Yes"], disabled=not has_internet
        )
        tech_support = service_cols[0].selectbox(
            "Tech support", ["No", "Yes"], disabled=not has_internet
        )
        streaming_tv = service_cols[1].selectbox(
            "Streaming TV", ["No", "Yes"], disabled=not has_internet
        )
        streaming_movies = service_cols[2].selectbox(
            "Streaming movies", ["No", "Yes"], disabled=not has_internet
        )

    submitted = st.button("Estimate churn probability", type="primary", width="stretch")

    if submitted:
        customer = {
            "gender": gender,
            "SeniorCitizen": int(senior == "Yes"),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
        try:
            result = predict_customer(artefact, customer, threshold)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.divider()
            r1, r2, r3 = st.columns([1.2, 1, 1])
            r1.metric("Modelled churn probability", f"{result.probability:.1%}")
            r2.metric("Active threshold", f"{result.threshold:.0%}")
            r3.metric(
            "Prediction outcome",
            "Flagged for review" if result.flagged else "Not flagged for review",
                        )
            bar_colour = "#b34a32" if result.flagged else "#24775a"
            st.markdown(
                f"""
                <div style="height:12px;background:#e8eef1;border-radius:8px;overflow:hidden;margin:.25rem 0 1rem 0">
                  <div style="width:{result.probability * 100:.1f}%;height:100%;background:{bar_colour}"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if result.flagged:
                st.markdown(
                    '<div class="risk-high"><b>Customer flagged for review.</b> The score is at or above '
                    "the active operating threshold. Review the customer context before considering "
                    "any retention action.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="risk-low"><b>Customer not flagged for review.</b> The score is below the '
                    "active operating threshold. This is not a guarantee that the customer will remain.</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("#### Largest contributions to the model score")
            st.caption(
                "Contributions are exact effects on the fitted Logistic Regression log-odds after "
                "standardisation. A zero dummy value can still have a non-zero contribution because "
                "StandardScaler centres it relative to the training mean. These are predictive "
                "associations, not causal effects."
            )
            shown = result.contributions.head(8).copy()
            shown["Model feature"] = shown["feature"].map(friendly_feature)
            shown["Contribution"] = shown["contribution"].round(3)
            chart = (
                alt.Chart(shown)
                .mark_bar(cornerRadiusEnd=3)
                .encode(
                    x=alt.X("Contribution:Q", title="Contribution to log-odds"),
                    y=alt.Y("Model feature:N", sort="-x", title=None),
                    color=alt.Color(
                        "direction:N",
                        scale=alt.Scale(
                            domain=["Raises modelled risk", "Lowers modelled risk"],
                            range=["#b34a32", "#24775a"],
                        ),
                        legend=alt.Legend(title=None, orient="bottom"),
                    ),
                    tooltip=["Model feature:N", "value:Q", "Contribution:Q", "direction:N"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, width="stretch")

            with st.expander("Inspect model-ready record"):
                st.write(
                    f"The record was aligned to all {result.encoded_input.shape[1]} predictors "
                    "expected by the persisted model."
                )
                aligned = result.encoded_input.T.reset_index()
                aligned.columns = ["Feature", "Value"]
                st.dataframe(aligned, hide_index=True, width="stretch")

with evidence_tab:
    st.subheader("Evaluation evidence")
    st.markdown(
    '<div class="evidence-note">The results below were obtained from the final '
    'executed dissertation notebook. Model selection and threshold optimisation '
    'used the development data, while final performance was measured once on the '
    'held-out test set of 1,409 customers.</div>',
    unsafe_allow_html=True,
)

    st.markdown("#### Deployed operating point")
    selected = metadata["held_out_metrics"]["Logistic Regression (t=0.56)"]
    metric_cols = st.columns(6)
    for column, label, key in zip(
        metric_cols,
        ["F1", "Recall", "Precision", "ROC-AUC", "PR-AUC", "MCC"],
        ["F1", "Recall", "Precision", "ROC_AUC", "PR_AUC", "MCC"],
    ):
        column.metric(label, f"{selected[key]:.4f}")
    st.caption(
        "F1, recall, precision and MCC use the 0.56 operating threshold. ROC-AUC and "
        "PR-AUC assess ranking from probabilities and therefore do not change with that threshold."
    )

    held_out_names = [
        "Logistic Regression (t=0.56)",
        "Random Forest",
        "AdaBoost",
        "XGBoost",
        "MLP Neural Network",
        "Majority Class (baseline)",
    ]
    held_out = pd.DataFrame(
        {name: metadata["held_out_metrics"][name] for name in held_out_names}
    ).T.reset_index(names="Model")
    metric_choice = st.selectbox(
        "Compare held-out models by",
        ["F1", "Recall", "Precision", "ROC_AUC", "PR_AUC", "BalancedAcc", "MCC"],
    )
    comparison_chart = (
        alt.Chart(held_out)
        .mark_bar(cornerRadiusEnd=4, color="#2b6b88")
        .encode(
            x=alt.X(f"{metric_choice}:Q", title=metric_choice.replace("_", "-"), scale=alt.Scale(zero=True)),
            y=alt.Y("Model:N", sort="-x", title=None),
            tooltip=["Model:N", alt.Tooltip(f"{metric_choice}:Q", format=".4f")],
        )
        .properties(height=300)
    )
    st.altair_chart(comparison_chart, width="stretch")
    st.dataframe(
        held_out.set_index("Model").round(4),
        width="stretch",
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Confusion matrix at 0.56")
        cm = metadata["held_out_confusion_matrix"]
        cm_table = pd.DataFrame(
            [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
            index=["Actual retained", "Actual churned"],
            columns=["Predicted retained", "Predicted churned"],
        )
        st.dataframe(cm_table, width="stretch")
        st.caption("Rows are actual classes; columns are predicted classes. Total n = 1,409.")
    with right:
        st.markdown("#### Development threshold evidence")
        dev = metadata["development"]
        st.write(
            f"The 0.56 threshold was selected from out-of-fold development predictions: "
            f"precision **{dev['threshold_precision']:.4f}**, recall "
            f"**{dev['threshold_recall']:.4f}**, and F1 **{dev['threshold_f1']:.4f}**."
        )
        st.write(
            "At the selected held-out operating point, Logistic Regression remains competitive. "
            "XGBoost has the strongest benchmark F1, but it was not adopted after seeing the test "
            "result because that would reuse the held-out set for model selection."
        )

    with st.expander("Five-fold development comparison"):
        cv = pd.DataFrame(metadata["development"]["cross_validation"]).T
        st.dataframe(cv.round(4), width="stretch")

with method_tab:
    st.subheader("Methodological boundary")
    steps = st.columns(4)
    steps[0].markdown(
        '<div class="method-step"><b>1 · Prepare</b><br><br>Clean 7,043 records and '
        "create four deterministic engineered features.</div>",
        unsafe_allow_html=True,
    )
    steps[1].markdown(
        '<div class="method-step"><b>2 · Develop</b><br><br>Use a stratified 80/20 '
        "split. Tune and select the threshold using development data only.</div>",
        unsafe_allow_html=True,
    )
    steps[2].markdown(
        '<div class="method-step"><b>3 · Evaluate</b><br><br>Apply fixed decisions once '
        "to the 1,409-customer held-out test partition.</div>",
        unsafe_allow_html=True,
    )
    steps[3].markdown(
        '<div class="method-step"><b>4 · Deploy</b><br><br>Load the frozen pipeline and '
        "36-feature schema. Do not retrain inside the app.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### What the application reproduces")
    st.markdown(
        """
- The notebook's four engineered variables: `TotalServices`, `TenureCohort`, `ChargeRatio` and `AutoPayment`.
- The exact 36-column model schema and column order.
- Standardisation learned from the development partition.
- Class-weighted Logistic Regression with `C=0.01` and `lbfgs`.
- The primary development-selected threshold of 0.56.
- Exact coefficient-based contributions to the fitted log-odds.
        """
    )

    st.markdown("#### Limitations and responsible use")
    st.markdown(
        """
- The IBM Telco data are a static benchmark. External validity to a live provider, another country or a future customer population is unknown.
- The output is a model probability, not a causal estimate of whether an intervention would prevent churn.
- Cost-sensitive thresholds use hypothetical relative costs. They are sensitivity analyses, not observed monetary findings.
- The notebook dummy-encodes predictors before splitting. This transformation does not use the target, but a stricter future pipeline should fit `OneHotEncoder(handle_unknown='ignore')` on development data only.
- A production system would require input contracts, access control, drift and calibration monitoring, outcome logging, fairness review and controlled evaluation of retention interventions.
        """
    )
    st.warning(
        "Academic prototype only. Predictions should support accountable human judgement and "
        "must not be used as autonomous consequential decisions."
    )

st.divider()
st.caption(
    "Developed by Ayodamola Olusanya as part of an MSc Artificial Intelligence "
    "dissertation at the University of South Wales. Built using Python, "
    "scikit-learn and Streamlit."
)
