from pathlib import Path

import numpy as np

from src.inference import engineer_and_encode, load_model_artefact, predict_customer


ROOT = Path(__file__).resolve().parents[1]
ARTEFACT = load_model_artefact(ROOT / "churn_model.joblib")


def customer(**overrides):
    record = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0, "TotalCharges": 840.0,
    }
    record.update(overrides)
    return record


def test_schema_and_engineered_features():
    encoded = engineer_and_encode(customer(), ARTEFACT["training_columns"])
    assert encoded.shape == (1, 36)
    assert encoded.columns.tolist() == ARTEFACT["training_columns"]
    assert encoded.loc[0, "TotalServices"] == 0
    assert encoded.loc[0, "AutoPayment"] == 0
    assert np.isclose(encoded.loc[0, "ChargeRatio"], 1.0)


def test_service_dependencies_are_normalised():
    result = predict_customer(
        ARTEFACT,
        customer(
            PhoneService="No", MultipleLines="Yes", InternetService="No",
            OnlineSecurity="Yes", StreamingTV="Yes",
        ),
    )
    encoded = result.encoded_input
    assert encoded.loc[0, "MultipleLines_No phone service"] == 1
    assert encoded.loc[0, "InternetService_No"] == 1
    assert encoded.loc[0, "OnlineSecurity_No internet service"] == 1
    assert encoded.loc[0, "StreamingTV_No internet service"] == 1
    assert encoded.loc[0, "TotalServices"] == 0


def test_threshold_changes_policy_not_probability():
    low = predict_customer(ARTEFACT, customer(), threshold=0.05)
    high = predict_customer(ARTEFACT, customer(), threshold=0.95)
    assert np.isclose(low.probability, high.probability)
    assert low.flagged and not high.flagged


def test_contributions_reconstruct_probability():
    result = predict_customer(ARTEFACT, customer())
    classifier = ARTEFACT["model"].named_steps["clf"]
    logit = float(classifier.intercept_[0]) + result.contributions["contribution"].sum()
    reconstructed = 1 / (1 + np.exp(-logit))
    assert np.isclose(reconstructed, result.probability)


def test_high_and_low_profiles_match_notebook_direction():
    high = predict_customer(
        ARTEFACT,
        customer(
            tenure=2, InternetService="Fiber optic", StreamingTV="Yes",
            StreamingMovies="Yes", MonthlyCharges=94.5, TotalCharges=189.0,
        ),
    )
    low = predict_customer(
        ARTEFACT,
        customer(
            tenure=60, Contract="Two year", InternetService="DSL",
            MonthlyCharges=45.0, TotalCharges=2700.0,
            PaymentMethod="Credit card (automatic)", OnlineSecurity="Yes",
            TechSupport="Yes",
        ),
    )
    assert high.probability > 0.90
    assert low.probability < 0.10
