"""Validated preprocessing and inference for the frozen churn model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

AUTOMATIC_PAYMENT_METHODS = {
    "Bank transfer (automatic)",
    "Credit card (automatic)",
}


@dataclass(frozen=True)
class PredictionResult:
    probability: float
    flagged: bool
    threshold: float
    contributions: pd.DataFrame
    encoded_input: pd.DataFrame
    normalised_customer: dict[str, Any]


def load_model_artefact(path: str | Path) -> dict[str, Any]:
    """Load the frozen model and verify its deployment contract."""
    artefact = joblib.load(path)
    required = {"model", "threshold", "training_columns", "feature_names"}
    missing = required.difference(artefact)
    if missing:
        raise ValueError(f"Model artefact is missing keys: {sorted(missing)}")
    if list(artefact["training_columns"]) != list(artefact["feature_names"]):
        raise ValueError("Training columns and feature names are inconsistent.")
    if len(artefact["training_columns"]) != 36:
        raise ValueError("The frozen model must use the verified 36-feature schema.")
    return artefact


def _tenure_cohort(tenure: int) -> str:
    if tenure <= 12:
        return "0-12 Months"
    if tenure <= 24:
        return "12-24 Months"
    if tenure <= 48:
        return "24-48 Months"
    return "48+ Months"


def validate_raw_customer(customer: Mapping[str, Any]) -> None:
    required = {
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
    }
    missing = required.difference(customer)
    if missing:
        raise ValueError(f"Customer record is missing fields: {sorted(missing)}")

    tenure = int(customer["tenure"])
    monthly = float(customer["MonthlyCharges"])
    total = float(customer["TotalCharges"])
    if not 0 <= tenure <= 72:
        raise ValueError("Tenure must be between 0 and 72 months.")
    if not 18.0 <= monthly <= 120.0:
        raise ValueError("Monthly charges must be between 18 and 120.")
    if not 0.0 <= total <= 9000.0:
        raise ValueError("Total charges must be between 0 and 9,000.")
    if tenure == 0 and total != 0:
        raise ValueError("Total charges must be 0 when tenure is 0 months.")


def normalise_service_dependencies(customer: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the same logical service states present in the source dataset."""
    clean = dict(customer)
    if clean["PhoneService"] == "No":
        clean["MultipleLines"] = "No phone service"
    if clean["InternetService"] == "No":
        for column in SERVICE_COLUMNS:
            clean[column] = "No internet service"
    return clean


def engineer_and_encode(
    customer: Mapping[str, Any], training_columns: list[str]
) -> pd.DataFrame:
    """Reproduce the notebook's four engineered features and dummy schema."""
    validate_raw_customer(customer)
    clean = normalise_service_dependencies(customer)
    frame = pd.DataFrame([clean])

    frame["TotalServices"] = frame[SERVICE_COLUMNS].eq("Yes").sum(axis=1)
    frame["TenureCohort"] = frame["tenure"].map(_tenure_cohort)
    expected_total = frame["MonthlyCharges"] * frame["tenure"]
    frame["ChargeRatio"] = np.where(
        expected_total > 0,
        frame["TotalCharges"] / expected_total,
        1.0,
    )
    frame["AutoPayment"] = (
        frame["PaymentMethod"].isin(AUTOMATIC_PAYMENT_METHODS).astype(int)
    )

    # Do not use drop_first on a single record. Creating the observed dummy
    # columns and reindexing to the training schema preserves reference levels.
    encoded = pd.get_dummies(frame)
    encoded = encoded.reindex(columns=training_columns, fill_value=0)
    return encoded.astype(float)


def _logistic_contributions(model: Any, encoded: pd.DataFrame) -> pd.DataFrame:
    """Calculate exact per-feature contributions to the fitted log-odds."""
    scaler = model.named_steps.get("scaler")
    classifier = model.named_steps.get("clf")
    if scaler is None or classifier is None or not hasattr(classifier, "coef_"):
        return pd.DataFrame(columns=["feature", "value", "contribution", "direction"])

    scaled = scaler.transform(encoded)[0]
    coefficients = np.asarray(classifier.coef_).reshape(-1)
    contribution = scaled * coefficients
    result = pd.DataFrame(
        {
            "feature": encoded.columns,
            "value": encoded.iloc[0].to_numpy(),
            "contribution": contribution,
        }
    )
    result["direction"] = np.where(
        result["contribution"] >= 0,
        "Raises modelled risk",
        "Lowers modelled risk",
    )
    result["absolute_contribution"] = result["contribution"].abs()
    return result.sort_values("absolute_contribution", ascending=False).reset_index(drop=True)


def predict_customer(
    artefact: Mapping[str, Any],
    customer: Mapping[str, Any],
    threshold: float | None = None,
) -> PredictionResult:
    chosen_threshold = float(artefact["threshold"] if threshold is None else threshold)
    if not 0.05 <= chosen_threshold <= 0.95:
        raise ValueError("Threshold must be between 0.05 and 0.95.")

    clean = normalise_service_dependencies(customer)
    encoded = engineer_and_encode(clean, list(artefact["training_columns"]))
    model = artefact["model"]
    probability = float(model.predict_proba(encoded)[0, 1])
    return PredictionResult(
        probability=probability,
        flagged=probability >= chosen_threshold,
        threshold=chosen_threshold,
        contributions=_logistic_contributions(model, encoded),
        encoded_input=encoded,
        normalised_customer=clean,
    )
