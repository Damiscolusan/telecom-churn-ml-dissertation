"""Rebuild and verify the frozen Logistic Regression deployment artefact."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42
SELECTED_THRESHOLD = 0.56
SERVICE_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def prepare_training_data(path: Path):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    df["TotalServices"] = df[SERVICE_COLUMNS].eq("Yes").sum(axis=1)
    df["TenureCohort"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["0-12 Months", "12-24 Months", "24-48 Months", "48+ Months"],
    )
    expected_total = df["MonthlyCharges"] * df["tenure"]
    df["ChargeRatio"] = np.where(
        expected_total > 0, df["TotalCharges"] / expected_total, 1.0
    )
    df["AutoPayment"] = df["PaymentMethod"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype(int)
    categories = df.select_dtypes(include=["object", "string", "category"]).columns
    encoded = pd.get_dummies(df, columns=categories, drop_first=True)
    return encoded.drop(columns=["Churn"]), encoded["Churn"]


def main() -> None:
    X, y = prepare_training_data(ROOT / "telcos.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.01,
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= SELECTED_THRESHOLD).astype(int)

    observed_f1 = f1_score(y_test, predictions)
    observed_cm = confusion_matrix(y_test, predictions).ravel().tolist()
    if not np.isclose(observed_f1, 0.6237288135593221):
        raise RuntimeError(f"F1 verification failed: {observed_f1}")
    if observed_cm != [800, 235, 98, 276]:
        raise RuntimeError(f"Confusion-matrix verification failed: {observed_cm}")

    artefact = {
        "model": model,
        "feature_names": X_train.columns.tolist(),
        "threshold": SELECTED_THRESHOLD,
        "random_state": RANDOM_STATE,
        "training_columns": X_train.columns.tolist(),
    }
    joblib.dump(artefact, ROOT / "churn_model.joblib")
    print("Verified deployment artefact written to churn_model.joblib")


if __name__ == "__main__":
    main()
