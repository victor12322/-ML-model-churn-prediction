"""
Telco Customer Churn Prediction
================================
End-to-end pipeline: cleaning -> feature engineering -> modeling -> evaluation.

The goal is not maximal accuracy but a defensible, business-oriented workflow:
churn is imbalanced, so we optimise for recall on the churn class and report
the precision/recall trade-off explicitly rather than relying on accuracy.

Dataset: IBM Telco Customer Churn (public).
Download: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Place `Telco-Customer-Churn.csv` in ../data/
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / \
    "data" / "Telco-Customer-Churn.csv"
RANDOM_STATE = 42


def load_and_clean(path: Path) -> pd.DataFrame:
    """Load the raw CSV and fix known data-quality issues."""
    df = pd.read_csv(path)

    # TotalCharges is stored as string and contains blanks for new customers.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop the ID column: no predictive value, risk of leakage/overfitting.
    df = df.drop(columns=["customerID"])

    # Target to binary.
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few features that encode domain intuition about churn."""
    # Tenure buckets: churn risk is highly non-linear in tenure.
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
    )

    # Average monthly spend vs. current charge: flags recent price changes.
    df["avg_monthly"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] /
        df["tenure"], df["MonthlyCharges"]
    )

    return df


def build_pipeline(numeric: list[str], categorical: list[str], model) -> Pipeline:
    """Wrap preprocessing + estimator so the whole thing is one object."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline([("prep", preprocessor), ("clf", model)])


def evaluate(name: str, pipe: Pipeline, X_test, y_test) -> None:
    """Print a business-oriented evaluation for one model."""
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print(f"\n{'=' * 50}\n{name}\n{'=' * 50}")
    print(classification_report(
        y_test, y_pred, target_names=["Stay", "Churn"]))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Download it from Kaggle (see module docstring) and place it in data/."
        )

    df = load_and_clean(DATA_PATH)
    df = engineer_features(df)

    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = X.select_dtypes(
        include=["object", "category"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Baseline: interpretable, fast. class_weight handles imbalance.
    logreg = build_pipeline(
        numeric,
        categorical,
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    logreg.fit(X_train, y_train)
    evaluate("Logistic Regression (baseline)", logreg, X_test, y_test)

    # Stronger model for comparison.
    rf = build_pipeline(
        numeric,
        categorical,
        RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
        ),
    )
    rf.fit(X_train, y_train)
    evaluate("Random Forest", rf, X_test, y_test)


if __name__ == "__main__":
    main()
