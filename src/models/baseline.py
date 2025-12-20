"""Baseline classification models and helpers for SPY experiments."""

from typing import Dict

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_baseline_pipeline(random_state: int = 42) -> Pipeline:
    """Create a standardized logistic regression pipeline."""

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, set_name: str) -> Dict[str, float]:
    """Compute basic classification metrics for a fitted model."""

    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    return {
        "Set": set_name,
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "ROC-AUC": roc_auc_score(y, y_pred_proba),
    }
