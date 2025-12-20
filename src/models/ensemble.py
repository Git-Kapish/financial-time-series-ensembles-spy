"""Ensemble model training and evaluation utilities."""

from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier


def tune_model_time_series(
    model, param_grid: Dict[str, Any], X_train, y_train, n_splits: int = 3
) -> Tuple[Any, pd.DataFrame]:
    """Grid search with time-series cross-validation and ROC-AUC scoring."""

    tscv = TimeSeriesSplit(n_splits=n_splits)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring="roc_auc",
        refit=True,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results_summary = cv_results[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].copy()
    cv_results_summary = cv_results_summary.sort_values("rank_test_score")

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, cv_results_summary


def train_random_forest(X_train, y_train):
    """Tune and fit a Random Forest classifier for time-series data."""

    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [4, 6, 8],
        "max_features": ["sqrt", 0.5],
    }

    return tune_model_time_series(rf_model, rf_param_grid, X_train, y_train, n_splits=3)


def make_xgb_base():
    """Base XGBoost classifier with sensible defaults."""

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_estimators=300,
    )


def train_xgboost(X_train, y_train):
    """Tune and fit an XGBoost classifier for time-series data."""

    xgb_model = make_xgb_base()
    xgb_param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
    }

    return tune_model_time_series(xgb_model, xgb_param_grid, X_train, y_train, n_splits=3)


def evaluate_ensemble(rf_model, xgb_model, X, y, set_name: str) -> Dict[str, float]:
    """Evaluate an ensemble by averaging RF and XGB predicted probabilities."""

    p_rf = rf_model.predict_proba(X)[:, 1]
    p_xgb = xgb_model.predict_proba(X)[:, 1]
    p_ens = 0.5 * (p_rf + p_xgb)
    y_pred = (p_ens > 0.5).astype(int)

    return {
        "Set": set_name,
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "ROC-AUC": roc_auc_score(y, p_ens),
    }
