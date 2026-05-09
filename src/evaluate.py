"""
evaluate.py
───────────
Evaluation metrics computation and reporting for trained orthopedic
disease prediction models.

Author : Akhi3011
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix,
)
from typing import Dict, Any


CLASS_NAMES = ["Normal", "Disk Hernia", "Spondylolisthesis"]


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    is_keras: bool = False,
) -> Dict[str, float]:
    """
    Compute standard classification metrics for a fitted model.

    Parameters
    ----------
    model      : fitted sklearn estimator or Keras model
    X_test     : scaled test feature matrix
    y_test     : integer-encoded true labels
    model_name : display name for reporting
    is_keras   : True if model is a Keras/TF model

    Returns
    -------
    Dictionary with keys: accuracy, precision, recall, f1, roc_auc
    """
    if is_keras:
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = (
            model.predict_proba(X_test)
            if hasattr(model, "predict_proba")
            else None
        )

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    if y_pred_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(
                y_test, y_pred_proba, multi_class="ovr", average="macro"
            )
        except Exception:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def evaluate_all(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    mlp_model=None,
) -> pd.DataFrame:
    """
    Evaluate all classical models (and optionally MLP) and return a
    sorted comparison DataFrame.
    """
    records = []

    for name, model in models.items():
        m = evaluate_model(model, X_test, y_test, name)
        m["Model"] = name
        records.append(m)

    if mlp_model is not None:
        m = evaluate_model(mlp_model, X_test, y_test, "MLP (Deep Learning)", is_keras=True)
        m["Model"] = "MLP (Deep Learning)"
        records.append(m)

    df = pd.DataFrame(records).set_index("Model")
    df = df[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    df.columns = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    df = df.sort_values("Accuracy", ascending=False)

    print("\n" + "═"*68)
    print("  MODEL PERFORMANCE SUMMARY")
    print("═"*68)
    print(df.round(4).to_string())
    print("═"*68 + "\n")

    return df


def print_classification_report(model, X_test, y_test):
    """Print per-class precision, recall, F1 report."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
