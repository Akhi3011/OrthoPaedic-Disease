"""
preprocessing.py
────────────────
Data loading, cleaning, normalization, and train/test split pipeline
for the Orthopedic Disease Prediction project.

Author : Akhi3011
Dataset: Orthopedic Patients (310 records, 6 biomechanical features)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "pelvic_incidence",
    "pelvic_tilt numeric",
    "lumbar_lordosis_angle",
    "sacral_slope",
    "pelvic_radius",
    "degree_spondylolisthesis",
]

TARGET_COL   = "class"
TEST_SIZE    = 0.20
RANDOM_STATE = 42

CLASS_MAP = {
    "Normal":            0,
    "Hernia":            1,
    "Spondylolisthesis": 2,
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame and perform basic quality checks."""
    df = pd.read_csv(csv_path)
    print(f"[load_raw]  Shape: {df.shape}")
    print(f"[load_raw]  Missing values:\n{df.isnull().sum()}\n")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map string class labels to integer codes and add a numeric target column."""
    df = df.copy()
    df["label"] = df[TARGET_COL].map(CLASS_MAP)
    if df["label"].isnull().any():
        unseen = df.loc[df["label"].isnull(), TARGET_COL].unique()
        raise ValueError(f"Unexpected class labels found: {unseen}")
    print(f"[encode_target]  Class distribution:\n{df[TARGET_COL].value_counts()}\n")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive two clinically-motivated composite features:

    PI_PT_Ratio        — Pelvic Incidence / Pelvic Tilt
                         Captures sacro-pelvic balance; imbalance correlates
                         with spondylolisthesis severity.

    Lumbar_Pelvic_Index — Lumbar Lordosis Angle × Sacral Slope
                          Reflects lumbar-pelvic coupling disruption associated
                          with spinal pathology.
    """
    df = df.copy()
    pt = df["pelvic_tilt numeric"].replace(0, np.nan)          # avoid div-by-zero
    df["PI_PT_Ratio"]         = df["pelvic_incidence"] / pt
    df["Lumbar_Pelvic_Index"] = df["lumbar_lordosis_angle"] * df["sacral_slope"]
    df["PI_PT_Ratio"].fillna(df["PI_PT_Ratio"].median(), inplace=True)
    print("[add_engineered_features]  Added: PI_PT_Ratio, Lumbar_Pelvic_Index")
    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and integer target vector y."""
    all_features = FEATURE_COLS + ["PI_PT_Ratio", "Lumbar_Pelvic_Index"]
    X = df[all_features]
    y = df["label"]
    return X, y


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Stratified 80/20 train-test split followed by StandardScaler fit on train
    only (to prevent data leakage).

    Returns
    -------
    X_train, X_test, y_train, y_test, fitted_scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[split_and_scale]  Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train.values, y_test.values, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(csv_path: str):
    """
    End-to-end preprocessing pipeline.

    Usage
    -----
    >>> X_train, X_test, y_train, y_test, scaler = load_and_preprocess("data/Orthopedic_patients.csv")
    """
    df = load_raw(csv_path)
    df = encode_target(df)
    df = add_engineered_features(df)
    X, y = build_feature_matrix(df)
    return split_and_scale(X, y)
