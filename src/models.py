"""
models.py
─────────
Model definitions and training wrappers for the Orthopedic Disease Prediction
project. Covers six classical ML algorithms and one MLP deep learning model.

Author : Akhi3011
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from typing import Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# Classical ML Models
# ─────────────────────────────────────────────────────────────────────────────

def get_classical_models() -> Dict[str, Any]:
    """
    Returns a dictionary of instantiated, hyperparameter-tuned classical ML models.

    Hyperparameters selected via grid search (see notebook for full search logs).
    random_state fixed at 42 for reproducibility across all stochastic models.
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="minkowski",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=4,
            criterion="gini",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=4,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        ),
    }


def train_classical_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """
    Fit all classical models on training data and report cross-validation accuracy.

    Parameters
    ----------
    models    : dict of model name → sklearn estimator
    X_train   : scaled training feature matrix
    y_train   : integer-encoded training labels
    cv_folds  : number of stratified CV folds (default=5)

    Returns
    -------
    Fitted model dictionary (in-place mutation, returned for convenience).
    """
    print(f"\n{'─'*60}")
    print(f"  Training {len(models)} classical models  (CV folds = {cv_folds})")
    print(f"{'─'*60}")

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy")
        model.fit(X_train, y_train)
        print(f"  {name:<28}  CV Acc: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Deep Learning — MLP
# ─────────────────────────────────────────────────────────────────────────────

def build_mlp(input_dim: int, num_classes: int = 3):
    """
    Build a regularized Multi-Layer Perceptron for 3-class classification.

    Architecture
    ────────────
    Input(input_dim)
      → Dense(128, ReLU)
      → BatchNormalization
      → Dropout(0.30)
      → Dense(64, ReLU)
      → BatchNormalization
      → Dropout(0.20)
      → Dense(num_classes, Softmax)

    Regularization strategy: Dropout + BatchNorm + Early Stopping (caller-side).
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            Dense(128, activation="relu", input_shape=(input_dim,), name="hidden_1"),
            BatchNormalization(),
            Dropout(0.30),
            Dense(64, activation="relu", name="hidden_2"),
            BatchNormalization(),
            Dropout(0.20),
            Dense(num_classes, activation="softmax", name="output"),
        ], name="MLP_OrthoClassifier")

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        print(f"\n[build_mlp]  Model summary:")
        model.summary()
        return model

    except ImportError:
        print("[build_mlp]  TensorFlow not installed — skipping MLP.")
        return None


def train_mlp(model, X_train, y_train, X_val, y_val,
              epochs: int = 100, batch_size: int = 32):
    """
    Train the MLP with early stopping on validation loss.

    Returns training history object for loss/accuracy curve plotting.
    """
    if model is None:
        return None

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history
