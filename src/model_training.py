""
model_training.py
-----------------
Train, evaluate, and persist machine learning models that predict whether
an electronics e-commerce order will be delivered on time or late.

Models supported:
    - Logistic Regression (baseline)
    - Random Forest Classifier
    - XGBoost Classifier
    - LightGBM Classifier

Author: hgabrali
Date: 2026
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
from optuna.samplers import TPESampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _ensure_models_dir() -> None:
    """Create the models directory if it does not exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)


def save_model(model: Any, name: str) -> str:
    """Persist a fitted model to disk using joblib.

    Parameters
    ----------
    model : Any
        A fitted scikit-learn-compatible estimator.
    name : str
        Base filename (without extension).

    Returns
    -------
    str
        Full path to the saved model file.
    """
    _ensure_models_dir()
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)
    return path


def load_model(name: str) -> Any:
    """Load a previously saved model from disk.

    Parameters
    ----------
    name : str
        Base filename (without extension).

    Returns
    -------
    Any
        Loaded estimator object.
    """
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    model = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """Compute and log classification metrics.

    Parameters
    ----------
    model : Any
        Fitted estimator.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True labels.
    model_name : str
        Human-readable model identifier for logging.

    Returns
    -------
    dict
        Dictionary containing accuracy, f1, roc_auc, and classification report.
    """
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    logger.info("=== %s Evaluation ===", model_name)
    logger.info("Accuracy : %.4f", acc)
    logger.info("F1 Score : %.4f", f1)
    if roc is not None:
        logger.info("ROC-AUC  : %.4f", roc)
    logger.info("Confusion Matrix:\n%s", cm)

    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "roc_auc": roc,
        "classification_report": report,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def get_logistic_regression(C: float = 1.0, max_iter: int = 1000) -> LogisticRegression:
    """Return a configured Logistic Regression estimator."""
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto",
    )


def get_random_forest(
    n_estimators: int = 300,
    max_depth: int = None,
    min_samples_split: int = 5,
) -> RandomForestClassifier:
    """Return a configured Random Forest estimator."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )


def get_xgboost(
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
) -> XGBClassifier:
    """Return a configured XGBoost estimator."""
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def get_lightgbm(
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    max_depth: int = -1,
) -> LGBMClassifier:
    """Return a configured LightGBM estimator."""
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------
def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> LogisticRegression:
    """Train and return a logistic regression baseline model."""
    logger.info("Training Logistic Regression baseline...")
    model = get_logistic_regression()
    model.fit(X_train, y_train)
    evaluate_model(model, X_val, y_val, "Logistic Regression (val)")
    save_model(model, "logistic_regression")
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> RandomForestClassifier:
    """Train and return a Random Forest classifier."""
    logger.info("Training Random Forest...")
    model = get_random_forest()
    model.fit(X_train, y_train)
    evaluate_model(model, X_val, y_val, "Random Forest (val)")
    save_model(model, "random_forest")
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> XGBClassifier:
    """Train and return an XGBoost classifier with early stopping."""
    logger.info("Training XGBoost...")
    model = get_xgboost()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    evaluate_model(model, X_val, y_val, "XGBoost (val)")
    save_model(model, "xgboost")
    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> LGBMClassifier:
    """Train and return a LightGBM classifier."""
    logger.info("Training LightGBM...")
    model = get_lightgbm()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
    )
    evaluate_model(model, X_val, y_val, "LightGBM (val)")
    save_model(model, "lightgbm")
    return model


# ---------------------------------------------------------------------------
# Hyperparameter tuning with Optuna
# ---------------------------------------------------------------------------
def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> XGBClassifier:
    """Use Optuna to find the best XGBoost hyperparameters.

    Parameters
    ----------
    X_train, y_train : array-like
        Training data.
    X_val, y_val : array-like
        Validation data used as the objective.
    n_trials : int
        Number of Optuna trials.

    Returns
    -------
    XGBClassifier
        Best model fitted on train data.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        clf = XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train, verbose=False)
        preds = clf.predict(X_val)
        return f1_score(y_val, preds, average="weighted")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info("Best Optuna trial — F1: %.4f", study.best_value)
    logger.info("Best params: %s", study.best_params)

    best_model = XGBClassifier(
        **study.best_params,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    best_model.fit(X_train, y_train, verbose=False)
    save_model(best_model, "xgboost_tuned")
    return best_model


# ---------------------------------------------------------------------------
# Train all models
# ---------------------------------------------------------------------------
def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Train all models and return them in a dictionary.

    Parameters
    ----------
    X_train, y_train : array-like
        Training split.
    X_val, y_val : array-like
        Validation split.

    Returns
    -------
    dict
        Mapping from model name to fitted estimator.
    """
    models = {
        "logistic_regression": train_baseline(X_train, y_train, X_val, y_val),
        "random_forest": train_random_forest(X_train, y_train, X_val, y_val),
        "xgboost": train_xgboost(X_train, y_train, X_val, y_val),
        "lightgbm": train_lightgbm(X_train, y_train, X_val, y_val),
    }
    return models


if __name__ == "__main__":
    from data_preprocessing import run_preprocessing_pipeline
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "../data/ecommerce_dataset.csv"
    X_train, X_val, X_test, y_train, y_val, y_test, _ = run_preprocessing_pipeline(data_path)

    trained = train_all_models(X_train, y_train, X_val, y_val)

    print("\n=== Final Test Set Evaluation ===")
    for name, model in trained.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        print(f"{name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_weighted']:.4f}")
