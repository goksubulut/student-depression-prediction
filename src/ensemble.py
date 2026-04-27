"""
BONUS — Stacking/Blending ensemble for Student Depression Prediction.

Stacks base learners (LR, RF, XGBoost, LightGBM) with a meta-learner
(Logistic Regression) using out-of-fold (OOF) predictions to avoid leakage.
"""

import logging
import os
import pickle
import time

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
EXPERIMENT_NAME = "student-depression-prediction"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = {
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
}


def build_stacking_classifier() -> StackingClassifier:
    """
    Build a stacking ensemble with:
    - Base learners: LogisticRegression, RandomForest, XGBoost, LightGBM
    - Meta-learner: LogisticRegression (trained on OOF predictions)

    Uses passthrough=True so meta-learner also sees original features.

    Returns
    -------
    StackingClassifier (unfitted)
    """
    estimators = [
        ("lr", LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        )),
        ("rf", RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        )),
    ]

    try:
        from xgboost import XGBClassifier
        estimators.append(("xgb", XGBClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            eval_metric="logloss", use_label_encoder=False, verbosity=0,
        )))
    except ImportError:
        logger.warning("XGBoost not available — skipped from stack.")

    try:
        from lightgbm import LGBMClassifier
        estimators.append(("lgb", LGBMClassifier(
            n_estimators=200, is_unbalance=True,
            random_state=RANDOM_STATE, verbose=-1,
        )))
    except ImportError:
        logger.warning("LightGBM not available — skipped from stack.")

    meta_learner = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    )

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=CV,
        stack_method="predict_proba",
        passthrough=True,
        n_jobs=-1,
    )
    logger.info("Stacking ensemble built with %d base learners + LR meta-learner.", len(estimators))
    return stack


def train_stacking(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Train stacking ensemble and log to MLflow.

    Parameters
    ----------
    X_train : preprocessed training features
    y_train : training labels

    Returns
    -------
    dict with model, run_id, cv_scores
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    stack = build_stacking_classifier()

    logger.info("Training stacking ensemble (this may take a few minutes)...")
    t0 = time.time()

    cv_results = cross_validate(stack, X_train, y_train, cv=CV,
                                scoring=SCORING, n_jobs=1, return_train_score=False)
    elapsed = time.time() - t0

    scores = {
        f"{m}_mean": float(np.mean(cv_results[f"test_{m}"]))
        for m in SCORING
    }
    scores.update({
        f"{m}_std": float(np.std(cv_results[f"test_{m}"]))
        for m in SCORING
    })

    logger.info("Stacking CV — F1-macro=%.4f±%.4f | ROC-AUC=%.4f",
                scores["f1_macro_mean"], scores["f1_macro_std"], scores["roc_auc_mean"])

    # Fit on full train set
    stack.fit(X_train, y_train)

    params = {
        "base_learners": "LR, RF, XGBoost, LightGBM",
        "meta_learner": "LogisticRegression(C=1.0)",
        "passthrough": True,
        "cv_folds": 5,
        "stack_method": "predict_proba",
    }

    with mlflow.start_run(run_name="StackingEnsemble") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(scores)
        mlflow.log_metric("train_time_s", elapsed)
        mlflow.sklearn.log_model(stack, artifact_path="model")
        run_id = run.info.run_id

    logger.info("StackingEnsemble run_id=%s", run_id)

    model_path = os.path.join(MODELS_DIR, "stacking_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(stack, f)
    logger.info("Stacking model saved to %s", model_path)

    return {"model": stack, "run_id": run_id, "scores": scores}


def compare_with_best(stack_scores: dict, best_f1: float, best_roc: float) -> None:
    """Log a side-by-side comparison of stacking vs best single model."""
    logger.info("=" * 50)
    logger.info("Model Comparison:")
    logger.info("  Best single (LR_Tuned)  F1=%.4f | ROC=%.4f", best_f1, best_roc)
    logger.info("  Stacking ensemble       F1=%.4f | ROC=%.4f",
                stack_scores["f1_macro_mean"], stack_scores["roc_auc_mean"])
    delta_f1 = stack_scores["f1_macro_mean"] - best_f1
    logger.info("  Delta F1: %+.4f", delta_f1)
    logger.info("=" * 50)
