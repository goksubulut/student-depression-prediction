"""Model training with MLflow tracking for Student Depression Prediction."""

import logging
import os
import pickle
import time

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logging.warning("xgboost not installed — XGBoost model will be skipped.")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logging.warning("lightgbm not installed — LightGBM model will be skipped.")

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
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
}


def _cv_scores(model, X, y) -> dict:
    """Run stratified 5-fold CV and return mean ± std for each metric."""
    results = cross_validate(model, X, y, cv=CV, scoring=SCORING, n_jobs=-1)
    scores = {}
    for metric in SCORING:
        vals = results[f"test_{metric}"]
        scores[f"{metric}_mean"] = float(np.mean(vals))
        scores[f"{metric}_std"] = float(np.std(vals))
    return scores


def _log_and_save(run_name: str, model, params: dict, scores: dict, train_time: float) -> str:
    """Log params/metrics to MLflow and return run_id."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(scores)
        mlflow.log_metric("train_time_s", train_time)
        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = run.info.run_id
    logger.info("[%s] run_id=%s | F1-macro=%.4f±%.4f", run_name, run_id, scores["f1_macro_mean"], scores["f1_macro_std"])
    return run_id


def train_all_models(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Train baseline, improved, and tuned models with MLflow tracking.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
    y_train : array-like of shape (n_samples,)

    Returns
    -------
    results : pd.DataFrame
        Comparison table sorted by F1-macro.
    best_model : fitted estimator
    best_run_id : str
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    models_to_train = [
        (
            "LogisticRegression",
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
            {"class_weight": "balanced", "max_iter": 1000},
        ),
        (
            "DecisionTree",
            DecisionTreeClassifier(class_weight="balanced", max_depth=5, random_state=RANDOM_STATE),
            {"class_weight": "balanced", "max_depth": 5},
        ),
        (
            "RandomForest",
            RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
            {"class_weight": "balanced", "n_estimators": 200},
        ),
    ]

    if HAS_XGB:
        models_to_train.append((
            "XGBoost",
            XGBClassifier(
                scale_pos_weight=scale_pos_weight, n_estimators=200,
                random_state=RANDOM_STATE, eval_metric="logloss",
                use_label_encoder=False, verbosity=0,
            ),
            {"scale_pos_weight": round(scale_pos_weight, 4), "n_estimators": 200},
        ))

    if HAS_LGB:
        models_to_train.append((
            "LightGBM",
            LGBMClassifier(
                is_unbalance=True, n_estimators=200,
                random_state=RANDOM_STATE, verbose=-1,
            ),
            {"is_unbalance": True, "n_estimators": 200},
        ))

    records = []
    run_ids = {}

    for name, model, params in models_to_train:
        logger.info("Training %s ...", name)
        t0 = time.time()
        scores = _cv_scores(model, X_train, y_train)
        elapsed = time.time() - t0
        run_id = _log_and_save(name, model, params, scores, elapsed)
        run_ids[name] = run_id
        records.append({"model": name, "run_id": run_id, **scores})

    # ── Hyperparameter tuning on best base model ──────────────────────────────
    # Identify best model by F1-macro mean
    results_df = pd.DataFrame(records).sort_values("f1_macro_mean", ascending=False)
    best_base_name = results_df.iloc[0]["model"]
    logger.info("Best base model: %s — tuning now...", best_base_name)

    tuned_model, tuned_params, tuned_run_id = _tune_best(
        best_base_name, X_train, y_train, scale_pos_weight
    )
    t0 = time.time()
    tuned_scores = _cv_scores(tuned_model, X_train, y_train)
    elapsed = time.time() - t0
    tuned_run_id = _log_and_save(f"{best_base_name}_Tuned", tuned_model, tuned_params, tuned_scores, elapsed)
    records.append({"model": f"{best_base_name}_Tuned", "run_id": tuned_run_id, **tuned_scores})

    results_df = pd.DataFrame(records).sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)
    logger.info("\nModel comparison (sorted by F1-macro):\n%s", results_df[["model", "f1_macro_mean", "roc_auc_mean"]].to_string(index=False))

    # Fit best tuned model on full train set
    tuned_model.fit(X_train, y_train)
    best_run_id = tuned_run_id

    # Save best model
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(tuned_model, f)
    logger.info("Best model saved to %s", model_path)

    return results_df, tuned_model, best_run_id


def _tune_best(name: str, X_train, y_train, scale_pos_weight: float) -> tuple:
    """Run GridSearchCV on the best base model type."""
    if "RandomForest" in name:
        base = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        }
    elif "XGBoost" in name and HAS_XGB:
        from xgboost import XGBClassifier
        base = XGBClassifier(
            scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
            eval_metric="logloss", use_label_encoder=False, verbosity=0,
        )
        param_grid = {
            "learning_rate": [0.05, 0.1],
            "max_depth": [4, 6],
            "n_estimators": [100, 200],
            "subsample": [0.8, 1.0],
        }
    elif "LightGBM" in name and HAS_LGB:
        from lightgbm import LGBMClassifier
        base = LGBMClassifier(is_unbalance=True, random_state=RANDOM_STATE, verbose=-1)
        param_grid = {
            "learning_rate": [0.05, 0.1],
            "max_depth": [4, 6],
            "n_estimators": [100, 200],
            "subsample": [0.8, 1.0],
        }
    else:
        # Fallback to LogReg
        base = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
        param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}

    grid = GridSearchCV(
        base, param_grid, cv=CV, scoring="f1_macro",
        n_jobs=-1, refit=True, verbose=0,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    best_params = grid.best_params_
    logger.info("Best params for %s: %s | CV F1=%.4f", name, best_params, grid.best_score_)
    return best, best_params, None
