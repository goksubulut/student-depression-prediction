"""
Bayesian Hyperparameter Optimization using scikit-optimize (skopt).
Optimizes LogisticRegression C parameter and RandomForest key params.
"""

import logging
import os
import pickle
import time

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
EXPERIMENT_NAME = "student-depression-prediction"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def run_bayesian_opt(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Run Bayesian optimization for LogisticRegression using BayesSearchCV.

    Search space:
    - C : log-uniform in [0.001, 100]
    - solver : lbfgs or saga
    - max_iter : 500 or 1000

    Parameters
    ----------
    X_train : preprocessed training features
    y_train : training labels

    Returns
    -------
    dict with best_model, best_params, best_score, run_id
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    search_space = {
        "C": Real(0.001, 100.0, prior="log-uniform"),
        "solver": Categorical(["lbfgs", "saga"]),
        "max_iter": Integer(500, 2000),
    }

    base = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)

    logger.info("Starting Bayesian optimization (n_iter=30)...")
    t0 = time.time()

    opt = BayesSearchCV(
        base,
        search_space,
        n_iter=30,
        cv=CV,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=0,
    )
    opt.fit(X_train, y_train)
    elapsed = time.time() - t0

    best_params = opt.best_params_
    best_score = opt.best_score_
    best_model = opt.best_estimator_

    logger.info("Bayesian opt done in %.1fs | Best params: %s | CV F1=%.4f",
                elapsed, best_params, best_score)

    # Full CV on best estimator
    full_scores = cross_val_score(best_model, X_train, y_train,
                                  cv=CV, scoring="f1_macro", n_jobs=-1)
    logger.info("Best model CV F1: %.4f ± %.4f", full_scores.mean(), full_scores.std())

    # Log to MLflow
    with mlflow.start_run(run_name="BayesOpt_LogisticRegression") as run:
        mlflow.log_params({str(k): str(v) for k, v in best_params.items()})
        mlflow.log_metric("cv_f1_macro_mean", float(full_scores.mean()))
        mlflow.log_metric("cv_f1_macro_std", float(full_scores.std()))
        mlflow.log_metric("bayes_best_score", float(best_score))
        mlflow.log_metric("search_time_s", elapsed)
        run_id = run.info.run_id

    # Save
    path = os.path.join(MODELS_DIR, "bayesopt_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(best_model, f)
    logger.info("Saved: %s | MLflow run: %s", path, run_id)

    return {
        "model": best_model,
        "best_params": best_params,
        "best_score": best_score,
        "cv_f1_mean": float(full_scores.mean()),
        "cv_f1_std": float(full_scores.std()),
        "run_id": run_id,
    }
