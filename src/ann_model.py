"""
Artificial Neural Network (ANN / MLP) for Student Depression Prediction.
Uses sklearn MLPClassifier for consistency with the existing pipeline.
"""

import logging
import os
import pickle
import time

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
EXPERIMENT_NAME = "student-depression-prediction"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = {"f1_macro": "f1_macro", "roc_auc": "roc_auc", "accuracy": "accuracy"}


def train_ann(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Train a Multi-Layer Perceptron (ANN) classifier.

    Architecture:
    - Input layer: n_features (15)
    - Hidden layers: (128, 64, 32)
    - Output: sigmoid (binary)
    - Activation: relu
    - Optimizer: adam
    - Regularization: L2 (alpha=0.001)

    Parameters
    ----------
    X_train : preprocessed training features
    y_train : training labels

    Returns
    -------
    dict with model, cv_scores, run_id
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,          # L2 regularization
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=RANDOM_STATE,
    )

    logger.info("Training ANN (MLP) — architecture: (128, 64, 32) hidden layers...")
    t0 = time.time()
    cv_results = cross_validate(model, X_train, y_train, cv=CV,
                                scoring=SCORING, n_jobs=-1)
    elapsed = time.time() - t0

    scores = {
        f"{m}_mean": float(np.mean(cv_results[f"test_{m}"])) for m in SCORING
    }
    scores.update({
        f"{m}_std": float(np.std(cv_results[f"test_{m}"])) for m in SCORING
    })

    logger.info("ANN CV — F1-macro=%.4f±%.4f | ROC-AUC=%.4f | Accuracy=%.4f",
                scores["f1_macro_mean"], scores["f1_macro_std"],
                scores["roc_auc_mean"], scores["accuracy_mean"])

    # Fit on full training set
    model.fit(X_train, y_train)

    params = {
        "hidden_layers": "(128, 64, 32)",
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.001,
        "learning_rate_init": 0.001,
        "early_stopping": True,
    }

    with mlflow.start_run(run_name="ANN_MLP") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(scores)
        mlflow.log_metric("train_time_s", elapsed)
        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = run.info.run_id

    path = os.path.join(MODELS_DIR, "ann_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("ANN model saved to %s | MLflow run: %s", path, run_id)

    return {"model": model, "scores": scores, "run_id": run_id}
