"""Evaluation and error analysis module for Student Depression Prediction."""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    f1_score,
    average_precision_score,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure to reports/figures/ and close."""
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


def full_classification_report(y_true, y_pred) -> str:
    """Print and return sklearn classification report."""
    report = classification_report(y_true, y_pred, target_names=["No Depression", "Depression"])
    logger.info("Classification Report:\n%s", report)
    return report


def plot_confusion_matrix(y_true, y_pred) -> None:
    """Plot and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["No Depression", "Depression"],
        cmap="Blues", ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=13)
    fig.tight_layout()
    _save(fig, "confusion_matrix.png")


def plot_roc_curve(y_true, y_prob) -> float:
    """Plot ROC curve and return AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Best Model", fontsize=13)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "roc_curve.png")
    logger.info("ROC-AUC: %.4f", auc)
    return auc


def plot_precision_recall_curve(y_true, y_prob) -> float:
    """Plot Precision-Recall curve and return average precision."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="steelblue", lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Best Model", fontsize=13)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "precision_recall_curve.png")
    logger.info("Average Precision: %.4f", ap)
    return ap


def plot_feature_importance(model, feature_names: list, X_test, y_test) -> None:
    """
    Plot top-15 feature importances.
    Uses model.feature_importances_ if available, else permutation importance.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = "Feature Importances (model built-in)"
    else:
        logger.info("Computing permutation importances (may take a moment)...")
        perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        importances = perm.importances_mean
        title = "Permutation Feature Importances"

    indices = np.argsort(importances)[::-1][:15]
    top_names = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(top_names)), top_vals[::-1], color="steelblue")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"{title} — Top 15", fontsize=13)
    fig.tight_layout()
    _save(fig, "feature_importance.png")


def error_analysis(X_test_orig: pd.DataFrame, y_true, y_pred) -> dict:
    """
    Identify false positives and false negatives and print pattern summary.

    Parameters
    ----------
    X_test_orig : DataFrame with original (pre-encoding) feature values
    y_true : true labels
    y_pred : predicted labels

    Returns
    -------
    dict with fp_df and fn_df DataFrames
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)

    if isinstance(X_test_orig, pd.DataFrame):
        fp_df = X_test_orig[fp_mask].head(10)
        fn_df = X_test_orig[fn_mask].head(10)
    else:
        fp_df = pd.DataFrame(X_test_orig[fp_mask][:10])
        fn_df = pd.DataFrame(X_test_orig[fn_mask][:10])

    logger.info("=== FALSE POSITIVES (predicted depressed, actually not) ===")
    logger.info("\n%s", fp_df.to_string())

    logger.info("=== FALSE NEGATIVES (predicted not depressed, actually depressed) ===")
    logger.info("\n%s", fn_df.to_string())

    # Pattern summary
    numeric_cols = [c for c in ["Age", "CGPA", "Academic Pressure", "Work/Study Hours", "Financial Stress"]
                    if c in fp_df.columns]
    if numeric_cols and len(fp_df) > 0:
        logger.info("--- FP pattern (means): %s", fp_df[numeric_cols].mean().to_dict())
    if numeric_cols and len(fn_df) > 0:
        logger.info("--- FN pattern (means): %s", fn_df[numeric_cols].mean().to_dict())

    total_fp = fp_mask.sum()
    total_fn = fn_mask.sum()
    logger.info("Total FP: %d | Total FN: %d", total_fp, total_fn)

    return {"fp_df": fp_df, "fn_df": fn_df, "total_fp": total_fp, "total_fn": total_fn}


def run_evaluation(model, X_test, y_test, feature_names: list, X_test_orig=None) -> dict:
    """
    Full evaluation of the best model on the held-out test set.

    Parameters
    ----------
    model : fitted estimator
    X_test : preprocessed test features
    y_test : true test labels
    feature_names : list of feature names (post RFE)
    X_test_orig : optional raw DataFrame for error analysis

    Returns
    -------
    metrics : dict with f1_macro, roc_auc, average_precision
    """
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )

    report = full_classification_report(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)
    auc = plot_roc_curve(y_test, y_prob)
    ap = plot_precision_recall_curve(y_test, y_prob)
    plot_feature_importance(model, feature_names, X_test, y_test)

    f1 = f1_score(y_test, y_pred, average="macro")

    if X_test_orig is not None:
        error_analysis(X_test_orig, y_test, y_pred)

    metrics = {"f1_macro": f1, "roc_auc": auc, "average_precision": ap}
    logger.info("Test F1-macro: %.4f | ROC-AUC: %.4f | Avg-Precision: %.4f", f1, auc, ap)
    return metrics
