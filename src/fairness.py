"""
Subgroup and fairness evaluation for Student Depression Prediction.

Evaluates model performance across demographic and academic subgroups
to detect disparate impact or performance gaps.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def _save(fig, name: str) -> None:
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


def _metrics_for_group(y_true, y_pred, y_prob=None) -> dict:
    """Compute classification metrics for a single subgroup."""
    metrics = {
        "n": len(y_true),
        "prevalence": float(y_true.mean()),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_dep": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_dep": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_dep": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics


def subgroup_evaluation(model, X_test: np.ndarray, y_test: np.ndarray,
                         X_test_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate model performance across demographic and academic subgroups.

    Subgroups analyzed:
    - Gender (Male / Female)
    - Age group (<=20, 21–25, 26–30, 31+)
    - Academic Pressure level (Low 1–2 / Medium 3 / High 4–5)
    - Financial Stress level (Low 1–2 / Medium 3 / High 4–5)
    - Family History of Mental Illness (Yes / No)

    Parameters
    ----------
    model : fitted estimator
    X_test : preprocessed test features (numpy array)
    y_test : true labels
    X_test_raw : original (pre-encoding) test DataFrame for subgroup filtering

    Returns
    -------
    results_df : DataFrame with metrics per subgroup
    """
    y_test = np.array(y_test)
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba") else None
    )
    X_raw = X_test_raw.reset_index(drop=True)

    records = []

    # ── Overall ───────────────────────────────────────────────────────────────
    m = _metrics_for_group(y_test, y_pred, y_prob)
    records.append({"subgroup": "Overall", "group": "All", **m})

    # ── Gender ────────────────────────────────────────────────────────────────
    if "Gender" in X_raw.columns:
        for g in X_raw["Gender"].dropna().unique():
            mask = X_raw["Gender"] == g
            if mask.sum() < 30:
                continue
            m = _metrics_for_group(y_test[mask], y_pred[mask],
                                   y_prob[mask] if y_prob is not None else None)
            records.append({"subgroup": "Gender", "group": g, **m})

    # ── Age group ─────────────────────────────────────────────────────────────
    if "Age" in X_raw.columns:
        bins = [0, 20, 25, 30, 100]
        labels = ["<=20", "21–25", "26–30", "31+"]
        age_group = pd.cut(X_raw["Age"], bins=bins, labels=labels, right=True)
        for g in labels:
            mask = age_group == g
            if mask.sum() < 30:
                continue
            m = _metrics_for_group(y_test[mask], y_pred[mask],
                                   y_prob[mask] if y_prob is not None else None)
            records.append({"subgroup": "Age Group", "group": g, **m})

    # ── Academic Pressure ─────────────────────────────────────────────────────
    if "Academic Pressure" in X_raw.columns:
        def ap_level(v):
            if v <= 2:
                return "Low (1–2)"
            elif v == 3:
                return "Medium (3)"
            else:
                return "High (4–5)"
        ap_group = X_raw["Academic Pressure"].apply(ap_level)
        for g in ["Low (1–2)", "Medium (3)", "High (4–5)"]:
            mask = ap_group == g
            if mask.sum() < 30:
                continue
            m = _metrics_for_group(y_test[mask], y_pred[mask],
                                   y_prob[mask] if y_prob is not None else None)
            records.append({"subgroup": "Academic Pressure", "group": g, **m})

    # ── Financial Stress ──────────────────────────────────────────────────────
    if "Financial Stress" in X_raw.columns:
        def fs_level(v):
            if v <= 2:
                return "Low (1–2)"
            elif v == 3:
                return "Medium (3)"
            else:
                return "High (4–5)"
        fs_group = X_raw["Financial Stress"].apply(fs_level)
        for g in ["Low (1–2)", "Medium (3)", "High (4–5)"]:
            mask = fs_group == g
            if mask.sum() < 30:
                continue
            m = _metrics_for_group(y_test[mask], y_pred[mask],
                                   y_prob[mask] if y_prob is not None else None)
            records.append({"subgroup": "Financial Stress", "group": g, **m})

    # ── Family History ─────────────────────────────────────────────────────────
    if "Family History of Mental Illness" in X_raw.columns:
        for g in ["Yes", "No"]:
            mask = X_raw["Family History of Mental Illness"] == g
            if mask.sum() < 30:
                continue
            m = _metrics_for_group(y_test[mask], y_pred[mask],
                                   y_prob[mask] if y_prob is not None else None)
            records.append({"subgroup": "Family History", "group": g, **m})

    results_df = pd.DataFrame(records)
    logger.info("Subgroup evaluation complete — %d groups analyzed.", len(results_df))
    return results_df


def plot_subgroup_metrics(results_df: pd.DataFrame) -> None:
    """
    Bar chart comparing F1-macro and Recall (Depression class) across subgroups.
    Separate subplot per subgroup category.
    """
    categories = results_df[results_df["subgroup"] != "Overall"]["subgroup"].unique()
    n = len(categories)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for ax, cat in zip(axes, categories):
        subset = results_df[results_df["subgroup"] == cat].copy()
        x = np.arange(len(subset))
        width = 0.3

        ax.bar(x - width, subset["f1_macro"], width, label="F1-macro", color="steelblue")
        ax.bar(x,          subset["recall_dep"], width, label="Recall (Depression)", color="tomato")
        ax.bar(x + width,  subset["precision_dep"], width, label="Precision (Depression)", color="seagreen")

        # Overall reference line
        overall_f1 = results_df[results_df["subgroup"] == "Overall"]["f1_macro"].values[0]
        ax.axhline(overall_f1, color="navy", linestyle="--", linewidth=1.2, label=f"Overall F1 ({overall_f1:.3f})")

        ax.set_xticks(x)
        ax.set_xticklabels(subset["group"], rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Subgroup: {cat}", fontsize=12)
        ax.set_ylabel("Score")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Fairness Evaluation — Model Performance Across Subgroups", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, "subgroup_fairness.png")


def plot_subgroup_heatmap(results_df: pd.DataFrame) -> None:
    """Heatmap of key metrics across all subgroups."""
    display_df = results_df[results_df["subgroup"] != "Overall"].copy()
    display_df["label"] = display_df["subgroup"] + " | " + display_df["group"]
    heat = display_df.set_index("label")[["f1_macro", "recall_dep", "precision_dep", "roc_auc"]].astype(float)

    fig, ax = plt.subplots(figsize=(10, max(6, len(heat) * 0.5)))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.6, vmax=1.0, ax=ax, linewidths=0.5)
    ax.set_title("Subgroup Metric Heatmap", fontsize=13)
    ax.set_xlabel("Metric")
    ax.set_ylabel("")
    fig.tight_layout()
    _save(fig, "subgroup_heatmap.png")


def fairness_report(results_df: pd.DataFrame) -> dict:
    """
    Compute fairness metrics:
    - Max performance gap within each subgroup category
    - Flag subgroups where recall drops > 5% below overall
    """
    overall_recall = results_df[results_df["subgroup"] == "Overall"]["recall_dep"].values[0]
    overall_f1 = results_df[results_df["subgroup"] == "Overall"]["f1_macro"].values[0]

    report = {}
    categories = results_df[results_df["subgroup"] != "Overall"]["subgroup"].unique()

    for cat in categories:
        subset = results_df[results_df["subgroup"] == cat]
        f1_gap = subset["f1_macro"].max() - subset["f1_macro"].min()
        recall_gap = subset["recall_dep"].max() - subset["recall_dep"].min()
        low_recall = subset[subset["recall_dep"] < overall_recall - 0.05]

        report[cat] = {
            "f1_gap": round(f1_gap, 4),
            "recall_gap": round(recall_gap, 4),
            "flagged_groups": low_recall["group"].tolist(),
        }
        logger.info("[%s] F1 gap=%.4f | Recall gap=%.4f | Flagged: %s",
                    cat, f1_gap, recall_gap, low_recall["group"].tolist())

    return report


def run_fairness_evaluation(model, X_test: np.ndarray, y_test: np.ndarray,
                             X_test_raw: pd.DataFrame) -> dict:
    """
    Run full subgroup fairness evaluation pipeline.

    Returns
    -------
    dict with results_df, fairness_report
    """
    logger.info("Running subgroup fairness evaluation...")

    results_df = subgroup_evaluation(model, X_test, y_test, X_test_raw)

    # Print summary table
    cols = ["subgroup", "group", "n", "prevalence", "f1_macro", "recall_dep", "precision_dep"]
    logger.info("Subgroup results:\n%s", results_df[cols].round(4).to_string(index=False))

    plot_subgroup_metrics(results_df)
    plot_subgroup_heatmap(results_df)

    report = fairness_report(results_df)

    return {"results_df": results_df, "fairness_report": report}
