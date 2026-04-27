"""
Statistical significance testing for model comparison.

Implements:
- McNemar's test: compare two classifiers on the same test set
- Paired t-test: compare CV F1 scores across folds (5-fold)
- Bootstrap confidence intervals: estimate metric uncertainty
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def _save(fig, name: str) -> None:
    """Save figure to reports/figures/."""
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray,
                 y_pred_b: np.ndarray, name_a: str = "Model A",
                 name_b: str = "Model B") -> dict:
    """
    McNemar's test for comparing two classifiers on the same test set.

    Tests H0: both models make the same errors.
    Significant p-value means the models differ in their error patterns.

    Parameters
    ----------
    y_true : true labels
    y_pred_a, y_pred_b : predictions from model A and B
    name_a, name_b : model names for logging

    Returns
    -------
    dict with contingency table, statistic, p_value, interpretation
    """
    y_true = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Contingency table
    # [both correct, A correct B wrong]
    # [A wrong B correct, both wrong ]
    n00 = int(np.sum(~correct_a & ~correct_b))  # both wrong
    n01 = int(np.sum(~correct_a & correct_b))   # only B correct
    n10 = int(np.sum(correct_a & ~correct_b))   # only A correct
    n11 = int(np.sum(correct_a & correct_b))    # both correct

    table = np.array([[n11, n10], [n01, n00]])
    result = mcnemar(table, exact=False, correction=True)

    interpretation = (
        f"Significant difference (p={result.pvalue:.4f} < 0.05) — "
        f"{name_a} and {name_b} differ in error patterns."
        if result.pvalue < 0.05
        else
        f"No significant difference (p={result.pvalue:.4f} >= 0.05) — "
        f"cannot distinguish {name_a} from {name_b} on this test set."
    )

    logger.info("McNemar test: %s vs %s", name_a, name_b)
    logger.info("  Contingency table:\n    Both correct: %d | Only %s: %d\n    Only %s: %d | Both wrong: %d",
                n11, name_a, n10, name_b, n01, n00)
    logger.info("  Statistic=%.4f | p-value=%.4f", result.statistic, result.pvalue)
    logger.info("  %s", interpretation)

    return {
        "name_a": name_a, "name_b": name_b,
        "table": table,
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "interpretation": interpretation,
    }


def paired_ttest_cv(scores_a: np.ndarray, scores_b: np.ndarray,
                    name_a: str = "Model A", name_b: str = "Model B") -> dict:
    """
    Paired t-test on per-fold CV scores (5 folds).

    Tests H0: mean CV score of A == mean CV score of B.
    Each fold is a paired observation.

    Parameters
    ----------
    scores_a, scores_b : per-fold metric values (length = n_folds)
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    t_stat, p_value = ttest_rel(scores_a, scores_b)

    # Also run Wilcoxon signed-rank test (non-parametric alternative)
    try:
        w_stat, w_p = wilcoxon(scores_a, scores_b)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    interpretation = (
        f"Significant (p={p_value:.4f}) — {name_a} and {name_b} differ across folds."
        if p_value < 0.05
        else
        f"Not significant (p={p_value:.4f}) — no reliable difference between {name_a} and {name_b}."
    )

    logger.info("Paired t-test: %s vs %s", name_a, name_b)
    logger.info("  %s mean=%.4f | %s mean=%.4f", name_a, scores_a.mean(), name_b, scores_b.mean())
    logger.info("  t=%.4f | p=%.4f | Wilcoxon p=%.4f", t_stat, p_value, w_p)
    logger.info("  %s", interpretation)

    return {
        "name_a": name_a, "name_b": name_b,
        "mean_a": float(scores_a.mean()), "mean_b": float(scores_b.mean()),
        "t_stat": float(t_stat), "p_value": float(p_value),
        "wilcoxon_p": float(w_p),
        "interpretation": interpretation,
    }


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 metric_fn, n_bootstrap: int = 1000,
                 ci: float = 0.95, random_state: int = 42) -> dict:
    """
    Bootstrap confidence interval for a given metric.

    Parameters
    ----------
    y_true : true labels
    y_pred : predicted labels or scores
    metric_fn : callable(y_true, y_pred) -> float
    n_bootstrap : number of bootstrap resamples
    ci : confidence level (default 0.95)

    Returns
    -------
    dict with mean, lower, upper bounds
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    boot_scores = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        score = metric_fn(y_true[idx], y_pred[idx])
        boot_scores.append(score)

    boot_scores = np.array(boot_scores)
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_scores, alpha * 100))
    upper = float(np.percentile(boot_scores, (1 - alpha) * 100))
    mean = float(np.mean(boot_scores))

    return {"mean": mean, "lower": lower, "upper": upper, "ci": ci, "n_bootstrap": n_bootstrap}


def plot_cv_score_comparison(cv_results: dict) -> None:
    """
    Box plot of per-fold CV F1-macro scores for all models.

    Parameters
    ----------
    cv_results : {model_name: [fold_score_1, ..., fold_score_5]}
    """
    names = list(cv_results.keys())
    scores = [cv_results[n] for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(scores, labels=names, patch_artist=True, notch=False)

    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_title("CV F1-macro Score Distribution per Model (5-Fold)", fontsize=13)
    ax.set_ylabel("F1-macro")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, "cv_score_boxplot.png")


def run_significance_tests(models: dict, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Run full significance testing pipeline.

    Steps
    -----
    1. Collect per-fold CV F1 scores for all models
    2. Paired t-test: best model vs each other model
    3. McNemar's test: best model vs second best
    4. Bootstrap CI for best model's test F1
    5. CV score box plot

    Parameters
    ----------
    models : {name: fitted_estimator}
    X_train, y_train : training data
    X_test, y_test : held-out test data

    Returns
    -------
    dict with all test results
    """
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Collect per-fold scores
    logger.info("Collecting per-fold CV scores...")
    cv_fold_scores = {}
    for name, model in models.items():
        fold_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                      scoring="f1_macro", n_jobs=-1)
        cv_fold_scores[name] = fold_scores
        logger.info("  %s: folds=%s | mean=%.4f", name, np.round(fold_scores, 4), fold_scores.mean())

    # 2. Plot CV distributions
    plot_cv_score_comparison(cv_fold_scores)

    # 3. Identify best and second-best
    means = {n: s.mean() for n, s in cv_fold_scores.items()}
    ranked = sorted(means, key=means.get, reverse=True)
    best_name = ranked[0]
    second_name = ranked[1]
    best_model = models[best_name]
    second_model = models[second_name]

    logger.info("Best: %s (%.4f) | Second: %s (%.4f)",
                best_name, means[best_name], second_name, means[second_name])

    # 4. Paired t-test: best vs all others
    ttest_results = []
    for name in ranked[1:]:
        r = paired_ttest_cv(cv_fold_scores[best_name], cv_fold_scores[name],
                            name_a=best_name, name_b=name)
        ttest_results.append(r)

    # 5. McNemar's test: best vs second-best on test set
    y_pred_best = best_model.predict(X_test)
    y_pred_second = second_model.predict(X_test)
    mcnemar_result = mcnemar_test(y_test, y_pred_best, y_pred_second,
                                  name_a=best_name, name_b=second_name)

    # 6. Bootstrap CI for best model test F1
    logger.info("Computing bootstrap CI for %s test F1-macro...", best_name)
    ci_result = bootstrap_ci(
        y_true=np.array(y_test),
        y_pred=y_pred_best,
        metric_fn=lambda yt, yp: f1_score(yt, yp, average="macro"),
        n_bootstrap=1000,
        ci=0.95,
    )
    logger.info("Bootstrap 95%% CI for F1-macro: [%.4f, %.4f] (mean=%.4f)",
                ci_result["lower"], ci_result["upper"], ci_result["mean"])

    return {
        "cv_fold_scores": cv_fold_scores,
        "ttest_results": ttest_results,
        "mcnemar_result": mcnemar_result,
        "bootstrap_ci": ci_result,
        "best_model_name": best_name,
    }
