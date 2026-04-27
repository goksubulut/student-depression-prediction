"""
Seed sensitivity analysis — tests how stable model performance is
across different random seeds.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

SEEDS = [0, 7, 21, 42, 99, 123, 256, 512, 1337, 2024]


def run_seed_sensitivity(X_train: np.ndarray, y_train: np.ndarray,
                         seeds: list = None) -> pd.DataFrame:
    """
    Train LogisticRegression with different random seeds and measure
    CV F1-macro variance. Tests whether seed=42 was a lucky pick.

    Parameters
    ----------
    X_train : preprocessed training features
    y_train : training labels
    seeds : list of int seeds to test (default: SEEDS)

    Returns
    -------
    pd.DataFrame with seed, mean F1, std F1 per seed
    """
    if seeds is None:
        seeds = SEEDS

    records = []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        model = LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=1000, random_state=seed
        )
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="f1_macro", n_jobs=-1)
        records.append({
            "seed": seed,
            "f1_mean": float(scores.mean()),
            "f1_std": float(scores.std()),
            "f1_min": float(scores.min()),
            "f1_max": float(scores.max()),
        })
        logger.info("Seed %4d | F1=%.4f ± %.4f", seed, scores.mean(), scores.std())

    df = pd.DataFrame(records)

    overall_mean = df["f1_mean"].mean()
    overall_std = df["f1_mean"].std()
    logger.info("Across all seeds — mean F1=%.4f | std=%.4f | range=[%.4f, %.4f]",
                overall_mean, overall_std, df["f1_mean"].min(), df["f1_mean"].max())

    return df


def plot_seed_sensitivity(results: pd.DataFrame) -> None:
    """Bar chart of F1-macro per seed with error bars."""
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(results))
    bars = ax.bar(x, results["f1_mean"], yerr=results["f1_std"],
                  capsize=4, color="steelblue", alpha=0.8, error_kw={"elinewidth": 1.5})

    # Highlight seed=42
    seed42_idx = results[results["seed"] == 42].index
    if len(seed42_idx) > 0:
        idx = results.index.get_loc(seed42_idx[0])
        bars[idx].set_color("tomato")
        bars[idx].set_label("seed=42 (project default)")

    ax.axhline(results["f1_mean"].mean(), color="navy", linestyle="--",
               linewidth=1.5, label=f"Mean across seeds ({results['f1_mean'].mean():.4f})")

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed={s}" for s in results["seed"]], rotation=30)
    ax.set_ylabel("CV F1-macro")
    ax.set_title("Seed Sensitivity Analysis — LogisticRegression (C=0.1)", fontsize=13)
    ax.set_ylim(results["f1_mean"].min() - 0.005, results["f1_mean"].max() + 0.01)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "seed_sensitivity.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)
