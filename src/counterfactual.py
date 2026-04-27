"""
Counterfactual explanations — "What would need to change to flip the prediction?"

Uses a simple gradient-based / nearest-neighbour approach without external libraries.
For each selected instance, finds the minimal feature perturbation that changes
the model output from Depression=1 to Depression=0 (or vice versa).
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def _save(fig, name: str) -> None:
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


def _nearest_counterfactual(instance: np.ndarray, X_pool: np.ndarray,
                             y_pool: np.ndarray, target_class: int) -> np.ndarray:
    """
    Find the nearest neighbour in X_pool that belongs to target_class.
    Uses L2 distance. Simple but effective for tabular data.
    """
    mask = y_pool == target_class
    candidates = X_pool[mask]
    dists = np.linalg.norm(candidates - instance, axis=1)
    return candidates[np.argmin(dists)]


def generate_counterfactuals(model, X_test: np.ndarray, y_test: np.ndarray,
                              X_train: np.ndarray, y_train: np.ndarray,
                              feature_names: list, n_instances: int = 4) -> pd.DataFrame:
    """
    Generate counterfactual explanations for n_instances test samples.

    For each selected instance (correctly predicted depressed),
    finds the nearest training sample predicted as non-depressed
    and shows the feature deltas required to flip the prediction.

    Parameters
    ----------
    model : fitted estimator
    X_test, y_test : test data
    X_train, y_train : training data (counterfactual pool)
    feature_names : list of feature names
    n_instances : number of instances to explain

    Returns
    -------
    pd.DataFrame with original, counterfactual, and delta per feature
    """
    y_pred = model.predict(X_test)
    y_test = np.array(y_test)

    # Select correctly predicted depressed samples
    tp_idx = np.where((y_pred == 1) & (y_test == 1))[0][:n_instances]

    records = []
    for idx in tp_idx:
        original = X_test[idx]
        cf = _nearest_counterfactual(original, X_train, y_train, target_class=0)
        delta = cf - original

        row = {"instance_idx": idx, "type": "original"}
        row.update({f: original[i] for i, f in enumerate(feature_names)})
        records.append(row)

        row_cf = {"instance_idx": idx, "type": "counterfactual"}
        row_cf.update({f: cf[i] for i, f in enumerate(feature_names)})
        records.append(row_cf)

        row_d = {"instance_idx": idx, "type": "delta"}
        row_d.update({f: delta[i] for i, f in enumerate(feature_names)})
        records.append(row_d)

    df = pd.DataFrame(records)
    return df


def plot_counterfactuals(cf_df: pd.DataFrame, feature_names: list,
                         n_instances: int = 4) -> None:
    """
    Bar chart showing feature deltas (original → counterfactual) for each instance.
    Only shows features with non-trivial change (|delta| > 0.01).
    """
    instances = cf_df["instance_idx"].unique()[:n_instances]
    fig, axes = plt.subplots(1, len(instances), figsize=(5 * len(instances), 6))
    if len(instances) == 1:
        axes = [axes]

    for ax, inst_idx in zip(axes, instances):
        subset = cf_df[cf_df["instance_idx"] == inst_idx]
        delta_row = subset[subset["type"] == "delta"].iloc[0]
        deltas = pd.Series(
            {f: delta_row[f] for f in feature_names},
            dtype=float
        )
        # Keep only features with meaningful change
        deltas = deltas[deltas.abs() > 0.01].sort_values()

        colors = ["steelblue" if v < 0 else "tomato" for v in deltas]
        ax.barh(range(len(deltas)), deltas.values, color=colors)
        ax.set_yticks(range(len(deltas)))
        ax.set_yticklabels(deltas.index, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Sample #{inst_idx}\nDelta to flip prediction", fontsize=9)
        ax.set_xlabel("Feature change required")

    fig.suptitle("Counterfactual Explanations — What needs to change to avoid Depression?",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, "counterfactual_explanations.png")


def run_counterfactual_analysis(model, X_test: np.ndarray, y_test: np.ndarray,
                                 X_train: np.ndarray, y_train: np.ndarray,
                                 feature_names: list) -> pd.DataFrame:
    """Run full counterfactual analysis pipeline."""
    logger.info("Generating counterfactual explanations...")
    cf_df = generate_counterfactuals(model, X_test, y_test, X_train, y_train,
                                      feature_names, n_instances=4)
    plot_counterfactuals(cf_df, feature_names)
    logger.info("Counterfactual analysis complete.")
    return cf_df
