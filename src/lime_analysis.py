"""
LIME (Local Interpretable Model-agnostic Explanations) for individual predictions.
Complements SHAP with a different local explanation approach.
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


def run_lime_analysis(model, X_train: np.ndarray, X_test: np.ndarray,
                      y_test: np.ndarray, feature_names: list,
                      n_samples: int = 5) -> dict:
    """
    Generate LIME explanations for selected test instances.

    Explains:
    - 3 correctly predicted depressed cases (True Positives)
    - 2 false negative cases (missed depression)

    Parameters
    ----------
    model : fitted estimator with predict_proba
    X_train : training data (for LIME background distribution)
    X_test : test features
    y_test : true test labels
    feature_names : list of feature names
    n_samples : number of instances to explain

    Returns
    -------
    dict with explanations and bar chart figure paths
    """
    try:
        import lime
        import lime.lime_tabular
    except ImportError:
        logger.error("lime not installed. Run: pip install lime")
        return {}

    y_pred = model.predict(X_test)
    y_test = np.array(y_test)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["No Depression", "Depression"],
        mode="classification",
        random_state=42,
    )

    # Select instances to explain
    tp_idx = np.where((y_pred == 1) & (y_test == 1))[0][:3]
    fn_idx = np.where((y_pred == 0) & (y_test == 1))[0][:2]
    explain_idx = list(tp_idx) + list(fn_idx)
    labels_map = {i: "tp" for i in tp_idx} | {i: "fn" for i in fn_idx}

    explanations = {}
    for idx in explain_idx:
        label = labels_map[idx]
        exp = explainer.explain_instance(
            X_test[idx],
            model.predict_proba,
            num_features=10,
            num_samples=1000,
        )

        # Extract feature contributions for class 1 (Depression)
        contrib = exp.as_list(label=1)
        features = [c[0] for c in contrib]
        weights = [c[1] for c in contrib]

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["tomato" if w > 0 else "steelblue" for w in weights]
        ax.barh(range(len(weights)), weights[::-1], color=colors[::-1])
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features[::-1], fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("LIME Weight (positive = towards Depression)")
        ax.set_title(
            f"LIME Explanation — {label.upper()} case (sample #{idx})\n"
            f"Predicted: {'Depression' if y_pred[idx]==1 else 'No Depression'} | "
            f"Actual: {'Depression' if y_test[idx]==1 else 'No Depression'}",
            fontsize=10,
        )
        fig.tight_layout()
        _save(fig, f"lime_{label}_sample{idx}.png")
        explanations[idx] = {"label": label, "contributions": contrib}

    logger.info("LIME analysis complete — %d explanations generated.", len(explanations))
    return explanations
