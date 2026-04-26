"""SHAP explainability analysis for the Student Depression Prediction project."""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def _save(fig, name: str) -> None:
    """Save figure to reports/figures/ and close."""
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


def compute_shap_values(model, X_train: np.ndarray, X_test: np.ndarray, feature_names: list):
    """
    Compute SHAP values using the appropriate explainer for the model type.

    Uses LinearExplainer for linear models, TreeExplainer for tree-based models,
    and KernelExplainer as fallback.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_train : training data (for background distribution)
    X_test : test data to explain
    feature_names : list of feature names

    Returns
    -------
    explainer : shap Explainer object
    shap_values : np.ndarray of shape (n_samples, n_features)
    X_test_df : pd.DataFrame with feature names
    """
    model_type = type(model).__name__
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    X_train_df = pd.DataFrame(X_train, columns=feature_names)

    logger.info("Computing SHAP values for %s ...", model_type)

    if model_type in ("LogisticRegression", "LinearSVC", "Ridge", "Lasso"):
        explainer = shap.LinearExplainer(model, X_train_df, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_test_df)
    elif model_type in ("RandomForestClassifier", "ExtraTreesClassifier",
                        "GradientBoostingClassifier", "XGBClassifier",
                        "LGBMClassifier", "DecisionTreeClassifier"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_df)
        # Tree models return list [class0, class1] — take class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        logger.info("Using KernelExplainer (slow) — sampling 200 background rows")
        background = shap.sample(X_train_df, 200, random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test_df[:500], nsamples=100)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    logger.info("SHAP values computed: shape=%s", np.array(shap_values).shape)
    return explainer, shap_values, X_test_df


def plot_shap_summary_beeswarm(shap_values, X_test_df: pd.DataFrame) -> None:
    """
    SHAP beeswarm summary plot — shows feature impact distribution across all samples.
    Each dot = one sample, color = feature value, x-position = SHAP value.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test_df,
        show=False, plot_size=None,
    )
    plt.title("SHAP Summary Plot (Beeswarm) — Feature Impact on Depression Prediction", fontsize=12)
    plt.tight_layout()
    _save(plt.gcf(), "shap_summary_beeswarm.png")


def plot_shap_bar(shap_values, X_test_df: pd.DataFrame) -> pd.Series:
    """
    SHAP mean absolute bar chart — global feature importance via SHAP.

    Returns
    -------
    importance : pd.Series sorted by mean |SHAP|
    """
    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_test_df.columns
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    mean_abs.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("SHAP Global Feature Importance (Mean |SHAP value|)", fontsize=12)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    _save(fig, "shap_bar_importance.png")

    logger.info("Top 5 features by mean |SHAP|:\n%s", mean_abs.head(5).to_string())
    return mean_abs


def plot_shap_waterfall(explainer, shap_values, X_test_df: pd.DataFrame,
                        sample_idx: int = 0, label: str = "sample") -> None:
    """
    SHAP waterfall plot for a single prediction — shows how each feature
    pushes the model output from the base value up or down.

    Parameters
    ----------
    sample_idx : index of the sample in X_test_df to explain
    label : string label for the filename (e.g., 'fp_case', 'fn_case')
    """
    base_value = explainer.expected_value
    # LinearExplainer returns scalar; TreeExplainer may return array
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    sv = shap_values[sample_idx]
    expl = shap.Explanation(
        values=sv,
        base_values=float(base_value),
        data=X_test_df.iloc[sample_idx].values,
        feature_names=X_test_df.columns.tolist(),
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(expl, show=False, max_display=15)
    plt.title(f"SHAP Waterfall — {label} (sample #{sample_idx})", fontsize=11)
    plt.tight_layout()
    _save(plt.gcf(), f"shap_waterfall_{label}.png")


def plot_shap_dependence(shap_values, X_test_df: pd.DataFrame,
                         feature: str, interaction_feature: str = None) -> None:
    """
    SHAP dependence plot for one feature — shows how SHAP value changes with feature value.
    Optionally color by an interaction feature.
    """
    if feature not in X_test_df.columns:
        logger.warning("Feature '%s' not found, skipping dependence plot.", feature)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        feature, shap_values, X_test_df,
        interaction_index=interaction_feature,
        ax=ax, show=False,
    )
    fname = feature.replace(" ", "_").replace("?", "").replace("/", "_")
    ax.set_title(f"SHAP Dependence — {feature}", fontsize=11)
    fig.tight_layout()
    _save(fig, f"shap_dependence_{fname}.png")


def run_shap_analysis(model, X_train: np.ndarray, X_test: np.ndarray,
                      y_test: np.ndarray, feature_names: list) -> dict:
    """
    Run full SHAP analysis pipeline.

    Generates:
    - Beeswarm summary plot
    - Bar importance plot
    - Waterfall plot for a correctly predicted depressed case
    - Waterfall plot for a false negative case
    - Dependence plots for top 2 features

    Returns
    -------
    dict with shap_values, explainer, mean_abs_importance
    """
    explainer, shap_values, X_test_df = compute_shap_values(
        model, X_train, X_test, feature_names
    )

    # 1. Beeswarm
    plot_shap_summary_beeswarm(shap_values, X_test_df)

    # 2. Bar chart
    mean_abs = plot_shap_bar(shap_values, X_test_df)
    top2_features = mean_abs.head(2).index.tolist()

    # 3. Waterfall — correctly predicted depressed sample
    y_pred = model.predict(X_test)
    correct_dep = np.where((y_pred == 1) & (y_test == 1))[0]
    if len(correct_dep) > 0:
        plot_shap_waterfall(explainer, shap_values, X_test_df,
                            sample_idx=correct_dep[0], label="true_positive")

    # 4. Waterfall — false negative
    false_neg = np.where((y_pred == 0) & (y_test == 1))[0]
    if len(false_neg) > 0:
        plot_shap_waterfall(explainer, shap_values, X_test_df,
                            sample_idx=false_neg[0], label="false_negative")

    # 5. Dependence plots for top 2 features
    for feat in top2_features:
        plot_shap_dependence(shap_values, X_test_df, feature=feat)

    logger.info("SHAP analysis complete. All figures saved to %s", FIGURES_DIR)

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "mean_abs_importance": mean_abs,
    }
