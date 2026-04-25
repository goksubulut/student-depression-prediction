"""Exploratory Data Analysis module for the Student Depression Prediction project."""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pointbiserialr

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

NUMERIC_COLS = ["Age", "CGPA", "Academic Pressure", "Work/Study Hours", "Financial Stress"]
CATEGORICAL_COLS = [
    "Gender", "Sleep Duration", "Dietary Habits", "Degree",
    "Have you ever had suicidal thoughts ?", "Family History of Mental Illness",
]


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure to reports/figures/ and close it."""
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def plot_univariate_numeric(df: pd.DataFrame) -> None:
    """Plot histograms + KDE for each numeric feature."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_COLS):
        ax = axes[i]
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution of {col}", fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
    # hide extra subplot
    axes[-1].set_visible(False)
    fig.suptitle("Univariate Distributions — Numeric Features", fontsize=14, y=1.01)
    _save(fig, "univariate_numeric.png")


def plot_univariate_categorical(df: pd.DataFrame) -> None:
    """Plot countplots for each categorical feature."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    for i, col in enumerate(CATEGORICAL_COLS):
        ax = axes[i]
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, order=order, ax=ax, palette="Set2")
        ax.set_title(f"Count of {col}", fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Univariate Distributions — Categorical Features", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, "univariate_categorical.png")


def plot_bivariate_numeric(df: pd.DataFrame) -> None:
    """Boxplots for each numeric feature split by Depression label."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_COLS):
        ax = axes[i]
        sns.boxplot(data=df, x="Depression", y=col, ax=ax, palette="Set1")
        ax.set_title(f"{col} by Depression", fontsize=11)
        ax.set_xlabel("Depression (0=No, 1=Yes)")
        ax.set_ylabel(col)
    axes[-1].set_visible(False)
    fig.suptitle("Numeric Features vs Depression", fontsize=14, y=1.01)
    _save(fig, "bivariate_numeric_boxplots.png")


def plot_bivariate_categorical(df: pd.DataFrame) -> None:
    """Countplots for each categorical feature split by Depression label."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    for i, col in enumerate(CATEGORICAL_COLS):
        ax = axes[i]
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, hue="Depression", order=order, ax=ax, palette="Set1")
        ax.set_title(f"{col} by Depression", fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Depression", labels=["No", "Yes"])
    fig.suptitle("Categorical Features vs Depression", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, "bivariate_categorical_countplots.png")


def plot_mean_by_depression(df: pd.DataFrame) -> None:
    """Bar chart of mean numeric feature values grouped by Depression."""
    means = df.groupby("Depression")[NUMERIC_COLS].mean().T
    means.columns = ["No Depression", "Depression"]
    fig, ax = plt.subplots(figsize=(10, 6))
    means.plot(kind="bar", ax=ax, colormap="Set1")
    ax.set_title("Mean Numeric Feature Values by Depression Class", fontsize=13)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean Value")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Class")
    fig.tight_layout()
    _save(fig, "mean_features_by_depression.png")


def encode_for_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns minimally for correlation analysis."""
    df_enc = df.copy()
    sleep_map = {
        "Less than 5 hours": 1, "5-6 hours": 2, "6-7 hours": 3,
        "7-8 hours": 4, "More than 8 hours": 5,
    }
    df_enc["Sleep Duration"] = df_enc["Sleep Duration"].map(sleep_map)
    binary_cols = ["Have you ever had suicidal thoughts ?", "Family History of Mental Illness", "Gender"]
    for col in binary_cols:
        if col in df_enc.columns:
            uniq = df_enc[col].dropna().unique()
            if len(uniq) == 2:
                mapping = {uniq[0]: 0, uniq[1]: 1}
                df_enc[col] = df_enc[col].map(mapping)
    # one-hot remaining object columns
    obj_cols = df_enc.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        df_enc = pd.get_dummies(df_enc, columns=obj_cols, drop_first=True)
    return df_enc


def plot_correlation_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and plot correlation heatmap of all numeric features."""
    df_enc = encode_for_correlation(df)
    # keep only numeric
    df_num = df_enc.select_dtypes(include=[np.number])
    corr = df_num.corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=False, cmap="coolwarm",
        vmin=-1, vmax=1, linewidths=0.5, ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14)
    fig.tight_layout()
    _save(fig, "correlation_heatmap.png")
    return corr


def compute_point_biserial(df: pd.DataFrame) -> pd.Series:
    """Compute point-biserial correlation of each feature with the Depression target."""
    df_enc = encode_for_correlation(df)
    df_num = df_enc.select_dtypes(include=[np.number])
    target = df_num["Depression"]
    correlations = {}
    for col in df_num.columns:
        if col == "Depression":
            continue
        valid = df_num[[col, "Depression"]].dropna()
        r, _ = pointbiserialr(valid["Depression"], valid[col])
        correlations[col] = r
    series = pd.Series(correlations).abs().sort_values(ascending=False)
    logger.info("Top 5 features by |point-biserial r| with Depression:\n%s", series.head(5).to_string())
    return series


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Pie chart of class distribution."""
    counts = df["Depression"].value_counts().sort_index()
    labels = [f"No Depression\n({counts[0]})", f"Depression\n({counts[1]})"]
    colors = ["#66b3ff", "#ff6666"]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(counts, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.set_title("Depression Class Distribution", fontsize=14)
    _save(fig, "class_distribution_pie.png")


def plot_pairplot(df: pd.DataFrame, top_features: list) -> None:
    """Pair plot of top correlated features colored by Depression."""
    cols = [f for f in top_features if f in df.columns][:5]
    cols_with_target = cols + ["Depression"]
    subset = df[cols_with_target].copy()
    subset["Depression"] = subset["Depression"].map({0: "No", 1: "Yes"})
    fig = sns.pairplot(subset, hue="Depression", palette={"No": "steelblue", "Yes": "tomato"}, diag_kind="kde")
    fig.fig.suptitle("Pair Plot — Top 5 Correlated Features", y=1.01, fontsize=13)
    _save(fig.fig, "pairplot_top5.png")


def run_eda(df: pd.DataFrame) -> pd.Series:
    """
    Run the full EDA pipeline. Saves all figures to reports/figures/.

    Returns
    -------
    correlations : pd.Series
        Feature correlations with target, sorted descending.
    """
    logger.info("Starting EDA...")

    plot_univariate_numeric(df)
    plot_univariate_categorical(df)
    plot_bivariate_numeric(df)
    plot_bivariate_categorical(df)
    plot_mean_by_depression(df)
    plot_correlation_heatmap(df)

    correlations = compute_point_biserial(df)
    top5 = correlations.head(5).index.tolist()
    logger.info("Top 5 predictive features: %s", top5)

    plot_class_distribution(df)
    plot_pairplot(df, top5)

    logger.info("EDA complete. All figures saved to %s", FIGURES_DIR)
    return correlations
