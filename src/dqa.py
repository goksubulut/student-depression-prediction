"""Data Quality Assessment module for the Student Depression Prediction project."""

import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV dataset from disk."""
    df = pd.read_csv(filepath)
    logger.info("Loaded dataset: %s rows, %s columns", *df.shape)
    return df


def check_shape_and_types(df: pd.DataFrame) -> dict:
    """Report shape, dtypes, and memory usage."""
    info = {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "dtypes": df.dtypes.to_dict(),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }
    logger.info("Shape: %d rows × %d cols | Memory: %.2f MB", info["rows"], info["cols"], info["memory_mb"])
    return info


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Compute missing value counts and percentages per column."""
    missing = df.isnull().sum()
    pct = missing / len(df) * 100
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    report = report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)
    logger.info("Columns with missing values:\n%s", report.to_string())
    return report


def drop_missing_financial_stress(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where Financial Stress is null (expected: ~3 rows)."""
    before = len(df)
    df = df.dropna(subset=["Financial Stress"])
    dropped = before - len(df)
    logger.info("Dropped %d rows with null Financial Stress", dropped)
    return df.reset_index(drop=True)


def check_and_drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and drop duplicate rows."""
    n_dups = df.duplicated().sum()
    logger.info("Duplicate rows found: %d", n_dups)
    if n_dups > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info("Dropped %d duplicate rows", n_dups)
    return df


def investigate_zero_cgpa(df: pd.DataFrame) -> pd.DataFrame:
    """Flag and drop rows with CGPA == 0 (data-entry errors)."""
    zero_cgpa = (df["CGPA"] == 0).sum()
    logger.info("Rows with CGPA == 0: %d", zero_cgpa)
    if zero_cgpa > 0:
        df = df[df["CGPA"] != 0].reset_index(drop=True)
        logger.info("Dropped %d rows with CGPA == 0", zero_cgpa)
    return df


def age_outlier_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    IQR-based outlier detection on Age.
    Drops rows where Age > 60 (impossible for a student).
    """
    Q1 = df["Age"].quantile(0.25)
    Q3 = df["Age"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df["Age"] < lower) | (df["Age"] > upper)]
    logger.info(
        "Age IQR bounds: [%.1f, %.1f] | Outlier rows: %d",
        lower, upper, len(outliers),
    )
    impossible = (df["Age"] > 60).sum()
    logger.info("Rows with Age > 60 (impossible): %d", impossible)
    df = df[df["Age"] <= 60].reset_index(drop=True)
    return df


def check_zero_work_study_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where Work/Study Hours == 0."""
    n_zero = (df["Work/Study Hours"] == 0).sum()
    logger.info("Rows with Work/Study Hours == 0: %d", n_zero)
    return df


def handle_rare_others(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where Sleep Duration or Dietary Habits == 'Others'
    (small groups: ~18 and ~12 rows respectively).
    """
    before = len(df)
    df = df[df["Sleep Duration"] != "Others"]
    df = df[df["Dietary Habits"] != "Others"]
    dropped = before - len(df)
    logger.info("Dropped %d rows with 'Others' in Sleep Duration / Dietary Habits", dropped)
    return df.reset_index(drop=True)


def check_near_zero_variance(df: pd.DataFrame, threshold: float = 0.95) -> list:
    """
    Identify columns where a single value comprises > threshold fraction of rows.
    Returns list of columns to drop.
    """
    cols_to_drop = []
    for col in ["Work Pressure", "Job Satisfaction"]:
        if col not in df.columns:
            continue
        top_freq = df[col].value_counts(normalize=True).iloc[0]
        logger.info("'%s' top-value frequency: %.2f%%", col, top_freq * 100)
        if top_freq > threshold:
            cols_to_drop.append(col)
            logger.info("Flagging '%s' for removal (near-zero variance)", col)
    return cols_to_drop


def class_balance_report(df: pd.DataFrame) -> pd.DataFrame:
    """Print and return Depression class counts and percentages."""
    counts = df["Depression"].value_counts().sort_index()
    pcts = df["Depression"].value_counts(normalize=True).sort_index() * 100
    report = pd.DataFrame({"count": counts, "pct": pcts})
    logger.info("Class balance:\n%s", report.to_string())
    return report


def run_dqa(filepath: str, output_path: str = "data/cleaned.csv") -> tuple:
    """
    Run the full DQA pipeline.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned dataframe.
    summary : dict
        Summary of all issues found and actions taken.
    """
    summary = {}
    df = load_data(filepath)

    # 1. Shape / dtypes
    summary["raw_shape"] = df.shape
    check_shape_and_types(df)

    # 2. Missing values
    missing_report = check_missing(df)
    summary["missing"] = missing_report.to_dict()
    df = drop_missing_financial_stress(df)

    # 3. Duplicates
    df = check_and_drop_duplicates(df)

    # 4. CGPA == 0
    df = investigate_zero_cgpa(df)

    # 5. Age outliers
    df = age_outlier_analysis(df)

    # 6. Work/Study Hours == 0
    df = check_zero_work_study_hours(df)

    # 7. Rare "Others" categories
    df = handle_rare_others(df)

    # 8. Near-zero variance
    nzv_cols = check_near_zero_variance(df)
    summary["nzv_cols_dropped"] = nzv_cols

    # 9. Class balance
    class_report = class_balance_report(df)
    summary["class_balance"] = class_report.to_dict()

    # Final shape
    summary["clean_shape"] = df.shape
    logger.info("Clean dataset shape: %s", df.shape)

    df.to_csv(output_path, index=False)
    logger.info("Saved cleaned data to %s", output_path)

    return df, summary
