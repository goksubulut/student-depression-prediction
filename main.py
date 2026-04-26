"""
End-to-end runner for the Student Depression Prediction pipeline.

Usage:
    python main.py
"""

import logging
import os
import sys

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DATA_RAW = os.path.join(ROOT, "data", "Student Depression Dataset.csv")
DATA_CLEAN = os.path.join(ROOT, "data", "cleaned.csv")
MODEL_PATH = os.path.join(ROOT, "models", "best_model.pkl")


def main() -> None:
    """Run the full ML pipeline end-to-end."""

    # ── 1. DQA ────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 — Data Quality Assessment")
    logger.info("=" * 60)
    from src.dqa import run_dqa
    df_clean, dqa_summary = run_dqa(DATA_RAW, DATA_CLEAN)
    logger.info("Clean shape: %s", df_clean.shape)

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 — Exploratory Data Analysis")
    logger.info("=" * 60)
    from src.eda import run_eda
    correlations = run_eda(df_clean)
    top5 = correlations.head(5).index.tolist()
    logger.info("Top 5 predictive features: %s", top5)

    # ── 3. Feature Engineering ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 — Feature Engineering")
    logger.info("=" * 60)
    from src.features import prepare_features
    X_train, X_test, y_train, y_test, selected_features, preprocessor, rfe = prepare_features(df_clean)
    logger.info("Selected features (%d): %s", len(selected_features), selected_features)

    # Keep raw test rows for error analysis
    df_clean_reset = df_clean.copy().reset_index(drop=True)
    from sklearn.model_selection import train_test_split
    _, X_test_raw, _, _ = train_test_split(
        df_clean_reset.drop(columns=["Depression"]),
        df_clean_reset["Depression"],
        test_size=0.2, random_state=42,
        stratify=df_clean_reset["Depression"],
    )

    # ── 4. Modeling ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 — Model Training & MLflow Tracking")
    logger.info("=" * 60)
    from src.train import train_all_models
    results_df, best_model, best_run_id = train_all_models(X_train, y_train)

    logger.info("\nFinal model comparison:\n%s",
                results_df[["model", "f1_macro_mean", "f1_macro_std", "roc_auc_mean"]].to_string(index=False))

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5 — Evaluation & Error Analysis")
    logger.info("=" * 60)
    from src.evaluate import run_evaluation
    metrics = run_evaluation(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        feature_names=selected_features,
        X_test_orig=X_test_raw.reset_index(drop=True),
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    best_model_name = results_df.iloc[0]["model"]

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE — FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info("Dataset shape after cleaning : %s", df_clean.shape)
    logger.info("Selected features (post-RFE) : %s", selected_features)
    logger.info("Best model                   : %s", best_model_name)
    logger.info("Test F1-macro                : %.4f", metrics["f1_macro"])
    logger.info("Test ROC-AUC                 : %.4f", metrics["roc_auc"])
    logger.info("MLflow best run ID           : %s", best_run_id)
    logger.info("=" * 60)

    sep = "=" * 62
    print(f"\n{sep}")
    print("  STUDENT DEPRESSION PREDICTION -- FINAL SUMMARY")
    print(sep)
    print(f"  Clean dataset shape  : {df_clean.shape}")
    print(f"  Selected features    : {len(selected_features)}")
    print(f"  Best model           : {best_model_name}")
    print(f"  Test F1-macro        : {metrics['f1_macro']:.4f}")
    print(f"  Test ROC-AUC         : {metrics['roc_auc']:.4f}")
    print(f"  MLflow run ID        : {best_run_id}")
    print(sep)


if __name__ == "__main__":
    main()
