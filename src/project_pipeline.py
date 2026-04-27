"""End-to-end helpers for the Student Depression Prediction term project."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False

RANDOM_STATE = 42
TARGET = "Depression"
DATA_PATH = Path("data/raw/Student Depression Dataset.csv")
FIGURES_DIR = Path("figures")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
PROCESSED_DIR = Path("data/processed")

DROP_COLUMNS = ["id", "Profession", "Work Pressure", "Job Satisfaction"]
SLEEP_HOURS_MAP = {
    "Less than 5 hours": 4.5,
    "5-6 hours": 5.5,
    "6-7 hours": 6.5,
    "7-8 hours": 7.5,
    "More than 8 hours": 8.5,
}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add engineered features while keeping the data as a pandas DataFrame."""

    def __init__(self, add_engineered_features: bool = True):
        self.add_engineered_features = add_engineered_features
        self.sleep_median_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        sleep_values = X.get("Sleep Duration", pd.Series(dtype=object)).map(SLEEP_HOURS_MAP)
        self.sleep_median_ = float(sleep_values.median()) if sleep_values.notna().any() else 6.5
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if not self.add_engineered_features:
            return X_out

        sleep_numeric = X_out["Sleep Duration"].map(SLEEP_HOURS_MAP).fillna(self.sleep_median_)
        X_out["Sleep_Hours_Estimate"] = sleep_numeric

        if {"Academic Pressure", "Sleep Duration"}.issubset(X_out.columns):
            X_out["Academic_Pressure_x_Sleep"] = X_out["Academic Pressure"] * sleep_numeric
        if {"Academic Pressure", "Financial Stress"}.issubset(X_out.columns):
            X_out["Stress_Load"] = X_out["Academic Pressure"] + X_out["Financial Stress"]
        if {"Academic Pressure", "Work/Study Hours"}.issubset(X_out.columns):
            X_out["Academic_Lifestyle_Load"] = X_out["Academic Pressure"] + X_out["Work/Study Hours"]
        if "CGPA" in X_out.columns:
            X_out["CGPA_Category"] = pd.cut(
                X_out["CGPA"],
                bins=[-np.inf, 6.0, 7.5, 8.5, np.inf],
                labels=["low", "medium", "high", "very_high"],
            ).astype(str)
        return X_out


def ensure_directories() -> None:
    for path in [FIGURES_DIR, MODELS_DIR, REPORTS_DIR, PROCESSED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_target(df: pd.DataFrame) -> dict[str, Any]:
    target_exists = TARGET in df.columns
    target_values = sorted(df[TARGET].dropna().unique().tolist()) if target_exists else []
    return {
        "shape": df.shape,
        "target_exists": target_exists,
        "target_values": target_values,
        "is_binary_target": target_exists and set(target_values).issubset({0, 1}),
    }


def data_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    outlier_rows = {}
    outlier_bounds = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_bounds[col] = {"lower": float(lower), "upper": float(upper)}
        outlier_rows[col] = int(((df[col] < lower) | (df[col] > upper)).sum())

    rare_categories = {}
    for col in ["Sleep Duration", "Dietary Habits"]:
        if col in df.columns:
            rare_categories[col] = df[col].value_counts(dropna=False).to_dict()

    near_zero_variance = {}
    for col in ["Work Pressure", "Job Satisfaction"]:
        if col in df.columns:
            top_share = df[col].value_counts(normalize=True, dropna=False).iloc[0]
            near_zero_variance[col] = {
                "top_value": df[col].value_counts(dropna=False).index[0],
                "top_share": float(top_share),
                "n_unique": int(df[col].nunique(dropna=False)),
            }

    return {
        "missing": df.isna().sum().to_dict(),
        "financial_stress_missing": int(df["Financial Stress"].isna().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "cgpa_zero_rows": int((df["CGPA"] == 0).sum()) if "CGPA" in df.columns else None,
        "age_59_or_higher_rows": int((df["Age"] >= 59).sum()) if "Age" in df.columns else None,
        "numeric_outlier_counts_iqr": outlier_rows,
        "numeric_outlier_bounds_iqr": outlier_bounds,
        "rare_categories": rare_categories,
        "near_zero_variance": near_zero_variance,
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply minimal cleaning that is justified by the DQA section."""
    clean = df.copy()
    if "Financial Stress" in clean.columns:
        missing_financial = clean["Financial Stress"].isna().sum()
        if 0 < missing_financial <= 10:
            clean = clean.dropna(subset=["Financial Stress"])
    if clean.duplicated().sum() > 0:
        clean = clean.drop_duplicates()
    return clean.reset_index(drop=True)


def save_eda_plots(df: pd.DataFrame) -> list[Path]:
    ensure_directories()
    sns.set_theme(style="whitegrid")
    saved: list[Path] = []

    def save_current(name: str) -> None:
        path = FIGURES_DIR / name
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        saved.append(path)

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x=TARGET, hue=TARGET, palette="Set2", legend=False)
    ax.set_title("Target Distribution")
    ax.set_xlabel("Depression")
    ax.set_ylabel("Count")
    save_current("target_distribution.png")

    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
    if numeric_cols:
        axes = df[numeric_cols].hist(figsize=(14, 10), bins=25, color="#4C78A8", edgecolor="white")
        for ax in np.ravel(axes):
            ax.set_title(ax.get_title(), fontsize=10)
        save_current("numeric_histograms.png")

        n_cols = 3
        n_rows = int(np.ceil(len(numeric_cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = np.ravel(axes)
        for i, col in enumerate(numeric_cols):
            sns.boxplot(data=df, y=col, ax=axes[i], color="#F58518")
            axes[i].set_title(col)
        for ax in axes[len(numeric_cols) :]:
            ax.axis("off")
        save_current("numeric_boxplots.png")

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = np.ravel(axes)
        for i, col in enumerate(numeric_cols):
            sns.boxplot(data=df, x=TARGET, y=col, ax=axes[i], hue=TARGET, palette="Set2", legend=False)
            axes[i].set_title(f"{col} by Depression")
        for ax in axes[len(numeric_cols) :]:
            ax.axis("off")
        save_current("numeric_boxplots_by_depression.png")

        plt.figure(figsize=(10, 8))
        corr = df[numeric_cols + [TARGET]].corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True)
        plt.title("Correlation Heatmap")
        save_current("correlation_heatmap.png")

    categorical_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c != TARGET]
    selected_cats = categorical_cols[:8]
    for col in selected_cats:
        plt.figure(figsize=(10, 4))
        order = df[col].value_counts().head(15).index
        ax = sns.countplot(data=df, y=col, order=order, color="#54A24B")
        ax.set_title(f"{col} Distribution")
        ax.set_xlabel("Count")
        ax.set_ylabel(col)
        save_current(f"categorical_{safe_name(col)}.png")

        plt.figure(figsize=(10, 4))
        ax = sns.countplot(data=df, y=col, hue=TARGET, order=order, palette="Set2")
        ax.set_title(f"{col} by Depression")
        ax.set_xlabel("Count")
        ax.set_ylabel(col)
        save_current(f"categorical_by_depression_{safe_name(col)}.png")

    selected_pair = [c for c in ["Academic Pressure", "Financial Stress", "Work/Study Hours", "CGPA", TARGET] if c in df.columns]
    if len(selected_pair) >= 4:
        sample = df[selected_pair].sample(min(1500, len(df)), random_state=RANDOM_STATE)
        grid = sns.pairplot(sample, hue=TARGET, diag_kind="hist", corner=True, plot_kws={"alpha": 0.35, "s": 12})
        grid.fig.suptitle("Pairplot of Selected Features", y=1.02)
        path = FIGURES_DIR / "selected_pairplot.png"
        grid.savefig(path, dpi=140, bbox_inches="tight")
        plt.close("all")
        saved.append(path)

    return saved


def safe_name(value: str) -> str:
    return (
        value.lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("?", "")
        .replace("__", "_")
    )


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def build_preprocessor(X: pd.DataFrame, add_engineered_features: bool = True) -> ColumnTransformer:
    working = FeatureEngineer(add_engineered_features=add_engineered_features).fit(X).transform(X)
    drop_cols = [c for c in DROP_COLUMNS if c in working.columns]
    working = working.drop(columns=drop_cols)
    numeric_features = working.select_dtypes(include="number").columns.tolist()
    categorical_features = working.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(model: Any, X_train: pd.DataFrame, add_engineered_features: bool = True) -> Pipeline:
    preprocessor = build_preprocessor(X_train, add_engineered_features=add_engineered_features)
    drop_cols = [c for c in DROP_COLUMNS if c in X_train.columns]
    return Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer(add_engineered_features=add_engineered_features)),
            ("drop_columns", ColumnDropper(drop_cols)),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop known non-informative columns inside the sklearn pipeline."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ColumnDropper":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.columns if c in X.columns], errors="ignore")


def candidate_models(y_train: pd.Series) -> dict[str, Any]:
    n_positive = int(y_train.sum())
    n_negative = int(len(y_train) - n_positive)
    scale_pos_weight = n_negative / n_positive if n_positive else 1.0

    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Logistic Regression Balanced": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree Balanced": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest Balanced": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "MLP Classifier": MLPClassifier(
            hidden_layer_sizes=(40,),
            max_iter=250,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
        )
    return models


def compare_feature_engineering(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    records = []
    for use_engineering in [False, True]:
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        pipe = build_pipeline(model, X_train, add_engineered_features=use_engineering)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring={"f1_macro": "f1_macro", "roc_auc": "roc_auc"})
        records.append(
            {
                "engineered_features": use_engineering,
                "cv_f1_macro_mean": scores["test_f1_macro"].mean(),
                "cv_roc_auc_mean": scores["test_roc_auc"].mean(),
            }
        )
    return pd.DataFrame(records)


def compare_models(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "f1_macro": "f1_macro",
        "roc_auc": "roc_auc",
    }
    records = []
    for name, model in candidate_models(y_train).items():
        pipe = build_pipeline(model, X_train, add_engineered_features=True)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        record = {"model": name}
        for metric in scoring:
            record[f"cv_{metric}_mean"] = scores[f"test_{metric}"].mean()
            record[f"cv_{metric}_std"] = scores[f"test_{metric}"].std()
        records.append(record)
    results = pd.DataFrame(records).sort_values("cv_f1_macro_mean", ascending=False)
    results.to_csv("model_comparison_results.csv", index=False)
    return results


def tune_models(X_train: pd.DataFrame, y_train: pd.Series, top_model_names: list[str]) -> tuple[pd.DataFrame, Pipeline]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grids: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "Logistic Regression Balanced": (
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs", "liblinear"],
            },
        ),
        "Decision Tree Balanced": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {
                "model__max_depth": [3, 5, 8, 12, None],
                "model__min_samples_split": [2, 10, 25],
                "model__min_samples_leaf": [1, 5, 10],
            },
        ),
        "Random Forest Balanced": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [150, 250],
                "model__max_depth": [8, 14, None],
                "model__min_samples_split": [2, 10],
                "model__min_samples_leaf": [1, 5],
                "model__class_weight": ["balanced", "balanced_subsample"],
            },
        ),
    }

    if HAS_XGBOOST:
        n_positive = int(y_train.sum())
        n_negative = int(len(y_train) - n_positive)
        scale_pos_weight = n_negative / n_positive if n_positive else 1.0
        grids["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE),
            {
                "model__n_estimators": [150, 250],
                "model__learning_rate": [0.03, 0.08],
                "model__max_depth": [3, 5],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
        )

    selected = [name for name in top_model_names if name in grids][:2]
    if len(selected) < 2:
        for fallback in ["Logistic Regression Balanced", "Random Forest Balanced", "Decision Tree Balanced", "XGBoost"]:
            if fallback in grids and fallback not in selected:
                selected.append(fallback)
            if len(selected) == 2:
                break

    records = []
    best_search: GridSearchCV | None = None
    for name in selected:
        model, grid = grids[name]
        pipe = build_pipeline(model, X_train, add_engineered_features=True)
        search = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        records.append(
            {
                "model": name,
                "best_score_f1_macro": search.best_score_,
                "best_params": json.dumps(search.best_params_),
            }
        )
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_search = search

    if best_search is None:
        raise RuntimeError("No tunable models were selected.")

    tuning_results = pd.DataFrame(records).sort_values("best_score_f1_macro", ascending=False)
    tuning_results.to_csv("best_parameters.csv", index=False)
    return tuning_results, best_search.best_estimator_


def evaluate_final_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_directories()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("final_test_metrics.csv", index=False)

    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(REPORTS_DIR / "classification_report.csv")

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.title("Final Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close()

    if y_proba is not None:
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title("Final Model ROC Curve")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=160, bbox_inches="tight")
        plt.close()

        PrecisionRecallDisplay.from_predictions(y_test, y_proba)
        plt.title("Final Model Precision-Recall Curve")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=160, bbox_inches="tight")
        plt.close()

    save_feature_importance(model, X_test, y_test)

    cm = confusion_matrix(y_test, y_pred)
    error_details = error_analysis_frame(model, X_test, y_test, y_pred, y_proba)
    return metrics_df, {"classification_report": report, "confusion_matrix": cm.tolist(), "errors": error_details}


def save_feature_importance(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    fitted_model = model.named_steps["model"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = None
    if hasattr(fitted_model, "feature_importances_"):
        importances = fitted_model.feature_importances_
    elif hasattr(fitted_model, "coef_"):
        importances = np.abs(fitted_model.coef_[0])

    if importances is not None:
        top = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(20)
        )
        top.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
        plt.figure(figsize=(9, 6))
        sns.barplot(data=top, y="feature", x="importance", color="#4C78A8")
        plt.title("Top Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=160, bbox_inches="tight")
        plt.close()
    else:
        result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
        top = (
            pd.DataFrame({"feature": X_test.columns, "importance": result.importances_mean})
            .sort_values("importance", ascending=False)
            .head(20)
        )
        top.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
        plt.figure(figsize=(9, 6))
        sns.barplot(data=top, y="feature", x="importance", color="#4C78A8")
        plt.title("Permutation Importance")
        plt.xlabel("Mean Importance")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "permutation_importance.png", dpi=160, bbox_inches="tight")
        plt.close()


def error_analysis_frame(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
) -> dict[str, Any]:
    errors = X_test.copy()
    errors["actual"] = y_test.values
    errors["predicted"] = y_pred
    if y_proba is not None:
        errors["predicted_probability_depression"] = y_proba
    errors["error_type"] = np.select(
        [(errors["actual"] == 0) & (errors["predicted"] == 1), (errors["actual"] == 1) & (errors["predicted"] == 0)],
        ["False Positive", "False Negative"],
        default="Correct",
    )
    misclassified = errors[errors["error_type"] != "Correct"].copy()
    misclassified.to_csv(REPORTS_DIR / "misclassified_samples.csv", index=False)

    group_columns = [
        c
        for c in [
            "Gender",
            "Sleep Duration",
            "Dietary Habits",
            "Have you ever had suicidal thoughts ?",
            "Family History of Mental Illness",
        ]
        if c in errors.columns
    ]
    group_tables = {}
    for col in group_columns:
        table = pd.crosstab(errors[col], errors["error_type"], normalize="index").round(3)
        table.to_csv(REPORTS_DIR / f"error_rates_by_{safe_name(col)}.csv")
        group_tables[col] = table.to_dict()

    return {
        "false_positives": int((errors["error_type"] == "False Positive").sum()),
        "false_negatives": int((errors["error_type"] == "False Negative").sum()),
        "group_error_tables": group_tables,
    }


def save_model(model: Pipeline) -> Path:
    ensure_directories()
    path = MODELS_DIR / "final_model.pkl"
    with path.open("wb") as f:
        pickle.dump(model, f)
    return path


def run_full_pipeline(path: Path | str = DATA_PATH) -> dict[str, Any]:
    ensure_directories()
    df = load_dataset(path)
    validation = validate_target(df)
    dqa = data_quality_report(df)
    clean = clean_dataset(df)
    clean.to_csv(PROCESSED_DIR / "student_depression_clean.csv", index=False)
    figures = save_eda_plots(clean)
    X_train, X_test, y_train, y_test = split_features(clean)
    feature_results = compare_feature_engineering(X_train, y_train)
    feature_results.to_csv("feature_engineering_comparison.csv", index=False)
    comparison = compare_models(X_train, y_train)
    tuning, best_model = tune_models(X_train, y_train, comparison["model"].tolist())
    metrics, evaluation_details = evaluate_final_model(best_model, X_test, y_test)
    model_path = save_model(best_model)
    return {
        "validation": validation,
        "data_quality": dqa,
        "clean_shape": clean.shape,
        "figures": [str(p) for p in figures],
        "feature_engineering_comparison": feature_results,
        "model_comparison": comparison,
        "tuning_results": tuning,
        "test_metrics": metrics,
        "evaluation_details": evaluation_details,
        "model_path": str(model_path),
    }
