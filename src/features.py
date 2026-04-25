"""Feature engineering and preprocessing pipeline for Student Depression Prediction."""

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# Columns to drop entirely
DROP_COLS = ["id", "Profession", "Work Pressure", "Job Satisfaction", "City"]

SLEEP_ORDER = [
    ["Less than 5 hours", "5-6 hours", "6-7 hours", "7-8 hours", "More than 8 hours"]
]
SLEEP_MAP = {v: i + 1 for i, v in enumerate(SLEEP_ORDER[0])}

OHE_COLS = ["Gender", "Dietary Habits", "Degree"]

BINARY_COLS = {
    "Have you ever had suicidal thoughts ?": {"Yes": 1, "No": 0},
    "Family History of Mental Illness": {"Yes": 1, "No": 0},
}

NUMERIC_COLS = ["Age", "CGPA", "Academic Pressure", "Work/Study Hours", "Financial Stress"]

N_FEATURES_RFE = 15


class BinaryMapper:
    """sklearn-compatible transformer that applies fixed yes/no mappings."""

    def __init__(self, mappings: dict):
        """
        Parameters
        ----------
        mappings : dict
            {column_name: {raw_value: encoded_value}}
        """
        self.mappings = mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, mapping in self.mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.mappings.keys()))


def add_interaction_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Academic_Pressure_x_Sleep interaction column BEFORE scaling.

    Sleep Duration is numerically encoded first (1–5).
    """
    df = df.copy()
    sleep_encoded = df["Sleep Duration"].map(SLEEP_MAP).fillna(3)
    df["AP_x_Sleep"] = df["Academic Pressure"] * sleep_encoded
    return df


def _drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop configured columns if they exist."""
    to_drop = [c for c in DROP_COLS if c in df.columns]
    # also drop City if >20 unique values (already included above, but safety check)
    if "City" in df.columns and df["City"].nunique() > 20 and "City" not in to_drop:
        to_drop.append("City")
    df = df.drop(columns=to_drop, errors="ignore")
    logger.info("Dropped columns: %s", to_drop)
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Full feature engineering pipeline.

    Steps
    -----
    1. Drop irrelevant columns
    2. Add interaction feature
    3. Binary-encode yes/no columns
    4. Ordinal-encode Sleep Duration
    5. One-hot encode Gender, Dietary Habits, Degree
    6. Min-max scale numeric columns
    7. 80/20 stratified split
    8. RFE on train set — select top N_FEATURES_RFE features

    Returns
    -------
    X_train_rfe, X_test_rfe, y_train, y_test : arrays / DataFrames
    selected_features : list[str]
    pipeline : fitted sklearn Pipeline
    """
    df = df.copy()

    # Target
    y = df["Depression"].values
    df = df.drop(columns=["Depression"])

    # 1. Drop columns
    df = _drop_columns(df)

    # 2. Interaction feature
    df = add_interaction_feature(df)

    # 3. Binary encode
    for col, mapping in BINARY_COLS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # 4. Ordinal encode Sleep Duration
    if "Sleep Duration" in df.columns:
        df["Sleep Duration"] = df["Sleep Duration"].map(SLEEP_MAP).fillna(3)

    # Identify columns after transformations
    numeric_features = [c for c in NUMERIC_COLS + ["AP_x_Sleep"] if c in df.columns]
    ohe_features = [c for c in OHE_COLS if c in df.columns]
    passthrough_features = [
        c for c in df.columns
        if c not in numeric_features and c not in ohe_features
    ]

    logger.info("Numeric features: %s", numeric_features)
    logger.info("OHE features: %s", ohe_features)
    logger.info("Passthrough features: %s", passthrough_features)

    # 5–6. Build sklearn ColumnTransformer
    transformers = [
        ("num", MinMaxScaler(), numeric_features),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ohe_features),
    ]
    if passthrough_features:
        transformers.append(("pass", "passthrough", passthrough_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    # 7. Split BEFORE fitting the pipeline (no leakage)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train: %d | Test: %d", len(y_train), len(y_test))

    # Fit preprocessor on train only
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)

    # Recover feature names
    ohe_names = (
        preprocessor.named_transformers_["ohe"].get_feature_names_out(ohe_features).tolist()
        if ohe_features else []
    )
    feature_names = numeric_features + ohe_names + passthrough_features

    logger.info("Total features after preprocessing: %d", X_train_proc.shape[1])

    # 8. RFE
    estimator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    rfe = RFE(estimator=estimator, n_features_to_select=min(N_FEATURES_RFE, X_train_proc.shape[1]), step=1)
    rfe.fit(X_train_proc, y_train)

    selected_mask = rfe.support_
    selected_features = [f for f, s in zip(feature_names, selected_mask) if s]
    dropped_features = [f for f, s in zip(feature_names, selected_mask) if not s]

    logger.info("RFE selected features (%d): %s", len(selected_features), selected_features)
    logger.info("RFE dropped features (%d): %s", len(dropped_features), dropped_features)

    X_train_rfe = rfe.transform(X_train_proc)
    X_test_rfe = rfe.transform(X_test_proc)

    return X_train_rfe, X_test_rfe, y_train, y_test, selected_features, preprocessor, rfe
