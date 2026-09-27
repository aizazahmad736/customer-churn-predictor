"""
ChurnGuard AI - Data Preprocessing & Pipeline Engineering
Builds robust Scikit-Learn transformers for imputation, scaling, and one-hot encoding.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from .config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw customer data:
      - Strips string whitespace.
      - Converts TotalCharges to numeric (handles blanks/spaces by filling with tenure * MonthlyCharges).
      - Ensures proper categorical types.
    """
    df_clean = df.copy()

    # TotalCharges cleaning
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
        # Impute missing TotalCharges with tenure * MonthlyCharges
        mask_missing = df_clean["TotalCharges"].isna()
        if mask_missing.any():
            if "tenure" in df_clean.columns and "MonthlyCharges" in df_clean.columns:
                df_clean.loc[mask_missing, "TotalCharges"] = (
                    df_clean.loc[mask_missing, "tenure"] * df_clean.loc[mask_missing, "MonthlyCharges"]
                )
            else:
                df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

    # Standardize string representations
    for col in df_clean.select_dtypes(include=["object"]).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    return df_clean

def build_preprocessor() -> ColumnTransformer:
    """
    Constructs an end-to-end ColumnTransformer:
      - Numeric: SimpleImputer (median) -> StandardScaler
      - Categoric: SimpleImputer (most_frequent) -> OneHotEncoder (handle_unknown='ignore')
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor

def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> list:
    """Extracts output feature names after ColumnTransformer fitting."""
    feature_names = []
    
    # Numeric features
    feature_names.extend(NUMERICAL_FEATURES)
    
    # Categorical features from OneHotEncoder
    cat_transformer = preprocessor.named_transformers_["cat"]
    encoder = cat_transformer.named_steps["encoder"]
    encoded_cats = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(encoded_cats.tolist())
    
    return feature_names
