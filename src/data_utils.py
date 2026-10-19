"""
Data utility functions for ICU Patient Mortality Prediction Dashboard.
Provides robust data loading, quality auditing, outlier calculation, and feature mapping.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import streamlit as st

NUMERICAL_CLINICAL_FEATURES = [
    "age",
    "heart_rate_mean",
    "heart_rate_min",
    "heart_rate_max",
    "resp_rate_mean",
    "resp_rate_min",
    "resp_rate_max",
    "spo2_mean",
    "sbp_mean",
    "dbp_mean",
    "creatinine_mean",
    "urea_nitrogen_mean",
    "wbc_mean",
    "sodium_mean",
    "potassium_mean",
    "lactate_mean",
]

CATEGORICAL_FEATURES = [
    "gender",
    "admission_type",
    "icu_type",
    "diabetes_history",
    "ventilation_status",
]

IDENTIFIER_COLUMNS = ["subject_id", "hadm_id", "stay_id"]
TARGET_COLUMN = "hospital_expire_flag"

NOTEBOOK_SCATTER_PAIRS = [
    ("heart_rate_mean", "heart_rate_min", "Mean Heart Rate vs Minimum Heart Rate"),
    ("resp_rate_mean", "resp_rate_min", "Mean Respiratory Rate vs Minimum Respiratory Rate"),
    ("spo2_mean", "sbp_mean", "Oxygen Saturation (SpO2) vs Systolic Blood Pressure"),
    ("creatinine_mean", "dbp_mean", "Mean Creatinine vs Diastolic Blood Pressure"),
    ("wbc_mean", "sodium_mean", "White Blood Cell Count vs Serum Sodium"),
    ("potassium_mean", "lactate_mean", "Serum Potassium vs Serum Lactate"),
]


def get_default_dataset_path() -> Path:
    """Return path to primary raw dataset with nulls and duplicates."""
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "icu_mortality_with_nulls_and_duplicates.csv"


def get_cleaned_dataset_path() -> Path:
    """Return path to cleaned reference dataset if present."""
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "synthetic_icu_mortality_5000_with_categories.csv"


@st.cache_data(show_spinner="Loading ICU clinical dataset...")
def load_default_dataset() -> Optional[pd.DataFrame]:
    """Load default dataset from project directory if available."""
    path = get_default_dataset_path()
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"Error reading dataset at {path.name}: {e}")
            return None
    return None


@st.cache_data(show_spinner="Loading cleaned reference dataset...")
def load_cleaned_reference_dataset() -> Optional[pd.DataFrame]:
    """Load cleaned reference dataset from project directory if available."""
    path = get_cleaned_dataset_path()
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def parse_uploaded_csv(uploaded_file) -> Optional[pd.DataFrame]:
    """Safely parse user-uploaded CSV file with UTF-8 encoding."""
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to parse uploaded CSV: {e}")
        return None


def compute_dataset_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute high-level summary KPIs for any loaded dataset."""
    total_records = len(df)
    total_features = df.shape[1]
    missing_value_count = int(df.isnull().sum().sum())
    duplicate_row_count = int(df.duplicated().sum())

    kpis = {
        "total_records": total_records,
        "total_features": total_features,
        "missing_count": missing_value_count,
        "missing_rate": (missing_value_count / (total_records * total_features) * 100) if total_records > 0 else 0.0,
        "duplicate_count": duplicate_row_count,
        "has_target": TARGET_COLUMN in df.columns,
    }

    if kpis["has_target"]:
        valid_targets = df[TARGET_COLUMN].dropna()
        survived_count = int((valid_targets == 0).sum())
        died_count = int((valid_targets == 1).sum())
        total_valid = len(valid_targets)

        kpis["survived_count"] = survived_count
        kpis["died_count"] = died_count
        kpis["survived_pct"] = (survived_count / total_valid * 100) if total_valid > 0 else 0.0
        kpis["died_pct"] = (died_count / total_valid * 100) if total_valid > 0 else 0.0

    return kpis


def compute_data_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate per-column data quality breakdown."""
    rows = []
    total_len = len(df)

    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / total_len) * 100, 2) if total_len > 0 else 0.0
        unique_count = int(df[col].nunique(dropna=True))
        dtype_str = str(df[col].dtype)

        role = "Identifier" if col in IDENTIFIER_COLUMNS else (
            "Target (Outcome)" if col == TARGET_COLUMN else (
                "Numerical Predictor" if col in NUMERICAL_CLINICAL_FEATURES else (
                    "Categorical Predictor" if col in CATEGORICAL_FEATURES else "Feature"
                )
            )
        )

        rows.append({
            "Column Name": col,
            "Role": role,
            "Data Type": dtype_str,
            "Missing Count": null_count,
            "Missing %": null_pct,
            "Unique Values": unique_count,
            "Status": "Complete" if null_count == 0 else f"{null_pct}% Missing",
        })

    return pd.DataFrame(rows).sort_values(by="Missing %", ascending=False).reset_index(drop=True)


def compute_iqr_outliers(df: pd.DataFrame, num_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate 1.5 * IQR outlier statistics for numerical columns.
    Excludes identifier columns.
    """
    if num_cols is None:
        num_cols = [c for c in NUMERICAL_CLINICAL_FEATURES if c in df.columns]

    outlier_records = []
    for col in num_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = round(q1 - 1.5 * iqr, 2)
        upper_bound = round(q3 + 1.5 * iqr, 2)

        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_count = int(len(outliers))
        outlier_pct = round((outlier_count / len(series)) * 100, 2)

        outlier_records.append({
            "Feature": col,
            "Valid Count": len(series),
            "Q1 (25th)": round(q1, 2),
            "Median (50th)": round(float(series.median()), 2),
            "Q3 (75th)": round(q3, 2),
            "IQR": round(iqr, 2),
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Outlier Count": outlier_count,
            "Outlier %": outlier_pct,
        })

    return pd.DataFrame(outlier_records).sort_values(by="Outlier Count", ascending=False).reset_index(drop=True)


def compute_correlation_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Compute Pearson correlation matrix for clinical features, excluding identifiers."""
    available_cols = [c for c in NUMERICAL_CLINICAL_FEATURES if c in df.columns]
    corr_df = df[available_cols].apply(pd.to_numeric, errors="coerce").corr()
    return corr_df, available_cols
