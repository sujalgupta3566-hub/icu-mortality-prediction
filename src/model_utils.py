"""
Model utility functions for ICU Patient In-Hospital Mortality Prediction.
Handles loading, caching, inspection, and inference for the scikit-learn pipeline artifact.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import joblib
import streamlit as st

EXPECTED_FEATURES = [
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
    "gender",
    "admission_type",
    "icu_type",
    "diabetes_history",
    "ventilation_status",
]

DEFAULT_CATEGORIES = {
    "gender": ["Female", "Male"],
    "admission_type": ["Elective", "Emergency", "Urgent"],
    "icu_type": ["Cardiac ICU", "Medical ICU", "Neuro ICU", "Surgical ICU"],
    "diabetes_history": ["No", "Yes"],
    "ventilation_status": ["No", "Yes"],
}


def get_default_model_path() -> Path:
    """Return relative path to saved model artifact."""
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "icu_mortality_pipeline.joblib"


@st.cache_resource(show_spinner="Loading trained ICU mortality pipeline...")
def load_model_bundle(model_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the fitted scikit-learn model bundle from disk.
    Cached across user sessions to ensure fast inference and minimal memory overhead.
    """
    if model_path is None:
        model_path = get_default_model_path()
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path.name}'. "
            "Please ensure 'icu_mortality_pipeline.joblib' is present in the project directory."
        )

    bundle = joblib.load(model_path)

    required_keys = {"model", "threshold", "feature_columns", "target_column"}
    missing_keys = required_keys - set(bundle.keys())
    if missing_keys:
        raise KeyError(
            f"Model bundle is missing expected keys: {missing_keys}. "
            "Artifact may be damaged or incomplete."
        )

    # Validate feature columns length
    if len(bundle["feature_columns"]) != len(EXPECTED_FEATURES):
        st.warning(
            f"Model artifact specifies {len(bundle['feature_columns'])} features "
            f"(expected {len(EXPECTED_FEATURES)})."
        )

    return bundle


def get_categorical_levels(bundle: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract exact category levels from the fitted OneHotEncoder in the pipeline.
    Avoids hardcoding and guarantees consistency with trained model.
    """
    try:
        pipeline = bundle["model"]
        preprocessor = pipeline.named_steps["preprocessor"]
        cat_pipeline = preprocessor.named_transformers_["categorical"]
        encoder = cat_pipeline.named_steps["encoder"]

        cat_feature_names = [
            "gender",
            "admission_type",
            "icu_type",
            "diabetes_history",
            "ventilation_status",
        ]

        categories_dict = {}
        for col_name, cat_array in zip(cat_feature_names, encoder.categories_):
            categories_dict[col_name] = [str(c) for c in cat_array]

        return categories_dict
    except Exception:
        # Fallback to known notebook values if internal structure cannot be accessed
        return DEFAULT_CATEGORIES


def get_model_coefficients(bundle: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract feature names and coefficients from the final Logistic Regression classifier.
    Computes Odds Ratios for clinical interpretability.
    """
    try:
        pipeline = bundle["model"]
        preprocessor = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["classifier"]

        feature_names = preprocessor.get_feature_names_out()
        weights = clf.coef_[0]

        df_coef = pd.DataFrame({
            "Feature_Raw": feature_names,
            "Coefficient": weights,
            "Odds_Ratio": np.exp(weights),
            "Absolute_Weight": np.abs(weights),
        })

        # Format display name
        def clean_name(name: str) -> str:
            name = name.replace("numeric__", "").replace("categorical__", "")
            return name.replace("_", " ").title()

        df_coef["Feature"] = df_coef["Feature_Raw"].apply(clean_name)
        df_coef["Impact"] = np.where(
            df_coef["Coefficient"] > 0, "Increases Mortality Risk", "Protective / Decreases Risk"
        )
        return df_coef.sort_values(by="Coefficient", ascending=False).reset_index(drop=True)
    except Exception as e:
        st.error(f"Could not extract model coefficients: {e}")
        return pd.DataFrame()


def predict_single_record(
    bundle: Dict[str, Any],
    input_values: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate mortality prediction for a single patient record.
    Preserves missing values for pipeline median/mode imputers.
    Never mutates original pipeline or fits new encoders.
    """
    pipeline = bundle["model"]
    feature_cols = bundle["feature_columns"]
    threshold = float(bundle.get("threshold", 0.50))

    # Construct single-row DataFrame with exact columns and order
    row_data = {}
    for col in feature_cols:
        val = input_values.get(col, np.nan)
        # Convert empty strings to NaN so simple imputer activates
        if val is None or (isinstance(val, str) and val.strip() == ""):
            row_data[col] = [np.nan]
        else:
            row_data[col] = [val]

    df_input = pd.DataFrame(row_data)[feature_cols]

    # Predict using trained pipeline
    prob_array = pipeline.predict_proba(df_input)[0]
    prob_survived = float(prob_array[0])
    prob_died = float(prob_array[1])

    predicted_class = int(prob_died >= threshold)
    predicted_label = "Died" if predicted_class == 1 else "Survived"

    return {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "prob_died": prob_died,
        "prob_survived": prob_survived,
        "threshold": threshold,
        "features_df": df_input,
    }


def predict_batch_dataframe(
    bundle: Dict[str, Any],
    df_raw: pd.DataFrame
) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], Dict[str, Any]]:
    """
    Execute batch prediction on an uploaded patient DataFrame.
    Validates feature presence, strips extraneous columns, and returns annotated results.
    """
    pipeline = bundle["model"]
    feature_cols = bundle["feature_columns"]
    threshold = float(bundle.get("threshold", 0.50))

    # Validation: check for required 21 columns
    missing_cols = [c for c in feature_cols if c not in df_raw.columns]
    if missing_cols:
        return None, missing_cols, {}

    # Extract required features in authoritative order
    X_input = df_raw[feature_cols].copy()

    # Predict probabilities directly using the pipeline
    prob_array = pipeline.predict_proba(X_input)
    prob_survived = prob_array[:, 0]
    prob_died = prob_array[:, 1]

    pred_classes = (prob_died >= threshold).astype(int)
    pred_labels = np.where(pred_classes == 1, "Died", "Survived")

    # Build annotated result DataFrame
    df_results = df_raw.copy()
    df_results["predicted_mortality_probability"] = np.round(prob_died, 4)
    df_results["predicted_class"] = pred_classes
    df_results["predicted_outcome"] = pred_labels

    summary_metrics = {
        "total_records": len(df_results),
        "predicted_died_count": int(np.sum(pred_classes == 1)),
        "predicted_survived_count": int(np.sum(pred_classes == 0)),
        "predicted_mortality_rate": float(np.mean(pred_classes == 1) * 100),
        "predicted_survival_rate": float(np.mean(pred_classes == 0) * 100),
        "avg_mortality_probability": float(np.mean(prob_died) * 100),
    }

    return df_results, None, summary_metrics
