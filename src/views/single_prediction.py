"""
Single Patient Prediction View for ICU Patient Mortality Prediction.
Provides a clinical input form with pre-filled sample profiles, validation,
and real-time inference using the cached scikit-learn pipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional

from src.ui_components import render_header, render_clinical_disclaimer, apply_plotly_theme
from src.model_utils import (
    predict_single_record,
    get_categorical_levels,
    EXPECTED_FEATURES,
)


# Clinically plausible preset patient profiles for convenient evaluation
SAMPLE_PROFILES = {
    "Select a pre-filled profile (Optional)...": None,
    "Patient A: Stable Post-Operative (Low Risk)": {
        "age": 45.0,
        "gender": "Female",
        "admission_type": "Elective",
        "icu_type": "Surgical ICU",
        "diabetes_history": "No",
        "ventilation_status": "No",
        "heart_rate_mean": 74.0,
        "heart_rate_min": 62.0,
        "heart_rate_max": 88.0,
        "resp_rate_mean": 15.0,
        "resp_rate_min": 12.0,
        "resp_rate_max": 18.0,
        "spo2_mean": 99.0,
        "sbp_mean": 124.0,
        "dbp_mean": 72.0,
        "creatinine_mean": 0.85,
        "urea_nitrogen_mean": 14.0,
        "wbc_mean": 7.5,
        "sodium_mean": 140.0,
        "potassium_mean": 4.1,
        "lactate_mean": 1.1,
    },
    "Patient B: Acute Sepsis / Severe Instability (High Risk)": {
        "age": 78.0,
        "gender": "Male",
        "admission_type": "Emergency",
        "icu_type": "Medical ICU",
        "diabetes_history": "Yes",
        "ventilation_status": "Yes",
        "heart_rate_mean": 118.0,
        "heart_rate_min": 98.0,
        "heart_rate_max": 142.0,
        "resp_rate_mean": 28.0,
        "resp_rate_min": 22.0,
        "resp_rate_max": 36.0,
        "spo2_mean": 91.5,
        "sbp_mean": 88.0,
        "dbp_mean": 52.0,
        "creatinine_mean": 3.4,
        "urea_nitrogen_mean": 54.0,
        "wbc_mean": 24.8,
        "sodium_mean": 132.0,
        "potassium_mean": 5.4,
        "lactate_mean": 4.8,
    },
    "Patient C: Cohort Median Patient (Baseline)": {
        "age": 64.0,
        "gender": "Male",
        "admission_type": "Emergency",
        "icu_type": "Medical ICU",
        "diabetes_history": "No",
        "ventilation_status": "No",
        "heart_rate_mean": 88.36,
        "heart_rate_min": 78.66,
        "heart_rate_max": 103.08,
        "resp_rate_mean": 19.15,
        "resp_rate_min": 16.14,
        "resp_rate_max": 23.91,
        "spo2_mean": 96.0,
        "sbp_mean": 118.70,
        "dbp_mean": 68.16,
        "creatinine_mean": 1.06,
        "urea_nitrogen_mean": 19.02,
        "wbc_mean": 9.85,
        "sodium_mean": 139.04,
        "potassium_mean": 4.09,
        "lactate_mean": 1.70,
    },
}


def render_single_prediction_view(bundle: Optional[Dict[str, Any]]):
    """Render Single Patient Clinical Prediction interface."""
    render_header(
        title="SINGLE PATIENT MORTALITY PREDICTION",
        subtitle="Individual Patient Clinical Parameter Assessment and In-Hospital Mortality Risk Estimation",
        badge_text="Inference Engine",
    )

    render_clinical_disclaimer()

    if bundle is None:
        st.error(
            "Model artifact could not be loaded. Please ensure 'icu_mortality_pipeline.joblib' "
            "is located in the project directory."
        )
        return

    # Extract dynamic categories directly from fitted OneHotEncoder
    category_options = get_categorical_levels(bundle)
    threshold = float(bundle.get("threshold", 0.50))

    # Pre-load helper selection
    st.markdown("#### Patient Data Input")
    st.caption("You can manually input all clinical parameters or load a pre-configured sample profile for rapid testing:")

    selected_preset = st.selectbox(
        "Load Sample Patient Profile (Optional):",
        options=list(SAMPLE_PROFILES.keys()),
        key="single_pred_preset",
    )

    preset_data = SAMPLE_PROFILES[selected_preset] if selected_preset in SAMPLE_PROFILES else None

    # Prediction Form
    with st.form("single_patient_prediction_form"):
        st.markdown("##### 1. Demographics & ICU Admission Information")
        d_col1, d_col2, d_col3 = st.columns(3)

        with d_col1:
            age_default = float(preset_data["age"]) if preset_data else 64.0
            age = st.number_input(
                "Patient Age (years)",
                min_value=18.0,
                max_value=110.0,
                value=age_default,
                step=1.0,
                help="Patient age at hospital admission (MIMIC-IV cohort).",
            )

            gender_opts = category_options.get("gender", ["Female", "Male"])
            gender_default_idx = gender_opts.index(preset_data["gender"]) if preset_data and preset_data.get("gender") in gender_opts else 0
            gender = st.selectbox("Biological Gender", options=gender_opts, index=gender_default_idx)

        with d_col2:
            adm_opts = category_options.get("admission_type", ["Elective", "Emergency", "Urgent"])
            adm_default_idx = adm_opts.index(preset_data["admission_type"]) if preset_data and preset_data.get("admission_type") in adm_opts else 1
            admission_type = st.selectbox("Admission Type", options=adm_opts, index=adm_default_idx)

            icu_opts = category_options.get("icu_type", ["Cardiac ICU", "Medical ICU", "Neuro ICU", "Surgical ICU"])
            icu_default_idx = icu_opts.index(preset_data["icu_type"]) if preset_data and preset_data.get("icu_type") in icu_opts else 1
            icu_type = st.selectbox("ICU Care Unit Type", options=icu_opts, index=icu_default_idx)

        with d_col3:
            diab_opts = category_options.get("diabetes_history", ["No", "Yes"])
            diab_default_idx = diab_opts.index(preset_data["diabetes_history"]) if preset_data and preset_data.get("diabetes_history") in diab_opts else 0
            diabetes_history = st.selectbox("Pre-existing Diabetes History", options=diab_opts, index=diab_default_idx)

            vent_opts = category_options.get("ventilation_status", ["No", "Yes"])
            vent_default_idx = vent_opts.index(preset_data["ventilation_status"]) if preset_data and preset_data.get("ventilation_status") in vent_opts else 0
            ventilation_status = st.selectbox("Mechanical Ventilation Status", options=vent_opts, index=vent_default_idx)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 2. Clinical Measurements (Vital Signs & Laboratory Panels)")
        st.caption(
            "Values represent summary statistics recorded during the initial 24 hours of ICU stay. "
            "Leave blank or enter valid values; the model pipeline contains a median imputer for missing physiology."
        )

        v_col1, v_col2, v_col3, v_col4 = st.columns(4)

        # Heart Rate Panel
        with v_col1:
            st.markdown("**Heart Rate (bpm)**")
            hr_mean = st.number_input("Mean Heart Rate", min_value=20.0, max_value=250.0, value=float(preset_data["heart_rate_mean"]) if preset_data else 88.0, step=1.0)
            hr_min = st.number_input("Min Heart Rate", min_value=15.0, max_value=200.0, value=float(preset_data["heart_rate_min"]) if preset_data else 78.0, step=1.0)
            hr_max = st.number_input("Max Heart Rate", min_value=30.0, max_value=280.0, value=float(preset_data["heart_rate_max"]) if preset_data else 103.0, step=1.0)

        # Respiratory & SpO2 Panel
        with v_col2:
            st.markdown("**Respiration & Oxygenation**")
            rr_mean = st.number_input("Mean Resp Rate (bpm)", min_value=4.0, max_value=70.0, value=float(preset_data["resp_rate_mean"]) if preset_data else 19.0, step=0.5)
            rr_min = st.number_input("Min Resp Rate (bpm)", min_value=2.0, max_value=50.0, value=float(preset_data["resp_rate_min"]) if preset_data else 16.0, step=0.5)
            rr_max = st.number_input("Max Resp Rate (bpm)", min_value=6.0, max_value=80.0, value=float(preset_data["resp_rate_max"]) if preset_data else 24.0, step=0.5)
            spo2_mean = st.number_input("Mean SpO2 (%)", min_value=60.0, max_value=100.0, value=float(preset_data["spo2_mean"]) if preset_data else 96.0, step=0.5)

        # Hemodynamics & Blood Pressure
        with v_col3:
            st.markdown("**Blood Pressure (mmHg)**")
            sbp_mean = st.number_input("Mean Systolic BP", min_value=40.0, max_value=260.0, value=float(preset_data["sbp_mean"]) if preset_data else 118.0, step=1.0)
            dbp_mean = st.number_input("Mean Diastolic BP", min_value=20.0, max_value=160.0, value=float(preset_data["dbp_mean"]) if preset_data else 68.0, step=1.0)
            st.markdown("**Metabolic Lab Panel**")
            lactate_mean = st.number_input("Mean Serum Lactate (mmol/L)", min_value=0.1, max_value=30.0, value=float(preset_data["lactate_mean"]) if preset_data else 1.70, step=0.1)

        # Renal & Hematology Lab Panel
        with v_col4:
            st.markdown("**Renal & Hematology Labs**")
            creatinine_mean = st.number_input("Mean Creatinine (mg/dL)", min_value=0.1, max_value=25.0, value=float(preset_data["creatinine_mean"]) if preset_data else 1.06, step=0.05)
            bun_mean = st.number_input("Mean BUN (mg/dL)", min_value=1.0, max_value=200.0, value=float(preset_data["urea_nitrogen_mean"]) if preset_data else 19.0, step=1.0)
            wbc_mean = st.number_input("Mean WBC (K/uL)", min_value=0.5, max_value=100.0, value=float(preset_data["wbc_mean"]) if preset_data else 9.85, step=0.1)
            sodium_mean = st.number_input("Mean Sodium (mEq/L)", min_value=100.0, max_value=180.0, value=float(preset_data["sodium_mean"]) if preset_data else 139.0, step=0.5)
            potassium_mean = st.number_input("Mean Potassium (mEq/L)", min_value=1.5, max_value=10.0, value=float(preset_data["potassium_mean"]) if preset_data else 4.10, step=0.1)

        submit_btn = st.form_submit_button(
            "Generate In-Hospital Mortality Prediction",
            width="stretch",
            type="primary",
        )

    # Process Prediction
    if submit_btn:
        input_dict = {
            "age": age,
            "gender": gender,
            "admission_type": admission_type,
            "icu_type": icu_type,
            "diabetes_history": diabetes_history,
            "ventilation_status": ventilation_status,
            "heart_rate_mean": hr_mean,
            "heart_rate_min": hr_min,
            "heart_rate_max": hr_max,
            "resp_rate_mean": rr_mean,
            "resp_rate_min": rr_min,
            "resp_rate_max": rr_max,
            "spo2_mean": spo2_mean,
            "sbp_mean": sbp_mean,
            "dbp_mean": dbp_mean,
            "creatinine_mean": creatinine_mean,
            "urea_nitrogen_mean": bun_mean,
            "wbc_mean": wbc_mean,
            "sodium_mean": sodium_mean,
            "potassium_mean": potassium_mean,
            "lactate_mean": lactate_mean,
        }

        with st.spinner("Executing pipeline preprocessing and model inference..."):
            results = predict_single_record(bundle, input_dict)

        pred_class = results["predicted_class"]
        pred_label = results["predicted_label"]
        prob_died = results["prob_died"]
        prob_survived = results["prob_survived"]

        st.markdown("### Prediction Outcome & Clinical Risk Assessment")

        # Result Banner
        if pred_class == 1:
            st.markdown(
                f"""
                <div class="prediction-card-died">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span class="icu-badge badge-red" style="font-size: 0.90rem; padding: 4px 12px;">HIGH MORTALITY RISK FLAG</span>
                            <h2 style="color: #F87171; margin: 8px 0 4px 0; font-size: 1.8rem;">
                                Predicted Class: 1 (Died)
                            </h2>
                            <p style="color: #CBD5E1; font-size: 0.95rem; margin: 0;">
                                The model's estimated probability of in-hospital mortality is 
                                <strong style="color: #F8FAFC;">{prob_died * 100:.1f}%</strong> (decision threshold: {threshold * 100:.0f}%).
                            </p>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="prediction-card-survived">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span class="icu-badge badge-green" style="font-size: 0.90rem; padding: 4px 12px;">FAVORABLE SURVIVAL PROFILE</span>
                            <h2 style="color: #34D399; margin: 8px 0 4px 0; font-size: 1.8rem;">
                                Predicted Class: 0 (Survived)
                            </h2>
                            <p style="color: #CBD5E1; font-size: 0.95rem; margin: 0;">
                                The model's estimated probability of in-hospital mortality is 
                                <strong style="color: #F8FAFC;">{prob_died * 100:.1f}%</strong> (below threshold of {threshold * 100:.0f}%).
                            </p>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Visual Probability Gauge
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob_died * 100, 1),
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Estimated Mortality Probability (%)", "font": {"size": 16, "family": "Inter", "color": "#F8FAFC"}},
                number={"suffix": "%", "font": {"size": 28, "color": "#F8FAFC", "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                    "bar": {"color": "#EF4444" if pred_class == 1 else "#14B8A6", "thickness": 0.4},
                    "bgcolor": "#151F33",
                    "borderwidth": 1,
                    "bordercolor": "#1E293B",
                    "steps": [
                        {"range": [0, threshold * 100], "color": "rgba(20, 184, 166, 0.2)"},
                        {"range": [threshold * 100, 100], "color": "rgba(239, 68, 68, 0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "#7C3AED", "width": 4},
                        "thickness": 0.85,
                        "value": threshold * 100,
                    },
                },
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
            apply_plotly_theme(fig_gauge)
            st.plotly_chart(fig_gauge, width="stretch")

        with res_col2:
            st.markdown("##### Clinical Interpretative Context")
            st.write(f"- **Model Estimated Survival Probability:** `{prob_survived * 100:.1f}%`")
            st.write(f"- **Model Estimated Mortality Probability:** `{prob_died * 100:.1f}%`")
            st.write(f"- **Operating Decision Threshold:** `{threshold:.2f}`")
            st.write(f"- **Predicted Mortality Classification:** `{'Class 1 (Died)' if pred_class == 1 else 'Class 0 (Survived)'}`")

            st.caption(
                "**Educational Model Disclaimer:** This probability is a statistical estimate produced by a "
                "class-weighted Logistic Regression model trained on retrospective data. It does **not** constitute "
                "a validated prognostic score (such as APACHE IV or SOFA) and must never guide actual clinical therapy."
            )

        # Download input + prediction as CSV
        export_df = results["features_df"].copy()
        export_df["predicted_mortality_probability"] = round(prob_died, 4)
        export_df["predicted_class"] = pred_class
        export_df["predicted_label"] = pred_label
        export_df["decision_threshold"] = threshold

        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Patient Prediction Record (CSV)",
            data=csv_data,
            file_name="icu_patient_prediction_result.csv",
            mime="text/csv",
        )
