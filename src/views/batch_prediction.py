"""
Batch CSV Prediction View for ICU Patient Mortality Prediction.
Enables high-throughput clinical batch scoring, column validation,
summary analytics, and CSV export while maintaining HIPAA-conscious in-memory processing.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional

from src.ui_components import (
    render_header,
    render_clinical_disclaimer,
    render_privacy_notice,
    render_kpi_card,
    apply_plotly_theme,
)
from src.model_utils import predict_batch_dataframe, EXPECTED_FEATURES
from src.data_utils import parse_uploaded_csv


def render_batch_prediction_view(bundle: Optional[Dict[str, Any]]):
    """Render Batch CSV Prediction page."""
    render_header(
        title="BATCH CSV PATIENT MORTALITY PREDICTION",
        subtitle="Automated High-Throughput Batch Scoring and In-Hospital Mortality Risk Classification",
        badge_text="Batch Engine",
    )

    render_clinical_disclaimer()

    if bundle is None:
        st.error(
            "Model artifact could not be loaded. Please ensure 'icu_mortality_pipeline.joblib' "
            "is present in the project directory."
        )
        return

    # In-memory confidentiality notice - dark themed
    render_privacy_notice()

    # Provide Batch Template for Easy Testing
    col_up, col_tmp = st.columns([3, 1])

    with col_tmp:
        st.markdown("##### Need a Test File?")
        sample_path = Path(__file__).resolve().parent.parent.parent / "sample_icu_batch_input.csv"
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                template_bytes = f.read()
            st.download_button(
                label="Download Sample Batch CSV",
                data=template_bytes,
                file_name="sample_icu_batch_input.csv",
                mime="text/csv",
                help="Download a verified 5-patient test CSV matching all 21 expected features.",
            )
        else:
            st.caption("Generate a template from the single prediction screen.")

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Clinical CSV File for Batch Prediction",
            type=["csv"],
            key="batch_file_uploader",
            help="Upload a CSV file containing the 21 required numerical and categorical ICU features.",
        )

    if uploaded_file is None:
        st.markdown("---")
        st.markdown("#### Expected Input Specifications")
        st.caption("The batch file must supply these 21 clinical predictors (additional identifier columns like `subject_id` are safely ignored):")

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("**16 Numerical Clinical Measurements:**")
            st.code(", ".join(EXPECTED_FEATURES[:16]), language="text")
        with exp_col2:
            st.markdown("**5 Categorical Patient & Unit Descriptors:**")
            st.code(", ".join(EXPECTED_FEATURES[16:]), language="text")
        return

    # Parse CSV
    df_upload = parse_uploaded_csv(uploaded_file)
    if df_upload is None or df_upload.empty:
        st.error("Uploaded CSV is empty or could not be parsed. Please check the file format.")
        return

    st.success(f"CSV uploaded successfully: **{len(df_upload):,} rows** and **{df_upload.shape[1]} columns** detected.")

    # Execute Batch Prediction
    with st.spinner("Validating feature columns and running pipeline inference..."):
        df_results, missing_cols, summary = predict_batch_dataframe(bundle, df_upload)

    # Column Validation Check
    if missing_cols:
        st.error(
            f"**Validation Error: Missing Required Columns ({len(missing_cols)})**\n\n"
            f"The uploaded CSV is missing the following features required by the trained model pipeline:\n\n"
            f"- `{', '.join(missing_cols)}`\n\n"
            "Please ensure your CSV includes all 21 expected clinical features."
        )
        return

    st.markdown("---")

    # Batch Prediction Summary KPIs
    st.markdown("#### Batch Prediction Analytics Summary")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card(
            title="Total Processed",
            value=f"{summary['total_records']:,}",
            subtitle="Patient admission records",
            badge="Cohort",
            badge_tone="blue",
        )
    with k2:
        render_kpi_card(
            title="Predicted Survived",
            value=f"{summary['predicted_survived_count']:,}",
            subtitle=f"{summary['predicted_survival_rate']:.1f}% of cohort",
            badge="Class 0",
            badge_tone="green",
            border_color="#10B981",
        )
    with k3:
        render_kpi_card(
            title="Predicted Died",
            value=f"{summary['predicted_died_count']:,}",
            subtitle=f"{summary['predicted_mortality_rate']:.1f}% of cohort",
            badge="Class 1 Flag",
            badge_tone="red",
            border_color="#EF4444",
        )
    with k4:
        render_kpi_card(
            title="Mean Mortality Prob",
            value=f"{summary['avg_mortality_probability']:.1f}%",
            subtitle=f"Threshold: {bundle.get('threshold', 0.50):.2f}",
            badge="Risk Mean",
            badge_tone="slate",
        )

    # Outcome Visual Distribution
    chart_c1, chart_c2 = st.columns([1, 1])

    with chart_c1:
        pred_dist = df_results["predicted_outcome"].value_counts().reset_index()
        pred_dist.columns = ["Predicted_Outcome", "Count"]

        fig_pred_bar = px.bar(
            pred_dist,
            x="Predicted_Outcome",
            y="Count",
            text="Count",
            color="Predicted_Outcome",
            color_discrete_map={"Survived": "#14B8A6", "Died": "#EF4444"},
            title="Batch Prediction Outcome Breakdown",
            labels={"Predicted_Outcome": "Predicted Outcome", "Count": "Number of Patients"},
        )
        fig_pred_bar.update_traces(texttemplate="%{text:,} patients", textposition="outside")
        fig_pred_bar.update_layout(height=350, showlegend=False)
        apply_plotly_theme(fig_pred_bar)
        st.plotly_chart(fig_pred_bar, width="stretch")

    with chart_c2:
        fig_prob_hist = px.histogram(
            df_results,
            x="predicted_mortality_probability",
            nbins=30,
            color="predicted_outcome",
            color_discrete_map={"Survived": "#14B8A6", "Died": "#EF4444"},
            title="Distribution of Model-Estimated Mortality Probabilities",
            labels={"predicted_mortality_probability": "Estimated Mortality Probability"},
        )
        fig_prob_hist.add_vline(
            x=bundle.get("threshold", 0.50),
            line_dash="dash",
            line_color="#7C3AED",
            annotation_text=f"Threshold ({bundle.get('threshold', 0.50):.2f})",
            annotation_font_color="#F8FAFC",
        )
        fig_prob_hist.update_layout(height=350)
        apply_plotly_theme(fig_prob_hist)
        st.plotly_chart(fig_prob_hist, width="stretch")

    # Detailed Results Table
    st.markdown("#### Patient Batch Prediction Results")
    st.caption("Sort and inspect predicted mortality risk alongside individual clinical parameters:")

    # Highlight risk in display
    display_cols = ["predicted_outcome", "predicted_mortality_probability", "predicted_class"] + [
        c for c in EXPECTED_FEATURES if c in df_results.columns
    ]

    st.dataframe(
        df_results[display_cols],
        width="stretch",
        hide_index=True,
    )

    # CSV Download Button
    export_csv = df_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Full Batch Predictions (CSV)",
        data=export_csv,
        file_name="icu_batch_mortality_predictions.csv",
        mime="text/csv",
        type="primary",
    )
