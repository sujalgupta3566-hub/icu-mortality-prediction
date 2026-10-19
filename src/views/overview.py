"""
Overview Dashboard View for ICU Patient Mortality Prediction.
Presents high-level clinical objectives, dataset KPIs, outcome distribution, and project workflow.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

from src.ui_components import (
    render_header,
    render_clinical_disclaimer,
    render_kpi_card,
    apply_plotly_theme,
)
from src.data_utils import compute_dataset_kpis, TARGET_COLUMN


def render_overview_view(df: Optional[pd.DataFrame]):
    """Render the Overview Dashboard page."""
    render_header(
        title="ICU PATIENT MORTALITY PREDICTION",
        subtitle="Machine Learning-Based Exploratory Analytics and Mortality Classification",
        badge_text="Overview Dashboard",
    )

    render_clinical_disclaimer()

    # Analytical Context & Objectives
    st.markdown(
        """
        ### Clinical Problem & Analytical Objectives
        Intensive Care Units (ICUs) care for critically ill patients whose clinical status can change rapidly. 
        Early identification of patients with elevated mortality risk enables clinical teams to allocate resources, 
        prioritize close surveillance, and intervene before irreversible organ decompensation occurs.

        This dashboard provides an end-to-end data science platform that explores ICU patient cohorts, audits data quality, 
        illustrates model evaluation across imbalanced clinical data, and serves real-time predictions using a 
        pre-trained, class-weighted machine learning pipeline.
        """
    )

    st.markdown("---")

    # Dataset KPIs
    if df is None:
        st.warning(
            "No dataset is currently loaded. Please upload a CSV file below or place "
            "'icu_mortality_with_nulls_and_duplicates.csv' in the project directory."
        )
        uploaded = st.file_uploader("Upload ICU Clinical CSV Dataset", type=["csv"], key="overview_uploader")
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.session_state["dataset"] = df
            st.rerun()
        return

    # Compute actual KPIs
    kpis = compute_dataset_kpis(df)

    st.markdown("#### Primary Cohort Key Performance Indicators")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        render_kpi_card(
            title="Total Records",
            value=f"{kpis['total_records']:,}",
            subtitle="Patient ICU admissions",
            badge="Cohort Size",
            badge_tone="purple",
            border_color="#7C3AED",
        )

    with col2:
        render_kpi_card(
            title="Clinical Features",
            value=f"{kpis['total_features']}",
            subtitle="Predictors & identifiers",
            badge="Dimensions",
            badge_tone="teal",
            border_color="#14B8A6",
        )

    with col3:
        render_kpi_card(
            title="Missing Values",
            value=f"{kpis['missing_count']:,}",
            subtitle=f"{kpis['missing_rate']:.1f}% of total cells",
            badge="Data Quality",
            badge_tone="amber" if kpis['missing_count'] > 0 else "teal",
            border_color="#F59E0B" if kpis['missing_count'] > 0 else "#14B8A6",
        )

    with col4:
        render_kpi_card(
            title="Duplicate Rows",
            value=f"{kpis['duplicate_count']:,}",
            subtitle="Identical admissions",
            badge="Cleaning",
            badge_tone="amber" if kpis['duplicate_count'] > 0 else "teal",
            border_color="#F59E0B" if kpis['duplicate_count'] > 0 else "#14B8A6",
        )

    with col5:
        if kpis["has_target"]:
            render_kpi_card(
                title="Survived Class",
                value=f"{kpis['survived_pct']:.1f}%",
                subtitle=f"{kpis['survived_count']:,} patients (Class 0)",
                badge="Majority",
                badge_tone="green",
                border_color="#14B8A6",
            )
        else:
            render_kpi_card("Target Outcome", "N/A", "Target not found")

    with col6:
        if kpis["has_target"]:
            render_kpi_card(
                title="Mortality Class",
                value=f"{kpis['died_pct']:.1f}%",
                subtitle=f"{kpis['died_count']:,} patients (Class 1)",
                badge="Minority",
                badge_tone="red",
                border_color="#EF4444",
            )
        else:
            render_kpi_card("Target Outcome", "N/A", "Target not found")

    st.markdown("<br>", unsafe_allow_html=True)

    # Target Outcome Distribution
    if kpis["has_target"]:
        st.markdown("#### Patient In-Hospital Mortality Outcome Distribution (`hospital_expire_flag`)")
        st.caption(
            "The target outcome is binary: **0 = Survived** during hospital stay, and "
            "**1 = Died** during hospital stay. ICU mortality exhibits pronounced class imbalance, "
            "requiring specialized machine learning cost-weighting."
        )

        target_counts = df[TARGET_COLUMN].value_counts().reset_index()
        target_counts.columns = ["Outcome_Code", "Patient_Count"]
        target_counts["Outcome_Label"] = target_counts["Outcome_Code"].map({0: "Survived (Class 0)", 1: "Died (Class 1)"})
        target_counts["Percentage"] = (target_counts["Patient_Count"] / len(df) * 100).round(2)

        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            fig_bar = px.bar(
                target_counts,
                x="Outcome_Label",
                y="Patient_Count",
                text="Patient_Count",
                color="Outcome_Label",
                color_discrete_map={
                    "Survived (Class 0)": "#14B8A6",
                    "Died (Class 1)": "#EF4444",
                },
                labels={"Outcome_Label": "Hospital Outcome", "Patient_Count": "Number of Patients"},
                title="Patient Count by Mortality Outcome",
            )
            fig_bar.update_traces(
                texttemplate="%{text:,} patients",
                textposition="outside",
                marker_line_width=1,
                marker_line_color="#1E293B",
            )
            fig_bar.update_layout(showlegend=False, height=360)
            apply_plotly_theme(fig_bar)
            st.plotly_chart(fig_bar, width="stretch")
            st.caption(
                f"**Observation:** Survived cases comprise {kpis['survived_pct']:.1f}% of observations, "
                f"while in-hospital deaths represent {kpis['died_pct']:.1f}%. Models trained on raw data without "
                "reweighting would suffer severe minority-class bias."
            )

        with chart_col2:
            fig_donut = px.pie(
                target_counts,
                names="Outcome_Label",
                values="Patient_Count",
                hole=0.55,
                color="Outcome_Label",
                color_discrete_map={
                    "Survived (Class 0)": "#14B8A6",
                    "Died (Class 1)": "#EF4444",
                },
                title="Mortality Class Ratio (Percentage)",
            )
            fig_donut.update_traces(
                textposition="inside",
                textinfo="percent+label",
                marker=dict(line=dict(color="#151F33", width=2)),
            )
            fig_donut.update_layout(height=360, showlegend=True, legend=dict(orientation="h", y=-0.1))
            apply_plotly_theme(fig_donut)
            st.plotly_chart(fig_donut, width="stretch")
            st.caption(
                "**Donut Representation:** The severe minority ratio (~1:12 death-to-survival ratio) "
                "dictates the use of Precision-Recall AUC (PR-AUC) over standard accuracy."
            )

    st.markdown("---")

    # Project Workflow Pipeline
    st.markdown("#### Machine Learning Development Workflow")
    st.caption("Standardized seven-stage methodology executed in `ICUPREDICTIONNEW.ipynb` and operationalized in this application:")

    workflow_steps = [
        ("1. Data Inspection", "Audit MIMIC-IV derived ICU variables, schemas, data types, and null patterns."),
        ("2. Cleaning & Imputation", "Deduplicate identical records and establish median/mode imputers for missing physiology."),
        ("3. Exploratory Analysis", "Analyze physiological distributions, patient demographics, and bivariate correlations."),
        ("4. Imbalance Strategy", "Benchmark SMOTENC synthetic oversampling versus cost-sensitive class weights."),
        ("5. Model Comparison", "Train and benchmark Logistic Regression, Random Forest, GBDT, Decision Tree, and SVM."),
        ("6. Holdout Evaluation", "Evaluate final model on untouched 20% holdout test set with threshold analysis."),
        ("7. Pipeline Deployment", "Serialize end-to-end scikit-learn pipeline for real-time and batch clinical inference."),
    ]

    cols = st.columns(len(workflow_steps))
    for i, (col, (step_title, step_desc)) in enumerate(zip(cols, workflow_steps)):
        with col:
            st.markdown(
                f"""
                <div style="background: #151F33; border: 1px solid #1E293B; border-top: 3px solid #14B8A6; 
                            border-radius: 8px; padding: 12px; height: 180px; display: flex; flex-direction: column; justify-content: space-between;">
                    <div style="font-weight: 700; font-size: 0.85rem; color: #F8FAFC; margin-bottom: 6px;">
                        {step_title}
                    </div>
                    <div style="font-size: 0.75rem; color: #CBD5E1; line-height: 1.35;">
                        {step_desc}
                    </div>
                    <div style="font-size: 0.70rem; font-weight: 600; color: #14B8A6; margin-top: 6px;">
                        Step 0{i+1}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
