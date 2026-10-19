"""
Data Quality View for ICU Patient Mortality Prediction.
Provides transparent audits of missingness, duplicate rows, data types, IQR outliers,
and an interactive before/after cleaning comparison when both datasets are available.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

from src.ui_components import render_header, render_kpi_card, apply_plotly_theme
from src.data_utils import (
    compute_dataset_kpis,
    compute_data_quality_table,
    compute_iqr_outliers,
    load_cleaned_reference_dataset,
    NUMERICAL_CLINICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


def render_data_quality_view(df: Optional[pd.DataFrame]):
    """Render the Data Quality and Audit page."""
    render_header(
        title="DATA QUALITY & AUDIT",
        subtitle="Verification of Dataset Completeness, Schema Integrity, Missingness, and Clinical Outliers",
        badge_text="Quality Audit",
    )

    if df is None:
        st.warning("Please load or upload a dataset to perform quality auditing.")
        return

    # Data Quality KPI Summary
    total_cells = df.shape[0] * df.shape[1]
    total_nulls = int(df.isnull().sum().sum())
    null_pct = (total_nulls / total_cells * 100) if total_cells > 0 else 0.0
    duplicate_rows = int(df.duplicated().sum())

    k_col1, k_col2, k_col3, k_col4 = st.columns(4)
    with k_col1:
        render_kpi_card("Dimensions", f"{df.shape[0]:,} × {df.shape[1]}", "Rows × Columns", "Shape", "purple", border_color="#7C3AED")
    with k_col2:
        render_kpi_card(
            "Missing Cells",
            f"{total_nulls:,}",
            f"{null_pct:.2f}% of total data matrix",
            "Completeness",
            "amber" if total_nulls > 0 else "teal",
            border_color="#F59E0B" if total_nulls > 0 else "#14B8A6",
        )
    with k_col3:
        render_kpi_card(
            "Duplicate Records",
            f"{duplicate_rows:,}",
            "Exact row matches across all fields",
            "Deduplication",
            "amber" if duplicate_rows > 0 else "teal",
            border_color="#F59E0B" if duplicate_rows > 0 else "#14B8A6",
        )
    with k_col4:
        cols_with_nulls = int((df.isnull().sum() > 0).sum())
        render_kpi_card(
            "Columns with Nulls",
            f"{cols_with_nulls} / {df.shape[1]}",
            "Features requiring imputation",
            "Imputation Target",
            "teal",
            border_color="#14B8A6",
        )

    st.markdown("---")

    dq_tab1, dq_tab2, dq_tab3, dq_tab4, dq_tab5 = st.tabs([
        "Missing Values Audit",
        "Duplicate Rows Inspection",
        "Outlier & Range Analysis",
        "Statistical Summaries",
        "Before / After Cleaning Comparison",
    ])

    # ==========================================
    # TAB 1: Missing Values Audit
    # ==========================================
    with dq_tab1:
        st.markdown("#### Missing Values by Clinical Feature")
        st.caption(
            "Detailed audit of unrecorded or null values. In real hospital records, missingness frequently "
            "occurs because clinicians only order specific laboratory panels when clinically indicated."
        )

        quality_table = compute_data_quality_table(df)

        # Missingness Bar Chart
        missing_subset = quality_table[quality_table["Missing Count"] > 0].copy()

        if len(missing_subset) > 0:
            fig_miss = px.bar(
                missing_subset,
                x="Column Name",
                y="Missing %",
                text="Missing %",
                color="Missing %",
                color_continuous_scale="Teal",
                labels={"Missing %": "Missing Percentage (%)", "Column Name": "Feature"},
                title="Features with Missing Values (%)",
            )
            fig_miss.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_miss.update_layout(height=380)
            apply_plotly_theme(fig_miss)
            st.plotly_chart(fig_miss, width="stretch")
        else:
            st.success("No missing values detected in the currently loaded dataset!")

        st.markdown("##### Comprehensive Data Schema & Completeness Table")
        st.dataframe(quality_table, hide_index=True, width="stretch")

        st.info(
            "**Data Imputation Integrity:** Intermediate cleaning cells in the project notebook show that "
            "even after initial imputation attempts, residual missingness may remain in secondary features. "
            "To ensure production safety without data loss, the saved production pipeline encapsulates its own "
            "fitted median (for numerical) and mode (for categorical) `SimpleImputer` steps."
        )

    # ==========================================
    # TAB 2: Duplicate Rows Inspection
    # ==========================================
    with dq_tab2:
        st.markdown("#### Duplicate Records Audit")
        st.caption(
            "Duplicate admissions distort training distributions and can lead to over-optimistic evaluation metrics "
            "if identical records fall into both train and test splits."
        )

        if duplicate_rows > 0:
            st.warning(
                f"**Identified {duplicate_rows:,} duplicate records** across the loaded dataset. "
                "The notebook deduplication protocol applies `df.drop_duplicates(keep='first')`."
            )
            dupe_df = df[df.duplicated(keep=False)].sort_values(by=list(df.columns[:3]))
            st.markdown("##### Preview of Duplicate Records (Sample):")
            st.dataframe(dupe_df.head(20), width="stretch")
        else:
            st.success("No duplicate records found in the current dataset.")

    # ==========================================
    # TAB 3: Outlier & Range Analysis
    # ==========================================
    with dq_tab3:
        st.markdown("#### Interquartile Range (IQR) Outlier Analysis")
        st.caption(
            "Outlier thresholds calculated using Tukey's method: Lower Limit = Q1 - 1.5×IQR, "
            "Upper Limit = Q3 + 1.5×IQR. Evaluates clinical measurements against statistical bounds."
        )

        iqr_table = compute_iqr_outliers(df)
        st.dataframe(iqr_table, hide_index=True, width="stretch")

        st.markdown("##### Interactive Outlier Boxplot Explorer")
        avail_num = [c for c in NUMERICAL_CLINICAL_FEATURES if c in df.columns]
        chosen_outlier_feat = st.selectbox(
            "Select Feature to Inspect Outliers & Limits:",
            options=avail_num,
            index=0,
            format_func=lambda x: x.replace("_", " ").title(),
            key="dq_outlier_select",
        )

        if chosen_outlier_feat:
            feat_series = pd.to_numeric(df[chosen_outlier_feat], errors="coerce").dropna()
            q1 = feat_series.quantile(0.25)
            q3 = feat_series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            fig_out_box = px.box(
                df,
                y=chosen_outlier_feat,
                points="all",
                color_discrete_sequence=["#14B8A6"],
                title=f"Outlier Distribution for {chosen_outlier_feat.replace('_', ' ').title()}",
                labels={chosen_outlier_feat: chosen_outlier_feat.replace("_", " ").title()},
            )
            # Add horizontal lines for IQR boundaries
            fig_out_box.add_hline(
                y=upper_bound,
                line_dash="dash",
                line_color="#EF4444",
                annotation_text=f"Upper Bound: {upper_bound:.2f}",
            )
            fig_out_box.add_hline(
                y=lower_bound,
                line_dash="dash",
                line_color="#EF4444",
                annotation_text=f"Lower Bound: {lower_bound:.2f}",
            )
            fig_out_box.update_layout(height=420)
            apply_plotly_theme(fig_out_box)
            st.plotly_chart(fig_out_box, width="stretch")

        st.warning(
            "**Clinical Warning on Outlier Treatment in Intensive Care:** "
            "In commercial and consumer data science, extreme values are frequently trimmed or capped. "
            "However, in critical care medicine, **extreme physiological readings (e.g., severe acidosis, "
            "extreme tachycardia, or elevated lactate) are genuine biological markers of acute shock or organ failure.** "
            "Capping or deleting them without clinical consultation risks obscuring critical early mortality signals."
        )

    # ==========================================
    # TAB 4: Statistical Summaries
    # ==========================================
    with dq_tab4:
        st.markdown("#### Numerical Feature Summary Statistics")
        num_summary = df[avail_num].describe().transpose().round(2)
        num_summary["IQR"] = (num_summary["75%"] - num_summary["25%"]).round(2)
        st.dataframe(num_summary, width="stretch")

        st.markdown("#### Categorical Value Frequencies")
        avail_cats = [c for c in CATEGORICAL_FEATURES if c in df.columns]
        for c in avail_cats:
            with st.expander(f"Frequency Distribution: {c.replace('_', ' ').title()}", expanded=False):
                freq = df[c].value_counts(dropna=False).reset_index()
                freq.columns = ["Category Level", "Patient Count"]
                freq["Percentage (%)"] = (freq["Patient Count"] / len(df) * 100).round(2)
                st.dataframe(freq, hide_index=True, width="stretch")

    # ==========================================
    # TAB 5: Before / After Cleaning Comparison
    # ==========================================
    with dq_tab5:
        st.markdown("#### Before / After Cleaning Comparative Audit")
        st.caption(
            "Compare the uncleaned raw cohort (with intentional simulated missingness and duplicate records) "
            "against the fully curated 5,000-record reference dataset."
        )

        df_cleaned = load_cleaned_reference_dataset()

        if df_cleaned is None:
            st.info(
                "Cleaned reference dataset `synthetic_icu_mortality_5000_with_categories.csv` was not found. "
                "Displaying statistics for currently active dataset only."
            )
        else:
            comp_col1, comp_col2 = st.columns(2)

            with comp_col1:
                st.markdown("##### Raw Dataset (`icu_mortality_with_nulls_and_duplicates.csv`)")
                raw_kpis = compute_dataset_kpis(df)
                st.write(f"- **Total Rows:** {raw_kpis['total_records']:,}")
                st.write(f"- **Duplicate Rows:** {raw_kpis['duplicate_count']:,}")
                st.write(f"- **Missing Value Cells:** {raw_kpis['missing_count']:,} ({raw_kpis['missing_rate']:.2f}%)")
                st.write(f"- **Mortality Rate:** {raw_kpis.get('died_pct', 0):.2f}%")

            with comp_col2:
                st.markdown("##### Cleaned Reference (`synthetic_icu_mortality_5000_with_categories.csv`)")
                clean_kpis = compute_dataset_kpis(df_cleaned)
                st.write(f"- **Total Rows:** {clean_kpis['total_records']:,}")
                st.write(f"- **Duplicate Rows:** {clean_kpis['duplicate_count']:,}")
                st.write(f"- **Missing Value Cells:** {clean_kpis['missing_count']:,} ({clean_kpis['missing_rate']:.2f}%)")
                st.write(f"- **Mortality Rate:** {clean_kpis.get('died_pct', 0):.2f}%")

            # Comparative table
            st.markdown("##### Comparative Metric Table")
            comp_df = pd.DataFrame({
                "Metric": [
                    "Total Patient Records",
                    "Total Features",
                    "Duplicate Rows",
                    "Total Null Cells",
                    "Null Cell Percentage",
                    "Survivors (Class 0)",
                    "In-Hospital Deaths (Class 1)",
                    "Empirical Mortality Rate",
                ],
                "Raw (Uncleaned)": [
                    f"{raw_kpis['total_records']:,}",
                    f"{raw_kpis['total_features']}",
                    f"{raw_kpis['duplicate_count']:,}",
                    f"{raw_kpis['missing_count']:,}",
                    f"{raw_kpis['missing_rate']:.2f}%",
                    f"{raw_kpis.get('survived_count', 0):,}",
                    f"{raw_kpis.get('died_count', 0):,}",
                    f"{raw_kpis.get('died_pct', 0):.2f}%",
                ],
                "Cleaned (Reference)": [
                    f"{clean_kpis['total_records']:,}",
                    f"{clean_kpis['total_features']}",
                    f"{clean_kpis['duplicate_count']:,}",
                    f"{clean_kpis['missing_count']:,}",
                    f"{clean_kpis['missing_rate']:.2f}%",
                    f"{clean_kpis.get('survived_count', 0):,}",
                    f"{clean_kpis.get('died_count', 0):,}",
                    f"{clean_kpis.get('died_pct', 0):.2f}%",
                ],
            })
            st.dataframe(comp_df, hide_index=True, width="stretch")
