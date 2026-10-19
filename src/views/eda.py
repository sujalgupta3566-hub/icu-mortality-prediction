"""
Exploratory Data Analysis View for ICU Patient Mortality Prediction.
Recreates all exploratory visualizations from ICUPREDICTIONNEW.ipynb with interactive Plotly graphics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

from src.ui_components import render_header, apply_plotly_theme
from src.data_utils import (
    NUMERICAL_CLINICAL_FEATURES,
    CATEGORICAL_FEATURES,
    NOTEBOOK_SCATTER_PAIRS,
    TARGET_COLUMN,
    compute_correlation_matrix,
)


def render_eda_view(df: Optional[pd.DataFrame]):
    """Render the Exploratory Data Analysis page."""
    render_header(
        title="EXPLORATORY DATA ANALYSIS",
        subtitle="Interactive Visualizations of ICU Patient Physiology, Demographics, and Clinical Relationships",
        badge_text="EDA Suite",
    )

    if df is None:
        st.warning("Please load or upload a dataset to explore clinical feature distributions.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Numerical Distributions",
        "Categorical Analysis",
        "Bivariate Relationships",
        "Correlation Heatmap",
        "Target vs. Physiology",
    ])

    # ==========================================
    # TAB 1: Numerical Feature Distributions
    # ==========================================
    with tab1:
        st.markdown("#### Clinical Feature Distributions")
        st.caption(
            "Select any of the 16 clinical physiological measurements to inspect its empirical distribution, "
            "central tendency, and clinical dispersion."
        )

        available_numerics = [col for col in NUMERICAL_CLINICAL_FEATURES if col in df.columns]

        selected_num = st.selectbox(
            "Select Numerical Feature:",
            options=available_numerics,
            index=0,
            format_func=lambda x: f"{x.replace('_', ' ').title()} ({x})",
            key="eda_num_feature",
        )

        if selected_num:
            series = pd.to_numeric(df[selected_num], errors="coerce").dropna()

            # Statistical Summary Metrics
            m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
            m_col1.metric("Valid Observations", f"{len(series):,}")
            m_col2.metric("Mean", f"{series.mean():.2f}")
            m_col3.metric("Median", f"{series.median():.2f}")
            m_col4.metric("Std. Deviation", f"{series.std():.2f}")
            m_col5.metric("Minimum", f"{series.min():.2f}")
            m_col6.metric("Maximum", f"{series.max():.2f}")

            chart_col1, chart_col2 = st.columns([3, 2])

            with chart_col1:
                fig_hist = px.histogram(
                    df,
                    x=selected_num,
                    nbins=40,
                    marginal="box",
                    opacity=0.85,
                    color_discrete_sequence=["#14B8A6"],
                    labels={selected_num: selected_num.replace("_", " ").title()},
                    title=f"Histogram & Boxplot: {selected_num.replace('_', ' ').title()}",
                )
                fig_hist.update_layout(height=420, showlegend=False)
                apply_plotly_theme(fig_hist)
                st.plotly_chart(fig_hist, width="stretch")

            with chart_col2:
                # Cumulative Distribution / ECDF
                fig_ecdf = px.ecdf(
                    df,
                    x=selected_num,
                    color_discrete_sequence=["#7C3AED"],
                    title=f"Empirical CDF: {selected_num.replace('_', ' ').title()}",
                    labels={selected_num: selected_num.replace("_", " ").title(), "probability": "Cumulative Prob"},
                )
                fig_ecdf.update_layout(height=420)
                apply_plotly_theme(fig_ecdf)
                st.plotly_chart(fig_ecdf, width="stretch")

            st.caption(
                f"**Clinical Insight for {selected_num.replace('_', ' ').title()}:** "
                f"The median value is {series.median():.2f} with an interquartile spread from "
                f"{series.quantile(0.25):.2f} to {series.quantile(0.75):.2f}. Extreme observations in "
                "critically ill patients often signal acute physiological instability."
            )

    # ==========================================
    # TAB 2: Categorical Feature Analysis
    # ==========================================
    with tab2:
        st.markdown("#### Patient Cohort Categorical Analysis")
        st.caption(
            "Distribution of demographic and ICU operational characteristics: Gender, Admission Type, "
            "ICU Care Unit, Pre-existing Diabetes, and Mechanical Ventilation Status."
        )

        available_cats = [col for col in CATEGORICAL_FEATURES if col in df.columns]

        cat_col1, cat_col2 = st.columns([1, 2])

        with cat_col1:
            selected_cat = st.radio(
                "Choose Categorical Variable:",
                options=available_cats,
                format_func=lambda x: x.replace("_", " ").title(),
                key="eda_cat_selector",
            )

        with cat_col2:
            if selected_cat:
                cat_counts = df[selected_cat].fillna("Missing / Unrecorded").value_counts().reset_index()
                cat_counts.columns = [selected_cat, "Count"]
                cat_counts["Percentage"] = (cat_counts["Count"] / len(df) * 100).round(2)

                fig_cat = px.bar(
                    cat_counts,
                    x=selected_cat,
                    y="Count",
                    text=cat_counts.apply(lambda r: f"{r['Count']:,} ({r['Percentage']}%)", axis=1),
                    color=selected_cat,
                    color_discrete_sequence=["#14B8A6", "#7C3AED", "#38BDF8", "#F59E0B", "#10B981"],
                    title=f"Frequency Count & Percentage: {selected_cat.replace('_', ' ').title()}",
                )
                fig_cat.update_traces(textposition="outside")
                fig_cat.update_layout(height=400, showlegend=False)
                apply_plotly_theme(fig_cat)
                st.plotly_chart(fig_cat, width="stretch")

        st.markdown("##### Complete Categorical Breakdown Grid")
        grid_cols = st.columns(len(available_cats))
        for col_name, c in zip(available_cats, grid_cols):
            with c:
                st.markdown(f"**{col_name.replace('_', ' ').title()}**")
                mini_counts = df[col_name].value_counts(dropna=False).reset_index()
                mini_counts.columns = ["Value", "Count"]
                mini_counts["%"] = (mini_counts["Count"] / len(df) * 100).round(1)
                st.dataframe(mini_counts, hide_index=True, width="stretch")

    # ==========================================
    # TAB 3: Bivariate Scatter & Relationships
    # ==========================================
    with tab3:
        st.markdown("#### Clinical Bivariate Relationships")
        st.caption(
            "Scatter plots with fitted Ordinary Least Squares (OLS) linear trendlines for key physiological "
            "pairs explored in the project notebook."
        )

        pair_options = {label: (x_col, y_col) for x_col, y_col, label in NOTEBOOK_SCATTER_PAIRS}
        selected_pair_label = st.selectbox(
            "Select Clinical Feature Pair:",
            options=list(pair_options.keys()),
            key="eda_pair_selector",
        )

        x_col, y_col = pair_options[selected_pair_label]

        if x_col in df.columns and y_col in df.columns:
            # Pair-specific non-null slice
            pair_df = df[[x_col, y_col]].dropna().copy()
            pair_df[x_col] = pd.to_numeric(pair_df[x_col], errors="coerce")
            pair_df[y_col] = pd.to_numeric(pair_df[y_col], errors="coerce")
            pair_df = pair_df.dropna()

            # Optional color by target if present
            color_arg = None
            if TARGET_COLUMN in df.columns:
                pair_df[TARGET_COLUMN] = df.loc[pair_df.index, TARGET_COLUMN].map({0: "Survived", 1: "Died"})
                color_arg = TARGET_COLUMN

            fig_scatter = px.scatter(
                pair_df,
                x=x_col,
                y=y_col,
                color=color_arg,
                color_discrete_map={"Survived": "#14B8A6", "Died": "#EF4444"},
                opacity=0.60,
                title=f"{selected_pair_label} (N = {len(pair_df):,})",
                labels={
                    x_col: x_col.replace("_", " ").title(),
                    y_col: y_col.replace("_", " ").title(),
                    TARGET_COLUMN: "Outcome",
                },
            )

            # Fit OLS linear trendline using robust closed-form arithmetic (avoids LAPACK SVD crash)
            try:
                x_arr = pair_df[x_col].to_numpy()
                y_arr = pair_df[y_col].to_numpy()
                x_m = float(np.mean(x_arr))
                y_m = float(np.mean(y_arr))
                denom = float(np.sum((x_arr - x_m) ** 2))
                if denom > 0:
                    slope = float(np.sum((x_arr - x_m) * (y_arr - y_m)) / denom)
                    intercept = y_m - slope * x_m
                    x_vals = np.linspace(float(x_arr.min()), float(x_arr.max()), 100)
                    y_vals = slope * x_vals + intercept
                    fig_scatter.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines",
                        name=f"OLS Trendline (slope = {slope:.2f})",
                        line=dict(color="#7C3AED", width=2.5, dash="dash"),
                    ))
            except Exception:
                pass

            fig_scatter.update_layout(height=480)
            apply_plotly_theme(fig_scatter)
            st.plotly_chart(fig_scatter, width="stretch")

            # Pearson correlation for this pair
            corr_val = pair_df[x_col].corr(pair_df[y_col])
            st.caption(
                f"**Statistical Note:** Pearson correlation coefficient r = **{corr_val:.3f}**. "
                "Fitted trendline illustrates the general direction of physiological co-variation; "
                "isolated rows missing one or both values were excluded only from this specific visualization."
            )

    # ==========================================
    # TAB 4: Correlation Heatmap
    # ==========================================
    with tab4:
        st.markdown("#### Clinical Feature Correlation Heatmap")
        st.caption(
            "Pairwise Pearson correlation matrix among all 16 numerical physiological predictors. "
            "Subject, hospital admission, and stay identifiers are strictly excluded."
        )

        corr_df, active_cols = compute_correlation_matrix(df)

        clean_labels = [c.replace("_", " ").title() for c in active_cols]

        fig_heat = px.imshow(
            corr_df,
            x=clean_labels,
            y=clean_labels,
            color_continuous_scale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            text_auto=".2f",
            aspect="auto",
            title="Pearson Correlation Matrix (Clinical Predictors)",
        )
        fig_heat.update_layout(height=650)
        apply_plotly_theme(fig_heat)
        st.plotly_chart(fig_heat, width="stretch")

        st.info(
            "**Causality & Multicollinearity Caveat:** Correlation reflects linear association between "
            "vitals (such as mean and minimum heart rates), but **correlation does not establish clinical causation**. "
            "The downstream regularized Logistic Regression pipeline applies L2 penalty scaling to stabilize coefficient estimation."
        )

    # ==========================================
    # TAB 5: Target vs. Clinical Physiology
    # ==========================================
    with tab5:
        st.markdown("#### Physiological Variations Stratified by Survival Outcome")
        st.caption(
            "Examine whether critical laboratory values and vital signs differ systematically between "
            "patients who survived and those who experienced in-hospital mortality."
        )

        if TARGET_COLUMN not in df.columns:
            st.warning("The target column `hospital_expire_flag` is not present in the loaded dataset.")
        else:
            strat_feat = st.selectbox(
                "Select Clinical Feature to Compare by Outcome:",
                options=available_numerics,
                index=available_numerics.index("lactate_mean") if "lactate_mean" in available_numerics else 0,
                format_func=lambda x: x.replace("_", " ").title(),
                key="eda_target_strat_feat",
            )

            plot_df = df[[strat_feat, TARGET_COLUMN]].dropna().copy()
            plot_df[strat_feat] = pd.to_numeric(plot_df[strat_feat], errors="coerce")
            plot_df = plot_df.dropna()
            plot_df["Outcome"] = plot_df[TARGET_COLUMN].map({0: "Survived (Class 0)", 1: "Died (Class 1)"})

            col_t1, col_t2 = st.columns(2)

            with col_t1:
                fig_box = px.box(
                    plot_df,
                    x="Outcome",
                    y=strat_feat,
                    color="Outcome",
                    color_discrete_map={
                        "Survived (Class 0)": "#14B8A6",
                        "Died (Class 1)": "#EF4444",
                    },
                    points="outliers",
                    title=f"Boxplot Comparison: {strat_feat.replace('_', ' ').title()} by Outcome",
                    labels={strat_feat: strat_feat.replace("_", " ").title()},
                )
                fig_box.update_layout(height=400, showlegend=False)
                apply_plotly_theme(fig_box)
                st.plotly_chart(fig_box, width="stretch")

            with col_t2:
                fig_hist_split = px.histogram(
                    plot_df,
                    x=strat_feat,
                    color="Outcome",
                    barmode="overlay",
                    opacity=0.65,
                    color_discrete_map={
                        "Survived (Class 0)": "#14B8A6",
                        "Died (Class 1)": "#EF4444",
                    },
                    title=f"Distribution Overlay: {strat_feat.replace('_', ' ').title()}",
                    labels={strat_feat: strat_feat.replace("_", " ").title()},
                )
                fig_hist_split.update_layout(height=400)
                apply_plotly_theme(fig_hist_split)
                st.plotly_chart(fig_hist_split, width="stretch")

            # Outcome group summary statistics
            st.markdown(f"##### Group Statistics for `{strat_feat}`")
            group_stats = plot_df.groupby("Outcome")[strat_feat].agg(
                ["count", "mean", "std", "median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
            ).rename(columns={"<lambda_0>": "25th Pct", "<lambda_1>": "75th Pct"}).round(2)
            st.dataframe(group_stats, width="stretch")
