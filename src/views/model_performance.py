"""
Model Performance View for ICU Patient Mortality Prediction.
Separates historical validation benchmarks from the final deployed pipeline's recorded holdout evaluation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional

from src.ui_components import render_header, render_kpi_card, apply_plotly_theme
from src.model_utils import get_model_coefficients


# Recorded validation benchmark results from notebook Cell 66
HISTORICAL_VALIDATION_MODELS = [
    {
        "Model": "Logistic Regression (Class-Weighted)",
        "Death Precision": 0.152,
        "Death Recall": 0.587,
        "Death F1": 0.241,
        "PR-AUC": 0.205,
        "ROC-AUC": 0.685,
        "Algorithm Family": "Linear / Generalized Linear",
        "Imbalance Handling": "class_weight='balanced'",
    },
    {
        "Model": "Random Forest (Class-Weighted)",
        "Death Precision": 0.200,
        "Death Recall": 0.016,
        "Death F1": 0.029,
        "PR-AUC": 0.154,
        "ROC-AUC": 0.643,
        "Algorithm Family": "Ensemble / Bagging",
        "Imbalance Handling": "class_weight='balanced'",
    },
    {
        "Model": "Gradient Boosting (Sample-Weighted)",
        "Death Precision": 0.150,
        "Death Recall": 0.349,
        "Death F1": 0.210,
        "PR-AUC": 0.132,
        "ROC-AUC": 0.632,
        "Algorithm Family": "Ensemble / Boosting",
        "Imbalance Handling": "sample_weight=compute_sample_weight()",
    },
    {
        "Model": "SVM (Class-Weighted)",
        "Death Precision": 0.000,
        "Death Recall": 0.000,
        "Death F1": 0.000,
        "PR-AUC": 0.106,
        "ROC-AUC": 0.578,
        "Algorithm Family": "Kernel Support Vector Machine",
        "Imbalance Handling": "class_weight='balanced'",
    },
    {
        "Model": "Decision Tree (Class-Weighted)",
        "Death Precision": 0.127,
        "Death Recall": 0.127,
        "Death F1": 0.127,
        "PR-AUC": 0.085,
        "ROC-AUC": 0.526,
        "Algorithm Family": "Single Tree",
        "Imbalance Handling": "class_weight='balanced'",
    },
]


def render_model_performance_view(bundle: Optional[Dict[str, Any]]):
    """Render Model Performance and Evaluation page."""
    render_header(
        title="MODEL PERFORMANCE & BENCHMARKING",
        subtitle="Historical Multi-Model Validation Benchmarks and Recorded Final Pipeline Holdout Results",
        badge_text="Evaluation Hub",
    )

    tab_eval, tab_hist, tab_interpret, tab_glossary = st.tabs([
        "Final Pipeline Holdout Results",
        "Historical Model Comparison",
        "Clinical Feature Importance & Odds Ratios",
        "Metrics & Clinical Glossary",
    ])

    # ==========================================
    # TAB 1: Final Pipeline Holdout Results
    # ==========================================
    with tab_eval:
        st.markdown("#### Final Deployed Pipeline Holdout Evaluation")
        st.caption(
            "Evaluation on an independent, untouched 20% stratified holdout test split (N = 1,000 patients) "
            "as recorded in the project notebook. Decision threshold: **0.50**."
        )

        h_col1, h_col2, h_col3, h_col4, h_col5 = st.columns(5)
        with h_col1:
            render_kpi_card("ROC-AUC", "0.753", "Holdout Discrimination", "Primary", "blue")
        with h_col2:
            render_kpi_card("PR-AUC", "0.225", "Precision-Recall AUC", "Benchmark", "teal")
        with h_col3:
            render_kpi_card("Death Recall", "75.6%", "59 / 78 deaths captured", "Sensitivity", "green")
        with h_col4:
            render_kpi_card("Death Precision", "17.1%", "59 / 345 flagged", "PPV", "amber")
        with h_col5:
            render_kpi_card("Overall Accuracy", "69.5%", "695 / 1,000 correct", "Accuracy", "slate")

        eval_col1, eval_col2 = st.columns([1, 1])

        with eval_col1:
            # Confusion Matrix Heatmap
            # Recorded: [[636, 286], [19, 59]]
            cm_values = [[636, 286], [19, 59]]
            cm_text = [
                ["True Negative (TN)<br><b>636</b> patients", "False Positive (FP)<br><b>286</b> patients"],
                ["False Negative (FN)<br><b>19</b> patients", "True Positive (TP)<br><b>59</b> patients"],
            ]

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm_values,
                x=["Predicted Survived", "Predicted Died"],
                y=["Actual Survived", "Actual Died"],
                text=cm_text,
                texttemplate="%{text}",
                textfont={"size": 13, "family": "Inter", "color": "#F8FAFC"},
                colorscale=[[0, "#1E293B"], [0.5, "#7C3AED"], [1, "#14B8A6"]],
                showscale=False,
            ))
            fig_cm.update_layout(
                title="Holdout Confusion Matrix (Threshold = 0.50)",
                height=380,
                xaxis_title="Predicted Outcome",
                yaxis_title="Actual Hospital Outcome",
                yaxis=dict(autorange="reversed"),
            )
            apply_plotly_theme(fig_cm)
            st.plotly_chart(fig_cm, width="stretch")

            st.caption(
                "**Matrix Breakdown:** The model successfully flags 59 of the 78 deceased patients (75.6% sensitivity). "
                "The 286 False Positives represent patients with elevated mortality indicators who survived, "
                "which in an ICU surveillance context is far preferable to missing unstable patients (19 False Negatives)."
            )

        with eval_col2:
            # Detailed Classification Report
            st.markdown("##### Recorded Classification Report")
            rep_df = pd.DataFrame({
                "Outcome Class": ["Survived (Class 0)", "Died (Class 1)", "Macro Average", "Weighted Average"],
                "Precision": [0.97, 0.17, 0.57, 0.91],
                "Recall": [0.69, 0.76, 0.72, 0.69],
                "F1-Score": [0.81, 0.28, 0.54, 0.77],
                "Support (Patients)": [922, 78, 1000, 1000],
            })
            st.dataframe(rep_df, hide_index=True, width="stretch")

            st.markdown("##### Clinical Performance Trade-Off Analysis")
            st.markdown(
                """
                - **High Sensitivity Priority:** In acute intensive care, the penalty for a **False Negative** 
                  (failing to recognize a deteriorating patient who will die without aggressive intervention) is catastrophic.
                - **Tolerable False Positive Rate:** Flagging a patient who subsequently survives triggers closer nurse surveillance, 
                  lab re-checks, or vital sign re-titration, which carries relatively low clinical penalty.
                - **Class-Weighted Optimization:** By incorporating balanced loss weighting, the final Logistic Regression pipeline 
                  increased minority death recall to 75.6% without requiring artificial SMOTE record fabrication.
                """
            )

    # ==========================================
    # TAB 2: Historical Model Comparison
    # ==========================================
    with tab_hist:
        st.markdown("#### Historical Validation Benchmarks Across 5 Algorithms")
        st.caption(
            "Comparison of models trained and validated during exploratory experiments in `ICUPREDICTIONNEW.ipynb` "
            "at the standardized 0.50 decision threshold. Labeled as historical notebook validation results."
        )

        df_bench = pd.DataFrame(HISTORICAL_VALIDATION_MODELS)

        st.dataframe(
            df_bench.sort_values(by="PR-AUC", ascending=False).reset_index(drop=True),
            hide_index=True,
            width="stretch",
        )

        st.markdown("##### Comparison of Key Imbalanced Metrics: Recall, F1, and PR-AUC")

        plot_bench = df_bench.melt(
            id_vars=["Model"],
            value_vars=["Death Recall", "Death F1", "PR-AUC", "ROC-AUC"],
            var_name="Metric",
            value_name="Score",
        )

        fig_bench = px.bar(
            plot_bench,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            text="Score",
            color_discrete_sequence=["#14B8A6", "#7C3AED", "#38BDF8", "#F59E0B"],
            title="Algorithm Comparison Across Key Evaluation Metrics",
        )
        fig_bench.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_bench.update_layout(height=420, legend=dict(orientation="h", y=1.15))
        apply_plotly_theme(fig_bench)
        st.plotly_chart(fig_bench, width="stretch")

        st.info(
            "**Deployment Rationale:** Logistic Regression with class weights emerged as the superior production candidate, "
            "achieving the highest Death Recall (0.587 on validation, 0.756 on holdout), highest Death F1 (0.241), and highest "
            "PR-AUC (0.205). More complex models like Random Forest collapsed in minority-class recall (0.016) due to severe "
            "sub-tree class dominance."
        )

    # ==========================================
    # TAB 3: Feature Importance & Odds Ratios
    # ==========================================
    with tab_interpret:
        st.markdown("#### Model Interpretability: Coefficients & Odds Ratios")
        st.caption(
            "Coefficients and Odds Ratios ($e^{\\beta}$) derived directly from the fitted Logistic Regression classifier. "
            "An Odds Ratio > 1.0 indicates elevated mortality odds per standard deviation change."
        )

        if bundle is not None:
            df_coef = get_model_coefficients(bundle)

            if not df_coef.empty:
                c_col1, c_col2 = st.columns([1, 1])

                with c_col1:
                    st.markdown("##### Top Risk Factors (Increasing Predicted Mortality)")
                    pos_coef = df_coef[df_coef["Coefficient"] > 0].sort_values(by="Coefficient", ascending=False)
                    fig_pos = px.bar(
                        pos_coef.head(8),
                        x="Coefficient",
                        y="Feature",
                        orientation="h",
                        text="Coefficient",
                        color_discrete_sequence=["#EF4444"],
                        title="Strongest Predictors of Mortality Risk (+β)",
                    )
                    fig_pos.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    fig_pos.update_layout(height=380, yaxis=dict(autorange="reversed"))
                    apply_plotly_theme(fig_pos)
                    st.plotly_chart(fig_pos, width="stretch")

                with c_col2:
                    st.markdown("##### Top Protective Factors (Decreasing Mortality Risk)")
                    neg_coef = df_coef[df_coef["Coefficient"] < 0].sort_values(by="Coefficient", ascending=True)
                    fig_neg = px.bar(
                        neg_coef.head(8),
                        x="Coefficient",
                        y="Feature",
                        orientation="h",
                        text="Coefficient",
                        color_discrete_sequence=["#10B981"],
                        title="Strongest Protective Predictors (-β)",
                    )
                    fig_neg.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    fig_neg.update_layout(height=380, yaxis=dict(autorange="reversed"))
                    apply_plotly_theme(fig_neg)
                    st.plotly_chart(fig_neg, width="stretch")

                st.markdown("##### Complete Fitted Model Feature Weights Table")
                st.dataframe(
                    df_coef[["Feature", "Coefficient", "Odds_Ratio", "Impact"]].round(3),
                    hide_index=True,
                    width="stretch",
                )
        else:
            st.warning("Model bundle not loaded. Cannot display fitted coefficients.")

    # ==========================================
    # TAB 4: Metrics & Clinical Glossary
    # ==========================================
    with tab_glossary:
        st.markdown("#### Clinical & Statistical Metric Explanations")
        st.caption("Plain-language explanations of evaluation concepts for healthcare practitioners and evaluators.")

        glossary_items = [
            ("Death Precision (Positive Predictive Value)", 
             "Out of all patients flagged by the model as 'Died', what proportion actually died? "
             "In our model, precision is 17.1% because the model adopts an aggressive surveillance threshold to capture deaths."),
            ("Death Recall / Sensitivity", 
             "Out of all patients who genuinely died, what proportion did the model successfully identify? "
             "Our model achieves 75.6% recall, capturing roughly 3 out of every 4 deteriorating patients."),
            ("False Positives (FP)", 
             "Patients predicted to die who ultimately survived. Clinically, this leads to extra monitoring and diagnostic testing "
             "(and potential 'alarm fatigue'), but does not cause acute medical harm."),
            ("False Negatives (FN)", 
             "Critically ill patients who died whom the model failed to flag as high-risk. This is the most dangerous error in ICU care, "
             "making high recall the priority objective."),
            ("ROC-AUC (Receiver Operating Characteristic)", 
             "Measures the model's ability to rank a randomly chosen deceased patient higher than a surviving patient across all possible thresholds. "
             "Holdout ROC-AUC = 0.753 (strong discrimination)."),
            ("PR-AUC (Precision-Recall Area Under Curve)", 
             "The gold standard for imbalanced classification. Evaluates precision and recall across thresholds without being inflated by "
             "the overwhelming majority of surviving patients. Baseline random guess = 7.8%; our model achieves 0.225 (nearly 3× random)."),
        ]

        for term, desc in glossary_items:
            with st.expander(f"**{term}**", expanded=False):
                st.write(desc)
