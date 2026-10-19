"""
Project Information View for ICU Patient Mortality Prediction.
Comprehensive technical documentation, methodological breakdown, algorithmic comparisons,
and academic project context.
"""

import streamlit as st
from src.ui_components import render_header, render_clinical_disclaimer


def render_project_info_view():
    """Render Project Information and Documentation page."""
    render_header(
        title="PROJECT INFORMATION & METHODOLOGY",
        subtitle="End-to-End Documentation of Clinical Problem, Engineering Pipeline, and Imbalance Experiments",
        badge_text="Documentation",
    )

    render_clinical_disclaimer()

    st.markdown(
        """
        ### 1. Problem Statement & Clinical Context
        In Intensive Care Units (ICUs), rapid physiological deterioration can occur within hours. 
        Traditional clinical risk scoring systems—such as **APACHE (Acute Physiology and Chronic Health Evaluation)**, 
        **SAPS (Simplified Acute Physiology Score)**, and **SOFA (Sequential Organ Failure Assessment)**—rely on 
        rule-based point allocations derived from statistical regression models developed decades ago.

        While widely used for cohort risk benchmarking, traditional scores often struggle to adapt to nonlinear 
        physiological interactions and the acute class imbalance inherent in hospital mortality data. This project 
        investigates the development, empirical benchmarking, and deployment of a modern machine learning pipeline to 
        predict **in-hospital mortality (`hospital_expire_flag`)** using physiological, laboratory, and demographic 
        measurements captured during early ICU care.

        ---

        ### 2. Objectives & Project Scope
        1. **End-to-End Data Pipeline:** Design an interactive analytics platform that audits raw ICU data, inspects data quality, and performs reproducible preprocessing.
        2. **Imbalance Benchmarking:** Empirically evaluate synthetic minority oversampling (SMOTENC) against cost-sensitive class weighting across multiple algorithmic families.
        3. **Clinical Interpretability:** Demystify model decision-making through coefficient analysis and Odds Ratios to confirm biological plausibility.
        4. **Interactive Clinical Inference:** Serve the final pre-trained pipeline for real-time individual assessment and high-throughput batch CSV scoring.
        5. **Ethical & Safety Guardrails:** Contextualize predictions as academic prototypes with clear regulatory warnings regarding false positives and false negatives.

        ---

        ### 3. Dataset Provenance & Structure
        The dataset utilized in this project is structured around the schema and physiological parameters of the 
        **MIMIC-IV (Medical Information Mart for Intensive Care IV)** clinical database, distributed as a curated 
        synthetic/de-identified cohort (`icu_mortality_with_nulls_and_duplicates.csv` and `synthetic_icu_mortality_5000_with_categories.csv`).

        #### Data Fields & Schema Architecture
        - **Target Variable:** `hospital_expire_flag`
          - `0` = Patient survived during the hospital admission (Majority class, ~92.2% of cohort).
          - `1` = Patient died during the hospital admission (Minority class, ~7.8% of cohort).
        - **Patient & Admission Descriptors (5 Features):**
          - `gender`: Biological sex (Female, Male).
          - `admission_type`: Categorization of admission urgency (Elective, Emergency, Urgent).
          - `icu_type`: Specialized care unit (Cardiac ICU, Medical ICU, Neuro ICU, Surgical ICU).
          - `diabetes_history`: Documented pre-existing diabetes diagnosis (No, Yes).
          - `ventilation_status`: Mechanical ventilation support during admission (No, Yes).
        - **Numerical Physiological Predictors (16 Features):**
          - Vital signs: `age`, `heart_rate_mean`, `heart_rate_min`, `heart_rate_max`, `resp_rate_mean`, `resp_rate_min`, `resp_rate_max`, `spo2_mean`, `sbp_mean`, `dbp_mean`.
          - Laboratory chemistry & hematology: `creatinine_mean` (renal filtration), `urea_nitrogen_mean` (BUN / metabolic waste), `wbc_mean` (immune response / sepsis), `sodium_mean` & `potassium_mean` (electrolyte balance), `lactate_mean` (tissue perfusion / metabolic acidosis).
        - **Identifiers (Excluded from Inference):**
          - `subject_id`, `hadm_id`, `stay_id` are strictly withheld from model feature matrices to prevent identity leakage.

        ---

        ### 4. Data Cleaning, Imputation & Outlier Handling
        #### A. Deduplication
        The raw dataset contains 150 duplicate admission records. Duplicate rows artificially inflate training sample density 
        and risk test-set contamination. The cleaning pipeline applies strict first-occurrence deduplication:
        $$\text{Deduplicated Size} = 5,150 - 150 = 5,000 \text{ unique patients}$$

        #### B. Missing Value Strategy
        In ICU workflows, missing values are rarely missing completely at random (MCAR); instead, tests are ordered 
        only when symptoms present. To eliminate data leakage and ensure production readiness:
        - Numerical features are imputed using training **median** statistics (resistant to extreme skew).
        - Categorical features are imputed using training **most frequent (mode)** category levels.
        - The imputer is encapsulated inside the fitted scikit-learn `ColumnTransformer` to guarantee identical handling for new inference records.

        #### C. Outlier Exploration vs. Clinical Capping
        Using Tukey's method ($[Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}]$), outliers were identified across all 16 numerical vitals. 
        While exploratory notebook cells tested quantile capping, **clinical caution must be exercised:**
        In an ICU patient, an elevated serum lactate of $8.0\text{ mmol/L}$ or heart rate of $160\text{ bpm}$ is not measurement error; 
        it is a critical indicator of impending circulatory collapse. Trimming these values attenuates the model's ability to recognize severe illness.

        ---

        ### 5. Class Imbalance Experiments: SMOTENC vs. Class Weighting
        Mortality prediction exhibits acute class imbalance (~1:12 death-to-survival ratio). Predicting the majority class alone 
        would yield a deceptive 92.2% accuracy while failing 100% of dying patients.

        Two primary counter-measures were investigated in the notebook:
        1. **SMOTENC (Synthetic Minority Over-sampling for Nominal and Continuous features):** 
           Generated interpolated synthetic records to balance the training set. However, in medicine, synthetic interpolations can 
           produce biologically implausible combinations (e.g., normal lactate with profound septic shock vitals).
        2. **Cost-Sensitive Class Weighting (`class_weight='balanced'`):** 
           Adjusts the loss function dynamically by inversely weighting each class relative to its empirical frequency:
           $$w_j = \frac{N}{2 \times N_j}$$
           This heavily penalizes errors on deceased patients without generating artificial patient data.

        **Result:** Class weighting produced cleaner probabilistic calibration and superior holdout recall without synthetic distortion.

        ---

        ### 6. Multi-Model Validation Benchmarking
        Five algorithms were trained and benchmarked at the standard $0.50$ decision threshold:

        | Model | Death Precision | Death Recall | Death F1 | PR-AUC | ROC-AUC |
        |---|---:|---:|---:|---:|---:|
        | **Logistic Regression (Class-Weighted)** | **0.152** | **0.587** | **0.241** | **0.205** | **0.685** |
        | Random Forest (Class-Weighted) | 0.200 | 0.016 | 0.029 | 0.154 | 0.643 |
        | Gradient Boosting (Sample-Weighted) | 0.150 | 0.349 | 0.210 | 0.132 | 0.632 |
        | Support Vector Machine (Class-Weighted) | 0.000 | 0.000 | 0.000 | 0.106 | 0.578 |
        | Decision Tree (Class-Weighted) | 0.127 | 0.127 | 0.127 | 0.085 | 0.526 |

        #### Why Class-Weighted Logistic Regression Won:
        1. **Highest Minority Recall & F1:** It achieved by far the highest sensitivity to in-hospital death on both validation (0.587) and holdout (0.756).
        2. **Tree Collapse on Imbalance:** Complex tree models (Random Forest, Decision Tree) collapsed on minority recall because individual splits were dominated by the 92% survival majority.
        3. **Complete Interpretability:** Linear coefficients allow clinicians to inspect Odds Ratios and verify biological coherence.
        4. **Inference Latency:** Logistic Regression evaluates in less than $2\text{ ms}$, ideal for bedside ICU deployment.

        ---

        ### 7. Final Pipeline Holdout Evaluation
        The production pipeline was evaluated on a held-out test split of 1,000 patients:
        - **Holdout ROC-AUC:** `0.753` (solid overall ranking capability)
        - **Holdout PR-AUC:** `0.225` (nearly 3× random guessing baseline of 0.078)
        - **Holdout Death Recall:** `75.6%` (captured 59 of 78 deceased patients)
        - **Confusion Matrix:** `[[636, 286], [19, 59]]`

        ---

        ### 8. Installation & Local Launch Instructions
        To run this application locally:

        ```bash
        # 1. Clone repository and navigate to folder
        cd reatail_sales

        # 2. Install dependencies
        python -m pip install -r requirements.txt

        # 3. Launch Streamlit web dashboard
        streamlit run app.py
        ```

        The application will automatically detect `icu_mortality_pipeline.joblib` and `icu_mortality_with_nulls_and_duplicates.csv` in the root folder.

        ---

        ### 9. Developer Attribution & Technical Profile
        - **Lead Developer:** **Sujal**
        - **Professional Title:** Aspiring Data Scientist
        - **Core Technical Focus:** Python, SQL, Machine Learning, Statistical Inference, Clinical Analytics
        """
    )
