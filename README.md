# ICU Patient In-Hospital Mortality Prediction Using Machine Learning

An enterprise-grade, interactive healthcare analytics dashboard and clinical decision support prototype developed with **Streamlit**, **Scikit-learn**, and **Plotly**. This application explores critical care patient physiology, audits dataset quality, benchmarks five machine learning algorithms under severe class imbalance, and provides real-time single-patient and high-throughput batch predictions using an existing, serialized end-to-end model pipeline.

![Project Status](https://img.shields.io/badge/Status-Production%20Ready-0D9488)
![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12-1E293B)
![Framework](https://img.shields.io/badge/Framework-Streamlit%201.66-FF4B4B)
![Model](https://img.shields.io/badge/Model-Class--Weighted%20Logistic%20Regression-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 1. Project Overview & Clinical Context

In Intensive Care Units (ICUs), clinical trajectories evolve rapidly. Early recognition of physiological deterioration is essential to prevent refractory organ failure and mortality. Traditional risk scores (e.g., APACHE, SAPS, SOFA) rely on static, linear point aggregations that frequently struggle with non-linear physiological interactions and acute minority-class imbalance.

This project delivers an interactive machine learning application that:
- Audits patient cohort characteristics, data types, missingness patterns, and clinical outliers.
- Recreates exploratory bivariate and multivariate analyses with interactive Plotly visualizations.
- Benchmarks five machine learning models evaluated under cost-sensitive class weights and SMOTENC oversampling.
- Operationalizes a pre-trained, regularized scikit-learn pipeline for real-time risk assessment without data leakage.
- Preserves patient confidentiality through HIPAA-conscious, session-only in-memory batch processing.

---

## 2. Application Architecture & Sidebar Sections

The application is structured around a wide, professional healthcare layout with custom navy-blue, white, and teal styling:

1. **Overview Dashboard:** High-level cohort KPIs (total admissions, features, missing values, duplicate counts, empirical survival and mortality rates), interactive outcome distribution bar and donut charts, and the 7-stage ML methodology workflow.
2. **Exploratory Data Analysis (EDA):** Five interactive modules covering numerical distributions (histogram, boxplot, ECDF), categorical frequencies, bivariate scatter plots with OLS trendlines for key physiological pairs, a full Pearson correlation heatmap, and outcome-stratified physiological comparisons.
3. **Data Quality & Audit:** Comprehensive schema and completeness auditing, per-column missingness analysis, duplicate record detection, Tukey IQR outlier analysis with interactive boundary boxplots, and an empirical before/after cleaning comparison.
4. **Model Performance & Benchmarking:** Historical validation comparisons across 5 algorithms (Logistic Regression, Random Forest, GBDT, Decision Tree, SVM) at the 0.50 threshold, alongside the final pipeline's recorded holdout evaluation (Confusion Matrix, ROC-AUC: `0.753`, PR-AUC: `0.225`, Death Recall: `75.6%`, Odds Ratios, and clinical metrics glossary).
5. **Single Patient Prediction:** Interactive clinical entry form organized into Demographics & Admission details and Vital Signs / Lab Chemistry. Includes one-click sample patient presets (Stable Post-Op, Acute Sepsis, Cohort Median), dynamic category validation, probability gauge visualizer, and single-record CSV download.
6. **Batch CSV Prediction:** High-throughput batch inference engine with 21-feature schema validation, automatic exclusion of identifiers and target leakage, missing value preservation for pipeline imputers, cohort summary analytics, probability distribution charts, and full CSV export.
7. **Project Information:** In-depth documentation covering problem statement, data provenance, deduplication, imputation rationale, outlier capping considerations, SMOTENC vs. class-weight experiments, algorithmic selection trade-offs, and run instructions.

---

## 3. Technology Stack

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/) (wide layout, custom CSS, state caching)
- **Data Manipulation & Analytics:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Interactive Visualizations:** [Plotly Express & Graph Objects](https://plotly.com/python/)
- **Machine Learning & Pipeline:** [Scikit-learn](https://scikit-learn.org/), [Joblib](https://joblib.readthedocs.io/)
- **Statistical Modeling:** [Statsmodels](https://www.statsmodels.org/) (OLS trendline estimation)

---

## 4. Model Pipeline Specification

The application integrates the pre-trained model bundle serialized in `icu_mortality_pipeline.joblib`:

- **Operating Decision Threshold:** `0.50`
- **Target Variable:** `hospital_expire_flag` (`0` = Survived, `1` = Died)
- **Features Handled (21 total):**
  - **16 Numerical Predictors:** `age`, `heart_rate_mean`, `heart_rate_min`, `heart_rate_max`, `resp_rate_mean`, `resp_rate_min`, `resp_rate_max`, `spo2_mean`, `sbp_mean`, `dbp_mean`, `creatinine_mean`, `urea_nitrogen_mean`, `wbc_mean`, `sodium_mean`, `potassium_mean`, `lactate_mean`
  - **5 Categorical Predictors:** `gender`, `admission_type`, `icu_type`, `diabetes_history`, `ventilation_status`
- **Pipeline Architecture:**
  1. *Numerical Pipeline:* `SimpleImputer(strategy="median")` $\rightarrow$ `StandardScaler()`
  2. *Categorical Pipeline:* `SimpleImputer(strategy="most_frequent")` $\rightarrow$ `OneHotEncoder(handle_unknown="ignore")`
  3. *Preprocessor:* `ColumnTransformer` assembling numeric and categorical transformations
  4. *Classifier:* `LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)`

---

## 5. Project Directory Structure

```text
reatail_sales/
├── app.py                                            # Main Streamlit dashboard application
├── requirements.txt                                  # Production environment dependencies
├── README.md                                         # Comprehensive documentation & setup guide
├── .gitignore                                        # Git exclusion rules
├── icu_mortality_pipeline.joblib                     # Fitted scikit-learn model bundle
├── icu_mortality_with_nulls_and_duplicates.csv       # Primary cohort dataset with nulls/duplicates
├── synthetic_icu_mortality_5000_with_categories.csv  # Curated clean reference dataset
├── synthetic_icu_data_dictionary.csv                 # Column descriptions and units
├── sample_icu_batch_input.csv                        # Verified 5-record batch testing template
├── ICUPREDICTIONNEW.ipynb                            # Source notebook with training & experiments
└── src/                                              # Modular application source code
    ├── __init__.py
    ├── model_utils.py                                # Cached model loading & inference helpers
    ├── data_utils.py                                 # Data loading, quality audits, IQR stats
    ├── ui_components.py                              # CSS injection, KPI cards, Plotly themes
    └── views/                                        # Individual dashboard section views
        ├── __init__.py
        ├── overview.py                               # 1. Overview Dashboard
        ├── eda.py                                    # 2. Exploratory Data Analysis
        ├── data_quality.py                           # 3. Data Quality & Audit
        ├── model_performance.py                      # 4. Model Performance & Benchmarks
        ├── single_prediction.py                      # 5. Single Patient Prediction Form
        ├── batch_prediction.py                       # 6. Batch CSV Prediction Engine
        └── project_info.py                           # 7. Project Information & Methodology
```

---

## 6. Installation & Local Launch

### Step 1: Clone the Repository & Navigate to Folder
```bash
git clone <repository-url>
cd reatail_sales
```

### Step 2: Create and Activate a Python Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application
```bash
streamlit run app.py
```

The application will launch automatically in your default web browser at `http://localhost:8501`.

---

## 7. Streamlit Cloud Deployment Instructions

To deploy this application to **Streamlit Community Cloud**:

1. Push this project repository to your GitHub account.
2. Log in to [share.streamlit.io](https://share.streamlit.io/).
3. Click **"New app"**.
4. Select your repository, branch (`main`), and set the main file path to `app.py`.
5. Ensure `icu_mortality_pipeline.joblib` and `icu_mortality_with_nulls_and_duplicates.csv` are tracked in Git (or stored via Git LFS / cloud storage if larger than GitHub standard file limits).
6. Click **"Deploy!"**. Streamlit Cloud will automatically install dependencies from `requirements.txt` and launch the live URL.

---

## 8. Clinical & Regulatory Disclaimer

> [!WARNING]
> **Educational & Research Prototype Only:** This software application and machine learning model are developed strictly for academic demonstration and research benchmarking as part of an MCA Capstone portfolio. 
> 
> The predictions, estimated probabilities, and risk classifications generated by this prototype **do not constitute medical advice, clinical diagnoses, or validated prognostic scores**. They must **never** be used to alter medical treatments, prioritize patient care, or substitute for professional judgment by qualified healthcare providers.

---

## 9. Author & Developer Attribution

**Created by Sujal** — Aspiring Data Scientist
- **Technical Specialization:** Python • SQL • Machine Learning • Healthcare Predictive Analytics • Streamlit UI/UX Engineering

