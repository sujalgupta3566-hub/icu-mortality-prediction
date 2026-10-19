"""
Main Streamlit Application: ICU Patient Mortality Prediction Dashboard.
Orchestrates sidebar navigation, state management, cached model loading,
and renders responsive views across all 7 analytical sections.
"""

import sys
import os
from pathlib import Path

# Ensure conda env's Library\bin (MKL, BLAS, LAPACK) is in Windows DLL search path
conda_prefix = sys.prefix
library_bin = os.path.join(conda_prefix, "Library", "bin")
if os.path.exists(library_bin):
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(library_bin)
        except Exception:
            pass
    os.environ["PATH"] = library_bin + os.pathsep + os.environ.get("PATH", "")

import streamlit as st
import pandas as pd

# Add project root to sys.path to enable clean src imports
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ui_components import inject_custom_css
from src.model_utils import load_model_bundle, get_default_model_path
from src.data_utils import load_default_dataset, get_default_dataset_path
from src.views.overview import render_overview_view
from src.views.eda import render_eda_view
from src.views.data_quality import render_data_quality_view
from src.views.model_performance import render_model_performance_view
from src.views.single_prediction import render_single_prediction_view
from src.views.batch_prediction import render_batch_prediction_view
from src.views.project_info import render_project_info_view


# 1. Page Configuration
st.set_page_config(
    page_title="ICU Patient Mortality Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Custom Healthcare Styling
inject_custom_css()

# 3. Model & Data State Management
if "dataset" not in st.session_state:
    st.session_state["dataset"] = load_default_dataset()

# Attempt model loading
model_bundle = None
model_load_error = None
try:
    model_bundle = load_model_bundle()
except Exception as e:
    model_load_error = str(e)


# 4. Sidebar Branding & Navigation
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #1E293B; margin-bottom: 16px;">
            <div style="font-size: 0.70rem; font-weight: 700; letter-spacing: 0.1em; color: #14B8A6; text-transform: uppercase;">
                CLINICAL DECISION SUPPORT
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin-top: 2px;">
                ICU PATIENT MORTALITY
            </div>
            <div style="font-size: 0.75rem; color: #94A3B8;">
                Predictive Analytics & Risk Assessment
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Navigation")

    nav_options = [
        "1. Overview Dashboard",
        "2. Exploratory Data Analysis",
        "3. Data Quality",
        "4. Model Performance",
        "5. Single Patient Prediction",
        "6. Batch CSV Prediction",
        "7. Project Information",
    ]

    selected_page = st.radio(
        "Go to page:",
        options=nav_options,
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # System Status Indicators
    st.markdown("##### System Status")

    # Model Artifact Status Pill
    if model_bundle is not None:
        st.markdown(
            """
            <div style="background-color: #151F33; border: 1px solid #1E293B; border-left: 3px solid #14B8A6; padding: 8px 12px; border-radius: 6px; margin-bottom: 8px;">
                <div style="font-size: 0.70rem; color: #94A3B8; text-transform: uppercase;">Model Pipeline</div>
                <div style="font-weight: 600; font-size: 0.85rem; color: #F8FAFC;">Active (Logistic Regression)</div>
                <div style="font-size: 0.75rem; color: #14B8A6;">Threshold: 0.50 | 21 Features</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background-color: #151F33; border: 1px solid #1E293B; border-left: 3px solid #EF4444; padding: 8px 12px; border-radius: 6px; margin-bottom: 8px;">
                <div style="font-size: 0.70rem; color: #94A3B8; text-transform: uppercase;">Model Pipeline</div>
                <div style="font-weight: 600; font-size: 0.85rem; color: #FCA5A5;">Offline / Not Found</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if model_load_error:
            st.caption(f"Error: {model_load_error}")

    # Dataset Status Pill
    current_df = st.session_state.get("dataset")
    if current_df is not None:
        st.markdown(
            f"""
            <div style="background-color: #151F33; border: 1px solid #1E293B; border-left: 3px solid #7C3AED; padding: 8px 12px; border-radius: 6px; margin-bottom: 8px;">
                <div style="font-size: 0.70rem; color: #94A3B8; text-transform: uppercase;">Active Dataset</div>
                <div style="font-weight: 600; font-size: 0.85rem; color: #F8FAFC;">{len(current_df):,} Records</div>
                <div style="font-size: 0.75rem; color: #A78BFA;">{current_df.shape[1]} Columns Loaded</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background-color: #151F33; border: 1px solid #1E293B; border-left: 3px solid #F59E0B; padding: 8px 12px; border-radius: 6px; margin-bottom: 8px;">
                <div style="font-size: 0.70rem; color: #94A3B8; text-transform: uppercase;">Active Dataset</div>
                <div style="font-weight: 600; font-size: 0.85rem; color: #FCD34D;">No Dataset Loaded</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Optional dataset reload/upload in sidebar
    with st.expander("Dataset Source Settings", expanded=False):
        uploaded_sidebar = st.file_uploader(
            "Upload Alternate Dataset (CSV)",
            type=["csv"],
            key="sidebar_data_uploader",
        )
        if uploaded_sidebar is not None:
            try:
                st.session_state["dataset"] = pd.read_csv(uploaded_sidebar)
                st.success("New dataset loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

        if st.button("Reset to Default Dataset", width="stretch"):
            st.session_state["dataset"] = load_default_dataset()
            st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div style="background-color: #151F33; border: 1px solid #1E293B; border-left: 3px solid #7C3AED; padding: 10px 12px; border-radius: 8px; margin-bottom: 12px;">
            <div style="font-size: 0.68rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Project Developer</div>
            <div style="font-weight: 700; font-size: 0.90rem; color: #F8FAFC;">Sujal</div>
            <div style="font-size: 0.75rem; color: #14B8A6; font-weight: 500;">Aspiring Data Scientist</div>
            <div style="font-size: 0.70rem; color: #CBD5E1; margin-top: 3px;">Python • SQL • Machine Learning</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("MCA Capstone Portfolio | Healthcare ML Dashboard\nMIT License • MIMIC-IV Clinical Cohort")


# 5. Route to Selected Page View
active_dataset = st.session_state.get("dataset")

if selected_page.startswith("1."):
    render_overview_view(active_dataset)
elif selected_page.startswith("2."):
    render_eda_view(active_dataset)
elif selected_page.startswith("3."):
    render_data_quality_view(active_dataset)
elif selected_page.startswith("4."):
    render_model_performance_view(model_bundle)
elif selected_page.startswith("5."):
    render_single_prediction_view(model_bundle)
elif selected_page.startswith("6."):
    render_batch_prediction_view(model_bundle)
elif selected_page.startswith("7."):
    render_project_info_view()

# 6. Global Page Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="border-top: 1px solid #1E293B; padding-top: 18px; margin-top: 36px; text-align: center; color: #94A3B8; font-size: 0.82rem; line-height: 1.6;">
        <strong style="color: #F8FAFC; letter-spacing: 0.04em;">ICU PATIENT IN-HOSPITAL MORTALITY PREDICTION DASHBOARD</strong><br>
        Machine Learning System for Academic Demonstration • Class-Weighted Regularized Logistic Regression Pipeline • MIMIC-IV Inspired Cohort<br>
        <div style="margin-top: 8px; font-size: 0.82rem; color: #CBD5E1;">
            Created by <span style="color: #14B8A6; font-weight: 700;">Sujal</span> — Aspiring Data Scientist • Python • SQL • Machine Learning
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
