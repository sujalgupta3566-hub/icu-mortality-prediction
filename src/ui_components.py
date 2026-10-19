"""
UI Components and styling for ICU Mortality Prediction Dashboard.
Applies premium dark healthcare analytics aesthetics:
- Main background: #0B1120
- Content sections & cards: #151F33
- Upload areas & input fields: #1E293B
- Borders & highlights: Purple #7C3AED & Teal #14B8A6
- Headings: #F8FAFC | Body text: #CBD5E1
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Optional, Dict, Any


def inject_custom_css():
    """Inject polished custom dark CSS adhering to premium healthcare presentation standards."""
    st.markdown(
        """
        <style>
        /* Base typography & page resets */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"], .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            background-color: #0B1120 !important;
            color: #CBD5E1 !important;
        }

        /* Top header bar accent */
        header[data-testid="stHeader"] {
            background-color: #0B1120 !important;
            border-bottom: 2px solid #7C3AED !important;
        }

        /* Main content container background */
        .main, .block-container {
            background-color: #0B1120 !important;
            color: #CBD5E1 !important;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0F172A !important;
            border-right: 1px solid #1E293B !important;
        }

        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] p {
            color: #CBD5E1 !important;
        }

        section[data-testid="stSidebar"] .stRadio > div {
            background-color: transparent !important;
            gap: 6px;
        }

        /* Radio option items in sidebar */
        section[data-testid="stSidebar"] .stRadio label {
            background-color: #151F33 !important;
            padding: 9px 14px !important;
            border-radius: 8px !important;
            border: 1px solid #1E293B !important;
            margin-bottom: 4px;
            transition: all 0.2s ease;
            cursor: pointer;
            color: #CBD5E1 !important;
        }

        section[data-testid="stSidebar"] .stRadio label:hover {
            border-color: #7C3AED !important;
            background-color: #1E293B !important;
            color: #F8FAFC !important;
        }

        /* Headings */
        h1, h2, h3, h4, h5, h6,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #F8FAFC !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em;
        }

        p, span, label, .stMarkdown p {
            color: #CBD5E1 !important;
        }

        .stCaption, small {
            color: #94A3B8 !important;
        }

        hr {
            border: none !important;
            border-top: 1px solid #1E293B !important;
            margin: 24px 0 !important;
        }

        /* Professional Metric / KPI Card */
        .icu-kpi-card {
            background: #151F33 !important;
            border: 1px solid #1E293B !important;
            border-radius: 10px;
            padding: 18px 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            margin-bottom: 12px;
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        }

        .icu-kpi-card:hover {
            box-shadow: 0 8px 16px -2px rgba(124, 58, 237, 0.15);
            border-color: #7C3AED !important;
        }

        .icu-kpi-title {
            font-size: 0.80rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94A3B8 !important;
            margin-bottom: 6px;
        }

        .icu-kpi-value {
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.2;
            color: #F8FAFC !important;
            margin-bottom: 4px;
        }

        .icu-kpi-subtitle {
            font-size: 0.82rem;
            color: #94A3B8 !important;
        }

        /* Dark Badges */
        .icu-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.03em;
        }

        .badge-teal { background-color: rgba(20, 184, 166, 0.15); color: #14B8A6 !important; border: 1px solid rgba(20, 184, 166, 0.35); }
        .badge-purple { background-color: rgba(124, 58, 237, 0.15); color: #A78BFA !important; border: 1px solid rgba(124, 58, 237, 0.35); }
        .badge-blue { background-color: rgba(56, 189, 248, 0.15); color: #38BDF8 !important; border: 1px solid rgba(56, 189, 248, 0.35); }
        .badge-red { background-color: rgba(239, 68, 68, 0.15); color: #F87171 !important; border: 1px solid rgba(239, 68, 68, 0.35); }
        .badge-green { background-color: rgba(16, 185, 129, 0.15); color: #34D399 !important; border: 1px solid rgba(16, 185, 129, 0.35); }
        .badge-amber { background-color: rgba(245, 158, 11, 0.15); color: #FBBF24 !important; border: 1px solid rgba(245, 158, 11, 0.35); }
        .badge-slate { background-color: rgba(148, 163, 184, 0.15); color: #CBD5E1 !important; border: 1px solid rgba(148, 163, 184, 0.35); }

        /* Banner Callouts - Dark themed */
        .clinical-disclaimer-banner {
            background-color: #151F33 !important;
            border-left: 4px solid #14B8A6 !important;
            border-top: 1px solid #1E293B !important;
            border-right: 1px solid #1E293B !important;
            border-bottom: 1px solid #1E293B !important;
            padding: 14px 18px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.25);
        }

        .clinical-disclaimer-text {
            font-size: 0.88rem;
            color: #CBD5E1 !important;
            margin: 0;
            line-height: 1.5;
        }

        .privacy-disclaimer-banner {
            background-color: #151F33 !important;
            border-left: 4px solid #7C3AED !important;
            border-top: 1px solid #1E293B !important;
            border-right: 1px solid #1E293B !important;
            border-bottom: 1px solid #1E293B !important;
            padding: 14px 18px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.25);
        }

        .privacy-disclaimer-text {
            font-size: 0.88rem;
            color: #CBD5E1 !important;
            margin: 0;
            line-height: 1.5;
        }

        /* Outcome result cards */
        .prediction-card-died {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, #151F33 100%) !important;
            border: 2px solid #EF4444 !important;
            border-radius: 10px;
            padding: 24px;
            margin-top: 16px;
            margin-bottom: 16px;
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2);
        }

        .prediction-card-survived {
            background: linear-gradient(135deg, rgba(20, 184, 166, 0.15) 0%, #151F33 100%) !important;
            border: 2px solid #14B8A6 !important;
            border-radius: 10px;
            padding: 24px;
            margin-top: 16px;
            margin-bottom: 16px;
            box-shadow: 0 4px 12px rgba(20, 184, 166, 0.2);
        }

        /* File Uploader Dark Theme */
        div[data-testid="stFileUploader"] {
            background-color: #151F33 !important;
            border-radius: 10px !important;
            padding: 16px !important;
            border: 1px solid #1E293B !important;
        }

        div[data-testid="stFileUploader"] section {
            background-color: #1E293B !important;
            border: 2px dashed #7C3AED !important;
            border-radius: 8px !important;
            padding: 20px !important;
            transition: border-color 0.2s ease;
        }

        div[data-testid="stFileUploader"] section:hover {
            border-color: #14B8A6 !important;
        }

        div[data-testid="stFileUploader"] span,
        div[data-testid="stFileUploader"] small,
        div[data-testid="stFileUploader"] p {
            color: #CBD5E1 !important;
        }

        div[data-testid="stFileUploader"] button {
            background-color: #7C3AED !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
        }

        div[data-testid="stFileUploader"] button:hover {
            background-color: #6D28D9 !important;
            color: #FFFFFF !important;
        }

        /* Buttons Styling */
        /* Primary buttons - Teal */
        button[kind="primary"], .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #14B8A6 0%, #0D9488 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid #14B8A6 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 8px 18px !important;
            transition: all 0.2s ease !important;
        }

        button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%) !important;
            box-shadow: 0 4px 14px rgba(20, 184, 166, 0.4) !important;
            border-color: #14B8A6 !important;
            color: #FFFFFF !important;
        }

        /* Secondary / Standard buttons - Purple accent */
        button[kind="secondary"], .stButton > button, div.stDownloadButton > button {
            background-color: #1E293B !important;
            color: #FFFFFF !important;
            border: 1px solid #7C3AED !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 8px 18px !important;
            transition: all 0.2s ease !important;
        }

        button[kind="secondary"]:hover, .stButton > button:hover, div.stDownloadButton > button:hover {
            background-color: #7C3AED !important;
            color: #FFFFFF !important;
            border-color: #7C3AED !important;
            box-shadow: 0 4px 14px rgba(124, 58, 237, 0.4) !important;
        }

        /* Form & Form Wrapper styling */
        form,
        form[data-testid="stForm"],
        [data-testid="stForm"],
        div[data-testid="stForm"],
        .stForm,
        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            background-color: #151F33 !important;
            background: #151F33 !important;
            border-color: #1E293B !important;
        }

        form[data-testid="stForm"],
        [data-testid="stForm"],
        .stForm {
            border: 1px solid #1E293B !important;
            border-radius: 12px !important;
            padding: 24px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35) !important;
        }

        /* Number input buttons (- and +) */
        div[data-testid="stNumberInput"] button {
            background-color: #1E293B !important;
            color: #F8FAFC !important;
            border-color: #334155 !important;
        }

        div[data-testid="stNumberInput"] button:hover {
            background-color: #7C3AED !important;
            color: #FFFFFF !important;
        }

        /* Form labels and text */
        form label, [data-testid="stForm"] label,
        form .stMarkdown h5, [data-testid="stForm"] .stMarkdown h5,
        form .stMarkdown strong, [data-testid="stForm"] .stMarkdown strong {
            color: #F8FAFC !important;
        }

        form .stMarkdown p, [data-testid="stForm"] .stMarkdown p,
        form .stCaption, [data-testid="stForm"] .stCaption {
            color: #CBD5E1 !important;
        }

        /* Input fields, selectboxes, number inputs */
        div[data-baseweb="input"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] {
            background-color: #1E293B !important;
            border-color: #334155 !important;
            color: #F8FAFC !important;
            border-radius: 8px !important;
        }

        div[data-baseweb="input"]:focus-within,
        div[data-baseweb="select"] > div:focus-within {
            border-color: #14B8A6 !important;
            box-shadow: 0 0 0 1px #14B8A6 !important;
        }

        input, select, textarea {
            color: #F8FAFC !important;
            background-color: transparent !important;
        }

        /* Popover dropdown menus */
        div[data-baseweb="popover"],
        div[data-baseweb="menu"] {
            background-color: #1E293B !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
        }

        div[data-baseweb="menu"] li {
            color: #CBD5E1 !important;
            background-color: #1E293B !important;
        }

        div[data-baseweb="menu"] li:hover {
            background-color: #2A3A52 !important;
            color: #F8FAFC !important;
        }

        /* Native Alerts Dark Themed */
        div[data-testid="stAlert"] {
            background-color: #151F33 !important;
            color: #CBD5E1 !important;
            border-radius: 8px !important;
            border: 1px solid #1E293B !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
        }

        div[data-testid="stAlert"] p {
            color: #CBD5E1 !important;
        }

        div[data-testid="stAlert"] svg {
            color: #14B8A6 !important;
        }

        /* Tab styling */
        button[data-baseweb="tab"] {
            font-weight: 500 !important;
            color: #94A3B8 !important;
            background-color: transparent !important;
        }

        button[data-baseweb="tab"]:hover {
            color: #CBD5E1 !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #14B8A6 !important;
            border-bottom-color: #14B8A6 !important;
            font-weight: 600 !important;
        }

        div[data-baseweb="tab-list"] {
            background-color: transparent !important;
            border-bottom: 1px solid #1E293B !important;
        }

        /* Expanders */
        div[data-testid="stExpander"] {
            background-color: #151F33 !important;
            border: 1px solid #1E293B !important;
            border-radius: 8px !important;
        }

        div[data-testid="stExpander"] summary {
            color: #F8FAFC !important;
            font-weight: 600 !important;
        }

        /* Dataframes */
        div[data-testid="stDataFrame"] {
            background-color: #151F33 !important;
            border: 1px solid #1E293B !important;
            border-radius: 8px !important;
            overflow: hidden !important;
        }

        /* Code snippets */
        code, pre {
            background-color: #1E293B !important;
            color: #14B8A6 !important;
            border: 1px solid #334155 !important;
            border-radius: 6px !important;
        }

        /* Plotly chart container styling */
        div.js-plotly-plot {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #1E293B;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str, badge_text: Optional[str] = None):
    """Render a consistent professional title and subtitle block in dark theme."""
    badge_html = f'<span class="icu-badge badge-teal" style="margin-left: 10px;">{badge_text}</span>' if badge_text else ""
    st.markdown(
        f"""
        <div style="margin-bottom: 22px;">
            <h1 style="font-size: 1.85rem; font-weight: 700; color: #F8FAFC; margin-bottom: 6px; display: flex; align-items: center;">
                {title} {badge_html}
            </h1>
            <p style="font-size: 0.95rem; color: #CBD5E1; margin: 0; line-height: 1.5;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_clinical_disclaimer():
    """Display standard regulatory and ethical disclaimer for educational ML prototype."""
    st.markdown(
        """
        <div class="clinical-disclaimer-banner">
            <p class="clinical-disclaimer-text">
                <strong style="color: #14B8A6;">Clinical Research Notice:</strong>
                This application is an educational data science prototype built for academic demonstration.
                Predictions and risk estimates are generated from a retrospective machine learning model and
                <strong>must not</strong> be used as medical advice, diagnostic tools, or to guide real-world patient care decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_privacy_notice():
    """Display dark-themed privacy and HIPAA data protection notice."""
    st.markdown(
        """
        <div class="privacy-disclaimer-banner">
            <p class="privacy-disclaimer-text">
                <strong style="color: #A78BFA;">Privacy & Data Protection Notice:</strong>
                Uploaded patient batches are processed strictly in-memory within your current browser session.
                No patient records, identifiers, or clinical data are saved to disk, logged, or transmitted to any external API or third-party service.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    title: str,
    value: str,
    subtitle: str = "",
    badge: Optional[str] = None,
    badge_tone: str = "teal",
    border_color: Optional[str] = None,
):
    """Render dark HTML KPI card with customizable badge and styling."""
    border_style = f"border-left: 4px solid {border_color};" if border_color else ""
    badge_html = f'<span class="icu-badge badge-{badge_tone}">{badge}</span>' if badge else ""

    html = f"""
    <div class="icu-kpi-card" style="{border_style}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="icu-kpi-title">{title}</div>
            {badge_html}
        </div>
        <div class="icu-kpi-value">{value}</div>
        <div class="icu-kpi-subtitle">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def get_plotly_theme() -> Dict[str, Any]:
    """Provide dark theme Plotly charts matching premium navy, purple, and teal aesthetic."""
    return {
        "plot_bgcolor": "#151F33",
        "paper_bgcolor": "#151F33",
        "font": {"family": "Inter, sans-serif", "color": "#CBD5E1", "size": 12},
        "margin": {"l": 45, "r": 25, "t": 45, "b": 45},
        "xaxis": {
            "gridcolor": "#1E293B",
            "zerolinecolor": "#334155",
            "linecolor": "#334155",
            "tickfont": {"color": "#94A3B8"},
            "title_font": {"color": "#F8FAFC"},
        },
        "yaxis": {
            "gridcolor": "#1E293B",
            "zerolinecolor": "#334155",
            "linecolor": "#334155",
            "tickfont": {"color": "#94A3B8"},
            "title_font": {"color": "#F8FAFC"},
        },
        "colorway": ["#14B8A6", "#7C3AED", "#38BDF8", "#F59E0B", "#EF4444", "#10B981"],
    }


def apply_plotly_theme(fig):
    """Apply unified dark healthcare palette and typography to Plotly figure."""
    theme = get_plotly_theme()
    fig.update_layout(
        plot_bgcolor=theme["plot_bgcolor"],
        paper_bgcolor=theme["paper_bgcolor"],
        font=theme["font"],
        title_font={"color": "#F8FAFC", "size": 14, "family": "Inter, sans-serif"},
        legend=dict(
            bgcolor="rgba(21, 31, 51, 0.8)",
            bordercolor="#1E293B",
            font=dict(color="#CBD5E1"),
        ),
        margin=theme["margin"],
        xaxis=theme["xaxis"],
        yaxis=theme["yaxis"],
    )
    return fig
