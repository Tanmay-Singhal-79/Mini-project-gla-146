import streamlit as st

def apply_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * { font-family: 'Inter', sans-serif; }
        
        /* Premium Dark Theme */
        .stApp { background-color: #0f1115; }
        h1, h2, h3, h4, h5, h6 { color: #f8fafc !important; font-weight: 600 !important; }
        
        [data-testid="stSidebar"] {
            background-color: #1a1d24;
            border-right: 1px solid #2d3748;
        }
        
        /* Cards styling */
        .course-card {
            background-color: #1e212b;
            border: 1px solid #2d3748;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.2s;
        }
        .course-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 20px -8px rgba(0, 0, 0, 0.6);
            border-color: #6366f1;
        }
        .course-title {
            font-size: 1.35rem; font-weight: 600; color: #818cf8;
            margin-bottom: 8px; letter-spacing: -0.025em;
        }
        
        /* Badges */
        .tag {
            display: inline-block; padding: 4px 10px; border-radius: 6px;
            background-color: #334155; color: #e2e8f0; font-size: 0.75rem;
            font-weight: 500; margin-right: 8px; margin-bottom: 8px;
        }
        .sim-score {
            display: inline-block; padding: 4px 10px; border-radius: 6px;
            background-color: rgba(52, 211, 153, 0.1); color: #34d399; 
            font-size: 0.75rem; font-weight: 600; margin-right: 8px; border: 1px solid #34d399;
        }
        
        /* Explainability Trace Text */
        .explain-text {
            font-size: 0.85rem; color: #94a3b8; font-style: italic; margin-top: 12px;
            border-left: 2px solid #6366f1; padding-left: 10px;
        }
        
        /* Metric blocks inner text */
        [data-testid="stMetricValue"] {
            color: #34d399 !important;
        }
    </style>
    """, unsafe_allow_html=True)
