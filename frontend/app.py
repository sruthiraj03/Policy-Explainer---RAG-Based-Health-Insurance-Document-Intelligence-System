"""
PolicyExplainer Streamlit App (Frontend Main Entry Point)
"""

import streamlit as st

from components.hero import render_hero_view
from components.dashboard import render_dashboard_view
from components.sidebar import render_sidebar
from utils.state import init_session_state
from utils.style import load_css

# App config (Must be the first Streamlit command)
st.set_page_config(page_title="Policy Explainer", layout="wide", page_icon="📄")

def main():
    # 1. Load Custom CSS
    load_css()

    # 2. Initialize Session State Variables
    init_session_state()

    # 3. Render the Sidebar (Controls & Metrics)
    render_sidebar()

    # 4. Main Page Routing Logic
    try:
        # If no document is uploaded yet, show the Hero view (upload screen)
        if st.session_state.get("doc_id") is None:
            render_hero_view()

        # Otherwise, show the Dashboard view (Summary, FAQs, Chat)
        else:
            render_dashboard_view()

    except Exception as e:
        st.error("🚨 Oops! Something went wrong while processing your request.")
        with st.expander("Technical Error Details"):
            st.write(f"Error: {type(e).__name__} - {str(e)}")

if __name__ == "__main__":
    main()