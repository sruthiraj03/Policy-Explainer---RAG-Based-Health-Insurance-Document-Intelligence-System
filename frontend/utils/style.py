"""
CSS Injection for the Streamlit Frontend.
"""
import streamlit as st

def load_css():
    """Injects custom CSS to override default Streamlit styling."""
    st.markdown("""
    <style>
      .stAppHeader { background-color: white; z-index: 99; }
      .block-container { padding-top: 1rem; }

      /* HERO */
      .hero-title { font-size: 60px !important; font-weight: 800; color: #0F172A; margin-bottom: 0px; line-height: 1.1; }
      .hero-subtitle { font-size: 28px !important; font-weight: 500; color: #334155; margin-bottom: 20px; margin-top: 10px; }
      .hero-text { font-size: 18px !important; line-height: 1.6; color: #475569; margin-bottom: 30px; }
      .dash-title { font-size: 32px !important; font-weight: 800; color: #0F172A; margin: 0; line-height: 1.2; }

      /* Hide Drag & Drop Text */
      [data-testid="stFileUploaderDropzone"] div div::before { content: "Select your insurance document "; visibility: visible; }
      [data-testid="stFileUploaderDropzone"] div div span { display: none; }

      /* Sticky chat column */
      div[data-testid="element-container"]:has(.chat-sticky-marker) { display: none !important; }
      div[data-testid="element-container"]:has(.chat-sticky-marker) + div {
        position: sticky !important;
        top: 110px !important;
        align-self: flex-start !important;
      }

      /* Main buttons */
      div.stButton > button, div.stDownloadButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
      }
    </style>
    """, unsafe_allow_html=True)