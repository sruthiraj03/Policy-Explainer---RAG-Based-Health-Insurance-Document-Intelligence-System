"""
Dashboard Component for the main post-upload interface.
"""
import os
import requests
import streamlit as st
from dotenv import load_dotenv

from utils.pdf_generator import generate_policy_pdf
from components.chat import render_chat_panel

# Load environment variables for the API URL
load_dotenv()
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")


def render_summary_content():
    """Renders the expandable sections of the policy summary."""
    st.subheader("Policy Highlights")
    summary_data = st.session_state.get("summary", {})

    for section in summary_data.get("sections", []):
        if section.get("present"):
            with st.expander(f"{section['section_name']}", expanded=True):
                conf = section.get("confidence", "low").upper()
                st.markdown(
                    f"**Confidence:** :{'green' if conf == 'HIGH' else 'orange' if conf == 'MEDIUM' else 'red'}[{conf}]"
                )

                for bullet in section.get("bullets", []):
                    cites = ", ".join([f"p. {c['page']}" for c in bullet.get("citations", [])])
                    safe_text = bullet["text"].replace("$", "\\$")
                    st.markdown(f"• {safe_text} *({cites})*")


def render_faq_content():
    """Renders the FAQ section, dynamically generating them on first load."""
    st.subheader("Frequently Asked Questions")

    # If FAQs aren't in memory yet, generate them!
    if "faqs" not in st.session_state:
        with st.spinner("🧠 Generating FAQs tailored to your policy..."):
            try:
                doc_id = st.session_state["doc_id"]
                r = requests.get(f"{API_BASE}/qa/{doc_id}/faqs", timeout=120)
                r.raise_for_status()
                st.session_state["faqs"] = r.json().get("faqs", [])
            except Exception as e:
                st.error("⚠️ Could not generate FAQs at this time. Please try again later.")
                st.session_state["faqs"] = []

    faqs = st.session_state.get("faqs", [])

    if not faqs:
        st.info("No FAQs available for this document.")
    else:
        for faq in faqs:
            with st.expander(faq.get("question", "Question")):
                st.write(faq.get("answer", ""))


def render_dashboard_view():
    """Orchestrates the main layout with columns, navigation, and content."""
    st.markdown('<h1 class="dash-title">Policy Explainer</h1>', unsafe_allow_html=True)
    st.divider()

    col_main, col_chat = st.columns([0.74, 0.26], gap="large")

    with col_main:
        # Navigation Bar
        col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1, 1, 1.5, 1])

        with col_nav1:
            if st.button("📑 Summary", type="primary" if st.session_state["active_tab"] == "Summary" else "secondary"):
                st.session_state["active_tab"] = "Summary"
                st.rerun()

        with col_nav2:
            if st.button("🔄 New Policy", key="dash_new"):
                st.session_state.clear()
                st.rerun()

        with col_nav3:
            # Generate the PDF using our new utility
            summary_data = st.session_state.get("summary", {})
            pdf_bytes = generate_policy_pdf(summary_data)

            st.download_button(
                label="📥 Save Insurance Summary",
                data=pdf_bytes,
                file_name=f"Policy Summary-{st.session_state.get('doc_id', 'Analysis')}.pdf",
                mime="application/pdf",
            )

        with col_nav4:
            if st.button("❓ FAQs", type="primary" if st.session_state["active_tab"] == "FAQs" else "secondary"):
                st.session_state["active_tab"] = "FAQs"
                st.rerun()

        st.write("")

        # Display the active tab's content
        if st.session_state["active_tab"] == "Summary":
            render_summary_content()
        else:
            render_faq_content()

    with col_chat:
        st.markdown('<div class="chat-sticky-marker"></div>', unsafe_allow_html=True)
        with st.container():
            render_chat_panel()