import os
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

# --- Page Config ---
st.set_page_config(page_title="Policy Explainer", layout="wide", page_icon="📄")

# --- Session State Management ---
if "doc_id" not in st.session_state:
    st.session_state["doc_id"] = None
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Summary"
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# --- CSS Styling ---
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; }

        /* --- HERO STYLES (Landing Page) --- */
        .hero-title { 
            font-size: 60px !important; 
            font-weight: 800; 
            color: #0F172A; 
            margin-bottom: 0px; 
            line-height: 1.1; 
        }
        .hero-subtitle { 
            font-size: 28px !important; 
            font-weight: 500; 
            color: #334155; 
            margin-bottom: 20px; 
            margin-top: 10px;
        }
        .hero-text { 
            font-size: 18px !important; 
            line-height: 1.6; 
            color: #475569; 
            margin-bottom: 30px;
        }

        /* --- DASHBOARD STYLES (Minimized Header) --- */
        .dash-title {
            font-size: 32px !important; 
            font-weight: 800; 
            color: #0F172A; 
            margin: 0;
            line-height: 1.2;
        }
        .dash-subtitle {
            font-size: 16px !important; 
            font-weight: 500; 
            color: #64748B; 
            margin-bottom: 10px;
        }

        /* --- Button Styling --- */
        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            height: 3em;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar (App Controls & Metrics) ---
with st.sidebar:
    st.header("App Controls")
    # This button allows the user to fully reset and go back to the Hero view
    if st.button("Reset Application", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    # Display evaluation metrics if they exist
    if "eval_data" in st.session_state:
        st.divider()
        st.header("AI Performance Metrics")
        st.caption("Automatically calculated by the Evaluation Engine.")

        f_score = st.session_state["eval_data"].get("faithfulness", 0) * 100
        c_score = st.session_state["eval_data"].get("completeness", 0) * 100

        st.metric("Faithfulness (Accuracy)", f"{f_score:.1f}%")
        st.metric("Completeness (Coverage)", f"{c_score:.1f}%")


# --- VIEW 1: HERO (Landing Page) ---
def render_hero_view():
    col_text, col_img = st.columns([1, 1], gap="large")

    with col_text:
        # 1. Big Header (Keep your existing markdown here)
        st.markdown("""
            <h1 class="hero-title">Policy Explainer</h1>
            <div class="hero-subtitle">Understand your health insurance.</div>
            <div class="hero-text">
                Decode your health insurance instantly. Upload your <b>Summary of Benefits</b> (PDF) 
                to reveal hidden costs, understand your coverage in plain English, 
                and ask questions about real-life scenarios.
            </div>
        """, unsafe_allow_html=True)

        # 3. File Uploader - MOVED Assignment outside the IF block
        uploaded_file = None
        if st.session_state["doc_id"] is None:
            uploaded_file = st.file_uploader(
                "Drop your policy PDF here",
                type=["pdf"],
                key=f"uploader_{st.session_state['uploader_key']}"
            )

        # Guard: Only process if a file is present and we aren't already processing
        if uploaded_file and "processing" not in st.session_state:
            st.session_state["processing"] = True
            with st.spinner("Processing document... this may take a moment."):
                try:
                    # 1. Ingest the file
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    ingest_r = requests.post(f"{API_BASE}/ingest", files=files, timeout=120)
                    ingest_r.raise_for_status()
                    doc_id = ingest_r.json().get("doc_id")

                    # 2. Generate and CAPTURE the Summary
                    summary_r = requests.post(f"{API_BASE}/summary/{doc_id}", timeout=120)
                    summary_r.raise_for_status()
                    summary_json = summary_r.json()  # Store the actual summary data

                    # 3. Get Evaluation
                    eval_r = requests.post(f"{API_BASE}/evaluate/{doc_id}", timeout=120)
                    eval_r.raise_for_status()

                    # 4. Update State with all received data
                    st.session_state["doc_id"] = doc_id
                    st.session_state["summary"] = summary_json  # This fixes the blank UI!
                    st.session_state["eval_data"] = eval_r.json()
                    st.session_state["active_tab"] = "Summary"
                    st.session_state["uploader_key"] += 1

                    # Cleanup and refresh
                    del st.session_state["processing"]
                    st.rerun()

                except Exception as e:
                    st.error(f"Error processing file: {e}")
                    if "processing" in st.session_state:
                        del st.session_state["processing"]

        # --- Enhanced CSS for Styled Buttons ---
        st.markdown("""
            <style>
                /* Base styling for all three buttons */
                div.stButton > button {
                    border-radius: 12px;
                    padding: 20px;
                    font-size: 18px;
                    font-weight: 600;
                    transition: all 0.3s ease;
                    border: 2px solid transparent;
                }

                /* Summary Button: Primary Filled Style */
                div[data-testid="stColumn"]:nth-of-type(1) button {
                    background-color: #3b66f5 !important;
                    color: white !important;
                    box-shadow: 0 4px 6px -1px rgba(59, 102, 245, 0.2);
                }

                /* Q&A Button: Outlined Style */
                div[data-testid="stColumn"]:nth-of-type(2) button {
                    background-color: transparent !important;
                    border: 2px solid #3b66f5 !important;
                    color: #3b66f5 !important;
                }

                /* FAQs Button: Soft Light Style */
                div[data-testid="stColumn"]:nth-of-type(3) button {
                    background-color: #eff6ff !important;
                    color: #3b66f5 !important;
                }

                /* Ensure disabled buttons don't look too washed out */
                div.stButton > button:disabled {
                    cursor: default;
                    opacity: 0.8;
                }
            </style>
        """, unsafe_allow_html=True)

        # 2. Feature Preview Buttons (Visual only in Hero mode)
        c1, c2, c3 = st.columns(3, gap="medium")
        # Check if a document is loaded to enable buttons
        is_disabled = st.session_state["doc_id"] is None

        with c1:
            if st.button("📑 Summary", disabled=is_disabled, key="nav_sum"):
                st.session_state["active_tab"] = "Summary"
        with c2:
            if st.button("💬 Q&A", disabled=is_disabled, key="nav_qa"):
                st.session_state["active_tab"] = "Q&A"
        with c3:
            if st.button("❓ FAQs", disabled=is_disabled, key="nav_faq"):
                st.session_state["active_tab"] = "FAQs"

    with col_img:
        image_path = os.path.join(os.path.dirname(__file__), "header_image.jpg")
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)


# --- VIEW 2: DASHBOARD (Minimized Header) ---
def render_dashboard_view():
    # 1. Minimized Header (Same text, smaller size, kept at top)
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown("""
            <h1 class="dash-title">Policy Explainer</h1>
            <div class="dash-subtitle">Analysis for: <b>Current Document</b></div>
        """, unsafe_allow_html=True)
    with c2:
        # Option to upload a new file easily
        if st.button("🔄 Upload New File"):
            st.session_state.clear()
            st.rerun()

    st.divider()

    # 2. Navigation Tabs (The "Pressed Button Regions")
    col_nav1, col_nav2, col_nav3, _ = st.columns([1, 1, 1, 3])

    with col_nav1:
        if st.button("📑 Summary", type="primary" if st.session_state["active_tab"] == "Summary" else "secondary"):
            st.session_state["active_tab"] = "Summary"
            st.rerun()
    with col_nav2:
        if st.button("💬 Q&A", type="primary" if st.session_state["active_tab"] == "Q&A" else "secondary"):
            st.session_state["active_tab"] = "Q&A"
            st.rerun()
    with col_nav3:
        if st.button("❓ FAQs", type="primary" if st.session_state["active_tab"] == "FAQs" else "secondary"):
            st.session_state["active_tab"] = "FAQs"
            st.rerun()

    st.write("")  # Spacer

    # 3. Content Area (Switch based on active tab)
    if st.session_state["active_tab"] == "Summary":
        render_summary_content()
    elif st.session_state["active_tab"] == "Q&A":
        render_qa_content()
    elif st.session_state["active_tab"] == "FAQs":
        render_faq_content()


# --- Content Renderers ---

def render_summary_content():
    st.subheader("Policy Highlights")
    summary_data = st.session_state.get("summary", {})

    # Render sections
    for section in summary_data.get("sections", []):
        if section.get("present"):
            with st.expander(f"{section['section_name']}", expanded=True):
                # Confidence Badge
                conf = section.get('confidence', 'low').upper()
                color = "green" if conf == "HIGH" else "orange" if conf == "MEDIUM" else "red"
                st.markdown(f"**Confidence:** :{color}[{conf}]")

                # Bullets
                for bullet in section.get("bullets", []):
                    cites = ", ".join([f"p. {c['page']}" for c in bullet.get("citations", [])])
                    st.markdown(f"• {bullet['text']} *({cites})*")


def render_qa_content():
    st.subheader("Ask a Question")
    st.info("Try asking about specific scenarios like: *'What happens if I visit an out-of-network cardiologist?'*")

    user_q = st.text_input("Type your question:", placeholder="e.g., What is my deductible?")

    if user_q:
        with st.spinner("Analyzing policy..."):
            try:
                qa_r = requests.post(f"{API_BASE}/qa/{st.session_state['doc_id']}", json={"question": user_q})
                qa_r.raise_for_status()
                out = qa_r.json()

                st.markdown("### Answer")

                if out.get("answer_type") == "scenario":
                    st.success(f"**{out.get('header', 'Scenario Breakdown')}**")
                    for step in out.get("steps", []):
                        cites = ", ".join([f"p. {c['page']}" for c in step.get("citations", [])])
                        st.markdown(f"**{step.get('step_number')}.** {step.get('text')} *({cites})*")
                else:
                    st.write(out.get("answer"))
                    if out.get("citations"):
                        cites = ", ".join([f"p. {c['page']}" for c in out.get("citations", [])])
                        st.caption(f"**Sources:** {cites}")

                    if out.get("disclaimer"):
                        st.warning(out["disclaimer"])
            except Exception as e:
                st.error(f"Error: {e}")


def render_faq_content():
    st.subheader("Frequently Asked Questions")
    st.info("ℹ️ To enable auto-generated FAQs, you will need to add an endpoint to the backend.")

    with st.expander("Does this plan cover pre-existing conditions?"):
        st.write(
            "Most ACA-compliant plans cover pre-existing conditions. Please check the 'Exclusions' section in the Summary tab for specific details on waiting periods.")

    with st.expander("How do I file a claim?"):
        st.write(
            "Look for the 'Claims & Appeals' section in the Summary tab. It typically outlines the address and time limits for filing.")


# --- MAIN EXECUTION WITH GLOBAL ERROR HANDLING ---
try:
    # Always show the main header/image area
    render_hero_view()

    # Only show the results area if a document is active
    if st.session_state["doc_id"] is not None:
        if st.session_state["active_tab"] == "Summary":
            render_summary_content()
        elif st.session_state["active_tab"] == "Q&A":
            render_qa_content()
        elif st.session_state["active_tab"] == "FAQs":
            render_faq_content()

except Exception as e:
    # Captures unexpected errors and shows a clean message instead of a crash
    st.error("🚨 Oops! Something went wrong while processing your request.")
    st.info("Please try clicking 'Reset Application' in the sidebar if the issue persists.")

    with st.expander("Technical Error Details"):
        st.write(f"Error: {type(e).__name__} - {str(e)}")