"""
Hero View Component for Landing Page and File Upload.
"""
import os
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables for the API URL
load_dotenv()
API_BASE = os.getenv("API_BASE", "[http://127.0.0.1:8000](http://127.0.0.1:8000)")

def render_hero_view():
    col_text, col_img = st.columns([1, 1], gap="large")

    with col_text:
        st.markdown(
            '<h1 class="hero-title">Policy Explainer</h1><div class="hero-subtitle">Understand your health insurance.</div>',
            unsafe_allow_html=True
        )

        if "upload_error" in st.session_state:
            st.error(f"🚫 **Validation Failed:** {st.session_state['upload_error']}")
            if st.button("❌ Reset & Try Again", key="exit_reset"):
                st.session_state.clear()
                st.rerun()
        else:
            st.markdown(
                '<div class="hero-text">Decode your health insurance instantly. Upload your <b>Summary of Benefits</b> (PDF) to reveal hidden costs.</div>',
                unsafe_allow_html=True
            )

            uploaded_file = st.file_uploader(
                "Drop your policy PDF here",
                type=["pdf"],
                key=f"uploader_{st.session_state.get('uploader_key', 0)}",
                label_visibility="collapsed",
            )

            if uploaded_file and "processing" not in st.session_state:
                st.session_state["processing"] = True
                status_help = st.empty()

                try:
                    with st.spinner("Processing..."):
                        # Step 1: Ingest
                        status_help.markdown("🔍 **Step 1/3:** Validating policy...")
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        ingest_r = requests.post(f"{API_BASE}/ingest", files=files, timeout=120)

                        if ingest_r.status_code == 400:
                            status_help.empty()
                            error_detail = ingest_r.json().get("detail", "Invalid document.")
                            st.session_state["upload_error"] = error_detail.replace("Validation Failed:", "").strip()
                            del st.session_state["processing"]
                            st.rerun()

                        ingest_r.raise_for_status()
                        doc_id = ingest_r.json().get("doc_id")

                        # Step 2: Summarize
                        status_help.markdown("📑 **Step 2/3:** Generating summary...")
                        summary_r = requests.post(f"{API_BASE}/summary/{doc_id}", timeout=120)
                        summary_r.raise_for_status()

                        # Step 3: Evaluate
                        status_help.markdown("⚖️ **Step 3/3:** Verifying accuracy...")
                        eval_r = requests.post(f"{API_BASE}/evaluate/{doc_id}", timeout=120)

                        # Update Session State
                        st.session_state["doc_id"] = doc_id
                        st.session_state["summary"] = summary_r.json()
                        st.session_state["eval_data"] = eval_r.json() if eval_r.status_code == 200 else {}

                        del st.session_state["processing"]
                        st.rerun()

                except requests.exceptions.ConnectionError:
                    st.error("🚨 **Connection Error:** Cannot connect to the backend API. Please make sure your FastAPI server is running (`uvicorn backend.main:app --reload`).")
                    if "processing" in st.session_state:
                        del st.session_state["processing"]
                except requests.exceptions.Timeout:
                    st.error("⏳ **Timeout:** The document processing took too long and timed out. Try a smaller PDF.")
                    if "processing" in st.session_state:
                        del st.session_state["processing"]
                except requests.exceptions.HTTPError as e:
                    st.error(f"⚠️ **Server Error:** The backend failed with status code {e.response.status_code}. Check the FastAPI console for details.")
                    if "processing" in st.session_state:
                        del st.session_state["processing"]
                except Exception as e:
                    st.error(f"🚨 **System Error:** {str(e)}")
                    if "processing" in st.session_state:
                        del st.session_state["processing"]

    with col_img:
        # Resolves the path relative to this file's location inside the components folder
        base_dir = os.path.dirname(os.path.dirname(__file__))
        image_path = os.path.join(base_dir, "assets", "header_image.jpg")

        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            fallback_path = os.path.join(base_dir, "header_image.jpg")
            if os.path.exists(fallback_path):
                st.image(fallback_path, use_container_width=True)