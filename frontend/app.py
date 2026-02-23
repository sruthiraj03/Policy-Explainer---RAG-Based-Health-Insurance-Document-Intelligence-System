import os
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

# --- UI Header ---
st.set_page_config(page_title="Policy Explainer", layout="wide", page_icon="📄")
st.title("Policy Explainer")
st.markdown("Upload your medical insurance **Summary of Benefits** to get an instant plain-English overview.")

# --- Sidebar for Evaluation & Controls ---
with st.sidebar:
    st.header("App Controls")
    if st.button("Reset Application", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    # Display evaluation metrics if they exist in the session
    if "eval_data" in st.session_state:
        st.divider()
        st.header("AI Performance Metrics")
        st.caption("Automatically calculated by the Evaluation Engine.")

        f_score = st.session_state["eval_data"].get("faithfulness", 0) * 100
        c_score = st.session_state["eval_data"].get("completeness", 0) * 100

        st.metric("Faithfulness (Accuracy)", f"{f_score:.1f}%")
        st.metric("Completeness (Coverage)", f"{c_score:.1f}%")

# --- Step 1: File Upload ---
uploaded_file = st.file_uploader("Drop your policy PDF here", type=["pdf"])

if uploaded_file:
    # Logic to process only if it's a new file
    if "current_file" not in st.session_state or st.session_state["current_file"] != uploaded_file.name:
        with st.spinner("Processing document, generating summary, and evaluating..."):
            try:
                # 1. Ingest
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                ingest_r = requests.post(f"{API_BASE}/ingest", files=files, timeout=120)
                ingest_r.raise_for_status()
                doc_id = ingest_r.json().get("doc_id")

                # 2. Full Summary
                summary_r = requests.post(f"{API_BASE}/summary/{doc_id}", timeout=120)
                summary_r.raise_for_status()

                # 3. Evaluate (Trigger the judge module)
                eval_r = requests.post(f"{API_BASE}/evaluate/{doc_id}", timeout=120)
                eval_r.raise_for_status()

                # Store in Session
                st.session_state["doc_id"] = doc_id
                st.session_state["summary"] = summary_r.json()
                st.session_state["eval_data"] = eval_r.json()
                st.session_state["current_file"] = uploaded_file.name
                st.rerun()  # Refresh to populate the sidebar metrics

            except Exception as e:
                st.error(f"Error: {e}")

# --- Step 2: Display Summary (Automatic) ---
if "summary" in st.session_state:
    st.divider()
    st.header("Policy Summary")

    summary_data = st.session_state["summary"]

    # Create columns for the summary layout
    for section in summary_data.get("sections", []):
        if section.get("present"):
            with st.expander(f"{section['section_name']}", expanded=True):
                # Display Confidence Badge
                conf = section.get('confidence', 'low').upper()

                # Color code confidence for better UI
                color = "green" if conf == "HIGH" else "orange" if conf == "MEDIUM" else "red"
                st.markdown(f"**Confidence:** :{color}[{conf}]")

                for bullet in section.get("bullets", []):
                    # Show bullet text with page citations
                    cites = ", ".join([f"p. {c['page']}" for c in bullet.get("citations", [])])
                    st.markdown(f"• {bullet['text']} *({cites})*")

    st.divider()

    # --- Step 3: Interactive Q&A ---
    st.header("Ask a Question")
    st.info("Try a scenario: *'What happens if I visit the ER?'* or a fact: *'What is my deductible?'*")

    user_q = st.text_input("Type your question here:", placeholder="e.g., What happens if I go to the Emergency Room?")

    if user_q:
        with st.spinner("Searching policy..."):
            try:
                doc_id = st.session_state["doc_id"]
                qa_r = requests.post(f"{API_BASE}/qa/{doc_id}", json={"question": user_q}, timeout=60)
                qa_r.raise_for_status()
                out = qa_r.json()

                st.markdown("### Answer")

                # --- NEW: Scenario Renderer ---
                if out.get("answer_type") == "scenario":
                    st.success(f"**{out.get('header', 'Hypothetical Scenario')}**")
                    steps = out.get("steps", [])

                    if not steps:
                        st.write(out.get("not_found_message", "Could not calculate scenario from this document."))

                    for step in steps:
                        cites = ", ".join([f"p. {c['page']}" for c in step.get("citations", [])])
                        st.markdown(f"**Step {step.get('step_number')}:** {step.get('text')} *({cites})*")

                # --- Standard RAG & Deep Dive Renderer ---
                else:
                    st.write(out.get("answer") or out.get("answer_display", "No answer provided."))
                    if out.get("citations"):
                        cites_str = ", ".join([f"p. {c['page']}" for c in out.get("citations", [])])
                        st.caption(f"**Sources:** {cites_str}")

                # Show Disclaimer
                if out.get("disclaimer"):
                    st.warning(out["disclaimer"])

            except Exception as e:
                st.error(f"QA Error: {e}")