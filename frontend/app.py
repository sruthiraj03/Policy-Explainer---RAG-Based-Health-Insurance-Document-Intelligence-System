"""
PolicyExplainer Streamlit Application

This file implements the frontend user interface for PolicyExplainer,
an AI-powered system that helps users understand health insurance
policies in plain English.

Main Features:
1. Generate Summary
2. Q&A
"""

import os
import streamlit as st
import requests

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

CORE_SECTIONS = (
    "Plan Snapshot",
    "Cost Summary",
    "Covered Services",
    "Administrative Conditions",
    "Exclusions & Limitations",
    "Claims/Appeals & Member Rights",
)

# Ensures if a policy PDF is uploaded before using app features
# and returns the current document ID or None if not available
def ensure_doc_id():
    if not st.session_state.get("doc_id"):
        st.info("Please upload your policy document")
        return None
    return st.session_state["doc_id"]

# Retrieve and cache document text chunks for displaying evidence and citations
def chunks_map(doc_id: str) -> dict[str, dict]:
    key = f"_chunks_{doc_id}"
    if key not in st.session_state:
        try:
            r = requests.get(f"{API_BASE}/chunks/{doc_id}", timeout=30)
            r.raise_for_status()
            data = r.json()
            st.session_state[key] = {
                c["chunk_id"]: {"page_number": c["page_number"], "chunk_text": c["chunk_text"]}
                for c in data.get("chunks", [])
            }
        except Exception:
            st.session_state[key] = {}
    return st.session_state[key]


def render_evidence(citations: list, doc_id: str | None):
    if not doc_id or not citations:
        return
    cmap = chunks_map(doc_id)
    chunks_by_page: dict[int, list[tuple[str, str]]] = {}
    for cid, info in cmap.items():
        p = info.get("page_number")
        if p not in chunks_by_page:
            chunks_by_page[p] = []
        chunks_by_page[p].append((cid, info.get("chunk_text", "")))
    seen: set[str] = set()
    for cite in citations:
        page = cite.get("page")
        cid = cite.get("chunk_id")
        if cid:
            if cid in seen:
                continue
            seen.add(cid)
            info = cmap.get(cid, {})
            text = (info.get("chunk_text") or "").strip()
            pnum = info.get("page_number", page)
            with st.expander(f"Evidence (p. {pnum}) — {cid}"):
                st.caption(f"Page {pnum} · {cid}")
                st.text(text if text else "(no text)")
        elif page:
            for sub_cid, text in chunks_by_page.get(page, []):
                if sub_cid in seen:
                    continue
                seen.add(sub_cid)
                with st.expander(f"Evidence (p. {page}) — {sub_cid}"):
                    st.caption(f"Page {page} · {sub_cid}")
                    st.text((text or "").strip() or "(no text)")


def confidence_badge(level: str) -> str:
    level = (level or "").lower()
    if level == "high":
        return "High confidence"
    if level == "medium":
        return "Medium confidence"
    if level == "low":
        return "Low confidence"
    return "Confidence: " + (level or "—")

# Main App Title
st.markdown(
    "<h1 style='text-align: center; margin-bottom: 0;'>Policy Explainer</h1>",
    unsafe_allow_html=True
)
# Subtitle
st.markdown(
    "<p style='text-align: center;'>Your guide to understanding health insurance policies.</p>",
    unsafe_allow_html=True
)

# ----------------Functionality to be included later as part of Q & A suggestions----------------
# from backend.qa import get_preventive_follow_up_questions
# if st.button("Show preventive care example questions"):
#     st.session_state["show_preventive_examples"] = not st.session_state.get("show_preventive_examples", False)
# if st.session_state.get("show_preventive_examples"):
#     st.subheader("Preventive care example questions")
#     st.caption("You can ask these (or similar) questions after uploading a policy.")
#     for i, q in enumerate(get_preventive_follow_up_questions(), 1):
#         st.markdown(f"{i}. {q}")
#     st.divider()
# ---------------------------------------------------------------------------------------------

tabSummary, tabQA = st.tabs(["Generate Summary","Q&A"])

with tabSummary:
    st.subheader("Policy Summary")

    uploaded = st.file_uploader(
        "Upload Your Policy Document (PDF)",
        type=["pdf"],
        key="upload_pdf"
    )

    if uploaded is not None and st.button("Upload PDF"):
        with st.spinner("Processing document..."):
            try:
                r = requests.post(
                    f"{API_BASE}/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    timeout=120,
                )

                r.raise_for_status()
                data = r.json()
                doc_id = data.get("doc_id")

                if doc_id:
                    st.session_state["doc_id"] = doc_id

                    # Clear cached chunks when a new document is uploaded
                    for k in list(st.session_state.keys()):
                        if k.startswith("_chunks_"):
                            del st.session_state[k]

                    st.success("Document processed successfully.")

                else:
                    st.error("Document processing failed.")

            except requests.RequestException as e:
                st.error(f"Ingest failed: {e}")


    if st.session_state.get("doc_id"):
        st.caption(
            f"Current document ID: {st.session_state['doc_id'][:12]}…"
        )

with tabQA:
    st.subheader("Ask Questions About Your Policy")
    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "General Q&A",
        "Section Deep Dive",
        "Scenario Generator",
        "Evaluation"
    ])



with subtab1:
    doc_id = ensure_doc_id()
    if doc_id:
        q = st.text_input("Ask a question", key="qa_question", placeholder="e.g. What is my deductible?")
        if st.button("Ask", key="qa_ask"):
            if not (q or "").strip():
                st.warning("Enter a question.")
            else:
                try:
                    r = requests.post(f"{API_BASE}/qa/{doc_id}", json={"question": q.strip()}, timeout=60)
                    r.raise_for_status()
                    out = r.json()
                    answer = out.get("answer") or out.get("answer_display") or ""
                    st.markdown("**Answer**")
                    st.write(answer)
                    st.caption(confidence_badge(out.get("confidence")))
                    if out.get("citations"):
                        st.caption("Citations: " + ", ".join(f"p. {c.get('page', '?')}" for c in out["citations"]))
                    if out.get("disclaimer"):
                        st.caption(out["disclaimer"])
                    render_evidence(out.get("citations", []), doc_id)
                except requests.RequestException as e:
                    st.error(f"Request failed: {e}")

with subtab2:
    doc_id = ensure_doc_id()
    if doc_id:
        section = st.selectbox("Section", options=CORE_SECTIONS, key="section_select")
        if st.button("Get detailed summary", key="section_btn"):
            try:
                r = requests.post(f"{API_BASE}/summary/{doc_id}/section/{section}", timeout=90)
                r.raise_for_status()
                out = r.json()
                if out.get("present"):
                    all_cites = []
                    for b in out.get("bullets", []):
                        st.markdown(f"- {b.get('text', '')}")
                        cites = b.get("citations", [])
                        if cites:
                            st.caption("  " + ", ".join(f"p. {c.get('page', '?')}" for c in cites))
                            all_cites.extend(cites)
                    if all_cites:
                        render_evidence(all_cites, doc_id)
                else:
                    st.info(out.get("not_found_message", "Not found in this document."))
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
        q_sec = st.text_input("Or ask about this section", key="section_qa", placeholder="e.g. What does Cost Summary say about copays?")
        if st.button("Ask about section", key="section_qa_btn") and (q_sec or "").strip():
            try:
                r = requests.post(f"{API_BASE}/qa/{doc_id}", json={"question": q_sec.strip()}, timeout=60)
                r.raise_for_status()
                out = r.json()
                if out.get("answer_type") == "section_detail":
                    st.markdown("**Answer**")
                    st.write(out.get("answer", ""))
                    bullet_cites = []
                    for b in out.get("bullets", []):
                        st.markdown(f"- {b.get('text', '')}")
                        bullet_cites.extend(b.get("citations", []))
                    if bullet_cites:
                        render_evidence(bullet_cites, doc_id)
                    elif out.get("citations"):
                        render_evidence(out["citations"], doc_id)
                else:
                    st.write(out.get("answer") or out.get("answer_display", ""))
                    st.caption(confidence_badge(out.get("confidence")))
                    if out.get("disclaimer"):
                        st.caption(out["disclaimer"])
                    render_evidence(out.get("citations", []), doc_id)
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

with subtab3:
    doc_id = ensure_doc_id()
    if doc_id:
        st.info("Scenario generator is a placeholder. Use General Q&A with example scenario questions (e.g. 'What would happen if I visit the ER?').")
    else:
        st.info("Upload a policy PDF first.")

with subtab4:
    doc_id = ensure_doc_id()
    if doc_id:
        if st.button("Run evaluation", key="eval_btn"):
            try:
                r = requests.post(f"{API_BASE}/evaluate/{doc_id}", timeout=120)
                r.raise_for_status()
                report = r.json()
                st.subheader("Evaluation report")
                st.metric("Faithfulness", report.get("faithfulness_score"))
                st.metric("Completeness", report.get("completeness_score"))
                st.metric("Simplicity", report.get("simplicity_score"))
                if report.get("errors"):
                    st.warning("Errors: " + "; ".join(report["errors"]))
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
    else:
        st.info("Upload a policy PDF first.")
