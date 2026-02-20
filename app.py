"""PolicyExplainer Streamlit UI: upload, Q&A, section deep-dive, scenario placeholder, evaluation."""

import os
import streamlit as st
import requests

from backend.qa import get_preventive_follow_up_questions

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

CORE_SECTIONS = (
    "Plan Snapshot",
    "Cost Summary",
    "Covered Services",
    "Administrative Conditions",
    "Exclusions & Limitations",
    "Claims/Appeals & Member Rights",
)


def ensure_doc_id():
    if not st.session_state.get("doc_id"):
        st.info("Upload a policy PDF first to use this tab.")
        return None
    return st.session_state["doc_id"]


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


st.set_page_config(page_title="PolicyExplainer", layout="wide")
st.title("PolicyExplainer")
st.caption("Understand your health insurance policy in plain English.")

if st.button("Show preventive care example questions"):
    st.session_state["show_preventive_examples"] = not st.session_state.get("show_preventive_examples", False)
if st.session_state.get("show_preventive_examples"):
    st.subheader("Preventive care example questions")
    st.caption("You can ask these (or similar) questions after uploading a policy.")
    for i, q in enumerate(get_preventive_follow_up_questions(), 1):
        st.markdown(f"{i}. {q}")
    st.divider()

with st.sidebar:
    st.subheader("Policy document")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], key="upload_pdf")
    if uploaded is not None and st.button("Ingest PDF"):
        with st.spinner("Ingesting…"):
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
                    for k in list(st.session_state.keys()):
                        if k.startswith("_chunks_"):
                            del st.session_state[k]
                    st.success(f"Document ingested. ID: `{doc_id[:8]}…`")
                else:
                    st.error("No doc_id in response.")
            except requests.RequestException as e:
                st.error(f"Ingest failed: {e}")
    if st.session_state.get("doc_id"):
        st.caption(f"Current doc: `{st.session_state['doc_id'][:12]}…`")

tab1, tab2, tab3, tab4 = st.tabs(["General Q&A", "Section Deep Dive", "Scenario Generator", "Evaluation"])

with tab1:
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

with tab2:
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

with tab3:
    doc_id = ensure_doc_id()
    if doc_id:
        st.info("Scenario generator is a placeholder. Use General Q&A with example scenario questions (e.g. 'What would happen if I visit the ER?').")
    else:
        st.info("Upload a policy PDF first.")

with tab4:
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
