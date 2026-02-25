import os
import streamlit as st
import requests
from dotenv import load_dotenv
from fpdf import FPDF  # Requires: pip install fpdf2

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

# Chatbot Session States
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [{
        "role": "assistant",
        "content": "Hello! I am your virtual AI policy explainer assistant. How can I help you understand your coverage today?"
    }]
if "chat_open" not in st.session_state:
    st.session_state["chat_open"] = True  # start open so it feels like texting
if "chat_input_text" not in st.session_state:
    st.session_state["chat_input_text"] = ""
if "pending_question" not in st.session_state:
    st.session_state["pending_question"] = None


# --- Helper: PDF Generation (Fixed Unicode & Blank Document Error) ---
def generate_policy_pdf():
    def clean_text(text):
        replacements = {
            "\u2022": "-", "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "Policy Explainer Summary", ln=True, align="C")
    pdf.set_font("Arial", "", 10)

    summary_data = st.session_state.get("summary", {})
    for section in summary_data.get("sections", []):
        if section.get("present"):
            pdf.set_fill_color(240, 240, 240)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, clean_text(f" {section['section_name']}"), ln=True, fill=True)
            pdf.ln(2)

            pdf.set_font("Arial", "", 11)
            for bullet in section.get("bullets", []):
                pdf.multi_cell(0, 7, clean_text(f"- {bullet['text']}"))
                pdf.ln(1)
            pdf.ln(5)

    pdf_out = pdf.output(dest="S")
    return pdf_out.encode("latin-1") if isinstance(pdf_out, str) else bytes(pdf_out)


# --- CSS Styling ---
st.markdown("""
<style>
  .stAppHeader { background-color: white; z-index: 99; }
  .block-container { padding-top: 1rem; }

  /* --- HERO STYLES --- */
  .hero-title { font-size: 60px !important; font-weight: 800; color: #0F172A; margin-bottom: 0px; line-height: 1.1; }
  .hero-subtitle { font-size: 28px !important; font-weight: 500; color: #334155; margin-bottom: 20px; margin-top: 10px; }
  .hero-text { font-size: 18px !important; line-height: 1.6; color: #475569; margin-bottom: 30px; }
  .dash-title { font-size: 32px !important; font-weight: 800; color: #0F172A; margin: 0; line-height: 1.2; }

  /* Hide Drag & Drop Text */
  [data-testid="stFileUploaderDropzone"] div div::before { content: "Select your insurance document "; visibility: visible; }
  [data-testid="stFileUploaderDropzone"] div div span { display: none; }

  /* ---------------------------------------------------------
     RIGHT COLUMN "PHONE" CHAT — STICKY (RELIABLE IN STREAMLIT)
     We place a marker element, then make the NEXT container sticky.
     --------------------------------------------------------- */

  /* Hide the marker's own element-container so it doesn't create spacing */
  div[data-testid="element-container"]:has(.chat-sticky-marker) {
    display: none !important;
  }

  /* Make the container immediately AFTER the marker sticky */
  div[data-testid="element-container"]:has(.chat-sticky-marker) + div {
    position: sticky !important;
    top: 110px !important;
    align-self: flex-start !important;
  }

  /* Phone-like panel */
  .chat-panel {
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 12px;
    background: white;
    box-shadow: 0px 10px 26px rgba(0,0,0,0.10);
  }

  .chat-header {
    font-size: 15px;
    font-weight: 650;
    color: #0F172A;
    margin-bottom: 6px;
  }

  .user-bubble-container { display: flex; justify-content: flex-end; width: 100%; margin-bottom: 8px; }
  .ai-bubble-container { display: flex; justify-content: flex-start; width: 100%; margin-bottom: 8px; }

  .chat-bubble {
    padding: 8px 10px;
    border-radius: 12px;
    font-size: 13px;
    max-width: 88%;
    line-height: 1.35;
    word-wrap: break-word;
  }
  .user-bubble { background-color: #3B82F6; color: white; border-bottom-right-radius: 3px; }
  .ai-bubble { background-color: #F8FAFC; color: #1E293B; border: 1px solid #E2E8F0; border-bottom-left-radius: 3px; }

  /* Compact toggle button */
  .chat-toggle button {
    width: 42px !important;
    height: 42px !important;
    border-radius: 21px !important;
    font-size: 18px !important;
    padding: 0 !important;
  }

  /* OPTIONAL: tighten summary expanders slightly */
  div[data-testid="stExpander"] summary {
    padding-top: 6px !important;
    padding-bottom: 6px !important;
  }

  /* Main App Buttons */
  div.stButton > button, div.stDownloadButton > button { width: 100%; border-radius: 8px; height: 3em; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.header("App Controls")
    if st.button("Reset Application", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    if "eval_data" in st.session_state:
        st.divider()
        st.header("AI Performance Metrics")
        st.caption("Automatically calculated by the Evaluation Engine.")
        f_score = st.session_state["eval_data"].get("faithfulness", 0) * 100
        c_score = st.session_state["eval_data"].get("completeness", 0) * 100
        st.metric("Faithfulness (Accuracy)", f"{f_score:.1f}%")
        st.metric("Completeness (Coverage)", f"{c_score:.1f}%")


# --- VIEW 1: HERO ---
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
                key=f"uploader_{st.session_state['uploader_key']}",
                label_visibility="collapsed",
            )

            if uploaded_file and "processing" not in st.session_state:
                st.session_state["processing"] = True
                status_help = st.empty()
                try:
                    with st.spinner("Processing..."):
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

                        status_help.markdown("📑 **Step 2/3:** Generating summary...")
                        summary_r = requests.post(f"{API_BASE}/summary/{doc_id}", timeout=120)

                        status_help.markdown("⚖️ **Step 3/3:** Verifying accuracy...")
                        eval_r = requests.post(f"{API_BASE}/evaluate/{doc_id}", timeout=120)

                        st.session_state["doc_id"] = doc_id
                        st.session_state["summary"] = summary_r.json()
                        st.session_state["eval_data"] = eval_r.json() if eval_r.status_code == 200 else {}
                        del st.session_state["processing"]
                        st.rerun()

                except Exception as e:
                    st.error(f"🚨 **System Error:** {str(e)}")
                    if "processing" in st.session_state:
                        del st.session_state["processing"]

    with col_img:
        image_path = os.path.join(os.path.dirname(__file__), "header_image.jpg")
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)


# --- Content Renderers ---
def render_summary_content():
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
    st.subheader("Frequently Asked Questions")
    st.info("ℹ️ To enable auto-generated FAQs, you will need to add an endpoint to the backend.")
    with st.expander("Does this plan cover pre-existing conditions?"):
        st.write("Most ACA-compliant plans cover pre-existing conditions. Please check the 'Exclusions' section.")
    with st.expander("How do I file a claim?"):
        st.write("Look for the 'Claims & Appeals' section in the Summary tab.")


# --- Chatbot Logic ---
def handle_chat_input():
    user_q = st.session_state.chat_input_text
    if not user_q.strip():
        return

    nonsense_keywords = ["asdf", "qwerty", "ghjk", "yewh", "test"]
    is_nonsense = len(user_q.strip()) < 4 or any(kw in user_q.lower() for kw in nonsense_keywords)

    st.session_state.chat_history.append({"role": "user", "content": user_q})
    st.session_state.chat_input_text = ""

    if is_nonsense:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "I'm sorry, I don't recognize that as a policy-related question. Could you please provide more detail?"
        })
    else:
        st.session_state.pending_question = user_q


def render_chat_panel():
    st.markdown('<div class="chat-toggle">', unsafe_allow_html=True)
    toggle_icon = "✖" if st.session_state.chat_open else "💬"
    if st.button(toggle_icon, key="toggle_chat"):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if not st.session_state.chat_open:
        return

    st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
    st.markdown('<div class="chat-header">Policy Assistant</div>', unsafe_allow_html=True)

    # smaller, phone-like history area
    hist_container = st.container(height=360)
    with hist_container:
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(
                    f'<div class="user-bubble-container"><div class="chat-bubble user-bubble">{chat["content"]}</div></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="ai-bubble-container"><div class="chat-bubble ai-bubble">{chat["content"]}</div></div>',
                    unsafe_allow_html=True
                )

        if st.session_state.get("pending_question"):
            st.markdown(
                '<div class="ai-bubble-container"><div class="chat-bubble ai-bubble">💭 <i>Thinking...</i></div></div>',
                unsafe_allow_html=True
            )

    st.text_input(
        "Type your question...",
        key="chat_input_text",
        on_change=handle_chat_input,
        label_visibility="collapsed"
    )

    if st.session_state.get("pending_question"):
        q = st.session_state.pending_question
        st.session_state.pending_question = None

        try:
            qa_r = requests.post(f"{API_BASE}/qa/{st.session_state['doc_id']}", json={"question": q})
            qa_r.raise_for_status()
            out = qa_r.json()

            if out.get("answer_type") == "scenario":
                answer = f"**{out.get('header', 'Scenario Breakdown')}**\n\n"
                for step in out.get("steps", []):
                    cites = ", ".join([f"p. {c['page']}" for c in step.get("citations", [])])
                    safe_text = step.get("text", "")
                    answer += f"**{step.get('step_number')}.** {safe_text} *({cites})*\n"
            else:
                raw_answer = out.get("answer", "").strip()
                if raw_answer.lower() == q.strip().lower():
                    answer = "I don’t see that information in your policy—could you rephrase?"
                else:
                    answer = raw_answer
                    fallback_phrases = [
                        "i'm just a system", "i am an ai", "i'm an ai", "i am a virtual",
                        "help you with your questions", "how can i help", "here to assist",
                        "here to help", "need assistance", "let me know", "feel free to ask",
                        "i'm sorry", "i don't see any", "couldn't find", "hello", "hi there", "how are you"
                    ]
                    is_conversational = any(p in raw_answer.lower() for p in fallback_phrases) and len(raw_answer) < 250
                    if out.get("citations") and not is_conversational:
                        cites = ", ".join([f"p. {c['page']}" for c in out.get("citations", [])])
                        answer += f"\n\n*Sources: {cites}*"

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        except Exception:
            st.session_state.chat_history.append({"role": "assistant", "content": "Assistant temporarily unavailable."})

        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# --- VIEW 2: DASHBOARD ---
def render_dashboard_view():
    st.markdown('<h1 class="dash-title">Policy Explainer</h1>', unsafe_allow_html=True)
    st.divider()

    # Split screen: main content (left) + chat (right)
    col_main, col_chat = st.columns([0.74, 0.26], gap="large")

    with col_main:
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
            pdf_bytes = generate_policy_pdf()
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
        if st.session_state["active_tab"] == "Summary":
            render_summary_content()
        elif st.session_state["active_tab"] == "FAQs":
            render_faq_content()

    with col_chat:
        # marker + next container becomes sticky (via CSS)
        st.markdown('<div class="chat-sticky-marker"></div>', unsafe_allow_html=True)
        with st.container():
            render_chat_panel()


# --- MAIN EXECUTION ---
try:
    if st.session_state["doc_id"] is None:
        render_hero_view()
    else:
        render_dashboard_view()

except Exception as e:
    st.error("🚨 Oops! Something went wrong while processing your request.")
    with st.expander("Technical Error Details"):
        st.write(f"Error: {type(e).__name__} - {str(e)}")