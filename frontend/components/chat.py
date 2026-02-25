"""
Chat Component for the interactive Q&A assistant.
"""
import os
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables for the API URL
load_dotenv()
API_BASE = os.getenv("API_BASE", "[http://127.0.0.1:8000](http://127.0.0.1:8000)")

def render_chat_panel():
    """Renders a sleek, native Streamlit chat interface."""

    # 1. Clean Header
    st.subheader("💬 Policy Assistant")
    st.caption("Ask questions about your coverage, limits, and costs. Answers are grounded in your document.")

    # 2. Message History Area (Using native container with a border)
    msg_container = st.container(height=480, border=True)

    with msg_container:
        for msg in st.session_state.chat_history:
            # st.chat_message provides native avatars and formatting!
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Show a typing indicator if a question is pending
        if st.session_state.get("pending_question"):
            with st.chat_message("assistant"):
                st.markdown("💭 *Thinking...*")

    # 3. Native Chat Input (Automatically handles "Enter" and pins to the bottom nicely)
    if prompt := st.chat_input("Ask a question about your policy..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # Set the pending question flag
        st.session_state.pending_question = prompt
        # Rerun to update the UI instantly (shows the user message & "Thinking...")
        st.rerun()

    # 4. Handle the API call AFTER the UI updates
    if st.session_state.get("pending_question"):
        q = st.session_state.pending_question
        st.session_state.pending_question = None

        try:
            qa_r = requests.post(
                f"{API_BASE}/qa/{st.session_state['doc_id']}",
                json={"question": q},
                timeout=120
            )
            qa_r.raise_for_status()
            out = qa_r.json()

            # Scenario format support
            if out.get("answer_type") == "scenario":
                answer = f"**{out.get('header', 'Scenario Breakdown')}**\n\n"
                for step in out.get("steps", []):
                    cites = ", ".join([f"p. {c['page']}" for c in step.get("citations", [])])
                    safe_text = (step.get("text", "") or "").replace("$", "\\$")
                    answer += f"**{step.get('step_number')}.** {safe_text} *({cites})*\n"
            else:
                raw_answer = (out.get("answer", "") or "").strip()
                if raw_answer.lower() == q.strip().lower():
                    answer = "I don’t see that information in your policy—could you rephrase?"
                else:
                    answer = raw_answer

                    is_valid_answer = out.get("answer_type") in ["answerable", "section_detail"]
                    if out.get("citations") and is_valid_answer:
                        cites = ", ".join([f"p. {c['page']}" for c in out.get("citations", [])])
                        answer += f"\n\n*Sources: {cites}*"

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        except requests.exceptions.ConnectionError:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "🚨 **Connection Error:** I couldn't reach the backend server. Please ensure the FastAPI server is running."
            })
        except requests.exceptions.Timeout:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "⏳ **Timeout:** The server took too long to respond. It might be processing a heavy load."
            })
        except requests.exceptions.HTTPError as e:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"⚠️ **Server Error:** The backend returned an error ({e.response.status_code}). Please try again."
            })
        except Exception as e:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"❌ **Unexpected Error:** Something went wrong ({str(e)})."
            })

        # Final rerun to display the API's answer
        st.rerun()