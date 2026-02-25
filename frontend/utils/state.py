"""
Session State Manager for the Streamlit Frontend.
"""
import streamlit as st

def init_session_state():
    """Initializes all required session state variables if they don't exist."""

    # Core app states
    if "doc_id" not in st.session_state:
        st.session_state["doc_id"] = None
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "Summary"
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    # Chat states
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [{
            "role": "assistant",
            "content": "Hi, I am your Policy Explainer AI Chatbot here to assist you. Feel free to "
                       "ask me any questions about your policy."
        }]
    if "chat_open" not in st.session_state:
        st.session_state["chat_open"] = True
    if "chat_input_text" not in st.session_state:
        st.session_state["chat_input_text"] = ""
    if "pending_question" not in st.session_state:
        st.session_state["pending_question"] = None