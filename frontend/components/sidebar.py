"""
Sidebar Component for App Controls and Metrics.
"""
import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.header("App Controls")

        if st.button("Reset Application", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        if "eval_data" in st.session_state and st.session_state["eval_data"]:
            st.divider()
            st.header("AI Performance Metrics")
            st.caption("Automatically calculated by the Evaluation Engine.")

            f_score = st.session_state["eval_data"].get("faithfulness", 0) * 100
            c_score = st.session_state["eval_data"].get("completeness", 0) * 100

            st.metric("Faithfulness (Accuracy)", f"{f_score:.1f}%")
            st.metric("Completeness (Coverage)", f"{c_score:.1f}%")