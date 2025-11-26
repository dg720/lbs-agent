"""Streamlit UI for the Evi NHS/LBS companion."""

import os

import streamlit as st
from dotenv import load_dotenv

from agent import AgentSession
from prompts import intro_prompt

load_dotenv()

# Prefer Streamlit secrets for cloud deploy, fall back to env/local .env
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def ensure_state() -> AgentSession:
    """Create or return the cached AgentSession and default prompt suggestions."""
    if "agent_session" not in st.session_state:
        st.session_state.agent_session = AgentSession()
        st.session_state.prompt_suggestions = [
            "Build my onboarding profile",
            "Start triage process",
            "How do I register with a GP?",
            "What NHS services am I eligible for?",
        ]
    return st.session_state.agent_session


def render_history(session: AgentSession) -> None:
    """Replay chat history into Streamlit chat containers."""
    for message in session.conversation_history:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        with st.chat_message(role):
            st.markdown(message.get("content", ""))


def main() -> None:
    """Render the Streamlit page and chat workflow."""
    st.set_page_config(
        page_title="Evi - LBS Healthcare companion", page_icon=":hospital:"
    )

    base_css = """
    <style>
    html, body, .main, [class*="stApp"] {
        background: #0e1117 !important;
        color: #f5f5f5 !important;
    }
    .main .block-container {
        max-width: 1100px;
        width: 75vw;
        padding-top: 12px;
        padding-bottom: 32px;
    }
    .evi-card {
        background: #161b23;
        border: 1px solid #28303d;
        border-radius: 12px;
        padding: 14px 16px;
        color: #ffffff;
    }
    .evi-hero {
        background: linear-gradient(135deg, #223043, #30201f);
        border: 1px solid #344152;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 12px;
        color: #ffffff;
    }
    .evi-hero * {
        color: #ffffff !important;
    }
    .evi-hero h1 {
        margin: 4px 0 6px 0;
        color: #ffffff;
        font-size: 2rem;
        line-height: 1.1;
        white-space: nowrap;
    }
    .evi-hero p {
        margin: 0;
        color: #ffffff;
        font-size: 0.98rem;
        line-height: 1.25;
        white-space: nowrap;
    }
    .stButton > button {
        width: 100%;
        min-height: 38px;
        border-radius: 8px;
        background: linear-gradient(135deg, #223043, #30201f);
        border: 1px solid #344152;
        color: #f7f9fb;
        padding: 7px 10px;
        font-size: 0.9rem;
        font-weight: 600;
        white-space: nowrap;
        box-shadow: 0 2px 6px rgba(0,0,0,0.18);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #29384d, #3a2624);
        border-color: #3f4e63;
        color: white;
    }
    .stChatMessage {
        background: #141820;
        border: 1px solid #222834;
        border-radius: 10px;
        padding: 10px 12px;
        color: #ffffff;
    }
    .stChatMessage p, .stChatMessage span {
        color: #f5f5f5 !important;
    }
    /* Dark chat input bar */
    [data-testid="stChatInput"], .stChatInputContainer, .stChatFloatingInputContainer {
        background: #0e1117 !important;
        border: none !important;
        color: #f5f5f5 !important;
    }
    [data-testid="stChatInput"] > div {
        background: #141820 !important;
        border: 1px solid #222834 !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    }
    [data-testid="stChatInput"] textarea,
    .stChatInputContainer textarea,
    .stChatFloatingInputContainer textarea {
        color: #f5f5f5 !important;
        background: #141820 !important;
    }
    .stTextInput input {
        color: #f5f5f5 !important;
    }
    </style>
    """
    st.markdown(base_css, unsafe_allow_html=True)

    if not OPENAI_API_KEY:
        st.error(
            "OPENAI_API_KEY is not set. Add it to your environment to chat with the assistant."
        )
        return

    session = ensure_state()

    st.markdown(
        """
        <div class="evi-hero">
            <div style="font-weight:700; color:#2f2f2f;">Evi - your LBS healthcare companion</div>
            <h1>Navigate NHS services with confidence</h1>
            <p>Fast, friendly guidance for GP registration, triage, eligibility, and next steps across UK care pathways.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Build marker: Streamlit app refreshed (ping #5).")

    st.markdown(
        f'<div class="evi-card"><strong>How I can help</strong><br>{intro_prompt}</div>',
        unsafe_allow_html=True,
    )

    suggestions = st.session_state.get("prompt_suggestions", [])
    if not session.conversation_history and suggestions:
        rows = [suggestions[i : i + 2] for i in range(0, len(suggestions), 2)]
        for row in rows:
            cols = st.columns(2, gap="medium")
            for idx, suggestion in enumerate(row):
                if cols[idx].button(suggestion, use_container_width=True):
                    st.session_state.queued_message = suggestion
                    st.session_state.prompt_suggestions = suggestions

    render_history(session)

    queued_message = st.session_state.pop("queued_message", None)
    chat_value = st.chat_input(
        "Ask a question or type 'onboarding' to update your details"
    )
    user_message = queued_message or chat_value

    if user_message:
        with st.chat_message("user"):
            st.markdown(user_message)
        with st.spinner("Thinking..."):
            reply = session.step(user_message)
            st.session_state.prompt_suggestions = session.prompt_suggestions
        with st.chat_message("assistant"):
            st.markdown(reply)


if __name__ == "__main__":
    main()
