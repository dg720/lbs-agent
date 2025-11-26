import os

import streamlit as st
from dotenv import load_dotenv

from agent import AgentSession
from prompts import intro_prompt

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def ensure_state() -> AgentSession:
    if "agent_session" not in st.session_state:
        st.session_state.agent_session = AgentSession()
        st.session_state.prompt_suggestions = [
            "Start onboarding",
            "Start triage",
            "How do I register with a GP?",
            "What am I eligible for?",
        ]
    return st.session_state.agent_session


def render_history(session: AgentSession) -> None:
    for message in session.conversation_history:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = message.get("content", "")
        with st.chat_message(role):
            st.markdown(content)


def main() -> None:
    st.set_page_config(
        page_title="Evi - your LBS Healthcare companion",
        page_icon=":hospital:",
        layout="wide",
    )

    brand_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap');
    :root {
        --evi-primary: #009aa8;
        --evi-ink: #3a3a3a;
        --evi-ink-strong: #292929;
        --evi-accent: #ec4f45;
        --evi-secondary: #216d74;
        --evi-soft: #fdf5f1;
        --evi-contrast: #ffffff;
        --evi-muted: #a4a4a4;
        --evi-border: #f6aea9;
    }
    .block-container {
        width: 72vw;
        max-width: 1180px;
        min-width: 840px;
        margin: 0 auto;
    }
    html, body, [class*="stApp"] {
        font-family: 'Manrope', sans-serif;
        background: radial-gradient(circle at 20% 20%, rgba(0,154,168,0.05), transparent 25%),
                    radial-gradient(circle at 80% 0%, rgba(236,79,69,0.06), transparent 28%),
                    var(--evi-soft);
        color: var(--evi-ink);
    }
    h1, h2, h3, h4 {
        color: var(--evi-ink-strong);
        letter-spacing: -0.01em;
    }
    .evi-hero {
        background: linear-gradient(120deg, rgba(0,154,168,0.16), rgba(236,79,69,0.12));
        border: 1px solid rgba(0,154,168,0.2);
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.06);
        margin-bottom: 10px;
    }
    .stButton > button {
        border-radius: 12px;
        border: 1px solid transparent;
        background: var(--evi-primary);
        color: white;
        padding: 0.7rem 1rem;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(0,154,168,0.25);
    }
    .stButton > button:hover {
        background: var(--evi-secondary);
        border-color: var(--evi-secondary);
    }
    .evi-secondary-btn > button {
        background: var(--evi-contrast);
        color: var(--evi-ink-strong);
        border: 1px solid #dce9eb;
        box-shadow: none;
    }
    .stChatMessage {
        border-radius: 14px;
        padding: 12px 14px;
        background: #ffffff;
        border: 1px solid #eef3f3;
        color: var(--evi-ink-strong);
    }
    .stChatMessage p, .stChatMessage span, .stMarkdown p {
        color: var(--evi-ink-strong) !important;
    }
    [data-testid="stChatMessageContent"] { color: var(--evi-ink-strong) !important; }
    /* Avatar overrides to avoid black backgrounds */
    [data-testid="stChatMessageAvatar"] {
        background: transparent !important;
        box-shadow: none !important;
    }
    [data-testid="stChatMessageAvatar"] > div {
        background: #009aa8 !important; /* Evida teal for user */
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12) !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
    }
    [data-testid="stChatMessage"][data-testid*="assistant"] [data-testid="stChatMessageAvatar"] > div {
        background: #f6aea9 !important; /* Soft coral for Evi */
        color: #292929 !important;
    }
    .evi-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: var(--evi-contrast);
        padding: 8px 12px;
        border-radius: 999px;
        border: 1px solid #dce9eb;
        font-weight: 600;
    }
    .evi-caption {
        color: var(--evi-muted);
        font-size: 0.95rem;
        margin-top: -8px;
    }
    .stChatInputContainer, .stChatFloatingInputContainer {
        background: var(--evi-soft) !important;
        border-top: 1px solid #dce9eb !important;
        padding: 12px 0 6px 0;
    }
    .stChatInputContainer textarea, .stChatFloatingInputContainer textarea {
        color: var(--evi-ink-strong) !important;
        background: #ffffff !important;
        caret-color: var(--evi-ink-strong) !important;
    }
    [data-testid="stChatInput"] {
        background: var(--evi-soft) !important;
        border-top: 1px solid #dce9eb !important;
        padding: 0 8px 8px 8px !important;
    }
    [data-testid="stChatInput"] > div {
        max-width: 72vw;
        width: 72vw;
        min-width: 840px;
        margin: 0 auto;
        background: #ffffff;
        border: 1px solid #dce9eb;
        border-radius: 12px;
        padding: 6px 10px;
        display: flex;
        align-items: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    [data-testid="stChatInput"] textarea {
        background: #ffffff !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        color: var(--evi-ink-strong) !important;
        padding: 0.25rem 0 !important;
        min-height: 1.6rem !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #5a5a5a !important;
    }
    [data-testid="stChatInput"] button {
        background: var(--evi-primary) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        margin-left: 10px;
        padding: 0.35rem 0.65rem !important;
        align-self: stretch;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }
    [data-testid="stBottom"] {
        background: var(--evi-soft) !important;
    }
    [data-testid="stBottom"] > div {
        background: var(--evi-soft) !important;
    }
    </style>
    """
    st.markdown(brand_css, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="evi-hero">
            <div class="evi-pill">Evi - your LBS healthcare companion</div>
            <h1 style="margin-bottom:6px;">Navigate NHS services with confidence</h1>
            <p style="max-width:640px; font-size:1.02rem; color:#2e2e2e;">
            Fast, friendly guidance for GP registration, triage, eligibility, and next steps across UK care pathways.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is not set. Add it to your environment to chat with the assistant.")
        return

    session = ensure_state()

    st.markdown(
        f"""
        <div style="margin-top:14px; margin-bottom:10px; padding:14px 16px; border-radius:14px; background:#ffffff; border:1px solid #eef3f3;">
            <strong>How I can help</strong><br>
            {intro_prompt}
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_history(session)

    queued_message = st.session_state.pop("queued_message", None)
    chat_value = st.chat_input("Ask a question or type 'onboarding' to update your details")
    user_message = queued_message or chat_value

    if user_message:
        with st.chat_message("user"):
            st.markdown(user_message)
        with st.spinner("Thinking..."):
            reply = session.step(user_message)
            st.session_state.prompt_suggestions = session.prompt_suggestions
        with st.chat_message("assistant"):
            st.markdown(reply)

    suggestions = st.session_state.get("prompt_suggestions", [])
    if not session.conversation_history and suggestions:
        st.subheader("Jump in with a quick ask")
        cols = st.columns(len(suggestions), gap="medium")
        for idx, suggestion in enumerate(suggestions):
            with cols[idx]:
                if st.button(suggestion, key=f"suggestion-{idx}", use_container_width=True):
                    st.session_state.queued_message = suggestion
                    st.session_state.prompt_suggestions = suggestions


if __name__ == "__main__":
    main()
