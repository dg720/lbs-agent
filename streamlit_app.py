import os
from typing import Any, Dict

import streamlit as st
from dotenv import load_dotenv

from agent import AgentSession
from prompts import intro_prompt

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def ensure_state() -> AgentSession:
    if "agent_session" not in st.session_state:
        st.session_state.agent_session = AgentSession()
    return st.session_state.agent_session


def render_history(session: AgentSession) -> None:
    for message in session.conversation_history:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        with st.chat_message(role):
            st.markdown(message.get("content", ""))


def main() -> None:
    st.set_page_config(page_title="NHS 101 (LBS Edition)")
    st.title("NHS 101 (LBS Edition)")
    st.caption("A simple local Streamlit chat interface")

    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is not set. Add it to your environment to chat with the assistant.")
        return

    session = ensure_state()

    st.info(intro_prompt)

    render_history(session)

    user_message = st.chat_input("Ask a question or type 'onboarding' to update your details")
    if user_message:
        with st.chat_message("user"):
            st.markdown(user_message)
        reply = session.step(user_message)
        with st.chat_message("assistant"):
            st.markdown(reply)


if __name__ == "__main__":
    main()
