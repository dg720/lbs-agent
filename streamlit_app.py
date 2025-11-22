import os
import json
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from agent import extract_profile, strip_profile_tag, execute_tool
from prompts import build_system_prompt, intro_prompt
from tools import tools

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Limit how many turns we send back to the model
MAX_CONTEXT_TURNS = 12


def ensure_state() -> None:
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history: List[Dict[str, str]] = []
    if "user_profile" not in st.session_state:
        st.session_state.user_profile: Dict[str, Any] = {}
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = build_system_prompt(st.session_state.user_profile)


def render_history() -> None:
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_assistant_reply(raw_reply: str) -> None:
    clean_reply = strip_profile_tag(raw_reply)
    st.session_state.conversation_history.append({"role": "assistant", "content": clean_reply})

    maybe_profile = extract_profile(raw_reply)
    if maybe_profile:
        st.session_state.user_profile = maybe_profile
        st.session_state.system_prompt = build_system_prompt(maybe_profile)
        st.session_state.conversation_history.append(
            {
                "role": "system",
                "content": f"Updated user profile for memory:\n{maybe_profile}",
            }
        )

    with st.chat_message("assistant"):
        st.markdown(clean_reply)


def run_model(user_message: str) -> None:
    st.session_state.conversation_history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": st.session_state.system_prompt}]
    messages.extend(st.session_state.conversation_history[-MAX_CONTEXT_TURNS:])

    resp = client.responses.create(
        model="gpt-4o-mini",
        store=True,
        input=messages,
        tools=tools,
        tool_choice="auto",
        max_output_tokens=500,
    )

    tool_call = next((item for item in resp.output if item.type == "function_call"), None)

    if tool_call is not None:
        tool_name = tool_call.name
        call_id = tool_call.call_id
        raw_args = tool_call.arguments

        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        else:
            args = raw_args or {}

        tool_result = execute_tool(tool_name, args)

        resp2 = client.responses.create(
            model="gpt-4o-mini",
            previous_response_id=resp.id,
            input=[
                {"role": "system", "content": st.session_state.system_prompt},
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": tool_result if isinstance(tool_result, str) else json.dumps(tool_result),
                },
            ],
            tools=tools,
            tool_choice="auto",
            max_output_tokens=500,
        )

        handle_assistant_reply(resp2.output_text)
    else:
        handle_assistant_reply(resp.output_text)


def main() -> None:
    st.set_page_config(page_title="NHS 101 (LBS Edition)")
    st.title("NHS 101 (LBS Edition)")
    st.caption("A simple local Streamlit chat interface")

    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is not set. Add it to your environment to chat with the assistant.")
        return

    ensure_state()

    st.info(intro_prompt)

    render_history()

    user_message = st.chat_input("Ask a question or type 'onboarding' to update your details")
    if user_message:
        with st.chat_message("user"):
            st.markdown(user_message)
        run_model(user_message)


if __name__ == "__main__":
    main()
