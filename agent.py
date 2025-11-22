from openai import OpenAI
import re
import json
from urllib.parse import quote_plus
from dotenv import load_dotenv
import os

from prompts import intro_prompt, build_system_prompt
from tools import tool_nearest_nhs_services, tool_safety, tool_onboarding, tools

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def extract_profile(text):
    m = re.search(r"<USER_PROFILE>(.*?)</USER_PROFILE>", text, re.DOTALL)
    if not m:
        return None

    raw = m.group(1)
    if not raw:
        return None  # avoid IndexError

    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def strip_profile_tag(text: str) -> str:
    return PROFILE_TAG_RE.sub("", text).strip()


PROFILE_TAG_RE = re.compile(r"<USER_PROFILE>.*?</USER_PROFILE>", re.DOTALL)


# --- Tool Registry for Python-side Execution ---
def execute_tool(tool_name, arguments):
    if tool_name == "nearest_nhs_services":
        return tool_nearest_nhs_services(arguments)
    elif tool_name == "trigger_safety_protocol":
        return tool_safety(arguments)
    elif tool_name == "onboarding":
        return tool_onboarding(
            arguments
        )  # LLM-driven onboarding tool (defined elsewhere)
    else:
        return f"[Error: Unknown tool '{tool_name}']"


def agent():
    import json  # ensure json is available

    # --- SESSION MEMORY ---
    conversation_history = []  # stores {"role": ..., "content": ...}

    # Start with no/empty profile; LLM will call onboarding tool if needed
    user_profile = {}

    print(intro_prompt + "\n")

    system_prompt = build_system_prompt(user_profile)

    # --- CONTINUOUS CHAT LOOP ---
    print("You can continue asking questions now. Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit", "stop"]:
            print("👋 Goodbye! Stay healthy.")
            break

        conversation_history.append({"role": "user", "content": user_input})

        # 1) First model call (store=True so we can chain tool outputs)
        resp = client.responses.create(
            model="gpt-4o-mini",
            store=True,
            input=[
                {"role": "system", "content": system_prompt},
                *conversation_history[-12:],
            ],
            tools=tools,
            tool_choice="auto",
            max_output_tokens=500,
        )

        # Find the first tool call, if any
        tool_call = None
        for item in resp.output:
            if item.type == "function_call":
                tool_call = item
                break

        # --- CASE 1: Tool call chosen by model ---
        if tool_call is not None:
            tool_name = tool_call.name
            call_id = tool_call.call_id
            raw_args = tool_call.arguments

            # parse args
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = raw_args or {}

            # Execute tool in Python
            tool_result = execute_tool(tool_name, args)

            # 2) Send tool output back CHAINED to prior response
            resp2 = client.responses.create(
                model="gpt-4o-mini",
                previous_response_id=resp.id,
                input=[
                    {"role": "system", "content": system_prompt},
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": tool_result
                        if isinstance(tool_result, str)
                        else json.dumps(tool_result),
                    },
                ],
                tools=tools,
                tool_choice="auto",
                max_output_tokens=500,
            )

            agent_reply = resp2.output_text

        # --- CASE 2: Normal assistant response ---
        else:
            agent_reply = resp.output_text

        clean_reply = strip_profile_tag(agent_reply)

        print("\nAssistant:", clean_reply, "\n")
        conversation_history.append({"role": "assistant", "content": clean_reply})

        # --- Detect completed onboarding ---
        maybe_profile = extract_profile(agent_reply)  # defined elsewhere
        if maybe_profile:
            user_profile = maybe_profile
            system_prompt = build_system_prompt(user_profile)
            conversation_history.append(
                {
                    "role": "system",
                    "content": f"Updated user profile for memory:\n{user_profile}",
                }
            )


agent()
