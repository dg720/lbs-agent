import json
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv
import os

from prompts import build_system_prompt
from tools import (
    tool_nearest_nhs_services,
    tool_safety,
    tool_onboarding,
    tools,
)
from extract import extract_profile, strip_profile_tag

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


PROFILE_UPDATE_SYSTEM_MESSAGE = "Updated user profile for memory:\n{profile}"


def execute_tool(tool_name: str, arguments: Dict) -> str:
    if tool_name == "nearest_nhs_services":
        return tool_nearest_nhs_services(arguments)
    if tool_name == "trigger_safety_protocol":
        return tool_safety(arguments)
    if tool_name == "onboarding":
        return tool_onboarding(arguments)
    return f"[Error: Unknown tool '{tool_name}']"


class ChatSession:
    def __init__(self) -> None:
        self.conversation_history: List[Dict[str, str]] = []
        self.user_profile: Dict = {}
        self.system_prompt = build_system_prompt(self.user_profile)

    def _run_model(self):
        return client.responses.create(
            model="gpt-4o-mini",
            store=True,
            input=[{"role": "system", "content": self.system_prompt}, *self.conversation_history[-12:]],
            tools=tools,
            tool_choice="auto",
            max_output_tokens=500,
        )

    def _run_tool_chain(self, initial_response, tool_call, tool_result):
        return client.responses.create(
            model="gpt-4o-mini",
            previous_response_id=initial_response.id,
            input=[
                {"role": "system", "content": self.system_prompt},
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": tool_result if isinstance(tool_result, str) else json.dumps(tool_result),
                },
            ],
            tools=tools,
            tool_choice="auto",
            max_output_tokens=500,
        )

    def send_message(self, message: str) -> Dict:
        self.conversation_history.append({"role": "user", "content": message})

        resp = self._run_model()

        tool_call = None
        for item in resp.output:
            if item.type == "function_call":
                tool_call = item
                break

        if tool_call is not None:
            raw_args = tool_call.arguments
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            tool_result = execute_tool(tool_call.name, args)
            resp2 = self._run_tool_chain(resp, tool_call, tool_result)
            agent_reply = resp2.output_text
        else:
            agent_reply = resp.output_text

        clean_reply = strip_profile_tag(agent_reply)
        self.conversation_history.append({"role": "assistant", "content": clean_reply})

        maybe_profile = extract_profile(agent_reply)
        if maybe_profile:
            self.user_profile = maybe_profile
            self.system_prompt = build_system_prompt(self.user_profile)
            self.conversation_history.append(
                {"role": "system", "content": PROFILE_UPDATE_SYSTEM_MESSAGE.format(profile=self.user_profile)}
            )

        return {"reply": clean_reply, "user_profile": self.user_profile}
