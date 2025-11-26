import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from prompts import intro_prompt, build_system_prompt
from tools import (
    guided_search,
    nhs_111_live_triage,
    tool_nearest_nhs_services,
    tool_onboarding,
    tool_safety,
    tools,
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


PROFILE_TAG_RE = re.compile(r"<USER_PROFILE>.*?</USER_PROFILE>", re.DOTALL)


def extract_profile(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"<USER_PROFILE>(.*?)</USER_PROFILE>", text, re.DOTALL)
    if not match:
        return None
    raw = match.group(1).strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def strip_profile_tag(text: str) -> str:
    return PROFILE_TAG_RE.sub("", text).strip()


# --- Tool Registry for Python-side Execution ---
def execute_tool(tool_name: str, arguments: Dict[str, Any]):
    if tool_name == "nearest_nhs_services":
        return tool_nearest_nhs_services(arguments)
    if tool_name == "trigger_safety_protocol":
        return tool_safety(arguments)
    if tool_name == "onboarding":
        return tool_onboarding(arguments)
    if tool_name == "guided_search":
        return guided_search(arguments)
    if tool_name == "nhs_111_live_triage":
        return nhs_111_live_triage(arguments)
    return f"[Error: Unknown tool '{tool_name}']"


class AgentSession:
    """
    Shared agent runner for CLI and Streamlit.
    Maintains conversation + onboarding/triage state across turns.
    """

    def __init__(self, client_override: Optional[OpenAI] = None):
        self.client = client_override or client
        self.conversation_history: List[Dict[str, str]] = []
        self.user_profile: Dict[str, Any] = {}
        self.system_prompt = build_system_prompt(self.user_profile)

        self.onboarding_active = False
        self.onboarding_spec: Optional[Dict[str, Any]] = None
        self.onboarding_state: Optional[Dict[str, Any]] = None

        self.triage_active = False
        self.triage_known_answers: Dict[str, Any] = {}

        self.HISTORY_WINDOW = 6
        self.MAX_OUT = 250
        self.MAX_TOOL_ROUNDS = 4
        self.MAX_RETRIES = 2

    # -----------------------------
    # SAFE MODEL CALL (TPM-aware)
    # -----------------------------
    def safe_create(self, **kwargs):
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                return self.client.responses.create(**kwargs)
            except RateLimitError:
                if "input" in kwargs and isinstance(kwargs["input"], list):
                    sys_and_pins = [x for x in kwargs["input"] if x.get("role") == "system"]
                    others = [x for x in kwargs["input"] if x.get("role") != "system"]
                    kwargs["input"] = sys_and_pins + others[-3:]
                kwargs["max_output_tokens"] = min(kwargs.get("max_output_tokens", self.MAX_OUT), 150)
                time.sleep(0.2)
        raise

    def _update_state_from_tool(self, tool_name: str, tool_result: Any):
        # onboarding activation
        if isinstance(tool_result, dict) and tool_result.get("mode") == "llm_multiturn_onboarding":
            self.onboarding_active = True
            self.onboarding_spec = tool_result
            self.onboarding_state = {
                "questions": tool_result.get("questions", []),
                "current_idx": 0,
                "answers": {},
                "expecting_answer": False,
                "reprompted": False,
            }
            self.triage_active = False

        # triage tracking
        parsed = None
        try:
            parsed = tool_result if isinstance(tool_result, dict) else json.loads(tool_result)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            if parsed.get("status") == "need_more_info":
                self.triage_active = True
                self.triage_known_answers.update(parsed.get("known_answers_update", {}))
            elif parsed.get("status") == "final":
                self.triage_active = False

        return parsed

    # -----------------------------
    # ONBOARDING HELPERS
    # -----------------------------
    def _onboarding_current_question(self) -> Optional[Dict[str, Any]]:
        if not self.onboarding_state:
            return None
        idx = self.onboarding_state.get("current_idx", 0)
        questions = self.onboarding_state.get("questions") or []
        if idx >= len(questions):
            return None
        return questions[idx]

    def _prompt_next_onboarding_question(self) -> str:
        question = self._onboarding_current_question()
        if not question:
            return self._finalize_onboarding_flow()
        self.onboarding_state["expecting_answer"] = True
        self.onboarding_state["reprompted"] = False
        return question.get("question", "").strip()

    def _normalize_onboarding_answer(self, raw: str):
        text = (raw or "").strip()
        if text == "":
            return None, True
        lowered = text.lower()
        if lowered in {"skip", "prefer not to say", "n/a", "na"}:
            return None, False
        return text, False

    def _finalize_onboarding_flow(self) -> str:
        questions = self.onboarding_state.get("questions") if self.onboarding_state else []
        answers = self.onboarding_state.get("answers") if self.onboarding_state else {}
        profile = {q.get("key"): answers.get(q.get("key")) for q in (questions or [])}
        profile_json = json.dumps(profile)
        completion_note = "Onboarding is complete. I have saved these details for future chats."
        return f"<USER_PROFILE>{profile_json}</USER_PROFILE>\n{completion_note}"

    def _handle_onboarding_answer(self, user_input: str) -> str:
        if not self.onboarding_state:
            return "I hit a snag loading the onboarding questions. Please say 'onboarding' to restart."

        question = self._onboarding_current_question()
        if not question:
            return self._finalize_onboarding_flow()

        answer, was_empty = self._normalize_onboarding_answer(user_input)
        if was_empty and not self.onboarding_state.get("reprompted", False):
            self.onboarding_state["reprompted"] = True
            self.onboarding_state["expecting_answer"] = True
            return f"I did not catch that. {question.get('question', '').strip()}"

        # store answer (None allowed for skips/empty after reprompt)
        self.onboarding_state["answers"][question.get("key")] = answer
        self.onboarding_state["current_idx"] += 1
        self.onboarding_state["expecting_answer"] = False
        self.onboarding_state["reprompted"] = False

        # ask next question or finish
        if self._onboarding_current_question():
            return self._prompt_next_onboarding_question()
        return self._finalize_onboarding_flow()

    def _process_final_reply(self, agent_reply: str) -> str:
        clean_reply = strip_profile_tag(agent_reply)
        self.conversation_history.append({"role": "assistant", "content": clean_reply})

        maybe_profile = extract_profile(agent_reply)
        if maybe_profile:
            self.user_profile = maybe_profile
            self.system_prompt = build_system_prompt(self.user_profile)

            # reset modes
            self.onboarding_active = False
            self.onboarding_spec = None
            self.onboarding_state = None
            self.triage_active = False
            self.triage_known_answers = {}

            self.conversation_history.append(
                {
                    "role": "system",
                    "content": f"Updated user profile for memory:\n{self.user_profile}",
                }
            )

        return clean_reply

    def step(self, user_input: str) -> str:
        """
        Process a single user turn and return the assistant reply (profile tags stripped).
        """
        self.conversation_history.append({"role": "user", "content": user_input})

        # -------------------------------
        # SHORT-CIRCUIT: ACTIVE ONBOARDING
        # -------------------------------
        if self.onboarding_active and self.onboarding_state:
            # If we have not yet asked the first question, do so now.
            if not self.onboarding_state.get("expecting_answer", False) and self.onboarding_state.get("current_idx", 0) == 0:
                reply = self._prompt_next_onboarding_question()
                return self._process_final_reply(reply)

            reply = self._handle_onboarding_answer(user_input)
            return self._process_final_reply(reply)

        # -------------------------------
        # PINNED CONTEXT (short!)
        # -------------------------------
        pinned = []

        if self.onboarding_active and self.onboarding_spec is not None:
            pinned.append(
                {
                    "role": "system",
                    "content": (
                        "ONBOARDING MODE IS ACTIVE. "
                        "Ask the next onboarding question verbatim, in order. "
                        "Do NOT start triage or search during onboarding."
                    ),
                }
            )

        if self.triage_active:
            pinned.append(
                {
                    "role": "system",
                    "content": (
                        "TRIAGE MODE IS ACTIVE. "
                        "Do NOT call onboarding unless user explicitly says 'onboarding'. "
                        "Ask only triage follow-up questions until triage status='final'."
                    ),
                }
            )

        # -------------------------------
        # FIRST MODEL CALL
        # -------------------------------
        resp = self.safe_create(
            model="gpt-4o-mini",
            store=True,
            input=[{"role": "system", "content": self.system_prompt}, *pinned, *self.conversation_history[-self.HISTORY_WINDOW :]],
            tools=tools,
            tool_choice="auto",
            max_output_tokens=self.MAX_OUT,
        )

        final_response = resp
        triage_called_this_turn = False
        tool_rounds = 0
        bailed_with_unresolved_calls = False

        # -------------------------------
        # BATCH TOOL HANDLING LOOP
        # -------------------------------
        while True:
            tool_rounds += 1
            if tool_rounds > self.MAX_TOOL_ROUNDS:
                bailed_with_unresolved_calls = True
                break

            tool_calls = [item for item in final_response.output if item.type == "function_call"]
            if not tool_calls:
                break

            # Guard: only one triage call per user turn
            if triage_called_this_turn and all(call.name == "nhs_111_live_triage" for call in tool_calls):
                bailed_with_unresolved_calls = True
                break

            outputs = [{"role": "system", "content": self.system_prompt}]

            for call in tool_calls:
                tool_name = call.name
                call_id = call.call_id
                raw_args = call.arguments

                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except Exception:
                        args = {}
                else:
                    args = raw_args or {}

                tool_result = execute_tool(tool_name, args)

                if tool_name == "nhs_111_live_triage":
                    triage_called_this_turn = True

                self._update_state_from_tool(tool_name, tool_result)

                tool_output_str = tool_result if isinstance(tool_result, str) else json.dumps(tool_result)

                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": tool_output_str,
                    }
                )

            final_response = self.safe_create(
                model="gpt-4o-mini",
                previous_response_id=final_response.id,
                input=outputs,
                tools=tools,
                tool_choice="auto",
                max_output_tokens=self.MAX_OUT,
            )

            # If onboarding just became active, break early and handle questions deterministically.
            if self.onboarding_active and self.onboarding_state:
                break

        # -------------------------------
        # FINAL TEXT RESPONSE
        # -------------------------------
        agent_reply = final_response.output_text or ""

        # If unresolved tool calls, force text-only reply
        if bailed_with_unresolved_calls:
            forced = self.safe_create(
                model="gpt-4o-mini",
                store=True,
                input=[
                    {"role": "system", "content": self.system_prompt},
                    *pinned,
                    *self.conversation_history[-self.HISTORY_WINDOW :],
                    {
                        "role": "system",
                        "content": (
                            "You MUST respond to the user now in plain text. "
                            "Do NOT call any tools. "
                            "If triage is incomplete, ask the next 1–3 triage follow-up questions. "
                            "If triage is complete, give routing and next steps."
                        ),
                    },
                ],
                tools=tools,
                tool_choice="none",
                max_output_tokens=200,
            )
            agent_reply = forced.output_text or ""

        # Blank-response fix
        elif agent_reply.strip() == "":
            forced = self.safe_create(
                model="gpt-4o-mini",
                previous_response_id=final_response.id,
                input=[
                    {
                        "role": "system",
                        "content": (
                            self.system_prompt
                            + "\n\nYou MUST respond to the user now in plain text. "
                            "Do NOT call any tools. "
                            "If triage is incomplete, ask the next 1–3 triage follow-up questions. "
                            "If triage is complete, give routing and next steps."
                        ),
                    }
                ],
                tools=tools,
                tool_choice="none",
                max_output_tokens=200,
            )
            agent_reply = forced.output_text or ""

        # Onboarding deterministic hand-off after tool activation
        if self.onboarding_active and self.onboarding_state:
            agent_reply = self._prompt_next_onboarding_question()

        return self._process_final_reply(agent_reply)


def run_cli():
    session = AgentSession()
    print(intro_prompt + "\n")
    print("You can continue asking questions now. Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit", "stop"]:
            print("👋 Goodbye! Stay healthy.")
            break
        reply = session.step(user_input)
        print("\nAssistant:", reply, "\n")


if __name__ == "__main__":
    run_cli()
