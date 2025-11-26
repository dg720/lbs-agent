"""NHS/LBS chat agent session management, tools, and deterministic flows."""

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
    """Extract JSON profile dict from assistant text, if present."""
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
    """Remove profile tag wrapper so only user-facing content remains."""
    return PROFILE_TAG_RE.sub("", text).strip()


# --- Tool Registry for Python-side Execution ---
def execute_tool(tool_name: str, arguments: Dict[str, Any]):
    """Dispatch supported tool calls by name."""
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
        self.triage_state: Optional[Dict[str, Any]] = None

        self.prompt_suggestions: List[str] = []

        self.HISTORY_WINDOW = 6
        self.MAX_OUT = 250
        self.MAX_TOOL_ROUNDS = 4
        self.MAX_RETRIES = 2

    # -----------------------------
    # SAFE MODEL CALL (TPM-aware)
    # -----------------------------
    def safe_create(self, **kwargs):
        """Call OpenAI with retries and trimmed history if rate limited."""
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
        """Integrate tool outputs into onboarding/triage state tracking."""
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
        """Return the current onboarding question or None if finished."""
        if not self.onboarding_state:
            return None
        idx = self.onboarding_state.get("current_idx", 0)
        questions = self.onboarding_state.get("questions") or []
        if idx >= len(questions):
            return None
        return questions[idx]

    def _prompt_next_onboarding_question(self) -> str:
        """Set expectation flag and surface the next onboarding question text."""
        question = self._onboarding_current_question()
        if not question:
            return self._finalize_onboarding_flow()
        self.onboarding_state["expecting_answer"] = True
        self.onboarding_state["reprompted"] = False
        return question.get("question", "").strip()

    def _normalize_onboarding_answer(self, raw: str):
        """Normalize onboarding input and indicate if a reprompt is needed."""
        text = (raw or "").strip()
        if text == "":
            return None, True
        lowered = text.lower()
        if lowered in {"skip", "prefer not to say", "n/a", "na"}:
            return None, False
        return text, False

    def _finalize_onboarding_flow(self) -> str:
        """Persist profile, add deterministic eligibility summary, and close onboarding."""
        questions = self.onboarding_state.get("questions") if self.onboarding_state else []
        answers = self.onboarding_state.get("answers") if self.onboarding_state else {}
        profile = {q.get("key"): answers.get(q.get("key")) for q in (questions or [])}
        profile_json = json.dumps(profile)
        completion_note = "Onboarding is complete. I have saved these details for future chats."
        eligibility = self._eligibility_summary_from_profile(profile)
        summary = f"{completion_note}\n\n{eligibility}" if eligibility else completion_note
        return f"<USER_PROFILE>{profile_json}</USER_PROFILE>\n{summary}"

    def _eligibility_summary_from_profile(self, profile: Dict[str, Any]) -> str:
        """Derive likely service eligibility based on stored onboarding answers."""
        stay = (profile.get("stay_length") or "").lower()
        visa = (profile.get("visa_status") or "").lower()
        postcode = profile.get("postcode") or ""
        gp_registered = str(profile.get("gp_registered") or "").lower()

        long_stay = any(k in stay for k in ["year", "yr", "6", "twelve", "12", "long", "permanent", "settled"])
        has_uk_status = any(k in visa for k in ["student", "work", "skilled", "settled", "ilr", "british", "uk"])
        has_address = bool(postcode.strip())

        gp_line = (
            "Likely eligible to register with a GP (typical for stays ~6+ months). "
            "Use your UK address/postcode; bring ID and proof of address if asked."
            if long_stay or has_uk_status
            else "May be asked about length of stay for GP registration; urgent/111/A&E are still available."
        )
        urgent_line = "Urgent and emergency care (NHS 111, A&E) are available regardless of GP registration."
        registered_note = "You already have a GP registered." if "yes" in gp_registered else ""
        location_note = f"Postcode on file: {postcode}" if postcode else ""

        return "\n".join(
            [
                "Based on your details, here are likely options:",
                f"- {gp_line}",
                f"- {urgent_line}",
                f"- {registered_note}".strip(" -"),
                f"- {location_note}".strip(" -"),
                "If you'd like, I can look up nearby GP practices or urgent care options.",
            ]
        )

    # -----------------------------
    # TRIAGE HELPERS (codified)
    # -----------------------------
    def _start_triage_flow(self) -> str:
        """Initialize deterministic triage question list (NHS 111 inspired)."""
        self.triage_state = {
            "questions": [
                ("severity", "On a scale of 0 to 10, how severe are your symptoms right now?"),
                ("fluids", "Have you been able to keep down food or fluids in the last 12 hours?"),
                ("onset", "When did the symptoms start (time or days ago)?"),
                (
                    "red_flags",
                    "Do you have any of these: chest pain, difficulty breathing, heavy bleeding, "
                    "sudden weakness/numbness, or a seizure in the last 24 hours?",
                ),
                ("other", "Any fever, rash, severe headache, pregnancy, or long-term conditions affecting immunity?"),
            ],
            "answers": {},
            "idx": 0,
        }
        self.triage_active = True
        return "I'll follow NHS 111 style triage. Please answer a few quick questions so I can route you correctly."

    def _triage_next_question(self) -> Optional[str]:
        """Return next triage question text or None when all asked."""
        if not self.triage_state:
            return None
        idx = self.triage_state.get("idx", 0)
        qs = self.triage_state.get("questions", [])
        if idx >= len(qs):
            return None
        return qs[idx][1]

    def _triage_record_answer(self, user_input: str) -> None:
        """Store user triage answer and advance the pointer."""
        if not self.triage_state:
            return
        idx = self.triage_state.get("idx", 0)
        qs = self.triage_state.get("questions", [])
        if idx >= len(qs):
            return
        key = qs[idx][0]
        self.triage_state["answers"][key] = user_input.strip()
        self.triage_state["idx"] = idx + 1

    def _triage_summary(self) -> str:
        """Summarize triage inputs, suggest routing, and offer postcode-based lookup."""
        answers = (self.triage_state or {}).get("answers", {})
        severity = answers.get("severity", "")
        fluids = answers.get("fluids", "").lower()
        red_flags = answers.get("red_flags", "").lower()
        onset = answers.get("onset", "")
        other = answers.get("other", "")
        postcode = (self.user_profile or {}).get("postcode") or ""

        def has_yes(text):
            return any(token in text for token in ["yes", "y", "true", "1"])

        red_flagged = has_yes(red_flags)
        severe = False
        try:
            severe = float(severity) >= 8
        except Exception:
            severe = False
        dehydrated = any(k in fluids for k in ["no", "not", "can't", "cannot", "unable"])

        if red_flagged:
            recommended_service = "A&E"
            route = (
                "This sounds like a red-flag situation. NHS 111 guidance is to call 999 or go to A&E if safe to travel. "
                "If you're unsure, contact NHS 111 immediately for clinical advice."
            )
        elif severe or dehydrated:
            recommended_service = "A&E"
            route = (
                "NHS 111 would typically advise urgent clinical review. Please contact NHS 111 now. "
                "They may direct you to urgent care/A&E if warranted."
            )
        else:
            recommended_service = "GP"
            route = (
                "You likely need non-emergency care. NHS 111 would usually suggest: "
                "register/contact your GP for review, and use NHS 111 online if symptoms change or worsen. "
                "If new severe symptoms arise, call 111 or 999 as appropriate."
            )

        postcode_note = (
            f"I can find the nearest {recommended_service} options using your postcode on record ({postcode})."
            if postcode
            else f"I can look up the nearest {recommended_service} options if you share your postcode."
        )

        summary = (
            "Thanks, here's a quick triage summary (following NHS 111 style steps):\n"
            f"- Severity: {severity or 'not given'}\n"
            f"- Fluids: {answers.get('fluids', 'not given')}\n"
            f"- Onset: {onset or 'not given'}\n"
            f"- Red flags: {answers.get('red_flags', 'not given')}\n"
            f"- Other: {other or 'not given'}\n\n"
            f"{route}\n"
            f"{postcode_note} If you want, I can also help with GP registration or local services."
        )
        return summary

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
        agent_reply = self._ensure_useful_links(agent_reply)
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

            follow_up = self._profile_followups()
            if follow_up:
                self.conversation_history.append({"role": "assistant", "content": follow_up})
                clean_reply = clean_reply + "\n\n" + follow_up

        return clean_reply

    def _ensure_useful_links(self, agent_reply: str) -> str:
        """
        Post-process assistant text to ensure common Useful links are concrete NHS URLs.
        """
        if "Useful links" not in agent_reply:
            return agent_reply

        lines = agent_reply.splitlines()
        if not any(line.strip().lower().startswith("useful links") for line in lines):
            return agent_reply

        canonical_links = [
            ("Find a GP", "https://www.nhs.uk/service-search/find-a-gp"),
            ("Register with a GP", "https://www.nhs.uk/nhs-services/gps/how-to-register-with-a-gp-surgery/"),
            ("Use NHS 111 online", "https://111.nhs.uk/"),
            ("NHS services guide", "https://www.nhs.uk/using-the-nhs/nhs-services/"),
            ("LBS health and wellbeing", "https://www.london.edu/masters-experience/student-support"),
            ("LBS mental wellbeing support", "https://www.london.edu/masters-experience/student-support/mental-health"),
        ]

        new_section = ["Useful links", *[f"- {title}: {url}" for title, url in canonical_links]]

        rebuilt = []
        idx = 0
        replaced = False
        while idx < len(lines):
            line = lines[idx]
            if not replaced and line.strip().lower().startswith("useful links"):
                # skip old section
                idx += 1
                while idx < len(lines) and lines[idx].strip() != "":
                    idx += 1
                rebuilt.extend(new_section)
                replaced = True
                continue
            rebuilt.append(line)
            idx += 1

        return "\n".join(rebuilt)

    def _profile_followups(self) -> str:
        if not self.user_profile:
            return ""

        prompt = (
            "Using the profile below, propose 3-5 concise follow-up suggestions tailored to the user. "
            "Keep it short (under ~120 words), use numbered bullets, and stay within wellbeing/health navigation "
            "topics relevant to UK NHS care. Do NOT ask for onboarding details again. "
            "End with a brief invitation to ask for help finding local services if relevant.\n\n"
            f"User profile: {json.dumps(self.user_profile)}"
        )

        try:
            resp = self.client.responses.create(
                model="gpt-4o-mini",
                input=prompt,
                max_output_tokens=220,
            )
            return resp.output_text or ""
        except Exception:
            fallback = (
                "Here are a few next steps you might find useful:\n"
                "1) Find nearby GP practices and register.\n"
                "2) Book a routine health check or vaccination if due.\n"
                "3) Explore local mental wellbeing resources.\n"
                "If you want, I can look up nearby services based on your postcode."
            )
            return fallback

    def _eligibility_response(self) -> str:
        """
        Deterministic, structured eligibility check with follow-up questions.
        """
        return (
            "Here's a structured check for NHS service eligibility:\n\n"
            "Key criteria:\n"
            "1) Residency/visa: UK resident, settled status, or valid visa (e.g., student or work).\n"
            "2) Location: Living within a UK postcode/catchment for local services (GP, urgent care).\n"
            "3) Duration: Planning to stay 6+ months (typical for GP registration).\n"
            "4) ID/proof: Ability to show ID plus address (e.g., bank statement/tenancy) if asked.\n"
            "5) Visitors: Short-stay visitors may still access urgent or emergency care.\n\n"
            "Want me to confirm with your details? I can start onboarding now to collect postcode, visa/status, UK stay length, and GP status, then I'll summarise what you're eligible for. Just say 'onboarding' to begin."
        )

    def _generate_prompt_suggestions(self, last_reply: str) -> List[str]:
        prompt = (
            "Generate 3 short follow-up prompts the user might want to ask next. "
            "Keep each under 80 characters. "
            "Return ONLY a JSON list of strings. "
            "Avoid duplicates. "
            f"User profile: {json.dumps(self.user_profile)}. "
            f"Last assistant reply: {last_reply}"
        )
        try:
            resp = self.client.responses.create(
                model="gpt-4o-mini",
                input=prompt,
                max_output_tokens=120,
            )
            raw = resp.output_text or "[]"
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                cleaned = [str(x).strip() for x in parsed if str(x).strip()]
                if cleaned:
                    return cleaned[:3]
        except Exception:
            pass

        # Fallback generic suggestions if LLM parsing fails
        fallback = [
            "Find nearby GP or A&E",
            "How to register with a GP",
            "What to do for my symptoms now",
        ]
        return fallback

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
        # SHORT-CIRCUIT: DETERMINISTIC TRIAGE
        # -------------------------------
        lower_input = user_input.lower()
        if self.triage_state:
            self._triage_record_answer(user_input)
            nxt = self._triage_next_question()
            if nxt:
                return self._process_final_reply(nxt)
            summary = self._triage_summary()
            self.triage_state = None
            self.triage_active = False
            return self._process_final_reply(summary)
        elif "triage" in lower_input or "feeling" in lower_input or "symptom" in lower_input:
            start_msg = self._start_triage_flow()
            first_q = self._triage_next_question()
            combined = f"{start_msg}\n\n{first_q}" if first_q else start_msg
            return self._process_final_reply(combined)

        # -------------------------------
        # SHORT-CIRCUIT: ELIGIBILITY QUERY
        # -------------------------------
        if "eligible" in lower_input or "eligibility" in lower_input:
            reply = self._eligibility_response()
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
        triage_lookup_done = False

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

                parsed_tool = self._update_state_from_tool(tool_name, tool_result)

                tool_output_str = tool_result if isinstance(tool_result, str) else json.dumps(tool_result)

                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": tool_output_str,
                    }
                )

                # Auto-chain to nearest services when triage recommends GP or A&E and has a postcode.
                if (
                    tool_name == "nhs_111_live_triage"
                    and not triage_lookup_done
                    and isinstance(parsed_tool, dict)
                    and parsed_tool.get("status") == "final"
                    and parsed_tool.get("should_lookup")
                    and parsed_tool.get("postcode_full")
                    and parsed_tool.get("suggested_service") in {"GP", "A&E"}
                ):
                    try:
                        lookup = tool_nearest_nhs_services(
                            {
                                "postcode_full": parsed_tool.get("postcode_full", ""),
                                "service_type": parsed_tool.get("suggested_service"),
                                "n": 3,
                            }
                        )
                        outputs.append(
                            {
                                "type": "function_call_output",
                                "call_id": f"{call_id}__nearest_services",
                                "output": lookup if isinstance(lookup, str) else json.dumps(lookup),
                            }
                        )
                        triage_lookup_done = True
                    except Exception:
                        pass

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

        clean = self._process_final_reply(agent_reply)
        self.prompt_suggestions = self._generate_prompt_suggestions(clean)
        return clean


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
