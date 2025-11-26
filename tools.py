"""Tool implementations for onboarding, safety, triage, search, and lookups."""

import json
import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)





def tool_onboarding(_args=None):
    """
    LLM-driven onboarding initializer.
    Returns a questionnaire & rules for the LLM to run multi-turn onboarding.
    """
    return {
        "mode": "llm_multiturn_onboarding",
        "questions": [
            # --- CONTEXT ---
            {"key": "name", "question": "What's your name? (optional - you can say 'skip')", "optional": True},
            {"key": "age_range", "question": "What's your age range?", "optional": False},
            {"key": "stay_length", "question": "How long will you stay in the UK?", "optional": False},
            {"key": "postcode", "question": "What's your London postcode / area?", "optional": False},
            {"key": "visa_status", "question": "Do you hold a UK visa/status (e.g., student, work, settled, visitor)?", "optional": False},
            {"key": "gp_registered", "question": "Do you already have a registered GP in the UK?", "optional": False},
            {"key": "conditions", "question": "Any long-term health conditions you'd like me to be aware of? (optional - say 'skip')", "optional": True},
            # --- MEDICAL ---
            {"key": "medications", "question": "Do you take any regular medications or receive ongoing treatment? (optional - say 'skip')", "optional": True},
            # --- LIFESTYLE ---
            {"key": "lifestyle_focus", "question": "Is there any lifestyle area you want to improve while in the UK?", "optional": False},
            # --- MENTAL HEALTH ---
            {"key": "mental_wellbeing", "question": "How has your mental wellbeing been recently? (optional - say 'skip')", "optional": True},
        ],
        "instructions_to_llm": """
You (the assistant) must run onboarding as a strict multi-turn Q&A.

CRITICAL RULES:
1) Ask ONLY the questions provided in the `questions` list.
2) Ask them in EXACT order.
3) Ask EXACTLY ONE question per turn.
4) Use the question text VERBATIM - do not rephrase, expand, or add examples.
5) Do NOT ask any extra questions (e.g., date of birth, phone number, email, gender, nationality, visas, etc.).
6) All answers are free text. Interpret/normalize internally if useful, but do not show options.
7) NEVER append the user's previous answer to the question line. Each assistant turn during onboarding should contain ONLY the next question.
8) If the user goes off-topic mid-onboarding, say you'll answer after onboarding and repeat the CURRENT question.
9) If optional and the user says 'skip', store null.
10) If the user gives an empty/unclear answer, gently reprompt ONCE with the same verbatim question.

When finished, output the final profile ONLY as JSON wrapped in:
<USER_PROFILE>{...}</USER_PROFILE>
Then briefly confirm onboarding is complete.
""",
    }

# ---------------------------------------------------------
# Safety Classifier (Red-Flag Detection)
# ---------------------------------------------------------
RED_FLAG_KEYWORDS = [
    "chest pain",
    "severe bleeding",
    "not breathing",
    "can't breathe",
    "suicidal",
    "harm myself",
    "overdose",
    "unconscious",
    "collapse",
    "stroke",
    "heart attack",
    "seizure",
    "very high fever",
    "severe allergic",
    "anaphylaxis",
]


def safety_check(message):
    msg = message.lower()
    return any(k in msg for k in RED_FLAG_KEYWORDS)






def emergency_response():
    return (
        "Important Safety Notice\n"
        "Your message includes symptoms that may be serious.\n\n"
        "**In the UK:**\n"
        "- Call **999** for emergencies.\n"
        "- If unsure but worried, call **NHS 111** for urgent advice.\n\n"
        "I can continue to provide general information once you're safe."
    )

# ---------------------------------------------------------
# Tools backed by OpenAI web_search_preview

# ---------------------------------------------------------
NHS_RESULTS_URLS = {
    "GP": "https://www.nhs.uk/service-search/find-a-gp/results/{pc}",
    "A&E": "https://www.nhs.uk/service-search/find-an-accident-and-emergency-service/results/{pc}",
}


def nearest_nhs_services(postcode_full: str, service_type: str, n: int = 3):
    """
    Opens NHS service-search results page and returns nearest n options.
    Uses OpenAI hosted web tool (web_search_preview).
    """
    stype = service_type.upper().strip()
    if stype not in NHS_RESULTS_URLS:
        raise ValueError(f"Unsupported service_type: {service_type}")

    url = NHS_RESULTS_URLS[stype].format(pc=quote_plus(postcode_full.strip().upper()))

    prompt = f"""
Open the NHS results page and extract the nearest {n} services.

URL: {url}

Return STRICT JSON: a list of up to {n} objects with:
- name
- distance (string, if shown)
- address
- phone (if shown)

The page is already nearest-first; take the top results.
"""

    resp = client.responses.create(
        model="gpt-4o",
        tools=[{"type": "web_search_preview"}],
        input=prompt,
    )

    text = (resp.output_text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text, "url": url}


ALLOWED_DOMAINS = [
    "gov.uk",
    "nhs.uk",
    "111.nhs.uk",
    "england.nhs.uk",
    "bartshealth.nhs.uk",
    "ukcisa.org.uk",
    "london.edu",
    "talkcampus.com",
]


def guided_search(args, max_results_default: int = 5):
    """
    Allowlist-first retrieval using ONLY OpenAI web_search_preview.
    """
    if isinstance(args, dict):
        query = args.get("query", "") or args.get("q", "")
        max_results = int(args.get("max_results", max_results_default) or max_results_default)
    else:
        query = str(args)
        max_results = max_results_default

    query = query.strip()
    if not query:
        return {"context": "", "sources": [], "fallback_used": False}

    def _has_allowlisted_domain(text: str) -> bool:
        lower_text = text.lower()
        return any(d in lower_text for d in ALLOWED_DOMAINS)

    site_filter = " OR ".join([f"site:{d}" for d in ALLOWED_DOMAINS])
    restricted_query = (
        f"({query}) ({site_filter}). "
        f"Prefer answers from these sites only. Return up to {max_results} relevant results with citations."
    )

    restricted_resp = client.responses.create(
        model="gpt-4o-mini",
        input=restricted_query,
        tools=[{"type": "web_search_preview"}],
        tool_choice={"type": "web_search_preview"},
        max_output_tokens=1200,
    )

    restricted_text = restricted_resp.output_text or ""
    restricted_sources = []

    too_short = len(restricted_text.strip()) < 200
    no_allowlisted_cites = not _has_allowlisted_domain(restricted_text)

    if not (too_short or no_allowlisted_cites):
        return {
            "context": restricted_text,
            "sources": restricted_sources,
            "fallback_used": False,
        }

    broad_query = f"{query}. Return up to {max_results} relevant results with citations."

    broad_resp = client.responses.create(
        model="gpt-4o-mini",
        input=broad_query,
        tools=[{"type": "web_search_preview"}],
        tool_choice={"type": "web_search_preview"},
        max_output_tokens=1200,
    )

    return {
        "context": broad_resp.output_text or "",
        "sources": [],
        "fallback_used": True,
    }




def nhs_111_live_triage(args):
    """
    Lightweight LLM-led triage + routing for NHS 111.
    Returns either follow-up questions (need_more_info) or a final routing decision.
    """
    presenting_issue = args.get("presenting_issue")
    postcode_full = args.get("postcode_full")
    known_answers = args.get("known_answers", {}) or {}

    prompt = f"""
You are NHS 101, a lightweight triage router for international students. NON-DIAGNOSTIC.

Goal:
- Ask only what you need to decide the most appropriate NHS service.
- Emergency red flags override everything.
- Use known_answers to avoid repeating questions.
- Keep it concise and stop once you have enough to route safely.

Emergency red flags (ANY => emergency / A&E / 999):
- severe chest pain, trouble breathing, blue lips
- heavy bleeding that won't stop
- stroke signs (face droop, arm weakness, speech trouble)
- seizure / fainting / unconsciousness
- sudden severe allergic reaction
- immediate danger / unsafe mental state / suicidal intent

You must return STRICT JSON in ONE of these two forms:

FORM A (need more info):
{{
  "status": "need_more_info",
  "follow_up_questions": ["ONLY_ONE_QUESTION"],
  "known_answers_update": {{}}
}}

FORM B (final):
{{
  "status": "final",
  "severity_level": "low|medium|high|emergency",
  "suggested_service": "A&E|GP|NHS_111|PHARMACY_SELFCARE|MENTAL_HEALTH_CRISIS",
  "rationale": "1-2 sentences",
  "postcode_full": "{postcode_full}",
  "should_lookup": true|false
}}

Inputs:
- presenting_issue: {presenting_issue}
- known_answers: {json.dumps(known_answers)}
Current_answer_count: {len(known_answers)}

Rules:
- If any red flag is present from presenting_issue or known_answers, return FORM B with:
  severity_level="emergency" and suggested_service="A&E".
- Otherwise, ask at most ONE short follow-up question IF needed, but keep total follow-ups to 5-8 and NEVER exceed 10.
- If len(known_answers) >= 5, do NOT ask more questions unless absolutely necessary; move to FORM B with your best judgment.
- If len(known_answers) >= 8, you MUST return FORM B (final) with your best judgment (no more questions).
- Do NOT repeat a topic already present in known_answers. Common keys: severity, onset, injury_trauma, functional ability (walking/using/weight-bearing), swelling/heat/bruising/deformity, red_flags, mental_health_safety.
- Examples of useful follow-ups (pick ONE at a time and only if not already covered):
  - severity 0-10
  - rapid onset vs gradual / time course
  - ability to function normally (walk/use/weight-bear/breathe/eat)
  - visible deformity, numbness, heavy swelling, heat/redness, locking/clicking, instability/buckling (MSK)
- clear mechanism of injury or recent trigger (fall, twist, overuse)
- mental health safety: self-harm thoughts / unsafe now
- Keep questions crisp, one-line, no preamble.
- Only return FORM B once enough info is available.

Browser instructions (web_search_preview):
- Open https://111.nhs.uk/ and follow the relevant triage path based on presenting_issue.
- Mirror the site prompts with very short single questions; do NOT add extra fluff.
- Stop browsing once you have enough to suggest a service (A&E/999, NHS 111, GP, pharmacy/self-care, mental health crisis).
- If the site flow is unclear or tool fails, fall back to your best judgment using the rules above.

Routing guidance:
- emergency/high + red flags or very severe rapid onset => A&E
- moderate symptoms, unsure urgency => NHS_111
- moderate/persistent but stable => GP
- mild + functioning OK => PHARMACY_SELFCARE
- mental health safety risk => MENTAL_HEALTH_CRISIS

should_lookup = true ONLY if:
- suggested_service is "GP" or "A&E"
- AND postcode_full is provided in inputs.
"""
    resp = client.responses.create(
        model="gpt-4o",
        input=prompt,
        tools=[{"type": "web_search_preview"}],
        tool_choice="auto",
        max_output_tokens=700,
    )

    raw = resp.output_text or ""
    try:
        parsed = json.loads(raw)
        return parsed
    except json.JSONDecodeError:
        return {"raw": raw, "error": "Could not parse triage result as JSON"}


tools = [
    {
        "type": "function",
        "name": "nearest_nhs_services",
        "description": (
            "Given a FULL UK postcode and service type, open the NHS service-search "
            "results page and return the nearest 2–3 options."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "postcode_full": {
                    "type": "string",
                    "description": "Full UK postcode, e.g. 'NW1 2BU'. Must be complete.",
                },
                "service_type": {
                    "type": "string",
                    "enum": ["GP", "A&E"],
                    "description": "Which NHS results page to use.",
                },
                "n": {"type": "integer", "default": 3, "minimum": 1, "maximum": 5},
            },
            "required": ["postcode_full", "service_type"],
        },
    },
    {
        "type": "function",
        "name": "trigger_safety_protocol",
        "description": "Safety response for dangerous symptoms.",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    },
    {
        "type": "function",
        "name": "onboarding",
        "description": (
            "Collect or refresh the user's profile so guidance can be personalised."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "guided_search",
        "description": (
            "Search approved NHS/LBS sites first using OpenAI Web Search. "
            "If nothing relevant is found, run a general OpenAI Web Search fallback. "
            "Never scrape manually."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "nhs_111_live_triage",
        "description": (
            "Lightweight triage + routing via live navigation of https://111.nhs.uk/ "
            "using OpenAI's web viewing/computer-use capability. "
            "Returns a structured routing result and a flag to chain to nearest_nhs_services "
            "when GP or A&E is recommended and full postcode is provided."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "presenting_issue": {
                    "type": "string",
                    "description": "User’s symptom or concern in natural language.",
                },
                "postcode_full": {
                    "type": "string",
                    "description": (
                        "Optional full UK postcode (e.g., 'NW1 2BU'). "
                        "If provided and 111 recommends GP or A&E, the agent should chain "
                        "to nearest_nhs_services."
                    ),
                },
                "known_answers": {
                    "type": "object",
                    "description": (
                        "Free-form key/value store of answers already collected "
                        "(e.g., {'severity_0_10': 6, 'red_flags': 'no'}). "
                        "Use to skip redundant questions on 111 where possible."
                    ),
                    "additionalProperties": True,
                },
            },
            "required": ["presenting_issue"],
        },
    },
]


def tool_nearest_nhs_services(args):
    """Wrapper to expose nearest_nhs_services to the tool dispatcher."""
    return nearest_nhs_services(
        postcode_full=args["postcode_full"],
        service_type=args["service_type"],
        n=args.get("n", 3),
    )


def tool_safety(_args):
    """Return the standard emergency safety response."""
    return emergency_response()
