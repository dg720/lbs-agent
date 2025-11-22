import json
from urllib.parse import quote_plus
from openai import OpenAI
import os
from dotenv import load_dotenv

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
            {
                "key": "name",
                "question": "What’s your name? (optional — you can say 'skip')",
                "optional": True,
            },
            {
                "key": "age_range",
                "question": "What’s your age range? (e.g., 18–24, 25–34, 35–44)",
                "optional": False,
            },
            {
                "key": "stay_length",
                "question": "How long will you stay in the UK? (e.g., 6 months, 1 year, 2 years)",
                "optional": False,
            },
            {
                "key": "postcode",
                "question": "What’s your London postcode / area? (e.g., NW1, W1H, E14)",
                "optional": False,
            },
            {
                "key": "ihs_paid",
                "question": "Have you paid the Immigration Health Surcharge (IHS)? (yes / no / not sure)",
                "optional": False,
                "allowed": ["yes", "no", "not sure"],
            },
            {
                "key": "gp_registered",
                "question": "Do you already have a registered GP in the UK? (yes / no / not sure)",
                "optional": False,
                "allowed": ["yes", "no", "not sure"],
            },
            {
                "key": "conditions",
                "question": "Any long-term health conditions you'd like me to be aware of? (optional — 'skip')",
                "optional": True,
            },
        ],
        "instructions_to_llm": (
            "You (the assistant) must run onboarding as a multi-turn Q&A. "
            "Ask exactly one question at a time in the given order. "
            "Wait for the user's answer before moving on. "
            "If optional and user says skip, store null. "
            "If allowed options exist, reprompt until valid. "
            "When finished, output the final profile ONLY as JSON wrapped in:\n"
            "<USER_PROFILE>{...}</USER_PROFILE>\n"
            "Then briefly confirm onboarding is complete."
        ),
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
    for k in RED_FLAG_KEYWORDS:
        if k in msg:
            return True
    return False


def emergency_response():
    return (
        "🚨 **Important Safety Notice**\n"
        "Your message includes symptoms that may be serious.\n\n"
        "**In the UK:**\n"
        "- Call **999** for emergencies.\n"
        "- If unsure but worried, call **NHS 111** for urgent advice.\n\n"
        "I can continue to provide general information once you're safe."
    )


NHS_RESULTS_URLS = {
    "GP": "https://www.nhs.uk/service-search/find-a-gp/results/{pc}",
    "A&E": "https://www.nhs.uk/service-search/find-an-accident-and-emergency-service/results/{pc}",
}


def nearest_nhs_services(postcode_full: str, service_type: str, n: int = 3):
    """
    Opens NHS service-search results page and returns nearest n options.
    Uses OpenAI hosted web tool (web_search_preview).
    """
    st = service_type.upper().strip()
    if st not in NHS_RESULTS_URLS:
        raise ValueError(f"Unsupported service_type: {service_type}")

    url = NHS_RESULTS_URLS[st].format(pc=quote_plus(postcode_full.strip().upper()))

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
        model="gpt-4o",  # web_search_preview is supported on tool-capable models :contentReference[oaicite:0]{index=0}
        tools=[{"type": "web_search_preview"}],
        input=prompt,
    )

    text = resp.output_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text, "url": url}


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
        "description": "Collect or refresh the user's profile (age range, stay length, postcode, IHS status, GP registration, conditions) so guidance can be personalised.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]


def tool_nearest_nhs_services(args):
    return nearest_nhs_services(
        postcode_full=args["postcode_full"],
        service_type=args["service_type"],
        n=args.get("n", 3),
    )


def tool_safety(args):
    return emergency_response()
