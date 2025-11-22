import re
import json


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
