# Evi – NHS Guidance Copilot (LBS AI Agent Cup)
Submission for the LBS AI Agent Cup (Team Evida). A Streamlit chat app that guides users through UK NHS navigation, onboarding, and triage with codified safety rules and curated links (NHS + LBS wellbeing).

## Quick start
- Install deps: `pip install -r requirements.txt`
- Run UI: `streamlit run streamlit_app.py`
- Set secrets: `OPENAI_API_KEY` (via `.env` for local or `st.secrets` on Streamlit Cloud).

## Repo map (essentials)
- `streamlit_app.py` – Streamlit UI (hero, chat history, prompt suggestions, theming).
- `agent.py` – Core agent session: onboarding, deterministic triage (NHS 111 style), eligibility check, link post-processing, prompt follow-ups, tool orchestration.
- `tools.py` – Tool definitions: onboarding questionnaire, safety check, NHS 111 triage stub, guided search (web), nearest NHS services, emergency response.
- `prompts.py` – System/intro text shown in the UI.
- `.env` – Local secrets (not tracked). Use `OPENAI_API_KEY`.
- `requirements.txt` – Python dependencies.
- `README.md` – You are here.
- `streamlit_app.py` + `agent.py` already read `OPENAI_API_KEY` from `st.secrets` or env for Streamlit Cloud.

## How it works
- Chat flow: user input → agent session (`agent.py`) → optional tools (triage, search, services) → post-processing (useful links, eligibility/onboarding summaries) → Streamlit render.
- Onboarding: strict, ordered questions; stored profile; post-onboarding eligibility summary.
- Triage: deterministic NHS 111-style Q&A with red-flag routing; offers GP/A&E lookup when complete.
- Useful links: replaces any “Useful links” section with a curated NHS/LBS list (no duplicates/truncation).

## Deploying to Streamlit Cloud
1) Add secret `OPENAI_API_KEY` in the Streamlit app settings.  
2) Run `streamlit_app.py` as the main entry point.  
3) (Optional) Add other env vars in `st.secrets` as needed; the app already checks `st.secrets` before env.
