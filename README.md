# Data Project Template


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                         <- Source code for this project
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    │    
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    ├── plots.py                <- Code to create visualizations 
    │
    └── services                <- Service classes to connect with external platforms, tools, or APIs
        └── __init__.py 
```

--------

## Running the chatbot API

Start a small FastAPI server that exposes the chatbot as a `/chat` endpoint:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

- `GET /health` — returns `{ "status": "ok" }` for uptime checks.
- `POST /chat` — send `{ "message": "hi", "session_id": "<optional>" }` and receive a reply plus the session id to reuse on subsequent turns.

Example request:

```bash
curl -X POST \
  http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, can you help me find a GP?"}'
```

The response contains the assistant reply and a `session_id` you can persist in your frontend or client to maintain conversation context.

## Hosting so Lovable can access the API

Lovable (or any other frontend) just needs an HTTPS URL that serves the `/chat` endpoint. A few quick hosting options are below; all require an `OPENAI_API_KEY` environment variable.

### One-click-ish: Render / Railway / Fly.io

#### Render without Docker (fastest)
1. Push this repo to GitHub.
2. In Render, click **New > Web Service**, pick your repo, and choose the **Python** environment.
3. Set build and start commands:
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn api:app --host 0.0.0.0 --port $PORT`
4. Environment variables:
   - `OPENAI_API_KEY` – your OpenAI key (required).
   - `PORT` – **do not** set manually; Render injects this automatically and the start command above uses it.
5. Health check path: `/health` (Render defaults to GET requests).
6. Deploy. Render will give you a URL like `https://your-app.onrender.com`; point Lovable to `https://your-app.onrender.com/chat`.

> You can skip Docker entirely with the above. If you prefer a fully containerized deploy, Render also accepts Dockerfiles—just pick “Docker” as the environment and keep the same start command.

### Docker container (works locally or on any VPS)

Build and run the container:

```bash
docker build -t nhs-chatbot .
docker run -it -p 8000:8000 -e OPENAI_API_KEY=sk-... nhs-chatbot
```

The API will be available at `http://localhost:8000/chat`. On a VPS, open port `8000` (or map it) and point Lovable to `https://<your-host>:8000/chat`.

### Quick tunnel for testing (no deploy)

If you just need a temporary public URL for Lovable, run the API locally and expose it with a tunnel tool such as `ngrok`:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
ngrok http 8000
```

Use the HTTPS forwarding URL provided by `ngrok` as your base URL.
