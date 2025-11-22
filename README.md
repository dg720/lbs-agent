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
