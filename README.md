# Supply Chain AI Agent

An educational, portfolio-ready agentic AI application for investigating supply
chain risks and preparing procurement recommendations.

The project is built incrementally. Each component is introduced only after its
purpose, trade-offs, and failure modes have been explained.

## Current status

Day 1 completed:

- direct LLM interaction through Groq;
- FastAPI endpoints with automatic validation;
- basic external-provider error handling;
- automated API tests that do not call the real LLM.

## Requirements

- Python 3.11 or 3.12
- A Groq API key

## Local setup

Create and activate a virtual environment, then install the project and its
development dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --editable ".[dev]"
```

Copy `.env.example` to `.env` and add your Groq API key. The `.env` file is
ignored by Git and must never be committed.

## Run the API

```powershell
uvicorn supply_chain_agent.api:app --reload --port 8001
```

Open `http://127.0.0.1:8001/docs` to use the interactive API documentation.

## Run the tests

```powershell
python -m pytest -v
```
