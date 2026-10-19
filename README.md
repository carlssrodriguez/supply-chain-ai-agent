# Supply Chain AI Agent

A portfolio-ready agentic AI application that answers internal policy questions,
reads exact operational data, assesses stockout risk, prepares replenishment
drafts, and enforces human approval.

The project intentionally combines probabilistic and deterministic components:

- the LLM selects tools and explains results;
- FAISS retrieves policy evidence by semantic similarity;
- Python performs inventory and replenishment calculations;
- Pydantic validates inputs and outputs;
- SQLite records immutable human decisions.

The agent never creates, approves, transmits, or modifies a real purchase order.

## Architecture

```text
User -> FastAPI -> Agent / Groq
                      |
          +-----------+------------+
          |                        |
     Policy question          Operational question
          |                        |
  Embeddings -> FAISS        Validated Python tools
          |                        |
       PDF chunks          CSV repository + calculations
          +-----------+------------+
                      |
             Pydantic validation
                      |
             Human approval + SQLite
```

## Implemented capabilities

- Direct Groq LLM integration without a framework abstraction.
- FastAPI endpoints and OpenAPI documentation.
- PDF extraction, structure-aware chunking, local embeddings, and FAISS search.
- RAG answers with document and page references.
- Retrieval evaluation with Hit@1, Hit@3, and MRR.
- Experimental BM25 + vector hybrid retrieval.
- Exact inventory, supplier, and purchase-order tools.
- Function calling with an allow-list and a maximum-step guardrail.
- Deterministic reorder point, target stock, MOQ, cost, and approver calculations.
- Stale-data and overdue-order warnings.
- Persistent, immutable human approvals and rejections in SQLite.
- Automated tests for calculations, APIs, retrieval, tools, and security.

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

Build the vector store once:

```powershell
python scripts/build_vector_store.py
```

## Run the API

```powershell
uvicorn supply_chain_agent.api:app --reload --port 8001
```

Open `http://127.0.0.1:8001/docs` to use the interactive API documentation.

Important endpoints:

- `GET /health`
- `POST /chat`
- `POST /rag/chat`
- `POST /agent/chat`
- `POST /recommendations`
- `GET /recommendations/{recommendation_id}`
- `POST /recommendations/{recommendation_id}/decision`

## Run the tests

```powershell
python -m pytest -v
```

On Windows, setting `OMP_NUM_THREADS=1` can avoid native FAISS/OpenMP conflicts:

```powershell
$env:OMP_NUM_THREADS="1"
python -m pytest -q
```

## Run with Docker

```powershell
docker compose up --build
```

Then open `http://127.0.0.1:8001/docs`.

## Example workflow

1. Ask the agent for the available stock of `SKU-104`.
2. The LLM requests the exact inventory tool.
3. Python calculates available stock and replenishment risk.
4. The system returns a draft with warnings and the required approver.
5. A human records an approval or rejection through the API.
6. The application stores the decision for audit; it does not place an order.

## Safety design

- The LLM can only request allow-listed tools.
- No tool can approve or transmit a purchase order.
- Company data is calculated by Python rather than invented by the LLM.
- Missing, stale, or contradictory data triggers human review.
- Approval authority is checked in deterministic code.
- Recorded decisions cannot be overwritten.
- Secrets, local databases, caches, and generated indexes are excluded from Git.

## Known limitations

- The project uses fictional training data, not a live ERP.
- Authentication is not implemented; approver identity is supplied by the API
  client and would require corporate identity integration in production.
- SQLite is appropriate for this demonstration but a production deployment
  would normally use a managed relational database.
- The evaluation dataset is intentionally small and should be expanded before
  production use.
- FAISS can have native OpenMP issues on some Windows environments; Docker/Linux
  provides a more reproducible runtime.

## Technical decisions

The core workflow was implemented directly before introducing LangChain. This
makes tool selection, message passing, validation, retrieval, and guardrails
visible and testable. LangChain can be added later as a comparison, but is not
required for the application to function.
