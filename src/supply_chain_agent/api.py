"""HTTP API for the Supply Chain AI Agent."""

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from supply_chain_agent.agent import answer_with_tools
from supply_chain_agent.approval import (
    ApprovalStore,
    DecisionAlreadyRecordedError,
    HumanDecision,
    InsufficientApprovalRoleError,
    RecommendationNotFoundError,
    RecommendationRecord,
)
from supply_chain_agent.data_repository import (
    ProductNotFoundError,
    SupplyChainRepository,
)
from supply_chain_agent.llm import LLMProviderError, generate_answer
from supply_chain_agent.rag import RAGAnswer, answer_with_rag
from supply_chain_agent.replenishment import assess_replenishment


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
APPROVAL_DATABASE_PATH = Path(
    os.getenv("APPROVAL_DATABASE_PATH", PROJECT_ROOT / "data" / "approvals.db")
)


app = FastAPI(
    title="Supply Chain AI Agent",
    description="An educational agentic AI application for supply chain workflows.",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    """Data that a client must send to the chat endpoint."""

    message: str = Field(min_length=1, description="The user's question")


class ChatResponse(BaseModel):
    """Data returned by the chat endpoint."""

    answer: str


class SourceResponse(BaseModel):
    """Location of a document chunk retrieved for the answer."""

    document: str
    page: int


class RAGChatResponse(BaseModel):
    """Answer grounded in retrieved document context."""

    answer: str
    retrieved_sources: list[SourceResponse]


class RecommendationRequest(BaseModel):
    """Exact product for which a draft recommendation should be prepared."""

    product_id: str = Field(min_length=1, examples=["SKU-104"])


@lru_cache(maxsize=1)
def get_vector_store():
    """Load the embedding model and vector store once per API process."""
    from supply_chain_agent.embeddings import LocalEmbeddingModel
    from supply_chain_agent.vector_store import FaissVectorStore

    return FaissVectorStore(VECTOR_STORE_DIR, LocalEmbeddingModel())


@lru_cache(maxsize=1)
def get_approval_store() -> ApprovalStore:
    """Open the persistent approval database once per API process."""
    return ApprovalStore(APPROVAL_DATABASE_PATH)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Confirm that the API process is running."""
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Send one user question to the language model."""
    try:
        answer = generate_answer(request.message)
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=502,
            detail="The language model provider is temporarily unavailable.",
        ) from exc

    return ChatResponse(answer=answer)


@app.post("/agent/chat", response_model=ChatResponse)
def agent_chat(request: ChatRequest) -> ChatResponse:
    """Answer a question while allowing the LLM to request safe data tools."""
    try:
        answer = answer_with_tools(request.message)
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=502,
            detail="The language model provider is temporarily unavailable.",
        ) from exc

    return ChatResponse(answer=answer)


@app.post("/recommendations", response_model=RecommendationRecord, status_code=201)
def create_recommendation(request: RecommendationRequest) -> RecommendationRecord:
    """Calculate and persist a draft that still requires a human decision."""
    try:
        assessment = assess_replenishment(
            request.product_id,
            SupplyChainRepository(),
        )
    except ProductNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Product not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return get_approval_store().create(assessment)


@app.get(
    "/recommendations/{recommendation_id}",
    response_model=RecommendationRecord,
)
def get_recommendation(recommendation_id: str) -> RecommendationRecord:
    """Read one recommendation and its recorded decision, if any."""
    try:
        return get_approval_store().get(recommendation_id)
    except RecommendationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Recommendation not found.") from exc


@app.post(
    "/recommendations/{recommendation_id}/decision",
    response_model=RecommendationRecord,
)
def decide_recommendation(
    recommendation_id: str,
    decision: HumanDecision,
) -> RecommendationRecord:
    """Record one explicit and immutable human approval or rejection."""
    try:
        return get_approval_store().record_decision(recommendation_id, decision)
    except RecommendationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Recommendation not found.") from exc
    except InsufficientApprovalRoleError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except DecisionAlreadyRecordedError as exc:
        raise HTTPException(
            status_code=409,
            detail="A decision has already been recorded.",
        ) from exc


@app.post("/rag/chat", response_model=RAGChatResponse)
def rag_chat(request: ChatRequest) -> RAGAnswer:
    """Answer one question using retrieved internal policy context."""
    try:
        return answer_with_rag(request.message, get_vector_store())
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="The vector store has not been built.",
        ) from exc
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=502,
            detail="The language model provider is temporarily unavailable.",
        ) from exc
