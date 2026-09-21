"""HTTP API for the Supply Chain AI Agent."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from supply_chain_agent.llm import LLMProviderError, generate_answer


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

