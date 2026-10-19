"""Agent loop that lets the LLM request safe, deterministic tools."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from groq import APIError, Groq
from pydantic import ValidationError

from supply_chain_agent.data_repository import (
    ProductNotFoundError,
    SupplyChainRepository,
)
from supply_chain_agent.llm import LLMProviderError
from supply_chain_agent.tools import TOOL_DEFINITIONS, UnknownToolError, execute_tool

if TYPE_CHECKING:
    from supply_chain_agent.vector_store import FaissVectorStore


AGENT_SYSTEM_PROMPT = (
    "You are a supply chain assistant. Use the available tools for exact "
    "inventory, supplier, and purchase-order data. Never invent company data. "
    "Explain the result concisely and distinguish on-hand from available stock. "
    "A replenishment assessment is only a draft. Never claim that an order was "
    "placed or approved. If its status is pending_human_review, clearly explain "
    "the warnings and request human review."
)
MAX_AGENT_STEPS = 4
VECTOR_STORE_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "vector_store"


def answer_with_tools(
    user_message: str,
    *,
    repository: SupplyChainRepository | None = None,
    client: Any | None = None,
    model: str | None = None,
    vector_store: FaissVectorStore | None = None,
) -> str:
    """Run the request/tool/result loop until the LLM produces a final answer."""
    load_dotenv()
    repository = repository or SupplyChainRepository()
    model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    if client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Copy .env.example to .env and add your key."
            )
        client = Groq(api_key=api_key)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        for _ in range(MAX_AGENT_STEPS):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0,
            )
            assistant_message = response.choices[0].message
            tool_calls = assistant_message.tool_calls or []

            if not tool_calls:
                if assistant_message.content is None:
                    raise RuntimeError("The LLM returned an empty response.")
                return assistant_message.content

            messages.append(assistant_message.model_dump(exclude_none=True))

            for tool_call in tool_calls:
                if (
                    tool_call.function.name == "search_policy"
                    and vector_store is None
                ):
                    from supply_chain_agent.embeddings import LocalEmbeddingModel
                    from supply_chain_agent.vector_store import FaissVectorStore

                    vector_store = FaissVectorStore(
                        VECTOR_STORE_DIRECTORY,
                        LocalEmbeddingModel(),
                    )
                result = _execute_requested_tool(
                    tool_call,
                    repository,
                    vector_store=vector_store,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )
    except APIError as exc:
        raise LLMProviderError("The LLM provider request failed.") from exc

    raise RuntimeError("The agent exceeded its maximum number of tool steps.")


def _execute_requested_tool(
    tool_call: Any,
    repository: SupplyChainRepository,
    *,
    vector_store: FaissVectorStore | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Parse, validate, and execute one tool request without crashing the loop."""
    try:
        arguments = json.loads(tool_call.function.arguments)
        return execute_tool(
            tool_call.function.name,
            arguments,
            repository,
            vector_store,
        )
    except json.JSONDecodeError:
        return {"error": "The tool arguments were not valid JSON."}
    except (ValidationError, ProductNotFoundError, UnknownToolError) as exc:
        return {"error": str(exc)}
