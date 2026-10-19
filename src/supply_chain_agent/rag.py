"""Retrieval-augmented generation over the internal policy documents."""

from dataclasses import dataclass

from supply_chain_agent.llm import generate_answer
from supply_chain_agent.tools import SearchableVectorStore


RAG_SYSTEM_PROMPT = """You are a supply chain policy assistant.
Answer the question using only the supplied context.
If the context does not contain enough information, say that you do not know.
Do not invent policies, thresholds, facts, or sources.
Keep the answer concise and mention the supporting source and page."""


@dataclass(frozen=True)
class SourceReference:
    """A source used as context for a generated answer."""

    document: str
    page: int


@dataclass(frozen=True)
class RAGAnswer:
    """A generated answer and the retrieved sources provided to the LLM."""

    answer: str
    retrieved_sources: list[SourceReference]


def _format_context(results) -> str:
    """Turn retrieved chunks into clearly labelled prompt context."""
    context_blocks = []

    for result in results:
        chunk = result.chunk
        context_blocks.append(
            f"[Source: {chunk.source}, page {chunk.page}]\n{chunk.text}"
        )

    return "\n\n".join(context_blocks)


def answer_with_rag(
    question: str,
    store: SearchableVectorStore,
    *,
    top_k: int = 3,
) -> RAGAnswer:
    """Retrieve relevant chunks and ask the LLM to answer from that context."""
    results = store.search(question, top_k=top_k)
    context = _format_context(results)
    prompt = f"""CONTEXT
{context}

QUESTION
{question}
"""
    answer = generate_answer(prompt, system_prompt=RAG_SYSTEM_PROMPT)

    unique_sources = []
    seen_sources = set()
    for result in results:
        source_key = (result.chunk.source, result.chunk.page)
        if source_key not in seen_sources:
            unique_sources.append(
                SourceReference(
                    document=result.chunk.source,
                    page=result.chunk.page,
                )
            )
            seen_sources.add(source_key)

    return RAGAnswer(answer=answer, retrieved_sources=unique_sources)
