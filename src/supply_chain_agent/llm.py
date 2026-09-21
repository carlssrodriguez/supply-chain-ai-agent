"""Direct communication with the language model provider."""

import os

from dotenv import load_dotenv
from groq import APIError, Groq


SYSTEM_PROMPT = (
    "You are a supply chain assistant. Be concise and clearly state when "
    "required company data is unavailable."
)


class LLMProviderError(Exception):
    """Raised when the external LLM provider cannot complete a request."""


def generate_answer(user_message: str) -> str:
    """Send a user message to the configured LLM and return its text response."""
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Copy .env.example to .env and add your key."
        )

    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )
    except APIError as exc:
        raise LLMProviderError("The LLM provider request failed.") from exc

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("The LLM returned an empty response.")

    return content
