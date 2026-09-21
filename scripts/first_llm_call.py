"""Make the project's first direct call to an LLM."""

import sys

from supply_chain_agent.llm import generate_answer


def main() -> None:
    """Send one prompt to Groq and print the model's response."""
    sys.stdout.reconfigure(encoding="utf-8")
    answer = generate_answer("How many units of SKU-104 do we currently have?")
    print(answer)


if __name__ == "__main__":
    main()
