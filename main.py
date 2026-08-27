"""Run one local chat request through Ollama."""

from llm.ollama_client import chat


def main() -> None:
    """Read one prompt, send it to Ollama, and print the response."""
    prompt = input("You: ")
    answer = chat([{"role": "user", "content": prompt}])
    print(answer)


if __name__ == "__main__":
    main()
