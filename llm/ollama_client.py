"""Minimal synchronous client for the local Ollama server."""

import os
from collections.abc import Mapping, Sequence

import ollama


DEFAULT_MODEL = "qwen3:8b"


def selected_model() -> str:
    """Return the model used for local chat requests."""
    return os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)


def chat(messages: Sequence[Mapping[str, str]]) -> str:
    """Send chat messages to Ollama and return the assistant text."""
    response = ollama.chat(model=selected_model(), messages=messages)
    content = response.message.content

    if not content:
        raise ValueError("Ollama returned a response without assistant text.")

    return content
