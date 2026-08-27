"""Tests for the minimal Ollama client wrapper."""

from types import SimpleNamespace

import pytest

from llm import ollama_client


def test_chat_forwards_messages_and_returns_assistant_text(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = [{"role": "user", "content": "Hello"}]
    captured: dict[str, object] = {}

    def fake_chat(*, model: str, messages: object) -> SimpleNamespace:
        captured["model"] = model
        captured["messages"] = messages
        return SimpleNamespace(message=SimpleNamespace(content="Hello from Qwen"))

    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setattr(ollama_client.ollama, "chat", fake_chat)

    result = ollama_client.chat(messages)

    assert result == "Hello from Qwen"
    assert captured == {"model": "qwen3:8b", "messages": messages}


def test_chat_uses_model_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_chat(*, model: str, messages: object) -> SimpleNamespace:
        captured["model"] = model
        return SimpleNamespace(message=SimpleNamespace(content="response"))

    monkeypatch.setenv("OLLAMA_MODEL", "another-local-model")
    monkeypatch.setattr(ollama_client.ollama, "chat", fake_chat)

    ollama_client.chat([{"role": "user", "content": "Hello"}])

    assert captured["model"] == "another-local-model"


def test_chat_rejects_response_without_assistant_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_chat(*, model: str, messages: object) -> SimpleNamespace:
        return SimpleNamespace(message=SimpleNamespace(content=None))

    monkeypatch.setattr(ollama_client.ollama, "chat", fake_chat)

    with pytest.raises(ValueError, match="without assistant text"):
        ollama_client.chat([{"role": "user", "content": "Hello"}])
