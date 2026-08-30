"""Tests for the explicit manual agent loop."""

from collections.abc import Iterator

import pytest

import agent.agent as agent_module
from agent.state import MAX_STEPS, AgentState
from tools import Tool, ToolRegistry, word_count


def test_agent_state_starts_with_explicit_defaults() -> None:
    messages = [{"role": "user", "content": "Hello"}]

    state = AgentState(messages=messages)

    assert state.messages == messages
    assert state.current_step == 0
    assert state.max_steps == MAX_STEPS
    assert state.observations == []


def test_run_agent_returns_immediate_final_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_messages: list[list[dict[str, str]]] = []

    def fake_chat(messages: list[dict[str, str]]) -> str:
        captured_messages.append([message.copy() for message in messages])
        return '{"type": "final", "answer": "Hello from the agent"}'

    monkeypatch.setattr(agent_module, "chat", fake_chat)

    answer = agent_module.run_agent("Hello")

    assert answer == "Hello from the agent"
    assert captured_messages[0][0]["role"] == "system"
    assert captured_messages[0][1] == {"role": "user", "content": "Hello"}


def test_run_agent_feeds_action_observation_back_to_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses: Iterator[str] = iter(
        [
            '{"type": "action", "tool": "word_count", "arguments": {"text": "one two three"}}',
            '{"type": "final", "answer": "The text contains 3 words."}',
        ]
    )
    calls: list[list[dict[str, str]]] = []
    created_states: list[AgentState] = []

    def fake_chat(messages: list[dict[str, str]]) -> str:
        calls.append([message.copy() for message in messages])
        return next(responses)

    def capture_state(*, messages: list[dict[str, str]], max_steps: int) -> AgentState:
        state = AgentState(messages=messages, max_steps=max_steps)
        created_states.append(state)
        return state

    monkeypatch.setattr(agent_module, "chat", fake_chat)
    monkeypatch.setattr(agent_module, "AgentState", capture_state)

    answer = agent_module.run_agent("How many words are in: one two three?")

    assert answer == "The text contains 3 words."
    assert len(calls) == 2
    assert created_states[0].current_step == 2
    assert created_states[0].observations == ["Tool word_count returned: 3"]
    assert created_states[0].messages == calls[1] + [
        {
            "role": "assistant",
            "content": '{"type": "final", "answer": "The text contains 3 words."}',
        }
    ]
    assert calls[1][-2]["role"] == "assistant"
    assert calls[1][-1] == {
        "role": "user",
        "content": "Observation: Tool word_count returned: 3. Decide what to do next.",
    }


def test_run_agent_shows_registered_tools_and_dispatches_model_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses: Iterator[str] = iter(
        [
            '{"type": "action", "tool": "repeat", "arguments": {"text": "hello"}}',
            '{"type": "final", "answer": "hellohello"}',
        ]
    )
    calls: list[list[dict[str, str]]] = []

    def fake_chat(messages: list[dict[str, str]]) -> str:
        calls.append([message.copy() for message in messages])
        return next(responses)

    repeat_tool = Tool(
        name="repeat",
        description="Repeat text twice.",
        argument_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        implementation=lambda text: text * 2,
    )
    registry = ToolRegistry([repeat_tool])
    monkeypatch.setattr(agent_module, "chat", fake_chat)

    answer = agent_module.run_agent("Repeat hello", tool_registry=registry)

    assert answer == "hellohello"
    assert '"name": "repeat"' in calls[0][0]["content"]
    assert '"description": "Repeat text twice."' in calls[0][0]["content"]
    assert calls[1][-1]["content"] == (
        "Observation: Tool repeat returned: hellohello. Decide what to do next."
    )


def test_run_agent_selects_filesystem_tools_through_generic_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses: Iterator[str] = iter(
        [
            '{"type": "action", "tool": "list_files", "arguments": {"path": "."}}',
            '{"type": "action", "tool": "read_file", "arguments": {"path": "README.md"}}',
            '{"type": "final", "answer": "This repository explores a local agent."}',
        ]
    )
    calls: list[list[dict[str, str]]] = []

    def fake_chat(messages: list[dict[str, str]]) -> str:
        calls.append([message.copy() for message in messages])
        return next(responses)

    monkeypatch.setattr(agent_module, "chat", fake_chat)

    answer = agent_module.run_agent("Explain what this repository does.")

    assert answer == "This repository explores a local agent."
    assert len(calls) == 3
    assert '"name": "list_files"' in calls[0][0]["content"]
    assert '"name": "read_file"' in calls[0][0]["content"]
    assert '"name": "search_code"' in calls[0][0]["content"]
    assert '"name": "word_count"' in calls[0][0]["content"]
    assert "Tool list_files returned:" in calls[1][-1]["content"]
    assert "README.md" in calls[1][-1]["content"]
    assert "Tool read_file returned:" in calls[2][-1]["content"]
    assert "agentic software engineering assistant" in calls[2][-1]["content"]


def test_existing_word_count_behavior_remains_available() -> None:
    assert word_count("one two three") == 3


def test_run_agent_stops_at_max_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0

    def fake_chat(messages: list[dict[str, str]]) -> str:
        nonlocal call_count
        call_count += 1
        return (
            '{"type": "action", "tool": "word_count", '
            '"arguments": {"text": "keep going"}}'
        )

    monkeypatch.setattr(agent_module, "chat", fake_chat)

    with pytest.raises(RuntimeError, match="maximum of 2 steps"):
        agent_module.run_agent("Never finish", max_steps=2)

    assert call_count == 2


@pytest.mark.parametrize(
    ("response", "error"),
    [
        ("not JSON", "invalid JSON"),
        ('{"type": "other"}', "Unknown decision type"),
        (
            '{"type": "action", "tool": "missing", "arguments": {"text": "text"}}',
            "Unknown tool",
        ),
        ('{"type": "action", "tool": "word_count"}', "requires an arguments object"),
        ('{"type": "final", "answer": ""}', "requires a non-empty answer"),
    ],
)
def test_run_agent_rejects_invalid_decisions(
    monkeypatch: pytest.MonkeyPatch,
    response: str,
    error: str,
) -> None:
    monkeypatch.setattr(agent_module, "chat", lambda messages: response)

    with pytest.raises(ValueError, match=error):
        agent_module.run_agent("Hello")
