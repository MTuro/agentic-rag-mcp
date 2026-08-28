"""A small, explicit agent loop built without an agent framework."""

import json
from typing import Any

from agent.state import MAX_STEPS, AgentState
from llm.ollama_client import chat


SYSTEM_PROMPT = """You are an educational agent that must reply with one JSON object only.

Choose exactly one of these decisions:
1. Give the final answer:
   {"type": "final", "answer": "your answer"}
2. Count the words in text:
   {"type": "action", "tool": "word_count", "input": "text to count"}

Use word_count when the user asks for an exact word count. After receiving an
observation, use it to produce a final answer. Do not wrap JSON in Markdown.
"""


def _word_count(text: str) -> int:
    """Count whitespace-separated words for the educational action."""
    return len(text.split())


def _parse_decision(response: str) -> dict[str, Any]:
    """Parse one JSON object returned by the language model."""
    try:
        decision = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError("The model returned invalid JSON.") from exc

    if not isinstance(decision, dict):
        raise ValueError("The model decision must be a JSON object.")

    return decision


def _execute_action(decision: dict[str, Any]) -> str:
    """Execute the single educational action available in this milestone."""
    tool_name = decision.get("tool")
    if tool_name != "word_count":
        raise ValueError(f"Unknown tool: {tool_name!r}.")

    tool_input = decision.get("input")
    if not isinstance(tool_input, str):
        raise ValueError("The word_count action requires a string input.")

    return f"word_count returned {_word_count(tool_input)}"


def run_agent(prompt: str, *, max_steps: int = MAX_STEPS) -> str:
    """Run the manual decision-action-observation loop for one user prompt."""
    state = AgentState(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_steps=max_steps,
    )

    while state.current_step < state.max_steps:
        state.current_step += 1
        response = chat(state.messages)
        state.messages.append({"role": "assistant", "content": response})
        decision = _parse_decision(response)
        decision_type = decision.get("type")

        if decision_type == "final":
            answer = decision.get("answer")
            if not isinstance(answer, str) or not answer:
                raise ValueError("A final decision requires a non-empty answer.")
            return answer

        if decision_type == "action":
            observation = _execute_action(decision)
            state.observations.append(observation)
            state.messages.append(
                {
                    "role": "user",
                    "content": f"Observation: {observation}. Decide what to do next.",
                }
            )
            continue

        raise ValueError(f"Unknown decision type: {decision_type!r}.")

    raise RuntimeError(f"Agent stopped after reaching the maximum of {state.max_steps} steps.")
