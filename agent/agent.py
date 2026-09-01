"""A small, explicit agent loop built without an agent framework."""

import json
from typing import Any

from agent.state import MAX_STEPS, AgentState
from llm.ollama_client import chat
from tools import (
    GIT_DIFF_TOOL,
    GIT_STATUS_TOOL,
    LIST_FILES_TOOL,
    READ_FILE_TOOL,
    SEARCH_CODE_TOOL,
    WORD_COUNT_TOOL,
    ToolRegistry,
)


BASE_SYSTEM_PROMPT = """You are an educational agent that must reply with one JSON object only.

Choose exactly one of these decisions:
1. Give the final answer:
   {"type": "final", "answer": "your answer"}
2. Call one available tool:
   {"type": "action", "tool": "tool_name", "arguments": {"argument_name": "value"}}

Choose tools from the definitions below. Use each tool's argument schema to build
the arguments object. After receiving an observation, use it to decide whether to
call another tool or produce a final answer. Do not wrap JSON in Markdown.
"""

DEFAULT_TOOL_REGISTRY = ToolRegistry(
    [
        WORD_COUNT_TOOL,
        LIST_FILES_TOOL,
        READ_FILE_TOOL,
        SEARCH_CODE_TOOL,
        GIT_STATUS_TOOL,
        GIT_DIFF_TOOL,
    ]
)


def _build_system_prompt(tool_registry: ToolRegistry) -> str:
    """Build instructions from the tools currently available to the agent."""
    definitions = json.dumps(tool_registry.model_definitions(), indent=2)
    return f"{BASE_SYSTEM_PROMPT}\nAvailable tools:\n{definitions}"


def _parse_decision(response: str) -> dict[str, Any]:
    """Parse one JSON object returned by the language model."""
    try:
        decision = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError("The model returned invalid JSON.") from exc

    if not isinstance(decision, dict):
        raise ValueError("The model decision must be a JSON object.")

    return decision


def _execute_action(decision: dict[str, Any], tool_registry: ToolRegistry) -> str:
    """Execute a model-selected tool through the generic registry."""
    tool_name = decision.get("tool")
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("An action decision requires a non-empty tool name.")

    arguments = decision.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("An action decision requires an arguments object.")

    result = tool_registry.execute(tool_name, arguments)
    return f"Tool {tool_name} returned: {result}"


def run_agent(
    prompt: str,
    *,
    max_steps: int = MAX_STEPS,
    tool_registry: ToolRegistry = DEFAULT_TOOL_REGISTRY,
) -> str:
    """Run the manual decision-action-observation loop for one user prompt."""
    state = AgentState(
        messages=[
            {"role": "system", "content": _build_system_prompt(tool_registry)},
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
            observation = _execute_action(decision, tool_registry)
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
