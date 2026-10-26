"""A small, explicit agent loop built without an agent framework."""

import asyncio
from collections.abc import Awaitable, Callable
import json
import logging
from typing import Any

from agent.state import MAX_STEPS, AgentState
from llm.ollama_client import chat
from mcp_client.client import connect
from tools.registry import ToolRegistry


BASE_SYSTEM_PROMPT = """You are an educational agent that must reply with one JSON object only.

Choose exactly one of these decisions:
1. Give the final answer:
   {"type": "final", "answer": "your answer"}
2. Call one available tool:
   {"type": "action", "tool": "tool_name", "arguments": {"argument_name": "value"}}

Choose tools from the definitions below. Use each tool's argument schema to build
the arguments object. After receiving an observation, use it to decide whether to
call another tool or produce a final answer. Do not repeat a successful tool call.
As soon as the available evidence is sufficient, return a final answer instead of
calling more tools. Do not wrap JSON in Markdown.

For requests about repository files or the current implementation, use tools to
inspect the requested files before making factual claims about their contents.
Tool definitions describe capabilities; they are not evidence of file contents.
Complete every requested inspection, listing, or comparison before answering.
When comparing documentation with code, inspect both and cite their file paths
and short excerpts from the observations. Never invent excerpts or claim to have
read a file without a successful observation. If evidence is missing, a tool
fails, or a result is truncated, state the limitation instead of guessing.
General questions that do not require repository evidence may be answered directly.
"""

logger = logging.getLogger(__name__)

MAX_OBSERVATION_LENGTH = 8_000
TRUNCATION_MARKER = "\n[tool result truncated]"


class ToolExecutionError(Exception):
    """A recoverable tool failure, distinct from a broken MCP connection."""


def _build_system_prompt(definitions: list[dict[str, Any]]) -> str:
    """Build instructions from the currently discovered or injected tools."""
    return f"{BASE_SYSTEM_PROMPT}\nAvailable tools:\n{json.dumps(definitions, indent=2)}"


def _parse_decision(response: str) -> dict[str, Any]:
    """Parse one JSON object returned by the language model."""
    try:
        decision = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError("The model returned invalid JSON.") from exc

    if not isinstance(decision, dict):
        raise ValueError("The model decision must be a JSON object.")

    return decision


def _tool_call_signature(decision: dict[str, Any]) -> str:
    """Return a stable identity for one validated tool call."""
    return json.dumps(
        [decision["tool"], decision["arguments"]],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _limit_observation(observation: str) -> str:
    """Bound evidence sent back to the model while preserving a clear marker."""
    if len(observation) <= MAX_OBSERVATION_LENGTH:
        return observation
    return observation[: MAX_OBSERVATION_LENGTH - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER


def run_agent(
    prompt: str,
    *,
    max_steps: int = MAX_STEPS,
    tool_registry: ToolRegistry | None = None,
) -> str:
    """Run synchronously; async callers should await ``run_agent_async``."""
    return asyncio.run(run_agent_async(prompt, max_steps=max_steps, tool_registry=tool_registry))


async def run_agent_async(
    prompt: str,
    *,
    max_steps: int = MAX_STEPS,
    tool_registry: ToolRegistry | None = None,
) -> str:
    """Discover MCP tools by default, or use an explicitly supplied local registry."""
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")

    if tool_registry is not None:
        async def execute_local(name: str, arguments: dict[str, Any]) -> Any:
            try:
                return tool_registry.execute(name, arguments)
            except Exception as exc:
                raise ToolExecutionError(str(exc)) from exc

        return await _run_loop(prompt, max_steps, tool_registry.model_definitions(), execute_local)

    async with connect() as client:
        discovery = await client.list_tools()
        definitions = [
            {"name": tool.name, "description": tool.description, "arguments": tool.input_schema}
            for tool in discovery.tools
        ]

        async def execute_mcp(name: str, arguments: dict[str, Any]) -> str:
            result = await client.call_tool(name, arguments)
            if result.structured_content is not None:
                evidence = json.dumps(result.structured_content, ensure_ascii=False)
            else:
                evidence = "\n".join(block.text for block in result.content if block.type == "text")
            if result.is_error:
                raise ToolExecutionError(evidence)
            return evidence

        return await _run_loop(prompt, max_steps, definitions, execute_mcp)


async def _run_loop(
    prompt: str,
    max_steps: int,
    definitions: list[dict[str, Any]],
    execute: Callable[[str, dict[str, Any]], Awaitable[Any]],
) -> str:
    """Share the same decisions and safeguards across both execution paths."""
    state = AgentState(
        messages=[
            {"role": "system", "content": _build_system_prompt(definitions)},
            {"role": "user", "content": prompt},
        ],
        max_steps=max_steps,
    )

    while state.current_step < state.max_steps:
        state.current_step += 1
        response = await asyncio.to_thread(chat, state.messages)
        state.messages.append({"role": "assistant", "content": response})
        try:
            decision = _parse_decision(response)
            decision_type = decision.get("type")

            if decision_type == "final":
                answer = decision.get("answer")
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError(
                        'A final decision requires a non-empty answer. Return '
                        '{"type": "final", "answer": "your answer"} with your answer as a string.'
                    )
                return answer

            if decision_type != "action":
                raise ValueError(f"Unknown decision type: {decision_type!r}.")

            tool_name = decision.get("tool")
            arguments = decision.get("arguments")
            if not isinstance(tool_name, str) or not tool_name:
                raise ValueError("An action decision requires a non-empty tool name.")
            if not isinstance(arguments, dict):
                raise ValueError("An action decision requires an arguments object.")

            signature = _tool_call_signature(decision)
            if signature in state.successful_tool_calls:
                raise ValueError(
                    "This successful tool call was already made; use its existing "
                    "evidence or choose another action."
                )

        except ValueError as exc:
            observation = f"Error: {exc}"
        else:
            logger.info("[tool call] %s %s", tool_name, json.dumps(arguments, ensure_ascii=False))
            try:
                result = await execute(tool_name, arguments)
            except ToolExecutionError as exc:
                observation = f"Error: {exc}"
            else:
                observation = f"Tool {tool_name} returned: {result}"
                state.successful_tool_calls.add(signature)

        observation = _limit_observation(observation)
        logger.info("[observation] %s", observation)
        state.observations.append(observation)
        state.messages.append(
            {
                "role": "user",
                "content": f"Observation: {observation}. Decide what to do next.",
            }
        )

    raise RuntimeError(f"Agent stopped after reaching the maximum of {state.max_steps} steps.")
