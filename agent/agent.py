"""The manual agent loop and operations shared with the graph workflow."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
import json
import logging
from typing import Any

from agent.state import MAX_STEPS, AgentResult, AgentState
from agent.trace import (
    FinalDecision, RejectedDecision, StepStarted, ToolAttempt, ToolObservation, TraceEvent,
)
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


type ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[Any]]
type TraceCallback = Callable[[TraceEvent], None]


@dataclass(frozen=True)
class ToolAction:
    name: str
    arguments: dict[str, Any]
    signature: str


class ToolExecutionError(Exception):
    """A recoverable tool failure, distinct from a broken MCP connection."""


class StepLimitError(RuntimeError):
    """The agent used its entire decision budget without giving a final answer."""


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
    """Return the answer text for callers that do not need tool metadata."""
    result = await run_agent_with_result(prompt, max_steps=max_steps, tool_registry=tool_registry)
    return result.answer


async def run_agent_with_result(
    prompt: str,
    *,
    max_steps: int = MAX_STEPS,
    tool_registry: ToolRegistry | None = None,
    trace: TraceCallback | None = None,
) -> AgentResult:
    """Discover tools and run the loop, optionally emitting synchronous trace events."""
    validate_max_steps(max_steps)

    if tool_registry is not None:
        async def execute_local(name: str, arguments: dict[str, Any]) -> Any:
            try:
                return tool_registry.execute(name, arguments)
            except Exception as exc:
                raise ToolExecutionError(str(exc)) from exc

        return await _run_loop(prompt, max_steps, tool_registry.model_definitions(), execute_local, trace)

    async with mcp_tools() as (definitions, execute):
        return await _run_loop(prompt, max_steps, definitions, execute, trace)


def validate_max_steps(max_steps: int) -> None:
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")


@asynccontextmanager
async def mcp_tools() -> AsyncIterator[tuple[list[dict[str, Any]], ToolExecutor]]:
    """Discover and execute tools through a single MCP session."""
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

        yield definitions, execute_mcp


def initial_state(prompt: str, max_steps: int, definitions: list[dict[str, Any]]) -> AgentState:
    return AgentState(
        messages=[
            {"role": "system", "content": _build_system_prompt(definitions)},
            {"role": "user", "content": prompt},
        ],
        max_steps=max_steps,
    )


def step_limit_error(state: AgentState) -> StepLimitError:
    return StepLimitError(f"Agent stopped after reaching the maximum of {state.max_steps} steps.")


def _validate_decision(
    state: AgentState, response: str, trace: TraceCallback | None,
) -> ToolAction | str:
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

    name = decision.get("tool")
    arguments = decision.get("arguments")
    if not isinstance(name, str) or not name:
        raise ValueError("An action decision requires a non-empty tool name.")
    if not isinstance(arguments, dict):
        raise ValueError("An action decision requires an arguments object.")

    signature = _tool_call_signature(decision)
    if trace is not None:
        trace(ToolAttempt(state.current_step, name, deepcopy(arguments)))
    if signature in state.successful_tool_calls:
        raise ValueError(
            "This successful tool call was already made; use its existing "
            "evidence or choose another action."
        )
    return ToolAction(name, arguments, signature)


async def decide(state: AgentState, trace: TraceCallback | None) -> ToolAction | str | None:
    """Consume one step and return an action, an answer, or a recorded rejection."""
    state.current_step += 1
    if trace is not None:
        trace(StepStarted(state.current_step))
    response = await asyncio.to_thread(chat, state.messages)
    state.messages.append({"role": "assistant", "content": response})

    try:
        decision = _validate_decision(state, response, trace)
        if isinstance(decision, str) and trace is not None:
            trace(FinalDecision(state.current_step, decision))
    except ValueError as exc:
        observation = _limit_observation(f"Error: {exc}")
        if trace is not None:
            trace(RejectedDecision(state.current_step, _limit_observation(response), observation))
        _add_observation(state, observation)
        return None

    return decision


async def execute_action(
    state: AgentState, action: ToolAction, execute: ToolExecutor, trace: TraceCallback | None,
) -> None:
    """Invoke one tool and feed its bounded result or recoverable error to Qwen."""
    logger.info("[tool call] %s %s", action.name, json.dumps(action.arguments, ensure_ascii=False))
    try:
        result = await execute(action.name, action.arguments)
    except ToolExecutionError as exc:
        succeeded = False
        observation = f"Error: {exc}"
    else:
        succeeded = True
        observation = f"Tool {action.name} returned: {result}"
        state.successful_tool_calls.add(action.signature)
        if action.name not in state.tools_used:
            state.tools_used.append(action.name)

    observation = _limit_observation(observation)
    if trace is not None:
        trace(ToolObservation(state.current_step, observation, succeeded))
    _add_observation(state, observation)


def _add_observation(state: AgentState, observation: str) -> None:
    logger.info("[observation] %s", observation)
    state.observations.append(observation)
    state.messages.append(
        {
            "role": "user",
            "content": f"Observation: {observation}. Decide what to do next.",
        }
    )


async def _run_loop(
    prompt: str,
    max_steps: int,
    definitions: list[dict[str, Any]],
    execute: ToolExecutor,
    trace: TraceCallback | None,
) -> AgentResult:
    state = initial_state(prompt, max_steps, definitions)
    while state.current_step < state.max_steps:
        decision = await decide(state, trace)
        if isinstance(decision, str):
            return AgentResult(answer=decision, tools_used=state.tools_used)
        if decision is not None:
            await execute_action(state, decision, execute, trace)

    raise step_limit_error(state)
