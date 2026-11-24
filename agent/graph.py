"""Run the agent's decision and tool steps as a local LangGraph workflow."""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context

from agent.agent import (
    ToolAction,
    TraceCallback,
    decide,
    execute_action,
    initial_state,
    mcp_tools,
    step_limit_error,
    validate_max_steps,
)
from agent.state import MAX_STEPS, AgentResult, AgentState


class GraphState(TypedDict):
    agent: AgentState
    pending_action: ToolAction | None
    answer: str | None


def after_observation(state: GraphState) -> Literal["decide", "__end__"]:
    agent = state["agent"]
    return "decide" if agent.current_step < agent.max_steps else END


def after_decision(state: GraphState) -> Literal["execute_tool", "decide", "__end__"]:
    if state["answer"] is not None:
        return END
    if state["pending_action"] is not None:
        return "execute_tool"
    return after_observation(state)


async def run_agent_graph(
    prompt: str,
    *,
    max_steps: int = MAX_STEPS,
    trace: TraceCallback | None = None,
) -> AgentResult:
    """Run one request with one discovered MCP catalog and one session."""
    validate_max_steps(max_steps)
    async with mcp_tools() as (definitions, execute):
        async def qwen_decision(state: GraphState) -> GraphState:
            agent = state["agent"]
            decision = await decide(agent, trace)
            return {
                "agent": agent,
                "pending_action": decision if isinstance(decision, ToolAction) else None,
                "answer": decision if isinstance(decision, str) else None,
            }

        async def mcp_execution(state: GraphState) -> GraphState:
            action = state["pending_action"]
            assert action is not None
            agent = state["agent"]
            await execute_action(agent, action, execute, trace)
            return {"agent": agent, "pending_action": None, "answer": None}

        workflow = StateGraph(GraphState)
        workflow.add_node("decide", qwen_decision)
        workflow.add_node("execute_tool", mcp_execution)
        workflow.add_edge(START, "decide")
        workflow.add_conditional_edges("decide", after_decision)
        workflow.add_conditional_edges("execute_tool", after_observation)

        state: GraphState = {
            "agent": initial_state(prompt, max_steps, definitions),
            "pending_action": None,
            "answer": None,
        }
        # A Qwen decision can schedule one more node for its tool call.
        # The application budget counts Qwen decisions, while LangGraph counts nodes.
        with tracing_context(enabled=False):
            result = await workflow.compile().ainvoke(
                state, config={"recursion_limit": 2 * max_steps + 1},
            )

    answer = result["answer"]
    if answer is None:
        raise step_limit_error(result["agent"])
    return AgentResult(answer=answer, tools_used=result["agent"].tools_used)
