"""Explicit state carried through the manual agent loop."""

from dataclasses import dataclass, field


MAX_STEPS = 5


@dataclass
class AgentState:
    """Information the agent retains between language-model decisions."""

    messages: list[dict[str, str]]
    current_step: int = 0
    max_steps: int = MAX_STEPS
    observations: list[str] = field(default_factory=list)
