"""Optional events emitted by one run of the manual agent loop."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StepStarted:
    step: int


@dataclass(frozen=True)
class RejectedDecision:
    step: int
    decision: str
    observation: str


@dataclass(frozen=True)
class ToolAttempt:
    step: int
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolObservation:
    step: int
    observation: str
    succeeded: bool


@dataclass(frozen=True)
class FinalDecision:
    step: int
    answer: str


type TraceEvent = StepStarted | RejectedDecision | ToolAttempt | ToolObservation | FinalDecision
