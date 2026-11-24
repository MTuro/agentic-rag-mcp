"""Run and grade repository tasks against the normal MCP-backed agent."""

from collections import Counter
from dataclasses import dataclass, field
import asyncio
import json
import re
from typing import Literal, TYPE_CHECKING

from agent.agent import MAX_STEPS, TRUNCATION_MARKER, StepLimitError, run_agent_with_result
from agent.graph import run_agent_graph
from agent.trace import RejectedDecision, StepStarted, ToolAttempt, ToolObservation, TraceEvent

if TYPE_CHECKING:
    from rag.vector_store import SearchResult


CODE_PROMPT = "Which Python file defines the retrieve_chunks function? Give its repository-relative path."
DOCS_SOURCE = "documents/README.md"
DOCS_TOP_K = 3

type Engine = Literal["manual", "langgraph"]


@dataclass(frozen=True)
class Case:
    id: str
    prompt: str
    required_tool: str | None = None
    optional_tool: str | None = None
    max_steps: int = MAX_STEPS


CASES = (
    Case("direct", "What is the sum of 2 and 2? Reply with just the number."),
    Case("code", CODE_PROMPT, "search_code", "read_file"),
    Case(
        "docs",
        "According to the indexed project documentation, which distance metric does ChromaDB "
        "use for retrieval, and what does a smaller distance mean?",
        "search_docs",
    ),
    Case(
        "git",
        "Inspect the current short Git working-tree status and return the tool's status text verbatim.",
        "git_status",
    ),
    Case("step-limit", CODE_PROMPT, "search_code", max_steps=1),
)


@dataclass(frozen=True)
class Check:
    description: str
    passed: bool


@dataclass
class Report:
    case: Case
    trace: list[TraceEvent] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    answer: str | None = None
    error: Exception | None = None
    unexpected_attempts: int = 0
    excess_attempts: int = 0
    replay_query: str | None = None
    replay_results: list[SearchResult] = field(default_factory=list)
    replay_hit: int | None = None
    replay_error: Exception | None = None

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def _replay_docs(query: str) -> list[SearchResult]:
    from rag.embeddings import load_embedding_model
    from rag.retriever import retrieve_chunks
    from rag.vector_store import open_collection

    return retrieve_chunks(query, open_collection(), model=load_embedding_model(), top_k=DOCS_TOP_K)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").rstrip("\n")


def _git_matches(answer: str, observation: str) -> bool:
    if observation.endswith(TRUNCATION_MARKER):
        return False
    evidence = observation.removeprefix("Tool git_status returned: ")
    if evidence.startswith("{"):
        try:
            evidence = json.loads(evidence)["result"]
        except (ValueError, KeyError, TypeError):
            return False
        if not isinstance(evidence, str):
            return False
    answer = _normalize_newlines(answer)
    fenced = re.fullmatch(r"[ \t]*```[^\n]*\n(.*?)\n[ \t]*```[ \t]*", answer, re.DOTALL)
    if fenced:
        answer = fenced[1]
    return _normalize_newlines(answer) == _normalize_newlines(evidence)


def _docs_facts(text: str) -> tuple[bool, bool]:
    """Check common positive statements of the two documented facts."""
    text = " ".join(re.sub(r"[*_`]", "", text.casefold()).split())
    metric = re.search(r"\bcosine distance\b", text) is not None
    negated_metric = re.search(
        r"\b(?:not|never|instead of|rather than|without(?: using)?)\s+"
        r"(?:us(?:e|es|ing)\s+)?cosine distance\b"
        r"|\bcosine distance\s+(?:is|was)\s+(?:not|never)\b",
        text,
    )
    distance = r"(?:smaller|lower)\s+(?:cosine\s+)?distance(?:s|\s+(?:value|score)s?)?"
    closer = (
        r"(?:closer\b|more\s+(?:semantically\s+)?similar\b|"
        r"(?:higher|greater|stronger)\s+(?:semantic\s+|cosine\s+)?similarity\b)"
    )
    match = re.search(
        rf"\b{distance}\s+(?:typically\s+)?(?:means?|indicates?|signifies?|implies?|"
        rf"corresponds? to|represents?|suggests?|reflects?)\s+"
        rf"(?:(?:a|the|that the)\s+)?(?:\w+\s+(?:is|are)\s+)?{closer}"
        rf"|\bthe {distance}\s*,?\s+the {closer}",
        text,
    )
    return metric and not negated_metric, match is not None


def _single_error(error: Exception) -> Exception:
    """Unwrap a single stdio task failure while retaining multiple failures."""
    while isinstance(error, ExceptionGroup) and len(error.exceptions) == 1:
        error = error.exceptions[0]
    return error


async def run_case(case: Case, *, engine: Engine = "manual") -> Report:
    report = Report(case)
    run = run_agent_graph if engine == "langgraph" else run_agent_with_result
    try:
        result = await run(case.prompt, max_steps=case.max_steps, trace=report.trace.append)
        report.answer = result.answer
    except Exception as exc:
        report.error = _single_error(exc)

    attempts = [event for event in report.trace if isinstance(event, ToolAttempt)]
    observations = {
        event.step: event.observation
        for event in report.trace
        if isinstance(event, ToolObservation) and event.succeeded
    }
    counts = Counter(attempt.name for attempt in attempts)
    allowed = {name for name in (case.required_tool, case.optional_tool) if name is not None}
    report.unexpected_attempts = sum(count for name, count in counts.items() if name not in allowed)
    report.excess_attempts = sum(max(0, counts[name] - 1) for name in allowed)
    report.checks.extend([
        Check("No unexpected tool attempts", report.unexpected_attempts == 0),
        Check("No attempts above the allowed count", report.excess_attempts == 0),
    ])

    successful = [
        attempt for attempt in attempts
        if attempt.name == case.required_tool and attempt.step in observations
    ]
    if case.required_tool:
        report.checks.append(Check(f"Exactly one successful {case.required_tool} call", len(successful) == 1))
    if case.id == "step-limit":
        report.checks.append(Check("Stopped with the typed step-limit error", isinstance(report.error, StepLimitError)))
        return report

    report.checks.append(Check("Completed with a final answer", report.error is None and report.answer is not None))
    answer = report.answer or ""
    if case.id == "direct":
        report.checks.append(Check("Answer is 4", answer.strip() == "4"))
    elif case.id == "code":
        has_path = re.search(r"(?<![\w./-])rag/retriever\.py(?![\w/-]|\.\w)", answer)
        report.checks.append(Check("Answer identifies rag/retriever.py", has_path is not None))
    elif case.id == "git":
        report.checks.append(Check(
            "Answer matches the observed short status",
            len(successful) == 1 and _git_matches(answer, observations[successful[0].step]),
        ))
    elif case.id == "docs":
        metric, meaning = _docs_facts(answer)
        report.checks.extend([
            Check("Answer identifies cosine distance", metric),
            Check("Answer says smaller distances indicate closer matches", meaning),
        ])
        if successful:
            report.replay_query = successful[0].arguments["query"]
            try:
                report.replay_results = await asyncio.to_thread(_replay_docs, report.replay_query)
            except Exception as exc:
                report.replay_error = exc
            for rank, result in enumerate(report.replay_results, 1):
                if result.chunk.metadata.get("source") == DOCS_SOURCE and all(_docs_facts(result.chunk.content)):
                    report.replay_hit = rank
                    break
        report.checks.append(Check(f"Top-{DOCS_TOP_K} replay found both facts in {DOCS_SOURCE}", report.replay_hit is not None))
    return report


def print_report(report: Report) -> None:
    steps = sum(isinstance(event, StepStarted) for event in report.trace)
    status = "PASS" if report.passed else "FAIL"
    print(f"\n[{status}] {report.case.id} ({steps} steps used)")
    print(f"  Prompt: {report.case.prompt}")
    if report.error is not None:
        print(f"  Terminal error: {type(report.error).__name__}: {report.error}")
    if report.answer is not None:
        print(f"  Answer: {report.answer}")
    print(f"  Unexpected attempts: {report.unexpected_attempts}; excess attempts: {report.excess_attempts}")

    for event in report.trace:
        if isinstance(event, ToolAttempt):
            print(f"  Step {event.step} attempted {event.name} {json.dumps(event.arguments, ensure_ascii=False)}")
        elif isinstance(event, ToolObservation):
            outcome = "succeeded" if event.succeeded else "failed"
            print(f"  Step {event.step} {outcome}: {event.observation}")
        elif isinstance(event, RejectedDecision):
            print(f"  Step {event.step} rejected {event.decision!r}: {event.observation}")

    if report.replay_query is not None:
        print(f"  Top-{DOCS_TOP_K} retrieval replay for {report.replay_query!r}")
        if report.replay_error is not None:
            print(f"    Replay error: {type(report.replay_error).__name__}: {report.replay_error}")
        elif not report.replay_results:
            print("    No chunks retrieved")
        for rank, result in enumerate(report.replay_results, 1):
            source = result.chunk.metadata.get("source", "(unknown source)")
            support = " [supports both facts]" if rank == report.replay_hit else ""
            print(f"    Rank {rank}: {source} ({result.chunk_id}, distance {result.distance:.4f}){support}")
            print(f"      {result.chunk.content.strip()!r}")

    for check in report.checks:
        print(f"  {'PASS' if check.passed else 'FAIL'} {check.description}")
