"""Evaluate live local model behavior through the MCP-backed agent."""

import argparse
import asyncio
from collections.abc import Sequence

from evals.runner import CASES, Case, Engine, print_report, run_case
from llm.ollama_client import selected_model


async def _run(cases: Sequence[Case], *, engine: Engine = "manual") -> int:
    print(f"Agent engine: {engine}")
    print(f"Ollama model: {selected_model()}")
    passed = 0
    for case in cases:
        report = await run_case(case, engine=engine)
        print_report(report)
        passed += report.passed
    print(f"\n{passed}/{len(cases)} cases passed")
    print(
        "Human review: Compare each factual claim, file path, and quotation with the "
        "observations and any retrieved excerpts. Flag unsupported claims, mismatched paths, "
        "and inaccurate quotations even when the automated checks pass."
    )
    return int(passed != len(cases))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case.id for case in CASES], help="Run only this case.")
    parser.add_argument("--engine", choices=("langgraph", "manual"), default="manual")
    args = parser.parse_args(argv)
    cases = [case for case in CASES if args.case is None or case.id == args.case]
    return asyncio.run(_run(cases, engine=args.engine))


if __name__ == "__main__":
    raise SystemExit(main())
