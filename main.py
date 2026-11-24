"""Run one user request through the local agent."""

import argparse
import asyncio
from collections.abc import Sequence
import logging

from agent import run_agent
from agent.graph import run_agent_graph


def main(argv: Sequence[str] | None = None) -> None:
    """Show tool evidence on stderr and print the final answer on stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("langgraph", "manual"), default="langgraph")
    args = parser.parse_args(argv)
    logging.basicConfig(format="%(message)s")
    logging.getLogger("agent.agent").setLevel(logging.INFO)
    prompt = input("You: ")
    if args.engine == "manual":
        answer = run_agent(prompt)
    else:
        answer = asyncio.run(run_agent_graph(prompt)).answer
    print(answer)


if __name__ == "__main__":
    main()
