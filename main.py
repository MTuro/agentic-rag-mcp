"""Run one user request through the local manual agent loop."""

import logging

from agent import run_agent


def main() -> None:
    """Show tool evidence on stderr and print the final answer on stdout."""
    logging.basicConfig(format="%(message)s")
    logging.getLogger("agent.agent").setLevel(logging.INFO)
    prompt = input("You: ")
    answer = run_agent(prompt)
    print(answer)


if __name__ == "__main__":
    main()
