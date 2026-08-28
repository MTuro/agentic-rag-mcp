"""Run one user request through the local manual agent loop."""

from agent import run_agent


def main() -> None:
    """Read one prompt, run the agent, and print its final answer."""
    prompt = input("You: ")
    answer = run_agent(prompt)
    print(answer)


if __name__ == "__main__":
    main()
