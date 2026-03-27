"""
Example 01: Basic Agent Framework Usage

A minimal example showing how to:
1. Initialize AgentFramework with an LLM
2. Register a tool using @tool decorator
3. Run a task
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_framework import AgentFramework, OpenAILLM, tool

# Check for API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Please set OPENAI_API_KEY environment variable")
    sys.exit(1)

# Initialize framework with OpenAI LLM
fw = AgentFramework(llm=OpenAILLM(api_key=api_key), mode="react")


# Register a tool — the decorator registers it with the global registry
@fw.tool(
    name="calculator",
    description="Perform basic math calculation. Args: expression (str), e.g. '2 + 2' or '10 * 5'"
)
def calc(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    try:
        # WARNING: eval is used here for simplicity — never use in production!
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@fw.tool(
    name="weather",
    description="Get weather for a city. Args: city (str)"
)
def weather(city: str) -> str:
    """Return weather for a city (simulated)."""
    return f"{city}: Sunny, 25°C"


def main():
    print("=" * 60)
    print("Example 01: Basic Usage")
    print("=" * 60)

    # Example 1: Simple math
    print("\n[1] Asking: What is 123 + 456?")
    result = fw.run("Calculate 123 + 456 using the calculator tool")
    print(f"Agent: {result}")

    # Example 2: Weather query
    print("\n[2] Asking: What's the weather in Beijing?")
    result = fw.run("What's the weather in Beijing? Use the weather tool.")
    print(f"Agent: {result}")

    # Example 3: Multi-step
    print("\n[3] Asking: Calculate 100 * 5, then tell me the result in Chinese")
    result = fw.run(
        "First calculate 100 times 5 using the calculator tool, "
        "then tell me the result in Chinese"
    )
    print(f"Agent: {result}")


if __name__ == "__main__":
    main()
