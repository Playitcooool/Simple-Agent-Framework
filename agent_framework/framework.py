"""
Framework entry point.

PURPOSE:
  This module provides the main user-facing interface to the Agent framework.
  The AgentFramework class is the "front door" — it's what most users
  will interact with.

KEY CONCEPTS:
  - Factory pattern: AgentFramework creates the right Agent based on mode
  - Dependency injection: LLM, registry, executor are wired up automatically
  - Simple API: one line to run a task

WHY HAVE A SEPARATE FRAMEWORK CLASS?
  Without this, users would need to:
    1. Import the right Agent class
    2. Create an Executor with the right registry
    3. Create Memory
    4. Wire everything together
    5. Call run()

  With AgentFramework:
    1. Create framework with an LLM
    2. Call run("task")

  The Framework handles all the "wiring" internally.

DESIGN DECISIONS:
  - mode="react" is the default because it's simpler and works well for most tasks
  - mode="plan" is available for complex multi-step tasks
  - max_turns=50 is a safety limit to prevent infinite loops
"""

from .core.llm import LLM
from .core.agent import BaseAgent, ReActAgent, PlanAndExecuteAgent
from .core.tool import get_registry, ActionExecutor
from .core.memory import SummarizationMemory


__all__ = ["AgentFramework"]


class AgentFramework:
    """
    Main entry point for the Agent framework.

    This class provides a simple, high-level API for running agents.
    It handles the boilerplate of setting up components so you don't have to.

    Usage:
        fw = AgentFramework(llm=OpenAILLM(api_key="..."))

        # Register tools (if not already decorated)
        @fw.tool(name="weather", description="Get weather")
        def get_weather(city: str) -> str:
            return f"{city} is sunny"

        # Run a task — returns the final answer
        result = fw.run("What's the weather in Beijing?")

    ATTRIBUTES:
        llm: The LLM used for decision-making
        mode: Execution mode ("react" or "plan")
        max_turns: Safety limit on iterations
    """

    def __init__(self, llm: LLM, mode: str = "react", max_turns: int = 50):
        """
        Initialize the framework.

        Args:
            llm: An LLM instance (e.g., OpenAILLM, AnthropicLLM, etc.)
                 This is the "brain" that will make decisions.
            mode: Execution strategy. Options:
                  - "react": ReAct loop (default, good for most tasks)
                  - "plan": Plan-then-Execute (better for complex multi-step tasks)
            max_turns: Maximum iterations before giving up.
                      Default 50 is usually sufficient. Increase for
                      complex tasks with many tool calls.
        """
        self.llm = llm
        self.mode = mode
        self.max_turns = max_turns

    def run(self, task: str) -> str:
        """
        Execute a task and return the result.

        This is the main API — one call to run() and you get a result.

        Under the hood, this:
          1. Gets the global tool registry
          2. Creates an ActionExecutor
          3. Creates a SummarizationMemory
          4. Creates the appropriate Agent (based on self.mode)
          5. Calls agent.run(task)

        Args:
            task: The user's task/instruction as a string

        Returns:
            The agent's final answer as a string

        Raises:
            ValueError: If mode is not "react" or "plan"
        """
        # Wire up the components
        registry = get_registry()
        executor = ActionExecutor(registry)
        memory = SummarizationMemory(llm=self.llm)

        # Create the appropriate agent based on mode
        # This is a factory pattern — one interface, multiple implementations
        if self.mode == "react":
            agent: BaseAgent = ReActAgent(
                llm=self.llm,
                executor=executor,
                memory=memory,
                max_turns=self.max_turns,
            )
        elif self.mode == "plan":
            agent = PlanAndExecuteAgent(
                llm=self.llm,
                executor=executor,
                memory=memory,
                max_turns=self.max_turns,
            )
        else:
            raise ValueError(
                f"Unknown mode: {self.mode}. Use 'react' or 'plan'."
            )

        # Delegate to the agent
        return agent.run(task)
