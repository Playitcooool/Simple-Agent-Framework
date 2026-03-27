"""
Framework entry point — AgentFramework class.

Usage:
    fw = AgentFramework(llm=OpenAILLM(api_key="..."))
    @fw.tool(name="weather", description="Get weather")
    def get_weather(city: str) -> str: ...
    result = fw.run("What's the weather in Beijing?")
"""

from .core.llm import LLM
from .core.agent import BaseAgent, ReActAgent, PlanAndExecuteAgent
from .core.tool import get_registry, ActionExecutor
from .core.memory import SummarizationMemory

__all__ = ["AgentFramework"]


class AgentFramework:
    """
    Main entry point — wires up LLM, tools, memory, and agents.

    Usage:
        fw = AgentFramework(llm=OpenAILLM(api_key="..."))
        result = fw.run("What is 6 * 7?")
    """

    def __init__(self, llm: LLM, mode: str = "react", max_turns: int = 50):
        """
        Args:
            llm: LLM instance (OpenAILLM, AnthropicLLM, etc.)
            mode: "react" (default) or "plan"
            max_turns: Safety limit on iterations
        """
        self.llm = llm
        self.mode = mode
        self.max_turns = max_turns

    def run(self, task: str) -> str:
        """Execute a task and return the result."""
        registry = get_registry()
        executor = ActionExecutor(registry)
        memory = SummarizationMemory(llm=self.llm)

        if self.mode == "react":
            agent: BaseAgent = ReActAgent(
                llm=self.llm, executor=executor, memory=memory, max_turns=self.max_turns,
            )
        elif self.mode == "plan":
            agent = PlanAndExecuteAgent(
                llm=self.llm, executor=executor, memory=memory, max_turns=self.max_turns,
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'react' or 'plan'.")

        return agent.run(task)
