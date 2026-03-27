from .core.llm import LLM
from .core.agent import BaseAgent, ReActAgent, PlanAndExecuteAgent
from .core.tool import get_registry, ActionExecutor
from .core.memory import SummarizationMemory

class AgentFramework:
    """
    Framework entry point providing chainable agent.run() interface.

    Usage example:
        fw = AgentFramework(llm=OpenAILLM(api_key="..."))
        result = fw.run("What is the weather in Beijing?")
    """

    def __init__(self, llm: LLM, mode: str = "react", max_turns: int = 50):
        """
        Args:
            llm: LLM instance
            mode: "react" or "plan"
            max_turns: Maximum number of loop iterations
        """
        self.llm = llm
        self.mode = mode
        self.max_turns = max_turns

    def run(self, task: str) -> str:
        """Synchronously execute a task"""
        registry = get_registry()
        executor = ActionExecutor(registry)
        memory = SummarizationMemory(llm=self.llm)

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
            raise ValueError(f"Unknown mode: {self.mode}. Use 'react' or 'plan'.")

        return agent.run(task)