from typing import Optional

from .core.llm import LLM
from .core.agent import BaseAgent, ReActAgent, PlanAndExecuteAgent
from .core.tool import get_registry, ActionExecutor
from .core.memory import SummarizationMemory

class AgentFramework:
    """
    框架入口，提供链式调用 agent.run()。

    使用示例:
        fw = AgentFramework(llm=OpenAILLM(api_key="..."))
        result = fw.run("What is the weather in Beijing?")
    """

    def __init__(self, llm: LLM, mode: str = "react", max_turns: int = 50):
        """
        Args:
            llm: LLM 实例
            mode: "react" 或 "plan"
            max_turns: 最大循环次数
        """
        self.llm = llm
        self.mode = mode
        self.max_turns = max_turns

    def run(self, task: str) -> str:
        """同步执行任务"""
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