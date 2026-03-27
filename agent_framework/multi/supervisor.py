# agent_framework/multi/supervisor.py
import re
from typing import List

from ..core.llm import LLM
from ..core.message import Message, MessageRole
from ..core.agent import BaseAgent


class SupervisorAgent:
    """
    主 Agent：接收任务，拆解为子任务，委托给子 Agent 执行，汇总结果。
    """

    def __init__(self, llm: LLM, sub_agents: List[BaseAgent]):
        if not sub_agents:
            raise ValueError("SupervisorAgent has no sub_agents")
        self.llm = llm
        self.sub_agents = {a.name: a for a in sub_agents}

    def run(self, task: str) -> str:
        # 1. LLM 分解任务
        decompose_prompt = (
            f"Break down the following task into subtasks, one per line, "
            f"each starting with 'Subtask N: '.\n"
            f"Task: {task}\n\n"
            f"Subtasks:"
        )
        plan_response = self.llm.generate([
            Message(role=MessageRole.USER, content=decompose_prompt)
        ])

        # 解析子任务行
        subtask_lines = re.findall(
            r"Subtask \d+[:\s]+(.+?)(?:\n|$)", plan_response, re.DOTALL
        )
        if not subtask_lines:
            subtask_lines = [task]

        # 2. 分发给子 Agent
        results: List[str] = []
        for line in subtask_lines:
            agent_name = self._select_agent(line)
            if agent_name not in self.sub_agents:
                raise ValueError(f"No sub-agent found for subtask: {line.strip()}")
            result = self.sub_agents[agent_name].run(line.strip())
            results.append(result)

        # 3. LLM 汇总
        synthesize_prompt = (
            f"Original task: {task}\n\n"
            f"Subtask results:\n" + "\n".join(f"- {r}" for r in results) + "\n\n"
            "Provide the final answer:"
        )
        final = self.llm.generate([Message(role=MessageRole.USER, content=synthesize_prompt)])
        return final

    def _select_agent(self, subtask: str) -> str:
        """简单按子任务关键词选择 Agent"""
        subtask_lower = subtask.lower()
        for name in self.sub_agents:
            if name.lower() in subtask_lower:
                return name
        return list(self.sub_agents.keys())[0] if self.sub_agents else ""