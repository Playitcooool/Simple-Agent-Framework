"""
Multi-agent module — SupervisorAgent for hierarchical collaboration.

SupervisorAgent:
1. Decomposes task via LLM
2. Delegates subtasks to sub-agents
3. Synthesizes results via LLM
"""

import re
from typing import List

from ..core.llm import LLM
from ..core.message import Message, MessageRole
from ..core.agent import BaseAgent

__all__ = ["SupervisorAgent"]


class SupervisorAgent:
    """
    Master agent that orchestrates sub-agents.

    Flow:
        1. LLM decomposes task into subtasks
        2. Each subtask dispatched to appropriate sub-agent
        3. LLM synthesizes all results into final answer
    """

    def __init__(self, llm: LLM, sub_agents: List[BaseAgent]):
        if not sub_agents:
            raise ValueError("SupervisorAgent requires at least one sub-agent")
        self.llm = llm
        self.sub_agents = {a.name: a for a in sub_agents}

    def run(self, task: str) -> str:
        # Phase 1: Decomposition
        decompose_prompt = (
            f"Break down the following task into subtasks, one per line, "
            f"each starting with 'Subtask N: '.\nTask: {task}\n\nSubtasks:"
        )
        plan_response = self.llm.generate([
            Message(role=MessageRole.USER, content=decompose_prompt)
        ])

        subtask_lines = re.findall(
            r"Subtask \d+[:\s]+(.+?)(?:\n|$)", plan_response, re.DOTALL
        )
        if not subtask_lines:
            subtask_lines = [task]

        # Phase 2: Dispatch to sub-agents
        results: List[str] = []
        for line in subtask_lines:
            agent_name = self._select_agent(line)
            if agent_name not in self.sub_agents:
                raise ValueError(f"No sub-agent found for subtask: {line.strip()}")
            result = self.sub_agents[agent_name].run(line.strip())
            results.append(result)

        # Phase 3: Synthesis
        synthesize_prompt = (
            f"Original task: {task}\n\nSubtask results:\n" +
            "\n".join(f"- {r}" for r in results) +
            "\n\nProvide the final answer:"
        )
        return self.llm.generate([Message(role=MessageRole.USER, content=synthesize_prompt)])

    def _select_agent(self, subtask: str) -> str:
        """Select agent by keyword matching agent name in subtask text."""
        subtask_lower = subtask.lower()
        for name in self.sub_agents:
            if name.lower() in subtask_lower:
                return name
        return list(self.sub_agents.keys())[0]
