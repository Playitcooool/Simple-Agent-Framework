"""
Agent module — core ReAct and Plan-and-Execute implementations.

Components:
- _parse_thought_output: parses LLM text into (thought, action, args)
- BaseAgent: abstract base with shared logic
- ReActAgent: Thought -> Action -> Observation loop
- PlanAndExecuteAgent: Plan -> Execute steps -> Synthesize
"""

import json
import re
from abc import ABC, abstractmethod
from typing import List, Optional

from .llm import LLM
from .message import Message, MessageRole
from .executor import ActionExecutor
from .memory import SummarizationMemory

MAX_TURNS_DEFAULT = 50


def _parse_thought_output(text: str) -> tuple[Optional[str], Optional[str], Optional[dict]]:
    """
    Parse LLM output for Thought/Action/Action Args.

    Expected format:
        Thought: I need to check weather
        Action: get_weather
        Action Args: {"city": "Beijing"}

    Or for final answer:
        Thought: I have enough info
        Final Answer: Beijing is sunny
    """
    thought_match = re.search(
        r"Thought[:\s]*(.+?)(?=\n(?:Action|Final Answer)|$)",
        text, re.DOTALL | re.IGNORECASE
    )
    thought = thought_match.group(1).strip() if thought_match else ""

    if re.search(r"Final Answer[:\s]*(.+)", text, re.DOTALL | re.IGNORECASE):
        return thought, "FINAL_ANSWER", None

    action_match = re.search(r"Action[:\s]*(.+?)(?:\n|$)", text, re.IGNORECASE)
    args_match = re.search(r"Action Args[:\s]*(.+)", text, re.IGNORECASE)

    action_name = action_match.group(1).strip() if action_match else None
    action_args = None
    if args_match:
        try:
            action_args = json.loads(args_match.group(1).strip())
        except json.JSONDecodeError:
            action_args = {}

    return thought, action_name, action_args


class BaseAgent(ABC):
    """Base class for all agents — shared wiring and utilities."""

    def __init__(self, llm: LLM, executor: ActionExecutor,
                 memory: Optional[SummarizationMemory] = None,
                 max_turns: int = MAX_TURNS_DEFAULT):
        self.llm = llm
        self.executor = executor
        self.memory = memory or SummarizationMemory(llm=self.llm)
        self.max_turns = max_turns

    @abstractmethod
    def run(self, task: str) -> str:
        pass

    def _build_system_prompt(self) -> str:
        tools = self.executor.registry.list_tools()
        tool_desc = "\n".join(f"- {t.name}: {t.description}" for t in tools)
        return (
            "You are a helpful AI agent.\n"
            "When you need to use tools, respond with:\n"
            "Thought: ...\n"
            "Action: tool_name\n"
            'Action Args: {"arg1": "value1"}\n\n'
            f"Available tools:\n{tool_desc}\n\n"
            "If you have enough information, respond with:\n"
            "Thought: ...\n"
            "Final Answer: ..."
        )

    def _get_messages(self, task: str) -> List[Message]:
        messages = [Message(role=MessageRole.SYSTEM, content=self._build_system_prompt())]
        messages.extend(self.memory.get_messages())
        messages.append(Message(role=MessageRole.USER, content=task))
        return messages


class ReActAgent(BaseAgent):
    """
    ReAct loop agent.

    Flow:
        while not done:
            1. LLM thinks and decides action (or final answer)
            2. If final answer -> return
            3. If action -> execute via executor, add result to messages
    """

    def run(self, task: str) -> str:
        messages = self._get_messages(task)
        turns = 0

        while turns < self.max_turns:
            turns += 1
            response = self.llm.generate(messages)
            _, action, args = _parse_thought_output(response)

            if action == "FINAL_ANSWER":
                match = re.search(r"Final Answer[:\s]*(.+)", response, re.DOTALL | re.IGNORECASE)
                answer = match.group(1).strip() if match else response
                self.memory.add(Message(role=MessageRole.ASSISTANT, content=answer))
                return answer

            if action and action != "FINAL_ANSWER":
                result = self.executor.run(action, args or {})
                messages.append(Message(role=MessageRole.ASSISTANT, content=response))
                messages.append(Message(role=MessageRole.TOOL_RESULT, content=result))
            else:
                self.memory.add(Message(role=MessageRole.ASSISTANT, content=response))
                return response

        return "Max turns reached without final answer."


class PlanAndExecuteAgent(BaseAgent):
    """
    Plan-then-Execute agent.

    Flow:
        Phase 1: LLM plans steps (e.g. "Step 1: ...\nStep 2: ...")
        Phase 2: Execute each step (may involve tools)
        Phase 3: LLM synthesizes results into final answer
    """

    def run(self, task: str) -> str:
        messages = self._get_messages(task)

        # Phase 1: Planning
        plan_response = self.llm.generate(messages)
        step_matches = re.findall(
            r"Step \d+[:\s]*(.+?)(?=(?:Step \d+)|$)",
            plan_response, re.DOTALL | re.IGNORECASE
        )
        steps = [s.strip() for s in step_matches if s.strip()]
        if not steps:
            steps = [task]

        # Phase 2: Execution
        messages.append(Message(role=MessageRole.ASSISTANT, content=plan_response))
        execution_results: List[str] = []

        for step in steps:
            step_msg = Message(role=MessageRole.USER, content=f"Execute this step: {step}")
            step_response = self.llm.generate(messages + [step_msg])
            messages.append(Message(role=MessageRole.ASSISTANT, content=step_response))

            _, action, args = _parse_thought_output(step_response)
            if action and action != "FINAL_ANSWER":
                result = self.executor.run(action, args or {})
                messages.append(Message(role=MessageRole.TOOL_RESULT, content=result))
                execution_results.append(result)
            else:
                execution_results.append(step_response)

        # Phase 3: Synthesis
        synthesize_prompt = (
            f"Original task: {task}\n"
            f"Execution results:\n" + "\n".join(f"- {r}" for r in execution_results)
        )
        return self.llm.generate([Message(role=MessageRole.USER, content=synthesize_prompt)])
