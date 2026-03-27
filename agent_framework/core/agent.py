# agent_framework/core/agent.py
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
    从 LLM 输出中解析 Thought / Action / Action Args。
    返回 (thought, action_name, action_args)
    """
    thought_match = re.search(r"Thought[:\s]*(.+?)(?=\n(?:Action|Final Answer)|$)", text, re.DOTALL | re.IGNORECASE)
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
        except Exception:
            action_args = {}

    return thought, action_name, action_args

class BaseAgent(ABC):
    """Agent 基类"""

    def __init__(
        self,
        llm: LLM,
        executor: ActionExecutor,
        memory: Optional[SummarizationMemory] = None,
        max_turns: int = MAX_TURNS_DEFAULT,
    ):
        self.llm = llm
        self.executor = executor
        self.memory = memory or SummarizationMemory(llm=llm)
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
            "Action Args: {\"arg1\": \"value1\"}\n\n"
            "Available tools:\n"
            f"{tool_desc}\n\n"
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
    """ReAct 模式的 Agent: Thought → Action → Observation 循环"""

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
                # 无法解析出 action，直接返回
                self.memory.add(Message(role=MessageRole.ASSISTANT, content=response))
                return response

        return "Max turns reached without final answer."

class PlanAndExecuteAgent(BaseAgent):
    """Plan-and-Execute 模式: 先规划，再执行"""

    def run(self, task: str) -> str:
        messages = self._get_messages(task)

        # 1. 规划阶段
        plan_response = self.llm.generate(messages)

        # 解析步骤
        step_matches = re.findall(r"Step \d+[:\s]*(.+?)(?=(?:Step \d+)|$)", plan_response, re.DOTALL | re.IGNORECASE)
        steps = [s.strip() for s in step_matches if s.strip()]

        if not steps:
            # 无法解析计划，退化为直接执行
            steps = [task]

        # 2. 执行阶段
        execution_results: List[str] = []
        messages.append(Message(role=MessageRole.ASSISTANT, content=plan_response))

        for step in steps:
            step_msg = Message(role=MessageRole.USER, content=f"Execute this step: {step}")
            step_response = self.llm.generate(messages + [step_msg])
            messages.append(Message(role=MessageRole.ASSISTANT, content=step_response))

            # 尝试从响应中提取工具调用
            _, action, args = _parse_thought_output(step_response)
            if action and action != "FINAL_ANSWER":
                result = self.executor.run(action, args or {})
                messages.append(Message(role=MessageRole.TOOL_RESULT, content=result))
                execution_results.append(result)
            else:
                execution_results.append(step_response)

        # 3. 汇总
        synthesize_prompt = (
            f"Original task: {task}\n"
            f"Execution results:\n" + "\n".join(f"- {r}" for r in execution_results)
        )
        final = self.llm.generate([Message(role=MessageRole.USER, content=synthesize_prompt)])
        return final