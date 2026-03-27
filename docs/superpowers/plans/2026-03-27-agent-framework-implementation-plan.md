# Agent Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零实现一个轻量级 Agent 开发框架，支持 ReAct 和 Plan-and-Execute 双模式、装饰器工具注册、摘要记忆、层级委托多 Agent。

**Architecture:** 纯 Python 无外部框架依赖。核心模块 `core/` 包含 LLM 接口、消息结构、工具系统、记忆系统和 Agent 实现。`multi/supervisor.py` 提供层级委托多 Agent 能力。入口 `framework.py` 提供链式调用 `agent.run()`。

**Tech Stack:** Python 标准库 + `requests`（LLM API 调用）

---

## 文件结构

```
agent_framework/
├── __init__.py
├── framework.py           # 入口：AgentFramework 类
├── core/
│   ├── __init__.py
│   ├── llm.py            # LLM 抽象基类 + OpenAI Adapter
│   ├── message.py        # Message / MessageRole
│   ├── tool.py           # @tool 装饰器 + ToolRegistry + ActionExecutor
│   ├── memory.py         # SummarizationMemory
│   ├── executor.py       # ActionExecutor
│   └── agent.py          # BaseAgent + ReActAgent + PlanAndExecuteAgent
└── multi/
    ├── __init__.py
    └── supervisor.py     # SupervisorAgent
tests/
├── __init__.py
├── test_message.py
├── test_tool.py
├── test_memory.py
├── test_executor.py
├── test_agent.py
└── test_supervisor.py
```

---

## Task 1: 消息结构 (core/message.py)

**Files:**
- Create: `agent_framework/core/message.py`
- Test: `tests/test_message.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_message.py
from agent_framework.core.message import Message, MessageRole

def test_message_creation():
    msg = Message(role=MessageRole.USER, content="Hello")
    assert msg.role == MessageRole.USER
    assert msg.content == "Hello"
    assert msg.tool_call is None

def test_message_with_tool_call():
    msg = Message(
        role=MessageRole.ASSISTANT,
        content="I'll check the weather",
        tool_call={"name": "weather", "args": {"city": "Beijing"}}
    )
    assert msg.tool_call["name"] == "weather"
    assert msg.tool_call["args"]["city"] == "Beijing"
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_message.py -v
```
Expected: FAIL — module not found

- [ ] **Step 3: 实现 message.py**

```python
# agent_framework/core/message.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"

@dataclass
class Message:
    role: MessageRole
    content: str
    tool_call: Optional[dict] = None
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_message.py -v
```
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git init && git add -A && git commit -m "feat: add Message and MessageRole"
```

---

## Task 2: LLM 接口 (core/llm.py)

**Files:**
- Create: `agent_framework/core/llm.py`
- Test: `tests/test_llm.py`

- [ ] **Step 1: 写测试（mock LLM）**

```python
# tests/test_llm.py
import pytest
from agent_framework.core.llm import LLM, OpenAILLM
from agent_framework.core.message import Message, MessageRole

def test_llm_is_abstract():
    with pytest.raises(TypeError):
        LLM()

class FakeLLM(LLM):
    def __init__(self, response: str):
        self.response = response
        self.called_with = None

    def generate(self, messages):
        self.called_with = messages
        return self.response

def test_openai_llm_init():
    llm = OpenAILLM(api_key="test-key", model="gpt-4o")
    assert llm.api_key == "test-key"
    assert llm.model == "gpt-4o"
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_llm.py -v
```
Expected: FAIL — module not found

- [ ] **Step 3: 实现 llm.py**

```python
# agent_framework/core/llm.py
from abc import ABC, abstractmethod
from typing import List
import requests

from .message import Message

DEFAULT_OPENAI_URL = "https://api.openai.com/v1/chat/completions"

class LLM(ABC):
    """LLM 抽象基类"""

    @abstractmethod
    def generate(self, messages: List[Message]) -> str:
        """给定消息列表，返回 LLM 生成的文本"""
        pass

class OpenAILLM(LLM):
    """OpenAI Chat Completions API"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        api_url: str = DEFAULT_OPENAI_URL,
        timeout: int = 60,
    ):
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.timeout = timeout

    def generate(self, messages: List[Message]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
        }
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_llm.py -v
```
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: add LLM abstract class and OpenAILLM adapter"
```

---

## Task 3: 工具系统 (core/tool.py + core/executor.py)

**Files:**
- Create: `agent_framework/core/tool.py`
- Create: `agent_framework/core/executor.py`
- Test: `tests/test_tool.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_tool.py
from agent_framework.core.tool import tool, ToolRegistry, ActionExecutor

registry = ToolRegistry()

def test_tool_decorator_basic():
    @tool(name="test_tool", description="A test tool")
    def my_tool(x: int, y: int) -> str:
        return str(x + y)

    assert "test_tool" in registry.tools
    t = registry.tools["test_tool"]
    assert t.description == "A test tool"
    assert t.fn is my_tool

def test_executor_run():
    @tool(name="add", description="Add two numbers")
    def add(a: int, b: int) -> str:
        return str(a + b)

    executor = ActionExecutor(registry)
    result = executor.run("add", {"a": 3, "b": 5})
    assert result == "8"

def test_executor_unknown_tool():
    executor = ActionExecutor(registry)
    with pytest.raises(ValueError, match="Unknown tool"):
        executor.run("nonexistent", {})
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_tool.py -v
```
Expected: FAIL — module not found

- [ ] **Step 3: 实现 tool.py**

```python
# agent_framework/core/tool.py
from typing import Callable, Any, Optional
from dataclasses import dataclass

@dataclass
class Tool:
    name: str
    description: str
    fn: Callable

class ToolRegistry:
    """全局工具注册表"""

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, fn: Callable) -> None:
        self.tools[name] = Tool(name=name, description=description, fn=fn)

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self.tools.values())

# 全局单例
_global_registry = ToolRegistry()

def tool(name: str = None, description: str = None):
    """装饰器：把函数注册为工具"""
    def decorator(fn: Callable) -> Callable:
        tool_name = name or fn.__name__
        tool_desc = description or fn.__doc__ or ""
        _global_registry.register(tool_name, tool_desc.strip(), fn)
        return fn
    return decorator

# 暴露全局注册表
def get_registry() -> ToolRegistry:
    return _global_registry
```

- [ ] **Step 4: 实现 executor.py**

```python
# agent_framework/core/executor.py
from typing import Any
from .tool import ToolRegistry, Tool

class ActionExecutor:
    """执行工具调用的组件"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def run(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        t = self.registry.get(tool_name)
        if t is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        try:
            result = t.fn(**tool_args)
            return str(result)
        except Exception as e:
            return f"Error executing tool {tool_name}: {e}"
```

- [ ] **Step 5: 运行测试，确认通过**

```bash
pytest tests/test_tool.py -v
```
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add -A && git commit -m "feat: add tool decorator, ToolRegistry and ActionExecutor"
```

---

## Task 4: 记忆系统 (core/memory.py)

**Files:**
- Create: `agent_framework/core/memory.py`
- Test: `tests/test_memory.py`

- [ ] **Step 1: 写测试（mock LLM）**

```python
# tests/test_memory.py
from agent_framework.core.memory import SummarizationMemory
from agent_framework.core.message import Message, MessageRole
from agent_framework.core.llm import LLM

class FakeLLMForMemory(LLM):
    def __init__(self, summary_response: str = "This is a summary."):
        self.response = summary_response
        self.called = False

    def generate(self, messages):
        self.called = True
        return self.response

def test_memory_add_and_get():
    llm = FakeLLMForMemory()
    mem = SummarizationMemory(llm=llm, max_messages_before_summary=3)

    mem.add(Message(role=MessageRole.USER, content="Hello"))
    mem.add(Message(role=MessageRole.ASSISTANT, content="Hi there!"))
    mem.add(Message(role=MessageRole.USER, content="How are you?"))

    msgs = mem.get_messages()
    assert len(msgs) == 3
    assert msgs[0].content == "Hello"

def test_memory_summarize_trigger():
    llm = FakeLLMForMemory(summary_response="Summarized conversation.")
    mem = SummarizationMemory(llm=llm, max_messages_before_summary=2)

    mem.add(Message(role=MessageRole.USER, content="msg1"))
    mem.add(Message(role=MessageRole.ASSISTANT, content="msg2"))
    # 触发摘要（2条消息达到阈值）

    msgs = mem.get_messages()
    assert llm.called is True  # 摘要已触发
    assert mem.summary == "Summarized conversation."
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_memory.py -v
```
Expected: FAIL — module not found

- [ ] **Step 3: 实现 memory.py**

```python
# agent_framework/core/memory.py
from typing import List, Optional
from .message import Message, MessageRole
from .llm import LLM

DEFAULT_SUMMARY_PROMPT = (
    "Summarize the following conversation concisely, "
    "preserving key information:\n{content}"
)

class SummarizationMemory:
    """
    对话记忆：定期对历史消息做摘要，节省 token。
    """

    def __init__(
        self,
        llm: LLM,
        max_messages_before_summary: int = 10,
        summary_prompt: str = None,
    ):
        self.llm = llm
        self.max_messages_before_summary = max_messages_before_summary
        self.summary_prompt = summary_prompt or DEFAULT_SUMMARY_PROMPT
        self.messages: List[Message] = []
        self.summary: Optional[str] = None

    def add(self, message: Message) -> None:
        self.messages.append(message)
        if self._should_summarize():
            self._summarize()

    def get_messages(self) -> List[Message]:
        """返回摘要（如果有）+ 最近消息"""
        result: List[Message] = []
        if self.summary:
            result.append(Message(
                role=MessageRole.SYSTEM,
                content=f"Previous conversation summary: {self.summary}"
            ))
        result.extend(self.messages)
        return result

    def _should_summarize(self) -> bool:
        return len(self.messages) >= self.max_messages_before_summary

    def _summarize(self) -> None:
        content = "\n".join(f"[{m.role.value}] {m.content}" for m in self.messages)
        prompt = self.summary_prompt.format(content=content)
        self.summary = self.llm.generate([Message(role=MessageRole.USER, content=prompt)])
        self.messages = []  # 摘要后清除原始消息
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_memory.py -v
```
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: add SummarizationMemory"
```

---

## Task 5: Agent (core/agent.py)

**Files:**
- Create: `agent_framework/core/agent.py`
- Test: `tests/test_agent.py`

- [ ] **Step 1: 写测试（mock LLM 和工具）**

```python
# tests/test_agent.py
from agent_framework.core.agent import BaseAgent, ReActAgent, PlanAndExecuteAgent
from agent_framework.core.llm import LLM
from agent_framework.core.message import Message, MessageRole
from agent_framework.core.tool import tool, ToolRegistry, get_registry, ActionExecutor

class FakeLLM(LLM):
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    def generate(self, messages):
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp

def test_react_agent_finishes_with_final_answer():
    @tool(name="weather", description="Get weather")
    def weather(city: str) -> str:
        return f"{city} is sunny"

    # LLM: 第一轮返回 thought + action，第二轮返回 FINAL_ANSWER
    llm = FakeLLM(responses=[
        'Thought: I should check the weather.\nAction: weather\nAction Args: {"city": "Beijing"}',
        'Thought: I have the info.\nFinal Answer: Beijing is sunny.',
    ])
    registry = get_registry()
    executor = ActionExecutor(registry)
    agent = ReActAgent(llm=llm, executor=executor, max_turns=10)

    result = agent.run("What's the weather in Beijing?")
    assert "sunny" in result

def test_plan_agent():
    @tool(name="search", description="Search")
    def search(query: str) -> str:
        return f"Results for {query}"

    llm = FakeLLM(responses=[
        'Step 1: search for "AI"\nStep 2: summarize results',  # plan
        'Summary: AI is great.',  # synthesize
    ])
    registry = get_registry()
    executor = ActionExecutor(registry)
    agent = PlanAndExecuteAgent(llm=llm, executor=executor, max_turns=10)

    result = agent.run("Tell me about AI")
    assert "great" in result
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_agent.py -v
```
Expected: FAIL — module not found

- [ ] **Step 3: 实现 agent.py**

```python
# agent_framework/core/agent.py
import re
from abc import ABC, abstractmethod
from typing import List, Optional

from .llm import LLM
from .message import Message, MessageRole
from .tool import ToolRegistry, ActionExecutor
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
        import json
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

            thought, action, args = _parse_thought_output(response)

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
            thought, action, args = _parse_thought_output(step_response)
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
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_agent.py -v
```
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: add BaseAgent, ReActAgent and PlanAndExecuteAgent"
```

---

## Task 6: 入口框架 (framework.py + __init__.py)

**Files:**
- Create: `agent_framework/framework.py`
- Create: `agent_framework/core/__init__.py`
- Create: `agent_framework/__init__.py`
- Create: `agent_framework/multi/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_framework.py
from agent_framework.framework import AgentFramework
from agent_framework.core.llm import LLM
from agent_framework.core.message import Message, MessageRole

class FakeLLM(LLM):
    def __init__(self, response: str):
        self.response = response
    def generate(self, messages):
        return self.response

def test_framework_run_react():
    fw = AgentFramework(llm=FakeLLM("Thought: done.\nFinal Answer: 42."), mode="react")
    result = fw.run("What is 6 * 7?")
    assert result == "42."

def test_framework_run_plan():
    fw = AgentFramework(llm=FakeLLM("Step 1: 6*7\nStep 2: answer\nSummary: 42"), mode="plan")
    result = fw.run("What is 6 * 7?")
    assert "42" in result
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_framework.py -v
```
Expected: FAIL — module not found

- [ ] **Step 3: 实现 framework.py**

```python
# agent_framework/framework.py
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
```

- [ ] **Step 4: 实现各层 __init__.py**

```python
# agent_framework/__init__.py
from .framework import AgentFramework
from .core.llm import LLM, OpenAILLM
from .core.message import Message, MessageRole
from .core.tool import tool, get_registry

__all__ = [
    "AgentFramework",
    "LLM",
    "OpenAILLM",
    "Message",
    "MessageRole",
    "tool",
    "get_registry",
]
```

```python
# agent_framework/core/__init__.py
from .llm import LLM, OpenAILLM
from .message import Message, MessageRole
from .agent import BaseAgent, ReActAgent, PlanAndExecuteAgent

__all__ = [
    "LLM",
    "OpenAILLM",
    "Message",
    "MessageRole",
    "BaseAgent",
    "ReActAgent",
    "PlanAndExecuteAgent",
]
```

```python
# agent_framework/multi/__init__.py
from .supervisor import SupervisorAgent

__all__ = ["SupervisorAgent"]
```

```python
# tests/__init__.py
# 空文件
```

- [ ] **Step 5: 运行测试，确认通过**

```bash
pytest tests/test_framework.py -v
```
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add -A && git commit -m "feat: add AgentFramework entry point and __init__.py files"
```

---

## Task 7: 多 Agent — SupervisorAgent (multi/supervisor.py)

**Files:**
- Create: `agent_framework/multi/supervisor.py`
- Test: `tests/test_supervisor.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_supervisor.py
from agent_framework.multi.supervisor import SupervisorAgent
from agent_framework.core.agent import BaseAgent
from agent_framework.core.llm import LLM
from agent_framework.core.message import Message, MessageRole
from agent_framework.core.tool import ActionExecutor, get_registry

class FakeLLM(LLM):
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    def generate(self, messages):
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp

class DummyAgent(BaseAgent):
    def __init__(self, name: str, llm: LLM, result: str):
        super().__init__(llm=llm, executor=ActionExecutor(get_registry()))
        self.name = name
        self._result = result

    def run(self, task: str) -> str:
        return self._result

def test_supervisor_decomposes_and_delegates():
    # LLM: 返回分解的子任务
    llm = FakeLLM(responses=[
        "Subtask 1: research, Subtask 2: write",
        "Final synthesized answer: Done.",
    ])
    sub1 = DummyAgent("researcher", llm, "Research complete.")
    sub2 = DummyAgent("writer", llm, "Writing complete.")

    supervisor = SupervisorAgent(llm=llm, sub_agents=[sub1, sub2])
    result = supervisor.run("Make me a report")

    assert "Done" in result
    assert llm.call_count == 2  # 分解 + 汇总
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_supervisor.py -v
```
Expected: FAIL — module not found

- [ ] **Step 3: 实现 supervisor.py**

```python
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
            if agent_name in self.sub_agents:
                result = self.sub_agents[agent_name].run(line.strip())
            else:
                result = f"[No agent for: {line.strip()}]"
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
        # 默认返回第一个
        return list(self.sub_agents.keys())[0] if self.sub_agents else ""
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_supervisor.py -v
```
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: add SupervisorAgent for multi-agent collaboration"
```

---

## 自我检查

**Spec 覆盖检查:**
- [x] LLM 接口 + OpenAI Adapter — Task 2
- [x] 消息结构 — Task 1
- [x] `@tool` 装饰器 + ToolRegistry + ActionExecutor — Task 3
- [x] SummarizationMemory — Task 4
- [x] BaseAgent + ReActAgent + PlanAndExecuteAgent — Task 5
- [x] AgentFramework 入口 — Task 6
- [x] SupervisorAgent — Task 7
- [x] 测试覆盖每个组件 — Tasks 1-7

**类型一致性检查:**
- `Message.role` 使用 `MessageRole` enum
- `LLM.generate(messages: List[Message])` 签名一致
- `ActionExecutor.run(tool_name, tool_args)` 参数顺序一致
- `SummarizationMemory.add(message: Message)` 使用 Message 类型
- `BaseAgent.__init__(llm, executor, memory)` 构造一致

**占位符扫描:** 无 "TBD"、"TODO"、未完成代码段。
