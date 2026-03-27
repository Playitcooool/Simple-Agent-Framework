# Agent Framework

A lightweight Agent development framework written in pure Python. No external framework dependencies.

---

## 框架简介 | Framework Overview

一个从零实现的轻量级 Agent 开发框架，纯 Python 编写，无外部框架依赖。

A lightweight Agent framework implemented from scratch in pure Python with no external framework dependencies.

### 核心特性 | Key Features

| 特性 / Feature | 描述 / Description |
|----------------|-------------------|
| **双执行模式 / Dual Execution Modes** | ReAct (Thought → Action → Observation) 和 Plan-and-Execute (规划 → 执行) |
| **装饰器工具注册 / Decorator-based Tool Registration** | `@tool` 装饰器，简单直观 / Simple and intuitive |
| **摘要记忆 / Summarization Memory** | 自动对对话历史做摘要，节省 token / Automatic conversation summarization |
| **层级多 Agent 协作 / Hierarchical Multi-Agent** | SupervisorAgent 委托子 Agent / Supervisor delegates to sub-agents |
| **链式调用入口 / Chainable Entry Point** | `agent.run("task")` 一行调用 / One-line interface |

---

## 安装 | Installation

```bash
pip install -e .
```

或者直接使用（无需安装）：

```bash
export OPENAI_API_KEY="your-api-key"
python -c "from agent_framework import *; ..."
```

Or use directly (no installation required):

```bash
export OPENAI_API_KEY="your-api-key"
python -c "from agent_framework import *; ..."
```

---

## 快速开始 | Quick Start

```python
from agent_framework import AgentFramework, OpenAILLM

# Initialize with OpenAI
fw = AgentFramework(llm=OpenAILLM(api_key="your-api-key"))

# Register a tool
@fw.tool(name="weather", description="Get weather for a city")
def get_weather(city: str) -> str:
    return f"{city} is sunny, 25°C"

# Run a task
result = fw.run("What's the weather in Beijing?")
print(result)
```

### 指定执行模式 | Specify Execution Mode

```python
# ReAct mode (default) - Thought → Action → Observation loop
fw = AgentFramework(llm=llm, mode="react")

# Plan-and-Execute mode - Plan first, then execute
fw = AgentFramework(llm=llm, mode="plan")
```

---

## 架构 | Architecture

```
agent_framework/
├── __init__.py              # Public API exports
├── framework.py             # AgentFramework entry point
├── core/
│   ├── llm.py              # LLM abstract class + OpenAI adapter
│   ├── message.py           # Message / MessageRole
│   ├── tool.py              # @tool decorator + ToolRegistry
│   ├── executor.py          # ActionExecutor
│   ├── memory.py            # SummarizationMemory
│   └── agent.py            # BaseAgent + ReActAgent + PlanAndExecuteAgent
└── multi/
    └── supervisor.py        # SupervisorAgent
```

### 核心组件 | Core Components

| 组件 / Component | 文件 / File | 说明 / Description |
|-----------------|-------------|-------------------|
| `LLM` | `core/llm.py` | Abstract base class for LLM adapters |
| `OpenAILLM` | `core/llm.py` | OpenAI Chat Completions API adapter |
| `Message` | `core/message.py` | Message dataclass with role and content |
| `@tool` | `core/tool.py` | Decorator for registering tools |
| `SummarizationMemory` | `core/memory.py` | Memory with periodic summarization |
| `ReActAgent` | `core/agent.py` | ReAct loop agent |
| `PlanAndExecuteAgent` | `core/agent.py` | Plan-and-execute agent |
| `SupervisorAgent` | `multi/supervisor.py` | Multi-agent orchestrator |

---

## 多 Agent 协作 | Multi-Agent Collaboration

```python
from agent_framework.multi import SupervisorAgent
from agent_framework import AgentFramework, OpenAILLM

# Create sub-agents
research_agent = ReActAgent(llm=llm, executor=executor)
writer_agent = ReActAgent(llm=llm, executor=executor)
research_agent.name = "researcher"
writer_agent.name = "writer"

# Create supervisor
supervisor = SupervisorAgent(llm=llm, sub_agents=[research_agent, writer_agent])

# Run collaborative task
result = supervisor.run("Write a report about AI trends")
```

---

## 测试 | Testing

```bash
pytest tests/ -v
```

---

## 扩展 | Extending

### 添加新的 LLM Adapter

```python
from agent_framework.core.llm import LLM

class MyLLM(LLM):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(self, messages: list[Message]) -> str:
        # Implement your LLM API call here
        return response_text
```

### 添加更多工具

```python
@fw.tool(name="my_tool", description="Description of my tool")
def my_tool(arg1: str, arg2: int) -> str:
    return f"Result: {arg1}, {arg2}"
```

---

## License

MIT
