# Agent Framework

A lightweight Agent development framework written in pure Python. No external framework dependencies.

---

## 框架简介 | Framework Overview

纯 Python 实现的轻量级 Agent 开发框架，支持：
- **ReAct 模式**：Thought → Action → Observation 循环
- **Plan-and-Execute 模式**：先规划后执行
- **装饰器工具注册**：`@tool` 简单直观
- **摘要记忆**：自动压缩对话历史节省 token
- **层级多 Agent**：Supervisor 委托子 Agent 协作
- **内置工具**：Bash、文件读写等常用工具开箱即用

---

## 架构图 | Architecture

```
┌─────────────────────────────────────────────────────┐
│                   AgentFramework                     │
│               (agent.run("task"))                    │
└─────────────────────┬───────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │                     │
    ┌─────▼─────┐         ┌─────▼─────┐
    │  ReActAgent │         │ PlanAnd   │
    │            │         │ Execute    │
    └─────┬─────┘         └─────┬─────┘
          │                     │
          └─────────┬───────────┘
                    │
         ┌──────────▼──────────┐
         │    ActionExecutor     │
         │  (executes tools)    │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │   ToolRegistry      │
         │ @tool decorators    │
         └─────────────────────┘
```

---

## 核心工作流 | Core Workflows

### ReAct Agent

```
User: "What's the weather in Beijing?"
         │
         ▼
┌────────────────────────┐
│  Build messages        │
│  (system + memory +    │
│   user task)          │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  LLM.generate()        │◄──────────────────┐
│  Response:             │                    │
│  "Thought: I should... │                    │
│   Action: weather      │
│   Action Args: {...}"   │                    │
└───────────┬────────────┘                    │
            │                                   │
            ▼                                   │
    ┌───────────────┐    Yes                   │
    │ Has Action?   │────────────────────────►│
    └───────┬───────┘                          │
            │ No                               │
            ▼                                   │
    ┌───────────────┐                          │
    │ Has Final     │    Yes                   │
    │ Answer?       │──────────┐               │
    └───────┬───────┘          │               │
            │ No               │               │
            ▼                   ▼               │
    ┌───────────────┐  ┌──────────────┐        │
    │ Execute tool  │  │ Return      │        │
    │ via Executor  │  │ answer       │────────┘
    └───────┬───────┘  └──────────────┘
            │
            ▼
    ┌───────────────┐
    │ Add result to │
    │ messages      │
    └───────┬───────┘
            │
            └────────────────── (loop back to LLM)
```

### Plan-and-Execute Agent

```
User: "Write a report about AI"
         │
         ▼
┌─────────────────────────┐
│  PHASE 1: PLANNING      │
│  LLM generates:         │
│  "Step 1: Research AI   │
│   Step 2: Outline       │
│   Step 3: Write"        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  PHASE 2: EXECUTION      │
│                         │
│  For each step:         │
│    LLM decides action    │
│    Execute tool          │
│    Collect result        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  PHASE 3: SYNTHESIS      │
│  LLM combines all       │
│  results into final       │
│  answer                  │
└───────────┬─────────────┘
            │
            ▼
       Final Answer
```

### SupervisorAgent (Multi-Agent)

```
User: "Write and publish a blog post"
         │
         ▼
┌─────────────────────────────────────┐
│  SupervisorAgent.run()              │
│                                      │
│  1. DECOMPOSE                        │
│     "Subtask 1: Research             │
│      Subtask 2: Write                │
│      Subtask 3: Publish"             │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  2. DISPATCH                         │
│                                      │
│  Subtask 1 ──► Researcher Agent      │
│  Subtask 2 ──► Writer Agent         │
│  Subtask 3 ──► Publisher Agent      │
│                                      │
│  (Each agent.run() independently)   │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  3. SYNTHESIZE                       │
│                                      │
│  LLM combines all subtask results   │
│  into cohesive final answer          │
└───────────────┬─────────────────────┘
                │
                ▼
           Final Answer
```

---

## 快速开始 | Quick Start

```python
from agent_framework import AgentFramework, OpenAILLM

# Initialize
fw = AgentFramework(llm=OpenAILLM(api_key="your-api-key"))

# Register a tool
@fw.tool(name="weather", description="Get weather for a city")
def get_weather(city: str) -> str:
    return f"{city} is sunny, 25°C"

# Run a task
result = fw.run("What's the weather in Beijing?")
print(result)
```

### 选择执行模式 | Choose Execution Mode

```python
# ReAct (default) — good for most tasks
fw = AgentFramework(llm=llm, mode="react")

# Plan-and-Execute — better for complex multi-step tasks
fw = AgentFramework(llm=llm, mode="plan")
```

---

## 目录结构 | Project Structure

```
agent_framework/
├── __init__.py          # Public API: AgentFramework, LLM, OpenAILLM, Message, @tool
├── framework.py          # AgentFramework entry point
├── core/
│   ├── llm.py           # LLM abstract class + OpenAI adapter
│   ├── message.py       # Message dataclass + MessageRole enum
│   ├── tool.py          # @tool decorator + ToolRegistry
│   ├── executor.py      # ActionExecutor — runs tools
│   ├── memory.py        # SummarizationMemory
│   └── agent.py         # BaseAgent, ReActAgent, PlanAndExecuteAgent
└── multi/
    └── supervisor.py    # SupervisorAgent
```

---

## 多 Agent 协作 | Multi-Agent

```python
from agent_framework.multi import SupervisorAgent

researcher = ReActAgent(llm=llm, executor=executor)
writer = ReActAgent(llm=llm, executor=executor)
researcher.name = "researcher"
writer.name = "writer"

supervisor = SupervisorAgent(llm=llm, sub_agents=[researcher, writer])
result = supervisor.run("Write a report about AI trends")
```

---

## 示例 | Examples

See `examples/` directory for runnable examples:

```bash
# Basic usage
OPENAI_API_KEY=sk-... python examples/01_basic_usage.py

# Built-in tools (Bash, Read, Write)
OPENAI_API_KEY=sk-... python examples/02_built_in_tools.py
```

---

## 测试 | Testing

## 扩展 | Extending

### 添加新的 LLM Adapter

```python
from agent_framework.core.llm import LLM

class AnthropicLLM(LLM):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(self, messages: list[Message]) -> str:
        # Implement Anthropic API call
        return response_text
```

### 内置工具

框架内置常用工具（位于 `agent_framework.tools`）：

```python
from agent_framework import AgentFramework, OpenAILLM
from agent_framework.tools import BashTool, ReadFileTool, WriteFileTool

fw = AgentFramework(llm=OpenAILLM(api_key="..."))

# Bash command
bash = BashTool(cwd="/tmp", timeout=10)
@fw.tool(name="bash", description="Run shell command")
def bash_cmd(command: str) -> str:
    return bash.run(command)

# Read file
read_tool = ReadFileTool(base_dir="/tmp")
@fw.tool(name="read", description="Read a file")
def read_cmd(path: str) -> str:
    return read_tool.run(path)

# Write file
write_tool = WriteFileTool(base_dir="/tmp")
@fw.tool(name="write", description="Write to a file")
def write_cmd(path: str, content: str) -> str:
    return write_tool.run(path, content)
```

### 添加工具

```python
@fw.tool(name="my_tool", description="Does something useful")
def my_tool(arg1: str, arg2: int) -> str:
    return f"Result: {arg1}, {arg2}"
```

---

## License

MIT
