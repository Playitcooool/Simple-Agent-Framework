# Agent Framework 设计文档

## 概述

从零实现一个轻量级 Agent 开发框架，纯 Python，无外部框架依赖。核心目标：

- 支持 ReAct 和 Plan-and-Execute 双执行模式
- 基于 Summarization 的上下文记忆管理
- 装饰器风格的工具注册
- 层级委托的多 Agent 协作
- 链式调用入口 `agent.run()`

---

## 技术决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 运行时 | 纯 Python，直接调用 LLM API | 零外部依赖，理解底层原理 |
| LLM 接口 | 可替换 Adapter（OpenAI / Anthropic） | 适配不同模型 |
| 执行模型 | ReAct + Plan-and-Execute | 兼顾简单任务和复杂任务 |
| 记忆 | SummarizationMemory（定期摘要） | 节省 token，实现简单 |
| 工具注册 | `@tool` 装饰器 | Pythonic，直观简洁 |
| 多 Agent | SupervisorAgent 层级委托 | 符合直觉，易于理解 |
| 入口 | `agent.run("任务")` 链式调用 | 一行解决简单任务 |

---

## 目录结构

```
agent_framework/
├── __init__.py
├── framework.py           # 入口：AgentFramework 类，agent.run()
├── core/
│   ├── __init__.py
│   ├── llm.py            # LLM 接口抽象 + OpenAI/ Anthropic Adapter
│   ├── message.py        # Message / MessageRole 消息结构
│   ├── tool.py           # @tool 装饰器 + Tool 基类
│   ├── memory.py         # SummarizationMemory 摘要记忆
│   ├── agent.py          # BaseAgent / ReActAgent / PlanAndExecuteAgent
│   └── executor.py       # ActionExecutor 工具执行器
└── multi/
    ├── __init__.py
    └── supervisor.py     # SupervisorAgent 主从 Agent
```

---

## 核心组件

### 1. LLM 接口 (`core/llm.py`)

抽象 `LLM` 基类，定义 `generate(messages) -> str` 接口。

```python
class LLM(ABC):
    @abstractmethod
    def generate(self, messages: list[Message]) -> str:
        pass
```

提供具体 Adapter：
- `OpenAILLM(api_key, model)` — 调用 OpenAI Chat Completions API
- `AnthropicLLM(api_key, model)` — 调用 Anthropic Messages API

### 2. 消息结构 (`core/message.py`)

```python
class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"

@dataclass
class Message:
    role: MessageRole
    content: str
    tool_call: dict | None = None  # 可选：{name, args}
```

### 3. 工具系统 (`core/tool.py`)

**装饰器注册** — 用 `@tool` 装饰函数，自动注册到全局 `ToolRegistry`。

```python
registry = ToolRegistry()

def tool(name: str = None, description: str = None):
    """装饰器：标记一个函数为工具"""
    def decorator(func):
        # 存入 registry
        return func
    return decorator
```

**工具执行** — `ActionExecutor` 负责调用被选中的工具函数。

### 4. 记忆系统 (`core/memory.py`)

```python
class SummarizationMemory:
    """
    对话历史 → 定期摘要 → 保留摘要，丢弃原始消息
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
        self.messages: list[Message] = []
        self.summary: str | None = None

    def add(self, message: Message) -> None: ...
    def get_messages(self) -> list[Message]: ...  # 返回摘要 + 最近消息
    def _should_summarize(self) -> bool: ...
    def _summarize(self) -> None: ...  # 调用 LLM 生成摘要
```

### 5. Agent (`core/agent.py`)

**BaseAgent** — 公共逻辑（工具注册、LLM 调用、记忆管理）。

**ReActAgent** — ReAct 循环：

```
while not done:
    thought = llm.think(messages)      # LLM 输出 thought
    if "FINAL_ANSWER" in thought:
        return extract_answer(thought)
    action, args = parse_action(thought)
    result = executor.run(action, args)
    messages.append(Message(TOOL_RESULT, result))
```

**PlanAndExecuteAgent** — 规划-执行分离：

```
plan = llm.plan(task)              # LLM 输出步骤列表
for step in plan:
    result = executor.run(step)    # 逐个执行
    messages.append(result)
final = llm.synthesize(task, messages)
```

### 6. 入口 (`framework.py`)

```python
class AgentFramework:
    """框架入口"""

    def __init__(self, llm: LLM, mode: str = "react"):
        self.llm = llm
        self.mode = mode
        self.tools: list[Tool] = []

    def tool(self, name: str = None, description: str = None):
        """装饰器入口"""
        return tool(name, description)

    def run(self, task: str) -> str:
        """同步执行任务"""
        agent = self._create_agent()
        return agent.run(task)

    def _create_agent(self) -> BaseAgent:
        if self.mode == "react":
            return ReActAgent(self.llm, self.tools)
        elif self.mode == "plan":
            return PlanAndExecuteAgent(self.llm, self.tools)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
```

### 7. 多 Agent — 层级委托 (`multi/supervisor.py`)

```python
class SupervisorAgent:
    """
    主 Agent：接收任务，拆解为子任务，委托给子 Agent 执行，汇总结果
    """

    def __init__(self, llm: LLM, sub_agents: list[BaseAgent]):
        self.llm = llm
        self.sub_agents = {a.name: a for a in sub_agents}

    def run(self, task: str) -> str:
        # 1. LLM 分解任务 → 子任务列表
        subtasks = self._decompose(task)

        # 2. 分发给对应子 Agent
        results = []
        for subtask in subtasks:
            agent = self._select_agent(subtask)
            result = agent.run(subtask)
            results.append(result)

        # 3. LLM 汇总结果
        return self._synthesize(task, results)
```

---

## 工具注册使用示例

```python
from agent_framework import AgentFramework

fw = AgentFramework(llm=OpenAILLM(api_key="..."))

@fw.tool(name="weather", description="查询城市天气")
def get_weather(city: str) -> str:
    return f"{city} 晴，25°C"

result = fw.run("北京今天天气怎么样？")
```

---

## 错误处理

- **LLM 调用失败** — 重试 3 次，指数退避
- **工具执行失败** — 返回错误信息，LLM 决定是否重试
- **无匹配工具** — LLM 输出 `FINAL_ANSWER` 表示无法完成
- **Agent 超时** — 最大循环次数保护（默认 50 轮）

---

## 测试策略

- `tests/test_llm.py` — LLM Adapter 单元测试（mock）
- `tests/test_tool.py` — 工具注册和执行测试
- `tests/test_memory.py` — 记忆摘要功能测试
- `tests/test_agent.py` — ReAct / Plan 模式集成测试（mock LLM）
- `tests/test_supervisor.py` — 多 Agent 协作测试

---

## 第一阶段实现范围

1. `core/llm.py` — LLM 接口 + OpenAI Adapter
2. `core/message.py` — 消息结构
3. `core/tool.py` — `@tool` 装饰器 + ToolRegistry + ActionExecutor
4. `core/memory.py` — SummarizationMemory
5. `core/agent.py` — BaseAgent + ReActAgent + PlanAndExecuteAgent
6. `framework.py` — AgentFramework 入口
7. `multi/supervisor.py` — SupervisorAgent
8. `tests/` — 各组件测试

后续扩展方向（不在本设计范围内）：
- Anthropic Adapter
- 向量检索记忆（VectorMemory）
- MCP 协议兼容层
- 异步执行支持
- 持久化状态
