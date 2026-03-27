import pytest
from agent_framework.core.tool import tool, ToolRegistry, ActionExecutor

def test_tool_decorator_basic():
    @tool(name="test_tool", description="A test tool")
    def my_tool(x: int, y: int) -> str:
        return str(x + y)

    from agent_framework.core.tool import get_registry
    registry = get_registry()
    assert "test_tool" in registry.tools
    t = registry.tools["test_tool"]
    assert t.description == "A test tool"
    assert t.fn is my_tool

def test_executor_run():
    from agent_framework.core.tool import get_registry

    @tool(name="add", description="Add two numbers")
    def add(a: int, b: int) -> str:
        return str(a + b)

    registry = get_registry()
    executor = ActionExecutor(registry)
    result = executor.run("add", {"a": 3, "b": 5})
    assert result == "8"

def test_executor_unknown_tool():
    from agent_framework.core.tool import get_registry
    registry = get_registry()
    executor = ActionExecutor(registry)
    with pytest.raises(ValueError, match="Unknown tool"):
        executor.run("nonexistent", {})