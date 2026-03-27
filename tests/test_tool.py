import pytest
from agent_framework.core.tool import tool, ToolRegistry, get_registry
from agent_framework.core.executor import ActionExecutor


@pytest.fixture(autouse=True)
def clean_registry():
    """Snapshot and restore the global registry before/after each test."""
    registry = get_registry()
    snapshot = dict(registry.tools)
    registry.clear()
    yield
    registry.clear()
    registry.tools.update(snapshot)


def test_tool_decorator_basic():
    @tool(name="test_tool", description="A test tool")
    def my_tool(x: int, y: int) -> str:
        return str(x + y)

    registry = get_registry()
    assert "test_tool" in registry.tools
    t = registry.tools["test_tool"]
    assert t.description == "A test tool"
    assert t.fn is my_tool


def test_executor_run():
    @tool(name="add", description="Add two numbers")
    def add(a: int, b: int) -> str:
        return str(a + b)

    registry = get_registry()
    executor = ActionExecutor(registry)
    result = executor.run("add", {"a": 3, "b": 5})
    assert result == "8"


def test_executor_unknown_tool():
    registry = get_registry()
    executor = ActionExecutor(registry)
    with pytest.raises(ValueError, match="Unknown tool"):
        executor.run("nonexistent", {})


def test_executor_returns_error_on_exception():
    @tool(name="failing_tool", description="A tool that fails")
    def failing_tool() -> str:
        raise RuntimeError("intentional failure")

    registry = get_registry()
    executor = ActionExecutor(registry)
    result = executor.run("failing_tool", {})
    assert "Error executing tool failing_tool" in result
    assert "intentional failure" in result
