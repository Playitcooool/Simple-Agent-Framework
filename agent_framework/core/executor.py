from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .tool import ToolRegistry, Tool

class ActionExecutor:
    """执行工具调用的组件"""
    def __init__(self, registry: "ToolRegistry"):
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