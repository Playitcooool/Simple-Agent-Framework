"""
Executor module — executes tool calls.

ActionExecutor looks up a tool by name in the registry and calls it.
Errors are returned as strings (not raised) so the Agent loop can handle them.
"""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .tool import ToolRegistry, Tool

__all__ = ["ActionExecutor"]


class ActionExecutor:
    """Looks up and executes a tool by name."""

    def __init__(self, registry: "ToolRegistry"):
        self.registry = registry

    def run(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        t = self.registry.get(tool_name)
        if t is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        try:
            return str(t.fn(**tool_args))
        except Exception as e:
            return f"Error executing tool {tool_name}: {e}"
