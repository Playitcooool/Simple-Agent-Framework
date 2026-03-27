"""
Tool module — decorator-based tool registration system.

Components:
- Tool: dataclass holding name, description, and function
- ToolRegistry: catalog of available tools
- @tool: decorator to register functions as tools
"""

from typing import Callable, Any, Optional
from dataclasses import dataclass

from .executor import ActionExecutor

__all__ = ["Tool", "ToolRegistry", "tool", "get_registry", "ActionExecutor"]


@dataclass
class Tool:
    """A callable tool with metadata."""
    name: str
    description: str
    fn: Callable


class ToolRegistry:
    """Catalog of registered tools. Maps name -> Tool."""

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, fn: Callable) -> None:
        self.tools[name] = Tool(name=name, description=description, fn=fn)

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self.tools.values())

    def clear(self) -> None:
        self.tools.clear()


_global_registry = ToolRegistry()


def tool(name: str = None, description: str = None):
    """
    Decorator that registers a function as a tool.

    Usage:
        @tool(name="weather", description="Get weather")
        def get_weather(city: str) -> str:
            return f"{city} is sunny"
    """
    def decorator(fn: Callable) -> Callable:
        tool_name = name or fn.__name__
        tool_desc = description or (fn.__doc__ or "").strip()
        _global_registry.register(tool_name, tool_desc, fn)
        return fn
    return decorator


def get_registry() -> ToolRegistry:
    return _global_registry
