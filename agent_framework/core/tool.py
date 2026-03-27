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

    def clear(self) -> None:
        self.tools.clear()

# 全局单例
_global_registry = ToolRegistry()

def tool(name: str = None, description: str = None):
    """装饰器：把函数注册为工具"""
    def decorator(fn: Callable) -> Callable:
        tool_name = name or fn.__name__
        tool_desc = description or (fn.__doc__ or "").strip()
        _global_registry.register(tool_name, tool_desc, fn)
        return fn
    return decorator

def get_registry() -> ToolRegistry:
    return _global_registry