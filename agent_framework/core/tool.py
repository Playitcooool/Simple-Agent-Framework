"""
Tool system for the Agent framework.

PURPOSE:
  Tools allow the Agent to interact with the outside world.
  Instead of just generating text, an Agent can "call functions"
  to get real information or perform actions.

KEY CONCEPTS:
  - Tool: A callable with metadata (name, description, function)
  - ToolRegistry: A catalog of all available tools
  - @tool: A decorator that turns any function into a registered tool

WHAT MAKES A GOOD TOOL?
  1. Clear name: Something that describes what it does
  2. Clear description: This is shown to the LLM so it knows when to use the tool
  3. Type hints: Help the LLM understand argument types
  4. Return type: Always return a string — keeps it simple and serializable

WHY USE A DECORATOR?
  The @tool decorator is the simplest way to register a function.
  Compare these two approaches:

  WITHOUT decorator (manual registration):
    def get_weather(city): ...
    registry.register("weather", "Get weather", get_weather)

  WITH decorator:
    @tool(name="weather", description="Get weather")
    def get_weather(city): ...

  The decorator version is cleaner and the registration happens
  automatically at import time. The user doesn't need to remember
  to register explicitly.

WHY A GLOBAL REGISTRY?
  We use a module-level global variable _global_registry as a singleton.
  This is simple and works well for single-agent use cases.
  Each @tool decorator automatically registers with this global registry.

  LIMITATION: In multi-threaded or multi-process environments,
  this global state could cause issues. For production, you might
  want to inject the registry instead.
"""

from typing import Callable, Any, Optional
from dataclasses import dataclass

# Import ActionExecutor here to make it re-exportable
# This lets users import ActionExecutor from agent_framework directly
# (e.g., from agent_framework import ActionExecutor)
from .executor import ActionExecutor

__all__ = ["Tool", "ToolRegistry", "tool", "get_registry", "ActionExecutor"]


@dataclass
class Tool:
    """
    Represents a callable tool that an Agent can use.

    A Tool packages:
    1. A name — how the LLM refers to it
    2. A description — shown to LLM so it knows when to use it
    3. The actual function to call

    WHY USE A DATACLASS?
      Tool is purely data — it holds no logic.
      Using @dataclass gives us __init__, __repr__, __eq__ for free.

    DESIGN NOTE — fn: Callable
      We store the raw function, not a wrapper or partial.
      This means the function works exactly as defined,
      with no interception or modification.
    """
    name: str
    description: str
    fn: Callable


class ToolRegistry:
    """
    A catalog of registered tools.

    Think of this like a phone book:
    - Name (the tool name) -> ContactInfo (the Tool object)

    WHY SEPARATE REGISTRY FROM TOOL?
      The Tool describes ONE tool. The Registry manages MANY tools.
      This separation of concerns keeps each class simple.

    KEY METHODS:
      register(): Add a new tool
      get(): Look up a tool by name
      list_tools(): See all available tools
      clear(): Remove all tools (useful for testing)
    """

    def __init__(self):
        """Create an empty registry."""
        # Maps tool name -> Tool object
        # dict provides O(1) lookup by name
        self.tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, fn: Callable) -> None:
        """
        Register a new tool.

        Args:
            name: Unique identifier for this tool
            description: Human-readable description (shown to LLM)
            fn: The actual function to call

        NOTE: If a tool with the same name already exists,
        it will be overwritten. This is usually not what you want.
        """
        self.tools[name] = Tool(name=name, description=description, fn=fn)

    def get(self, name: str) -> Optional[Tool]:
        """
        Look up a tool by name.

        Returns:
            The Tool object if found, None otherwise.

        Why return Optional[Tool] instead of raising?
          It's often convenient to try getting a tool and handle
          the None case, rather than catching an exception.
        """
        return self.tools.get(name)

    def list_tools(self) -> list[Tool]:
        """Get a list of all registered tools."""
        return list(self.tools.values())

    def clear(self) -> None:
        """
        Remove all tools from the registry.

        This is primarily useful in tests, where we want
        a clean slate between test cases.
        """
        self.tools.clear()


# ============================================================================
# GLOBAL REGISTRY — Singleton pattern for simplicity
# ============================================================================
#
# A module-level global variable serves as the default registry.
# All @tool-decorated functions automatically register here.
#
# PROS:
#   - Simple, no setup required
#   - Decorator works with zero configuration
#
# CONS:
#   - Global mutable state (can cause issues in testing or multi-instance scenarios)
#   - Not thread-safe without additional locking
#
# For a production system, consider dependency injection instead.
#
_global_registry = ToolRegistry()


def tool(name: str = None, description: str = None):
    """
    Decorator that registers a function as a tool.

    Usage:
        @tool(name="weather", description="Get weather for a city")
        def get_weather(city: str) -> str:
            ...

    Args:
        name: The tool name. If None, uses the function's __name__.
              Example: @tool() on function "get_weather" -> tool name "get_weather"
        description: The tool description. If None, uses the function's docstring.
                     The description is shown to the LLM, so be clear and specific!

    HOW IT WORKS:
      1. The outer function (tool()) receives the decorator arguments
      2. The inner function (decorator()) receives the function being decorated
      3. We register the function with the global registry
      4. We return the original function (unchanged — decorators can wrap,
         but don't have to)

    WHY RETURN THE ORIGINAL FUNCTION?
      We want @tool to be transparent — after decoration, the function
      still works exactly the same way. We didn't modify its behavior,
      just registered it in the catalog.
    """
    def decorator(fn: Callable) -> Callable:
        # Use function name as default tool name
        tool_name = name or fn.__name__
        # Use docstring as default description (first line only, stripped)
        tool_desc = description or (fn.__doc__ or "").strip()
        _global_registry.register(tool_name, tool_desc, fn)
        return fn
    return decorator


def get_registry() -> ToolRegistry:
    """
    Get the global tool registry.

    This is how the Agent (specifically ActionExecutor) accesses
    all registered tools. It retrieves the tool by name and calls it.

    For simple use cases, the global registry is convenient.
    For more complex scenarios (multiple agents with different tools),
    you might want to create your own ToolRegistry instances.
    """
    return _global_registry
