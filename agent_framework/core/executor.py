"""
Action Executor module.

PURPOSE:
  Takes a tool name and arguments, looks up the tool in the registry,
  calls it, and returns the result as a string.

THIS IS THE BRIDGE BETWEEN:
  - Agent's decision "I should call the 'weather' tool"
  - The actual Python function that implements weather lookup

KEY CONCEPTS:
  - The Agent doesn't call tools directly — it goes through the Executor
  - The Executor looks up the tool in the Registry
  - The Executor handles errors gracefully (doesn't crash the Agent)
  - Results are always converted to strings (standard interface)

WHY HANDLE ERRORS BY RETURNING A STRING?
  In an Agent loop, we want the LLM to see the error and decide what to do.
  If we raised an exception, the loop would crash.
  By returning a string like "Error: division by zero", the LLM can:
  1. See that something went wrong
  2. Try a different approach
  3. Report the error to the user

  This is more robust than crashing for transient errors.

WHY TYPE HINTS ON tool_args?
  tool_args: dict[str, Any] tells us:
  - It's a dict (mapping argument name to value)
  - Keys are strings (argument names)
  - Values can be anything (int, str, bool, etc.)

  The LLM is expected to provide the correct types based on the
  tool's function signature. Our job is just to pass them through.
"""

from typing import Any, TYPE_CHECKING

# TYPE_CHECKING block:
#   Tool and ToolRegistry are only used for type hints, not runtime.
#   This avoids circular imports — tool.py imports executor.py,
#   so we can't have executor.py import tool.py at the top level.
#   By putting it in TYPE_CHECKING, Python skips the import at runtime
#   but type checkers (like mypy, Pyright) still see it.
if TYPE_CHECKING:
    from .tool import ToolRegistry, Tool


__all__ = ["ActionExecutor"]


class ActionExecutor:
    """
    Executes tool calls by name.

    Think of this as the "operator" who receives a command like
    "call the weather tool with city=Beijing" and actually performs it.

    The Agent doesn't call tools directly because:
    1. It shouldn't need to know about the registry
    2. Errors should be handled gracefully
    3. We might want to add logging, rate limiting, etc.
    """

    def __init__(self, registry: "ToolRegistry"):
        """
        Create an executor with a reference to the tool registry.

        Args:
            registry: The catalog of available tools to look up from.
        """
        self.registry = registry

    def run(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """
        Execute a tool by name with the given arguments.

        Args:
            tool_name: The name of the tool to execute (must be registered)
            tool_args: Dictionary of argument names to values

        Returns:
            The tool's result as a string, or an error message if it fails.

        Raises:
            ValueError: If the tool is not found in the registry

        DESIGN NOTE — WHY RETURN STRING NOT RESULT TYPE?
          Returning a string is simpler and more uniform.
          The LLM doesn't need to know what type the tool returned.
          We convert everything to string representation.
        """
        # Look up the tool in the registry
        t = self.registry.get(tool_name)
        if t is None:
            # Raise — this indicates a programming error (wrong tool name)
            # The Agent should never request an unknown tool if prompts are correct
            raise ValueError(f"Unknown tool: {tool_name}")

        try:
            # Call the underlying function with the provided arguments
            # **tool_args unpacks the dict: {"city": "Beijing"} -> city="Beijing"
            result = t.fn(**tool_args)
            # Always return as string — keeps interface uniform
            return str(result)

        except Exception as e:
            # Catch ALL exceptions and return as error string.
            # Why catch everything?
            #   Tools can raise ANY exception (ValueError, RuntimeError, etc.)
            #   We don't want the Agent loop to crash.
            #   The LLM can decide how to handle the error.
            return f"Error executing tool {tool_name}: {e}"
