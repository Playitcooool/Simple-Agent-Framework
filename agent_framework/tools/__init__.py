"""
Built-in tools for the Agent framework.

Provides common tools that most agents need:
- Bash: execute shell commands
- ReadFile: read file contents
- WriteFile: write content to a file
- SearchTool: search for patterns in files
- ListDirTool: list directory contents
- WebSearchTool: web search via Tavily
- CalculatorTool: math calculations
- DateTimeTool: get current date/time
"""

from .bash import bash_tool, BashTool
from .file import read_file_tool, write_file_tool, ReadFileTool, WriteFileTool
from .search import search_tool, SearchTool
from .list_dir import list_dir_tool, ListDirTool
from .web_search import web_search_tool, WebSearchTool
from .calculator import calculator_tool, CalculatorTool
from .datetime_tool import datetime_tool, DateTimeTool

__all__ = [
    "BashTool",
    "bash_tool",
    "ReadFileTool",
    "read_file_tool",
    "WriteFileTool",
    "write_file_tool",
    "SearchTool",
    "search_tool",
    "ListDirTool",
    "list_dir_tool",
    "WebSearchTool",
    "web_search_tool",
    "CalculatorTool",
    "calculator_tool",
    "DateTimeTool",
    "datetime_tool",
]
