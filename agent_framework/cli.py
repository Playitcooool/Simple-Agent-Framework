"""
Interactive Chat CLI

A command-line interface that allows users to chat with an agent
through the terminal. Supports:
- Multi-turn conversations with memory
- Built-in tools (Bash, Read, Write, Search, List, WebSearch, Calculator, DateTime)
- Exit commands
"""

import os
import sys

from agent_framework import AgentFramework, OpenAILLM
from agent_framework.tools import (
    BashTool, ReadFileTool, WriteFileTool,
    SearchTool, ListDirTool, WebSearchTool,
    CalculatorTool, DateTimeTool
)


def create_agent() -> AgentFramework:
    """Create and configure an agent with built-in tools."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    fw = AgentFramework(llm=OpenAILLM(api_key=api_key), mode="react")

    # Tool instances
    bash = BashTool(cwd=os.getcwd(), timeout=30)
    read = ReadFileTool(base_dir=os.getcwd())
    write = WriteFileTool(base_dir=os.getcwd())
    search = SearchTool(base_dir=os.getcwd())
    list_dir = ListDirTool(base_dir=os.getcwd())
    calc = CalculatorTool()
    datetime = DateTimeTool()

    # Register built-in tools
    @fw.tool(name="bash", description="Execute a shell command. Args: command (str)")
    def bash_cmd(command: str) -> str:
        return bash.run(command)

    @fw.tool(name="read", description="Read a file. Args: path (str), lines (int, optional)")
    def read_cmd(path: str, lines: int = None) -> str:
        return read.run(path, lines)

    @fw.tool(name="write", description="Write content to a file. Args: path (str), content (str)")
    def write_cmd(path: str, content: str) -> str:
        return write.run(path, content)

    @fw.tool(name="search", description="Search for text in files. Args: pattern (str), path (str, default '.'), case_sensitive (bool)")
    def search_cmd(pattern: str, path: str = ".", case_sensitive: bool = False) -> str:
        return search.run(pattern, path, case_sensitive)

    @fw.tool(name="ls", description="List directory contents. Args: path (str, default '.'), all (bool), long (bool)")
    def ls_cmd(path: str = ".", all: bool = False, long: bool = False) -> str:
        return list_dir.run(path, all, long)

    @fw.tool(name="calculator", description="Calculate math expression. Args: expression (str)")
    def calc_cmd(expression: str) -> str:
        return calc.run(expression)

    @fw.tool(name="datetime", description="Get current datetime. Args: format (str), timezone (str, optional)")
    def datetime_cmd(format: str = "%Y-%m-%d %H:%M:%S", timezone: str = None) -> str:
        return datetime.run(format, timezone)

    @fw.tool(name="web_search", description="Search the web. Args: query (str), max_results (int, default 5)")
    def web_search_cmd(query: str, max_results: int = 5) -> str:
        return WebSearchTool().run(query, max_results)

    return fw


def print_welcome():
    """Print welcome message."""
    print("=" * 60)
    print("  Agent Chat CLI")
    print("=" * 60)
    print("  Type your messages and press Enter to chat.")
    print("  Commands:")
    print("    /exit or /quit - Exit the chat")
    print("    /reset       - Reset conversation history")
    print("  Built-in tools: bash, read, write, search, ls, calculator, datetime, web_search")
    print("=" * 60)
    print()


def chat_loop(agent: AgentFramework):
    """Main chat loop."""
    print_welcome()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
            print("Goodbye!")
            break

        if user_input.lower() in ["/reset", "reset"]:
            # Recreate agent to reset memory
            agent = create_agent()
            print("Conversation history reset.\n")
            continue

        # Send to agent
        try:
            print("Agent: ", end="", flush=True)
            response = agent.run(user_input)
            print(f"{response}\n")
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    try:
        agent = create_agent()
        chat_loop(agent)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
