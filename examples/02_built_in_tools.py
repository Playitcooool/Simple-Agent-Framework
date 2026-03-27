"""
Example: Using built-in tools (Bash, Read, Write)

This example demonstrates:
1. Using BashTool to run shell commands
2. Using ReadFileTool to read files
3. Using WriteFileTool to write files
4. Registering tools with the @tool decorator
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_framework import AgentFramework, OpenAILLM, tool, BashTool, ReadFileTool, WriteFileTool

# Check for API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Please set OPENAI_API_KEY environment variable")
    sys.exit(1)

# Initialize framework
fw = AgentFramework(llm=OpenAILLM(api_key=api_key), mode="react")


# --- Register built-in tools using the BashTool, ReadFileTool, WriteFileTool classes ---
# These are class-based tools with state (e.g., cwd for BashTool)

bash = BashTool(cwd="/tmp", timeout=10)
read_file = ReadFileTool(base_dir="/tmp")
write_file = WriteFileTool(base_dir="/tmp")


@fw.tool(name="bash", description="Execute a shell command in /tmp. Args: command (str)")
def bash_cmd(command: str) -> str:
    return bash.run(command)


@fw.tool(name="read", description="Read a file from /tmp. Args: path (str), lines (optional int)")
def read_cmd(path: str, lines: int = None) -> str:
    return read_file.run(path, lines)


@fw.tool(name="write", description="Write to a file in /tmp. Args: path (str), content (str), append (bool, default False)")
def write_cmd(path: str, content: str, append: bool = False) -> str:
    return write_file.run(path, content, append)


def main():
    print("=" * 60)
    print("Example: Built-in Tools (Bash, Read, Write)")
    print("=" * 60)

    # Example 1: Write a file
    print("\n[1] Writing a file...")
    result = fw.run(
        'Write "Hello from Agent Framework!" to a file called greeting.txt in /tmp'
    )
    print(f"Result: {result}")

    # Example 2: Read the file back
    print("\n[2] Reading the file back...")
    result = fw.run("Read the file greeting.txt from /tmp")
    print(f"Result: {result}")

    # Example 3: Run a bash command
    print("\n[3] Running a bash command (ls -la /tmp)...")
    result = fw.run("Run the command: ls -la /tmp")
    print(f"Result: {result}")

    # Example 4: Combined operation — write and then execute
    print("\n[4] Writing a script and running it...")
    result = fw.run(
        'Write "#!/bin/bash\necho \\"Hello World!\\"" to a file hello.sh, '
        'then make it executable and run it'
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
