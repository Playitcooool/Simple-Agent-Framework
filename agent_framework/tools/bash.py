"""
Bash tool — executes shell commands.
"""

import subprocess
from dataclasses import dataclass


@dataclass
class BashTool:
    """
    Execute shell commands and return output.

    Attributes:
        timeout: Max seconds before killing process (default 30)
        cwd: Working directory for the command
    """

    timeout: int = 30
    cwd: str = None

    def run(self, command: str) -> str:
        """
        Execute a shell command.

        Args:
            command: The shell command to execute

        Returns:
            stdout + stderr combined as string
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.cwd,
            )
            output = result.stdout + result.stderr
            if result.returncode != 0:
                return f"[Exit {result.returncode}] {output}"
            return output if output else "[Command completed with no output]"
        except subprocess.TimeoutExpired:
            return f"[Error: Command timed out after {self.timeout}s]"
        except Exception as e:
            return f"[Error: {e}]"


# Decorator-based tool for global registration
def bash_tool(command: str, timeout: int = 30) -> str:
    """
    Execute a shell command.

    Args:
        command: Shell command string
        timeout: Seconds before killing (default 30)
    """
    tool = BashTool(timeout=timeout)
    return tool.run(command)
