"""
File tools — read and write files.
"""

import os
from dataclasses import dataclass


@dataclass
class ReadFileTool:
    """
    Read the contents of a file.

    Attributes:
        base_dir: Base directory for relative paths (default: current dir)
    """

    base_dir: str = "."

    def run(self, path: str, lines: int = None) -> str:
        """
        Read a file.

        Args:
            path: File path (absolute or relative to base_dir)
            lines: If set, read only first N lines

        Returns:
            File contents as string, or error message
        """
        full_path = os.path.join(self.base_dir, path) if not os.path.isabs(path) else path
        if ".." in path:
            return "[Error: Path traversal not allowed]"
        try:
            if lines is not None:
                with open(full_path, "r") as f:
                    return "".join(f.readline() for _ in range(lines))
            with open(full_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            return f"[Error: File not found: {path}]"
        except IsADirectoryError:
            return f"[Error: Path is a directory: {path}]"
        except Exception as e:
            return f"[Error reading {path}: {e}]"


@dataclass
class WriteFileTool:
    """
    Write content to a file.

    Attributes:
        base_dir: Base directory for relative paths (default: current dir)
    """

    base_dir: str = "."

    def run(self, path: str, content: str, append: bool = False) -> str:
        """
        Write content to a file.

        Args:
            path: File path (absolute or relative to base_dir)
            content: Text content to write
            append: If True, append instead of overwrite

        Returns:
            Success or error message
        """
        full_path = os.path.join(self.base_dir, path) if not os.path.isabs(path) else path
        if ".." in path:
            return "[Error: Path traversal not allowed]"
        try:
            mode = "a" if append else "w"
            # Ensure parent directory exists
            parent = os.path.dirname(full_path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            with open(full_path, mode) as f:
                f.write(content)
            action = "Appended to" if append else "Wrote"
            return f"[OK: {action} {path} ({len(content)} chars)]"
        except Exception as e:
            return f"[Error writing {path}: {e}]"


# Decorator-based tools
def read_file_tool(path: str, lines: int = None) -> str:
    """Read a file. Args: path (str), lines (optional int)."""
    tool = ReadFileTool()
    return tool.run(path, lines)


def write_file_tool(path: str, content: str, append: bool = False) -> str:
    """Write to a file. Args: path (str), content (str), append (bool, default False)."""
    tool = WriteFileTool()
    return tool.run(path, content, append)
