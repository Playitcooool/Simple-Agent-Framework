"""
List directory tool — list files in a directory.
"""

import os
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ListDirTool:
    """
    List directory contents (ls-like).

    Attributes:
        base_dir: Base directory for relative paths (default: current dir)
    """

    base_dir: str = "."

    def run(self, path: str = ".", all: bool = False, long: bool = False) -> str:
        """
        List directory contents.

        Args:
            path: Directory path to list
            all: If True, show hidden files (starting with .)
            long: If True, show detailed output (ls -l style)

        Returns:
            Formatted directory listing or error message
        """
        if ".." in path:
            return "[Error: Path traversal not allowed]"

        full_path = os.path.join(self.base_dir, path) if not os.path.isabs(path) else path

        if not os.path.exists(full_path):
            return f"[Error: Path does not exist: {path}]"

        if not os.path.isdir(full_path):
            return f"[Error: Path is not a directory: {path}]"

        try:
            entries = os.listdir(full_path)
            if not all:
                entries = [e for e in entries if not e.startswith('.')]

            if long:
                # Detailed output
                lines = []
                total = 0
                for entry in sorted(entries):
                    full_entry = os.path.join(full_path, entry)
                    try:
                        stat = os.stat(full_entry)
                        is_dir = os.path.isdir(full_entry)
                        size = stat.st_size
                        mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
                        prefix = 'd' if is_dir else '-'
                        lines.append(f"{prefix} {size:>10} {mtime} {entry}")
                        total += size
                    except Exception:
                        lines.append(f"?              ? {entry}")
                lines.insert(0, f"total {total}")
                return "\n".join(lines)
            else:
                # Simple output
                return "  ".join(sorted(entries)) or "[Empty directory]"

        except PermissionError:
            return f"[Error: Permission denied: {path}]"
        except Exception as e:
            return f"[Error listing {path}: {e}]"


# Decorator-based tool
def list_dir_tool(path: str = ".", all: bool = False, long: bool = False) -> str:
    """List directory. Args: path (str, default '.'), all (bool), long (bool)."""
    tool = ListDirTool()
    return tool.run(path, all, long)
