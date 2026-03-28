"""
Search tool — search for text patterns in files.
"""

import os
import re
from dataclasses import dataclass


@dataclass
class SearchTool:
    """
    Search for text patterns in files (grep-like).

    Attributes:
        base_dir: Base directory for relative paths (default: current dir)
    """

    base_dir: str = "."

    def run(self, pattern: str, path: str = ".", case_sensitive: bool = False) -> str:
        """
        Search for a pattern in files.

        Args:
            pattern: Regular expression pattern to search for
            path: File or directory path to search in
            case_sensitive: Whether to match case (default False)

        Returns:
            Matching lines with line numbers, or error message
        """
        if ".." in path:
            return "[Error: Path traversal not allowed]"

        full_path = os.path.join(self.base_dir, path) if not os.path.isabs(path) else path

        flags = 0 if case_sensitive else re.IGNORECASE

        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"[Error: Invalid regex pattern: {e}]"

        results = []

        if os.path.isfile(full_path):
            files_to_search = [full_path]
        elif os.path.isdir(full_path):
            files_to_search = []
            for root, dirs, files in os.walk(full_path):
                # Skip hidden directories and common exclusions
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules', '.git')]
                for file in files:
                    if not file.startswith('.'):
                        files_to_search.append(os.path.join(root, file))
        else:
            return f"[Error: Path does not exist: {path}]"

        for filepath in files_to_search:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            # Escape control characters for display
                            display_line = line.rstrip()[:200]
                            results.append(f"{filepath}:{line_num}: {display_line}")
            except Exception as e:
                results.append(f"[Error reading {filepath}: {e}]")

        if not results:
            return "[No matches found]"

        return "\n".join(results[:100])  # Limit to 100 results


# Decorator-based tool
def search_tool(pattern: str, path: str = ".", case_sensitive: bool = False) -> str:
    """Search for text in files. Args: pattern (str), path (str, default '.'), case_sensitive (bool)."""
    tool = SearchTool()
    return tool.run(pattern, path, case_sensitive)
