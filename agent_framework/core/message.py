"""
Message module for the Agent framework.

Provides the Message dataclass and MessageRole enum for representing
conversation messages between agents, users, and tools.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

__all__ = ["Message", "MessageRole"]


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"

@dataclass
class Message:
    role: MessageRole
    content: Optional[str] = None
    tool_call: Optional[dict] = None