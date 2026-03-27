"""
Message module — defines Message and MessageRole for conversation representation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

__all__ = ["Message", "MessageRole"]


class MessageRole(Enum):
    """Role of the message sender in a conversation."""
    SYSTEM = "system"       # System instructions
    USER = "user"           # End user
    ASSISTANT = "assistant" # LLM response
    TOOL_RESULT = "tool_result"  # Tool execution output


@dataclass
class Message:
    """
    A single message in a conversation.

    Attributes:
        role: Who is sending this message
        content: Text content (can be None for tool-only responses)
        tool_call: Optional tool invocation, e.g. {"name": "weather", "args": {"city": "Beijing"}}
    """
    role: MessageRole
    content: Optional[str] = None
    tool_call: Optional[dict] = None
