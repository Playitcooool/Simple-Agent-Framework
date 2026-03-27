# agent_framework/core/message.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"

@dataclass
class Message:
    role: MessageRole
    content: str
    tool_call: Optional[dict] = None