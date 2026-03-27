"""
Memory module — conversation context management with summarization.

SummarizationMemory accumulates messages, then compresses them via LLM
when the count exceeds max_messages_before_summary.
"""

from typing import List, Optional
from .message import Message, MessageRole
from .llm import LLM

DEFAULT_SUMMARY_PROMPT = (
    "Summarize the following conversation concisely, "
    "preserving key information:\n{content}"
)

__all__ = ["SummarizationMemory"]


class SummarizationMemory:
    """
    Conversation memory that periodically summarizes to save tokens.

    Flow:
    1. Messages accumulate in self.messages
    2. When count >= max_messages_before_summary, summarize via LLM
    3. Original messages cleared, summary stored
    4. get_messages() returns [summary_msg] + recent_messages
    """

    def __init__(self, llm: LLM, max_messages_before_summary: int = 10,
                 summary_prompt: str = None):
        self.llm = llm
        self.max_messages_before_summary = max_messages_before_summary
        self.summary_prompt = summary_prompt or DEFAULT_SUMMARY_PROMPT
        self.messages: List[Message] = []
        self.summary: Optional[str] = None

    def add(self, message: Message) -> None:
        self.messages.append(message)
        if self._should_summarize():
            self._summarize()

    def get_messages(self) -> List[Message]:
        result: List[Message] = []
        if self.summary:
            result.append(Message(
                role=MessageRole.SYSTEM,
                content=f"Previous conversation summary: {self.summary}"
            ))
        result.extend(self.messages)
        return result

    def _should_summarize(self) -> bool:
        return len(self.messages) >= self.max_messages_before_summary

    def _summarize(self) -> None:
        content = "\n".join(
            f"[{m.role.value}] {m.content or ''}"
            for m in self.messages
        )
        prompt = self.summary_prompt.format(content=content)
        self.summary = self.llm.generate([Message(role=MessageRole.USER, content=prompt)])
        self.messages = []
