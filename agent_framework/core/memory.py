# agent_framework/core/memory.py
from typing import List, Optional
from .message import Message, MessageRole
from .llm import LLM

DEFAULT_SUMMARY_PROMPT = (
    "Summarize the following conversation concisely, "
    "preserving key information:\n{content}"
)

class SummarizationMemory:
    """
    Conversation memory with periodic summarization to save tokens.
    """

    def __init__(
        self,
        llm: LLM,
        max_messages_before_summary: int = 10,
        summary_prompt: str = None,
    ):
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
        """Returns summary (if any) followed by recent messages"""
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
        content = "\n".join(f"[{m.role.value}] {m.content or ''}" for m in self.messages)
        prompt = self.summary_prompt.format(content=content)
        self.summary = self.llm.generate([Message(role=MessageRole.USER, content=prompt)])
        self.messages = []  # Clear original messages after summarization