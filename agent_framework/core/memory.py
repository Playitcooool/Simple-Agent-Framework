"""
Memory module for conversation context management.

PURPOSE:
 LLMs have a context window (e.g., 128K tokens for GPT-4o).
 As conversations grow, we might exceed this limit.
 Memory manages the conversation history intelligently.

KEY CONCEPTS:
  - SummarizationMemory: Periodically compresses conversation history
  - Trade-off: We lose some detail but save tokens

WHY SUMMARIZATION?
  Alternative approaches:
  1. Truncate to last N messages — loses early context
  2. Vector embeddings + retrieval — complex, needs a vector DB
  3. Summarization — middle ground, preserves key info in compressed form

  Summarization is simple, requires no external services,
  and preserves a "gist" of the conversation.

HOW IT WORKS:
  1. We accumulate messages in a list
  2. When we reach max_messages_before_summary, we call the LLM
     to summarize the entire conversation
  3. We discard the original messages and keep only the summary
  4. Future messages are added on top of the summary

TRADE-OFFS TO UNDERSTAND:
  - Pros: Bounded memory usage, no external dependencies
  - Cons: Summary may lose nuance, summarization costs tokens/calls

  For production, consider hybrid approaches:
  - Recent messages: Keep as-is (important for current context)
  - Old messages: Store embeddings, retrieve relevant ones
  - Very old: Discard or summarize
"""

from typing import List, Optional
from .message import Message, MessageRole
from .llm import LLM

# Default prompt for summarization
# The LLM is asked to preserve "key information"
# This is intentionally vague — what counts as "key" depends on the conversation
DEFAULT_SUMMARY_PROMPT = (
    "Summarize the following conversation concisely, "
    "preserving key information:\n{content}"
)


__all__ = ["SummarizationMemory"]


class SummarizationMemory:
    """
    Manages conversation history with periodic summarization.

    This memory strategy:
    1. Stores messages in a list
    2. When the list gets too long, summarizes and replaces with a short summary
    3. Returns both summary (as a SYSTEM message) and recent messages

    WHY USE A SYSTEM MESSAGE FOR SUMMARY?
      The summary becomes part of the conversation context.
      By putting it as a SYSTEM message, it appears before all other messages,
      ensuring the LLM sees the summary first and understands the conversation
      context before processing recent messages.
    """

    def __init__(
        self,
        llm: LLM,
        max_messages_before_summary: int = 10,
        summary_prompt: str = None,
    ):
        """
        Initialize the memory.

        Args:
            llm: The LLM to use for summarization.
                 Note: This is typically a different (possibly cheaper/faster)
                 model than the main agent's model.
            max_messages_before_summary: How many messages to accumulate
                                         before summarizing. Default 10.
            summary_prompt: Custom prompt for the summarization LLM call.
                          If None, uses DEFAULT_SUMMARY_PROMPT.
        """
        self.llm = llm
        self.max_messages_before_summary = max_messages_before_summary
        # Use custom prompt if provided, otherwise default
        self.summary_prompt = summary_prompt or DEFAULT_SUMMARY_PROMPT
        self.messages: List[Message] = []  # Current un-summarized messages
        self.summary: Optional[str] = None  # The compressed summary

    def add(self, message: Message) -> None:
        """
        Add a message to memory.

        This is called after every LLM turn (user message or assistant response).
        If we've accumulated enough messages, we automatically summarize.

        SIDE EFFECT:
          This may trigger summarization, which:
          1. Calls the LLM (side effect — API call)
          2. Clears self.messages
          3. Sets self.summary
        """
        self.messages.append(message)
        if self._should_summarize():
            self._summarize()

    def get_messages(self) -> List[Message]:
        """
        Get the current conversation context for the Agent.

        Returns:
            A list of messages: [summary (as SYSTEM msg), ...recent messages]

        IMPORTANT — ORDER MATTERS:
          The summary comes FIRST, followed by recent messages.
          This ensures the LLM sees:
          1. "Here's the gist of what we discussed before..."
          2. "And here are the most recent messages..."
        """
        result: List[Message] = []
        if self.summary:
            # Include the summary as a SYSTEM message
            # SYSTEM messages are special — they set the context/instructions
            result.append(Message(
                role=MessageRole.SYSTEM,
                content=f"Previous conversation summary: {self.summary}"
            ))
        result.extend(self.messages)
        return result

    def _should_summarize(self) -> bool:
        """
        Check if we have enough messages to trigger summarization.

        We summarize when we have >= max_messages_before_summary.
        Using >= (not >) means we summarize AT the threshold.
        """
        return len(self.messages) >= self.max_messages_before_summary

    def _summarize(self) -> None:
        """
        Compress the current messages into a summary.

        This is the core summarization logic:
        1. Format all messages as text
        2. Send to LLM with summarization prompt
        3. Store the result as self.summary
        4. Clear the original messages (they're now summarized)

        SIDE EFFECT: This calls the LLM.
        """
        # Format messages for the summarization prompt
        # Format: "[role] message_content"
        content = "\n".join(
            f"[{m.role.value}] {m.content or ''}"
            for m in self.messages
        )

        # Build the prompt with the actual content
        prompt = self.summary_prompt.format(content=content)

        # Call the LLM to summarize
        self.summary = self.llm.generate([Message(role=MessageRole.USER, content=prompt)])

        # Clear the original messages — they're now represented by the summary
        self.messages = []
