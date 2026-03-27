"""
LLM module — abstraction layer for LLM providers.

Provides:
- LLM: Abstract base class (Strategy pattern)
- OpenAILLM: OpenAI Chat Completions implementation
- LLMError: Unified exception for API errors
"""

from abc import ABC, abstractmethod
from typing import List
import requests

from .message import Message

DEFAULT_OPENAI_URL = "https://api.openai.com/v1/chat/completions"


class LLMError(Exception):
    """Raised when LLM API call fails (network or API error)."""


class LLM(ABC):
    """Abstract base class for LLM implementations."""

    @abstractmethod
    def generate(self, messages: List[Message]) -> str:
        """Send messages to LLM, return generated text."""
        pass


class OpenAILLM(LLM):
    """OpenAI Chat Completions API implementation."""

    def __init__(self, api_key: str, model: str = "gpt-4o",
                 api_url: str = DEFAULT_OPENAI_URL, timeout: int = 60):
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.timeout = timeout

    def generate(self, messages: List[Message]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": m.role.value, "content": m.content or ""} for m in messages],
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            raise LLMError(f"Network error during API request: {e}")
        except (KeyError, ValueError, IndexError) as e:
            raise LLMError(f"Unexpected API response structure: {e}")
