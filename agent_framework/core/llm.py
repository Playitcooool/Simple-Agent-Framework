from abc import ABC, abstractmethod
from typing import List
import requests

from .message import Message

DEFAULT_OPENAI_URL = "https://api.openai.com/v1/chat/completions"


class LLMError(Exception):
    """LLM API 调用错误"""
    pass


class LLM(ABC):
    """LLM abstract base class"""

    @abstractmethod
    def generate(self, messages: List[Message]) -> str:
        """Given a list of messages, return the LLM generated text"""
        pass


class OpenAILLM(LLM):
    """OpenAI Chat Completions API"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        api_url: str = DEFAULT_OPENAI_URL,
        timeout: int = 60,
    ):
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
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            raise LLMError(f"Network error during API request: {e}")
        except (KeyError, ValueError, IndexError) as e:
            raise LLMError(f"Unexpected API response structure: {e}")