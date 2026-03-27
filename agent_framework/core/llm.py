"""
LLM (Large Language Model) module.

PURPOSE:
  Provides an abstraction layer between our Agent framework and the actual
  LLM provider (OpenAI, Anthropic, local model, etc.).

KEY CONCEPTS:
  - LLM: Abstract base class defining the interface
  - OpenAILLM: Concrete implementation for OpenAI's API
  - LLMError: Custom exception for LLM-related errors

WHY AN ABSTRACT BASE CLASS (ABC)?
  We want our Agent code to work with ANY LLM, not just OpenAI.
  By defining an abstract interface, we can:
  1. Write Agents once, using only the `LLM` interface
  2. Swap LLMs at runtime (e.g., OpenAI for production, a mock for testing)
  3. Add new LLM providers without changing Agent code

  This is the "Strategy Pattern" — different algorithms (LLMs)
  that implement the same interface.

WHY SEPARATE ERROR HANDLING?
  LLM API calls can fail in two fundamentally different ways:
  1. Network problems (no internet, server down) — RequestException
  2. The API returned something unexpected — KeyError, IndexError

  Both are serious but require different handling. By wrapping them in
  LLMError, we give callers a clean way to catch any LLM problem.
"""

from abc import ABC, abstractmethod
from typing import List
import requests

from .message import Message

# Default endpoint for OpenAI's Chat Completions API
# Note: This can be overridden to point to a proxy or custom endpoint
DEFAULT_OPENAI_URL = "https://api.openai.com/v1/chat/completions"


class LLMError(Exception):
    """
    Raised when an LLM API call fails.

    This is a wrapper exception that captures both network errors
    and unexpected API response formats under one error type.

    Why wrap instead of letting original exceptions propagate?
    1. Calling code only needs to catch LLMError, not RequestException + KeyError...
    2. We can provide more context in the error message
    3. The original exception details are preserved in the chain (via `from e`)
    """


class LLM(ABC):
    """
    Abstract base class for all LLM implementations.

    DESIGN PRINCIPLE — ONE METHOD, ONE RESPONSIBILITY:
      The `generate` method is the ONLY public interface.
      Everything else (API keys, model names, endpoints) is configuration.

      This keeps the interface simple: you give it messages, you get text back.
      How that happens is an implementation detail.

    ABSTRACT METHOD — WHAT IT MEANS:
      Any class that inherits from LLM MUST implement generate().
      Trying to instantiate LLM directly raises TypeError.
      This is enforced by Python's ABC machinery.

    WHY ACCEPT List[Message]?
      The LLM needs full conversation context to generate good responses.
      Each message tells the LLM who said what, so it can maintain
      coherence across the conversation.
    """

    @abstractmethod
    def generate(self, messages: List[Message]) -> str:
        """
        Generate a response given a list of conversation messages.

        Args:
            messages: A list of Message objects representing the
                     entire conversation history.

        Returns:
            The LLM's generated text response.

        Raises:
            LLMError: If the API call fails or returns an unexpected format.
        """
        pass


class OpenAILLM(LLM):
    """
    OpenAI Chat Completions API implementation.

    This class handles:
    1. Constructing the correct HTTP request
    2. Converting our Message objects to OpenAI's format
    3. Extracting the response from OpenAI's JSON structure

    API REFERENCE:
      OpenAI's Chat Completions API expects:
      {
        "model": "gpt-4o",
        "messages": [
          {"role": "system", "content": "You are helpful."},
          {"role": "user", "content": "Hello!"}
        ]
      }

      And returns:
      {
        "choices": [
          {"message": {"role": "assistant", "content": "Hello! How can I help?"}}
        ]
      }
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        api_url: str = DEFAULT_OPENAI_URL,
        timeout: int = 60,
    ):
        """
        Initialize the OpenAI LLM adapter.

        Args:
            api_key: Your OpenAI API key. Keep this secret!
                     In production, consider using environment variables.
            model: The OpenAI model to use. Options include "gpt-4o", "gpt-4",
                   "gpt-3.5-turbo". Default is "gpt-4o".
            api_url: The API endpoint URL. Defaults to OpenAI's server.
                     Can be overridden for testing or proxies.
            timeout: How long to wait (in seconds) before giving up on the API.
                     Default 60s is usually sufficient, but complex tasks
                     may need more time.
        """
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.timeout = timeout

    def generate(self, messages: List[Message]) -> str:
        """
        Send messages to OpenAI's API and return the generated response.

        Implementation notes:
        1. We use "Bearer" authentication — standard OAuth2 style
        2. The model name goes in the request body (not the URL)
        3. We convert Message objects: role -> MessageRole.value, content -> content
        4. m.content or "" handles None content safely for the API
        """
        # HTTP headers — "Bearer {api_key}" is the standard auth format
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # The request body for OpenAI's Chat Completions API
        # Note: We convert Message objects to dicts that OpenAI expects
        # The 'or ""' handles None content — OpenAI requires a string
        payload = {
            "model": self.model,
            "messages": [
                {"role": m.role.value, "content": m.content or ""}
                for m in messages
            ],
        }

        try:
            # Make the HTTP POST request
            # requests.post returns a Response object
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,  # requests automatically serializes dict to JSON
                timeout=self.timeout,
            )

            # raise_for_status() raises an exception for 4xx/5xx responses
            # This saves us from silently ignoring HTTP errors
            response.raise_for_status()

            # Parse the JSON response
            # OpenAI returns: {"choices": [{"message": {"content": "..."}}]}
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.RequestException as e:
            # Network errors (ConnectionError, Timeout, etc.)
            # These are different from API errors — the request didn't even complete
            raise LLMError(f"Network error during API request: {e}")

        except (KeyError, ValueError, IndexError) as e:
            # API returned something unexpected:
            # - KeyError: missing expected JSON keys
            # - ValueError: JSON decode failed (shouldn't happen with valid API response)
            # - IndexError: "choices" list is empty or malformed
            raise LLMError(f"Unexpected API response structure: {e}")
