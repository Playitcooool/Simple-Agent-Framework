import pytest
from unittest.mock import patch, MagicMock
from agent_framework.core.llm import LLM, OpenAILLM, LLMError
from agent_framework.core.message import Message, MessageRole


def test_llm_is_abstract():
    with pytest.raises(TypeError):
        LLM()


class FakeLLM(LLM):
    def __init__(self, response: str):
        self.response = response
        self.called_with = None

    def generate(self, messages):
        self.called_with = messages
        return self.response


def test_openai_llm_init():
    llm = OpenAILLM(api_key="test-key", model="gpt-4o")
    assert llm.api_key == "test-key"
    assert llm.model == "gpt-4o"


@patch("agent_framework.core.llm.requests.post")
def test_openai_llm_generate_returns_response_text(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hello, world!"}}]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    llm = OpenAILLM(api_key="test-key")
    messages = [Message(role=MessageRole.USER, content="Hi")]
    result = llm.generate(messages)

    assert result == "Hello, world!"
    mock_post.assert_called_once()


@patch("agent_framework.core.llm.requests.post")
def test_openai_llm_generate_raises_llm_error_on_malformed_response(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"choices": []}  # Missing expected structure
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    llm = OpenAILLM(api_key="test-key")
    messages = [Message(role=MessageRole.USER, content="Hi")]

    with pytest.raises(LLMError, match="Unexpected API response structure"):
        llm.generate(messages)


@patch("agent_framework.core.llm.requests.post")
def test_openai_llm_generate_handles_network_error(mock_post):
    import requests
    mock_post.side_effect = requests.RequestException("Connection failed")

    llm = OpenAILLM(api_key="test-key")
    messages = [Message(role=MessageRole.USER, content="Hi")]

    with pytest.raises(LLMError, match="Network error during API request"):
        llm.generate(messages)