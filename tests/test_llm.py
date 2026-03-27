import pytest
from agent_framework.core.llm import LLM, OpenAILLM
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