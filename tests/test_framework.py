import pytest
from agent_framework.framework import AgentFramework
from agent_framework.core.llm import LLM
from agent_framework.core.message import Message, MessageRole

class FakeLLM(LLM):
    def __init__(self, response: str):
        self.response = response
    def generate(self, messages):
        return self.response

def test_framework_run_react():
    fw = AgentFramework(llm=FakeLLM("Thought: done.\nFinal Answer: 42."), mode="react")
    result = fw.run("What is 6 * 7?")
    assert result == "42."

def test_framework_run_plan():
    fw = AgentFramework(llm=FakeLLM("Step 1: 6*7\nStep 2: answer\nSummary: 42"), mode="plan")
    result = fw.run("What is 6 * 7?")
    assert "42" in result

def test_framework_unknown_mode():
    fw = AgentFramework(llm=FakeLLM("response"), mode="invalid")
    with pytest.raises(ValueError, match="Unknown mode"):
        fw.run("test task")