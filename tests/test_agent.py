# tests/test_agent.py
import pytest
from agent_framework.core.agent import BaseAgent, ReActAgent, PlanAndExecuteAgent
from agent_framework.core.llm import LLM
from agent_framework.core.message import Message, MessageRole
from agent_framework.core.tool import tool, ActionExecutor, get_registry

class FakeLLM(LLM):
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    def generate(self, messages):
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp

def test_react_agent_finishes_with_final_answer():
    @tool(name="weather", description="Get weather")
    def weather(city: str) -> str:
        return f"{city} is sunny"

    # LLM: 第一轮返回 thought + action，第二轮返回 FINAL_ANSWER
    llm = FakeLLM(responses=[
        'Thought: I should check the weather.\nAction: weather\nAction Args: {"city": "Beijing"}',
        'Thought: I have the info.\nFinal Answer: Beijing is sunny.',
    ])
    registry = get_registry()
    executor = ActionExecutor(registry)
    agent = ReActAgent(llm=llm, executor=executor, max_turns=10)

    result = agent.run("What's the weather in Beijing?")
    assert "sunny" in result

def test_plan_agent():
    @tool(name="search", description="Search")
    def search(query: str) -> str:
        return f"Results for {query}"

    llm = FakeLLM(responses=[
        'Step 1: search for "AI"\nStep 2: summarize results',  # plan
        'Summary: AI is great.',  # synthesize
    ])
    registry = get_registry()
    executor = ActionExecutor(registry)
    agent = PlanAndExecuteAgent(llm=llm, executor=executor, max_turns=10)

    result = agent.run("Tell me about AI")
    assert "great" in result