# tests/test_supervisor.py
from agent_framework.multi.supervisor import SupervisorAgent
from agent_framework.core.agent import BaseAgent
from agent_framework.core.llm import LLM
from agent_framework.core.message import Message, MessageRole
from agent_framework.core.tool import ActionExecutor, get_registry


class FakeLLM(LLM):
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    def generate(self, messages):
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp


class DummyAgent(BaseAgent):
    def __init__(self, name: str, llm: LLM, result: str):
        super().__init__(llm=llm, executor=ActionExecutor(get_registry()))
        self.name = name
        self._result = result

    def run(self, task: str) -> str:
        return self._result


def test_supervisor_decomposes_and_delegates():
    llm = FakeLLM(responses=[
        "Subtask 1: research, Subtask 2: write",
        "Final synthesized answer: Done.",
    ])
    sub1 = DummyAgent("researcher", llm, "Research complete.")
    sub2 = DummyAgent("writer", llm, "Writing complete.")

    supervisor = SupervisorAgent(llm=llm, sub_agents=[sub1, sub2])
    result = supervisor.run("Make me a report")

    assert "Done" in result
    assert llm.call_count == 2  # 分解 + 汇总