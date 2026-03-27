# tests/test_memory.py
from agent_framework.core.memory import SummarizationMemory
from agent_framework.core.message import Message, MessageRole
from agent_framework.core.llm import LLM

class FakeLLMForMemory(LLM):
    def __init__(self, summary_response: str = "This is a summary."):
        self.response = summary_response
        self.called = False
        self.call_count = 0

    def generate(self, messages):
        self.called = True
        self.call_count += 1
        return self.response

def test_memory_add_and_get():
    llm = FakeLLMForMemory()
    mem = SummarizationMemory(llm=llm, max_messages_before_summary=4)

    mem.add(Message(role=MessageRole.USER, content="Hello"))
    mem.add(Message(role=MessageRole.ASSISTANT, content="Hi there!"))
    mem.add(Message(role=MessageRole.USER, content="How are you?"))

    msgs = mem.get_messages()
    assert len(msgs) == 3
    assert msgs[0].content == "Hello"

def test_memory_summarize_trigger():
    llm = FakeLLMForMemory(summary_response="Summarized conversation.")
    mem = SummarizationMemory(llm=llm, max_messages_before_summary=2)

    mem.add(Message(role=MessageRole.USER, content="msg1"))
    mem.add(Message(role=MessageRole.ASSISTANT, content="msg2"))

    # 触发摘要（2条消息达到阈值）
    assert llm.called is True  # 摘要已触发
    assert mem.summary == "Summarized conversation."
    assert mem.messages == []  # 原始消息已清除