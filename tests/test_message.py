# tests/test_message.py
from agent_framework.core.message import Message, MessageRole

def test_message_creation():
    msg = Message(role=MessageRole.USER, content="Hello")
    assert msg.role == MessageRole.USER
    assert msg.content == "Hello"
    assert msg.tool_call is None

def test_message_with_tool_call():
    msg = Message(
        role=MessageRole.ASSISTANT,
        content="I'll check the weather",
        tool_call={"name": "weather", "args": {"city": "Beijing"}}
    )
    assert msg.tool_call["name"] == "weather"
    assert msg.tool_call["args"]["city"] == "Beijing"