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


def test_message_role_values():
    assert MessageRole.SYSTEM.value == "system"
    assert MessageRole.USER.value == "user"
    assert MessageRole.ASSISTANT.value == "assistant"
    assert MessageRole.TOOL_RESULT.value == "tool_result"


def test_message_empty_content():
    msg = Message(role=MessageRole.ASSISTANT, content=None)
    assert msg.content is None