# -*- coding: utf-8 -*-
"""Regression test for issue #388: Kimi 2.5 reasoning_content preservation.

This test ensures that reasoning_content is preserved when formatting messages
with tool calls for Moonshot API (Kimi models with thinking mode enabled).
"""
import pytest
from agentscope.message import Msg

from copaw.agents.memory.memory_manager import TimestampedDashScopeChatFormatter


@pytest.mark.asyncio
async def test_reasoning_content_preserved_with_tool_calls():
    """Test that reasoning_content is preserved in assistant messages with tool_calls.

    When Kimi's thinking mode is enabled, the Moonshot API requires that every
    assistant message containing tool calls must also include a reasoning_content
    field in the conversation history. This test ensures the formatter preserves
    this field from message metadata.

    Regression test for: https://github.com/agentscope-ai/CoPaw/issues/388
    """
    formatter = TimestampedDashScopeChatFormatter(
        memory_compact_threshold=100000,
    )

    # Create a message simulating Kimi's response with reasoning_content and tool_calls
    msg = Msg(
        name="assistant",
        role="assistant",
        content=[
            {
                "type": "text",
                "text": "I need to read the file to help you.",
            },
            {
                "type": "tool_use",
                "id": "call_123",
                "name": "read_file",
                "input": {"path": "/tmp/test.txt"},
            },
        ],
        timestamp="2024-01-01T00:00:00",
    )

    # Add reasoning_content to message metadata (simulates Kimi API response)
    msg.metadata = {
        "reasoning_content": "The user wants to know about the file content. I should use the read_file tool to retrieve it.",
    }

    # Format the messages
    formatted = await formatter._format([msg])

    # Find the assistant message with tool_calls
    assistant_msg_with_tools = None
    for formatted_msg in formatted:
        if (
            formatted_msg.get("role") == "assistant"
            and "tool_calls" in formatted_msg
        ):
            assistant_msg_with_tools = formatted_msg
            break

    # Verify message was found
    assert assistant_msg_with_tools is not None, "No assistant message with tool_calls found"

    # Verify reasoning_content is preserved
    assert "reasoning_content" in assistant_msg_with_tools, (
        "reasoning_content is missing in assistant message with tool_calls. "
        "This will cause Moonshot API to reject the request with 400 error when "
        "thinking mode is enabled."
    )

    # Verify the reasoning_content value is correct
    expected_reasoning = "The user wants to know about the file content. I should use the read_file tool to retrieve it."
    assert assistant_msg_with_tools["reasoning_content"] == expected_reasoning, (
        f"reasoning_content value mismatch: expected '{expected_reasoning}', "
        f"got '{assistant_msg_with_tools.get('reasoning_content')}'"
    )


@pytest.mark.asyncio
async def test_reasoning_content_from_thinking_block():
    """Test that reasoning_content is extracted from ThinkingBlock.

    When models return structured ThinkingBlock, the formatter should extract
    the thinking content and add it as reasoning_content field.
    """
    formatter = TimestampedDashScopeChatFormatter(
        memory_compact_threshold=100000,
    )

    # Create a message with ThinkingBlock
    msg = Msg(
        name="assistant",
        role="assistant",
        content=[
            {
                "type": "thinking",
                "thinking": "Let me analyze the user's request carefully.",
            },
            {
                "type": "text",
                "text": "I'll help you with that.",
            },
            {
                "type": "tool_use",
                "id": "call_456",
                "name": "search",
                "input": {"query": "test"},
            },
        ],
        timestamp="2024-01-01T00:00:00",
    )

    formatted = await formatter._format([msg])

    assistant_msg = None
    for formatted_msg in formatted:
        if formatted_msg.get("role") == "assistant":
            assistant_msg = formatted_msg
            break

    assert assistant_msg is not None
    assert "reasoning_content" in assistant_msg
    assert assistant_msg["reasoning_content"] == "Let me analyze the user's request carefully."


@pytest.mark.asyncio
async def test_reasoning_content_priority():
    """Test that ThinkingBlock takes priority over metadata.

    When both ThinkingBlock and metadata contain reasoning_content,
    ThinkingBlock should take priority as it's the structured source.
    """
    formatter = TimestampedDashScopeChatFormatter(
        memory_compact_threshold=100000,
    )

    msg = Msg(
        name="assistant",
        role="assistant",
        content=[
            {
                "type": "thinking",
                "thinking": "Thinking from block",
            },
            {
                "type": "text",
                "text": "Response text",
            },
        ],
        timestamp="2024-01-01T00:00:00",
    )
    msg.metadata = {
        "reasoning_content": "Thinking from metadata",
    }

    formatted = await formatter._format([msg])

    assistant_msg = formatted[0]
    # ThinkingBlock should take priority
    assert assistant_msg["reasoning_content"] == "Thinking from block"


@pytest.mark.asyncio
async def test_reasoning_content_type_validation():
    """Test that non-string reasoning_content in metadata is ignored.

    Only string values should be preserved as reasoning_content.
    """
    formatter = TimestampedDashScopeChatFormatter(
        memory_compact_threshold=100000,
    )

    # Test with non-string metadata
    msg = Msg(
        name="assistant",
        role="assistant",
        content=[
            {
                "type": "text",
                "text": "Test message",
            },
        ],
        timestamp="2024-01-01T00:00:00",
    )
    msg.metadata = {
        "reasoning_content": 123,  # Invalid: not a string
    }

    formatted = await formatter._format([msg])

    assistant_msg = formatted[0]
    # Non-string value should be ignored
    assert "reasoning_content" not in assistant_msg


@pytest.mark.asyncio
async def test_reasoning_content_empty_string():
    """Test that empty string reasoning_content IS preserved.

    Even empty strings should be preserved because Kimi API requires
    the reasoning_content field to be present when thinking mode is enabled,
    regardless of whether the content is empty or not.
    """
    formatter = TimestampedDashScopeChatFormatter(
        memory_compact_threshold=100000,
    )

    msg = Msg(
        name="assistant",
        role="assistant",
        content=[
            {
                "type": "text",
                "text": "Test message",
            },
        ],
        timestamp="2024-01-01T00:00:00",
    )
    msg.metadata = {
        "reasoning_content": "",  # Empty string should be preserved
    }

    formatted = await formatter._format([msg])

    assistant_msg = formatted[0]
    # Empty string SHOULD be preserved for Kimi API compatibility
    assert "reasoning_content" in assistant_msg, (
        "Empty string reasoning_content should be preserved. "
        "Kimi API requires the field to be present even if empty."
    )
    assert assistant_msg["reasoning_content"] == ""


@pytest.mark.asyncio
async def test_reasoning_content_without_tool_calls():
    """Test that reasoning_content is preserved even without tool_calls.

    While Kimi requires reasoning_content with tool_calls, the formatter
    should preserve it in all cases for consistency.
    """
    formatter = TimestampedDashScopeChatFormatter(
        memory_compact_threshold=100000,
    )

    msg = Msg(
        name="assistant",
        role="assistant",
        content=[
            {
                "type": "text",
                "text": "Just a regular response",
            },
        ],
        timestamp="2024-01-01T00:00:00",
    )
    msg.metadata = {
        "reasoning_content": "My reasoning process",
    }

    formatted = await formatter._format([msg])

    assistant_msg = formatted[0]
    # Should preserve reasoning_content even without tool_calls
    assert "reasoning_content" in assistant_msg
    assert assistant_msg["reasoning_content"] == "My reasoning process"


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_reasoning_content_preserved_with_tool_calls())
    asyncio.run(test_reasoning_content_from_thinking_block())
    asyncio.run(test_reasoning_content_priority())
    asyncio.run(test_reasoning_content_type_validation())
    asyncio.run(test_reasoning_content_empty_string())
    asyncio.run(test_reasoning_content_without_tool_calls())

