"""
claude_sdk_worker.py

Standalone subprocess worker for the Claude Agent SDK.
Reads a JSON payload from stdin, streams SDK messages, and writes
newline-delimited JSON events to stdout.

Event types:
  {"type": "tool_start", "name": "<tool>", "id": "<id>", "input": {}}
  {"type": "tool_end",   "id": "<id>"}
  {"type": "result",     "text": "<final result>"}
  {"type": "system",     "text": "<system message>"}
  {"type": "error",      "text": "<error message>"}
  {"type": "session",    "id": "<session_id>"}
"""

import sys
import json
import anyio

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    UserMessage,
    SystemMessage,
    ResultMessage,
    ToolUseBlock,
    ToolResultBlock,
)


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


async def process_messages(
    client: ClaudeSDKClient, pending_tools: set[str]
) -> tuple[bool, str | None]:
    """
    Drain receive_messages() for one query() turn.

    Returns:
        (needs_continuation, raw_tool_call_text)
        - needs_continuation=True means a <tool_call> result was received and
          we should re-query the client with the raw text so the SDK can handle
          the tool execution in the same session.
        - raw_tool_call_text is the unmodified result string to re-submit.
    """
    async for message in client.receive_messages():

        if hasattr(message, "session_id") and message.session_id:
            emit({"type": "session", "id": message.session_id})

        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    pending_tools.add(block.id)
                    emit(
                        {
                            "type": "tool_start",
                            "name": block.name,
                            "id": block.id,
                            "input": block.input,
                        }
                    )

        elif isinstance(message, UserMessage):
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    pending_tools.discard(block.tool_use_id)
                    emit({"type": "tool_end", "id": block.tool_use_id})

        elif isinstance(message, SystemMessage):
            emit({"type": "system", "text": str(message)})

        elif isinstance(message, ResultMessage):
            if message.result:
                if "</tool_call>" in message.result:
                    # The SDK serialized a tool call as raw text instead of
                    # executing it via the normal ToolUseBlock path. Re-submit
                    # the raw text as a new query on the same session so the
                    # SDK agent loop can execute the tool and continue properly.
                    return True, message.result
                else:
                    emit({"type": "result", "text": message.result.strip()})
                    break

    return False, None


async def main() -> None:
    payload = json.loads(sys.stdin.readline())

    user_input: str = payload["user_input"]
    is_first: bool = payload["is_first"]
    continue_conv: bool = payload["continue_conversation"]
    last_session_id = payload.get("last_session_id")
    setting_sources: list = payload["setting_sources"]

    options = ClaudeAgentOptions(
        continue_conversation=(
            continue_conv and not is_first and last_session_id is None
        ),
        resume=(None if is_first or not continue_conv else last_session_id),
        setting_sources=setting_sources,
    )

    try:
        async with ClaudeSDKClient(options) as client:
            await client.query(user_input)

            pending_tools: set[str] = set()

            # Re-query loop: if the SDK emits a <tool_call> as a text result,
            # feed it back into the same session so the agent loop executes it.
            # A well-behaved turn emits no <tool_call> and exits the loop.
            while True:
                needs_continuation, tool_call_text = await process_messages(
                    client, pending_tools
                )
                if not needs_continuation:
                    break
                # Re-submit the raw tool-call text within the same session.
                # The SDK will parse it, execute the tool, and continue the
                # agent loop, eventually producing a clean ResultMessage.
                await client.query(tool_call_text)

    except Exception as exc:
        emit({"type": "error", "text": str(exc)})
        sys.exit(1)


anyio.run(main)
