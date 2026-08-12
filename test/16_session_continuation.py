"""
Test 16: Verify session store conversation ID persistence and continuation logic
across BaseResponder subclasses (Antigravity, Claude CLI/SDK, Codex, OpenCode).
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from spych import Spych
from spych.responders import BaseResponder, AgentResponse
from spych.agents.claude import (
    LocalClaudeCodeCLIResponder,
    LocalClaudeCodeSDKResponder,
)
from spych.agents.agy import LocalAntigravityCLIResponder
from spych.agents.codex import LocalCodexCLIResponder
from spych.agents.opencode import LocalOpenCodeCLIResponder
from spych.session_store import load_session, save_session, new_session_id


def test_codex_session_continuation(tmp_path):
    spych_mock = MagicMock(spec=Spych)
    session_id = new_session_id()
    workspace = str(tmp_path)

    # Save a initial session with a conversation ID
    save_session(
        session_id=session_id,
        workspace=workspace,
        conversation_id="thread_abc123",
        agent_name="Codex",
    )

    with patch("os.getcwd", return_value=workspace):
        responder = LocalCodexCLIResponder(
            spych_object=spych_mock,
            session_id=session_id,
            continue_conversation=True,
            use_speaker=False,
        )

        assert responder._last_session_id == "thread_abc123"

        # Mock StreamJsonCommand to simulate a Codex execution event
        mock_events = [
            {"type": "thread.started", "thread_id": "thread_abc123"},
            {
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": '{"response": "ok", "summary": "ok", "requires_user_feedback": false}',
                },
            },
        ]

        with patch("spych.agents.codex.StreamJsonCommand") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream.__iter__.return_value = iter(mock_events)
            mock_stream.stderr_lines = []
            mock_stream_cls.return_value = mock_stream

            resp = responder.respond("Hello Codex")

            # Verify that cmd included resume thread_abc123
            cmd_args = mock_stream_cls.call_args[0][0]
            assert "resume" in cmd_args
            assert "thread_abc123" in cmd_args

        # Now simulate a new thread returned
        mock_events_new = [
            {"type": "thread.started", "thread_id": "thread_xyz789"},
            {
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": '{"response": "ok", "summary": "ok", "requires_user_feedback": false}',
                },
            },
        ]
        with patch("spych.agents.codex.StreamJsonCommand") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream.__iter__.return_value = iter(mock_events_new)
            mock_stream.stderr_lines = []
            mock_stream_cls.return_value = mock_stream

            resp = responder.respond("Another prompt")
            assert responder._last_session_id == "thread_xyz789"

            saved = load_session(session_id)
            assert saved.get("conversation_id") == "thread_xyz789"


def test_opencode_session_continuation(tmp_path):
    spych_mock = MagicMock(spec=Spych)
    session_id = new_session_id()
    workspace = str(tmp_path)

    save_session(
        session_id=session_id,
        workspace=workspace,
        conversation_id="session_opencode_111",
        agent_name="OpenCode",
    )

    with patch("os.getcwd", return_value=workspace):
        responder = LocalOpenCodeCLIResponder(
            spych_object=spych_mock,
            session_id=session_id,
            continue_conversation=True,
            use_speaker=False,
        )

        assert responder._last_session_id == "session_opencode_111"

        mock_events = [
            {"type": "step_start", "sessionID": "session_opencode_111"},
            {
                "type": "text",
                "part": {
                    "text": '{"response": "hello", "summary": "hi", "requires_user_feedback": false}'
                },
            },
            {"type": "step_finish", "part": {"reason": "stop"}},
        ]

        with patch(
            "spych.agents.opencode.StreamJsonCommand"
        ) as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream.__iter__.return_value = iter(mock_events)
            mock_stream.stderr_lines = []
            mock_stream_cls.return_value = mock_stream

            resp = responder.respond("Hello OpenCode")

            cmd_args = mock_stream_cls.call_args[0][0]
            assert "--session" in cmd_args
            assert "session_opencode_111" in cmd_args


def test_claude_sdk_worker_payload(tmp_path):
    spych_mock = MagicMock(spec=Spych)
    session_id = new_session_id()
    workspace = str(tmp_path)

    save_session(
        session_id=session_id,
        workspace=workspace,
        conversation_id="claude_sess_999",
        agent_name="Claude",
    )

    with patch("os.getcwd", return_value=workspace):
        responder = LocalClaudeCodeSDKResponder(
            spych_object=spych_mock,
            session_id=session_id,
            continue_conversation=True,
            use_speaker=False,
        )

        assert responder._last_session_id == "claude_sess_999"

        with patch("spych.agents.claude.StreamJsonCommand") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream.__iter__.return_value = iter(
                [
                    {"type": "session", "id": "claude_sess_999"},
                    {
                        "type": "result",
                        "text": '{"response": "ok", "summary": "ok", "requires_user_feedback": false}',
                    },
                ]
            )
            mock_stream.stderr_lines = []
            mock_stream_cls.return_value = mock_stream

            resp = responder.respond("Hi Claude")

            input_payload = mock_stream_cls.call_args[1]["input_text"]
            import json

            payload_dict = json.loads(input_payload)
            assert payload_dict["last_session_id"] == "claude_sess_999"
            assert payload_dict["continue_conversation"] is True
