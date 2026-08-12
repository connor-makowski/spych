"""
Test 17: Verify session store list, clear, clear-all functions and CLI commands.
Run via: pytest test/17_session_commands.py
"""

import os
from unittest.mock import patch
from spych.session_store import (
    save_session,
    list_sessions,
    delete_session,
    clear_workspace_sessions,
    clear_all_sessions,
    new_session_id,
)
from spych.cli import main


def test_session_store_operations(tmp_path):
    ws1 = str(tmp_path / "project_a")
    ws2 = str(tmp_path / "project_b")

    s1 = new_session_id()
    s2 = new_session_id()
    s3 = new_session_id()

    save_session(s1, ws1, "conv_1", agent_name="Claude", personality="default")
    save_session(s2, ws1, "conv_2", agent_name="Codex", personality="jarvis")
    save_session(
        s3, ws2, "conv_3", agent_name="Antigravity", personality="default"
    )

    # List for ws1
    ws1_sessions = list_sessions(workspace=ws1)
    assert len(ws1_sessions) == 2
    ids_ws1 = {s["session_id"] for s in ws1_sessions}
    assert ids_ws1 == {s1, s2}

    # List all
    all_sessions = list_sessions(workspace=None)
    assert len(all_sessions) >= 3

    # Delete single session
    assert delete_session(s1) is True
    assert len(list_sessions(workspace=ws1)) == 1

    # Clear workspace sessions
    cleared = clear_workspace_sessions(ws1)
    assert cleared == 1
    assert len(list_sessions(workspace=ws1)) == 0

    # Clear all sessions
    cleared_all = clear_all_sessions()
    assert cleared_all >= 1
    assert len(list_sessions(workspace=None)) == 0


def test_cli_sessions_dispatch(tmp_path, capsys):
    ws = str(tmp_path / "my_project")
    s1 = new_session_id()
    save_session(s1, ws, "conv_100", agent_name="OpenCode")

    with patch("sys.argv", ["spych", "sessions", "--workspace", ws]):
        main()
        captured = capsys.readouterr().out
        assert "Saved Sessions" in captured
        assert s1 in captured
        assert "OpenCode" in captured

    with patch("sys.argv", ["spych", "clear", "--workspace", ws]):
        main()
        captured = capsys.readouterr().out
        assert "Cleared 1 session(s)" in captured

    assert len(list_sessions(workspace=ws)) == 0
