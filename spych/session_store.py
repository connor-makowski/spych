"""
Session persistence for spych agent conversations.

Each invocation of a spych agent gets a UUID session ID.  The ID is used as
the filename under ``~/.config/spych/sessions/`` and the file records the
workspace path plus the underlying agent conversation ID so the session can
be resumed across restarts.
"""

import json
import os
import uuid
from typing import Optional

_SESSIONS_DIR = os.path.join(
    os.path.expanduser("~"), ".config", "spych", "sessions"
)


def _sessions_dir() -> str:
    """
    Usage:

    - Return the sessions directory path, creating it if needed.

    Returns:

    - `path`:
        - Type: str
        - What: Absolute path to ``~/.config/spych/sessions/``.
    """
    os.makedirs(_SESSIONS_DIR, exist_ok=True)
    return _SESSIONS_DIR


def _session_path(session_id: str) -> str:
    return os.path.join(_sessions_dir(), f"{session_id}.json")


def new_session_id() -> str:
    """
    Usage:

    - Generate a fresh UUID to use as a spych session ID for this invocation.

    Returns:

    - `session_id`:
        - Type: str
        - What: A new UUID4 string.
    """
    return str(uuid.uuid4())


def load_session(session_id: str) -> dict:
    """
    Usage:

    - Load a previously saved session by its UUID.

    Requires:

    - `session_id`:
        - Type: str
        - What: UUID of the session to load.

    Returns:

    - `data`:
        - Type: dict
        - What: Session data dict with keys ``session_id``, ``workspace``, ``agent_name``,
          ``personality``, ``conversation_id`` (may be None), ``history`` (list of turn dicts), and timestamps.

    Notes:

    - Returns an empty dict if the file does not exist or cannot be parsed.
    """
    path = _session_path(session_id)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_session(
    session_id: str,
    workspace: str,
    conversation_id: Optional[str],
    agent_name: Optional[str] = None,
    personality: Optional[str] = None,
    history: Optional[list[dict]] = None,
) -> None:
    """
    Usage:

    - Persist a session's workspace, agent name, personality, conversation ID, and history to disk.

    Requires:

    - `session_id`:
        - Type: str
        - What: UUID of this spych session (used as the filename).

    - `workspace`:
        - Type: str
        - What: Absolute path to the workspace directory.

    - `conversation_id`:
        - Type: str | None
        - What: The agent's conversation ID to resume on the next run.

    Optional:

    - `agent_name`:
        - Type: str | None
        - What: Agent class/display name (e.g. "Antigravity", "Claude").
        - Default: None

    - `personality`:
        - Type: str | None
        - What: Personality preset name (e.g. "jarvis").
        - Default: None

    - `history`:
        - Type: list[dict] | None
        - What: List of past interaction turns.
        - Default: None (preserves existing history if present)

    Notes:

    - Overwrites any existing file for the same ``session_id``.
    """
    path = _session_path(session_id)
    existing = load_session(session_id)
    current_history = (
        history if history is not None else existing.get("history", [])
    )
    data = {
        "session_id": session_id,
        "workspace": workspace,
        "agent_name": agent_name or existing.get("agent_name"),
        "personality": personality or existing.get("personality"),
        "conversation_id": conversation_id,
        "history": current_history,
        "created_at": existing.get("created_at") or _now_iso(),
        "updated_at": _now_iso(),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def append_turn_to_session(
    session_id: str,
    workspace: str,
    conversation_id: Optional[str],
    user_input: str,
    response: str,
    summary: Optional[str] = None,
    agent_name: Optional[str] = None,
    personality: Optional[str] = None,
) -> None:
    """
    Usage:

    - Append a single interaction turn to a session's saved history.

    Requires:

    - `session_id`:
        - Type: str
        - What: UUID of the session.
    - `workspace`:
        - Type: str
        - What: Workspace path.
    - `conversation_id`:
        - Type: str | None
        - What: Current agent conversation ID.
    - `user_input`:
        - Type: str
        - What: User prompt / voice transcription.
    - `response`:
        - Type: str
        - What: Agent text response.

    Optional:

    - `summary`:
        - Type: str | None
        - What: Spoken summary text.
    - `agent_name`:
        - Type: str | None
        - What: Agent name.
    - `personality`:
        - Type: str | None
        - What: Personality preset.
    """
    existing = load_session(session_id)
    history = existing.get("history", [])
    history.append(
        {
            "timestamp": _now_iso(),
            "user_input": user_input,
            "response": response,
            "summary": summary or response,
        }
    )
    save_session(
        session_id,
        workspace,
        conversation_id,
        agent_name=agent_name,
        personality=personality,
        history=history,
    )


def latest_session_for_workspace(
    workspace: str,
    agent_name: Optional[str] = None,
    personality: Optional[str] = None,
) -> Optional[str]:
    """
    Usage:

    - Find the most recently updated session ID for a given workspace, agent_name, and personality.

    Requires:

    - `workspace`:
        - Type: str
        - What: Absolute path to the workspace directory.

    Optional:

    - `agent_name`:
        - Type: str | None
        - What: Agent name to match.
        - Default: None (matches any)

    - `personality`:
        - Type: str | None
        - What: Personality preset to match.
        - Default: None (matches any)

    Returns:

    - `session_id`:
        - Type: str | None
        - What: The UUID of the most recent matching session, or None if no match.
    """
    sessions_dir = _sessions_dir()
    best_id: Optional[str] = None
    best_mtime: float = -1.0

    try:
        entries = os.listdir(sessions_dir)
    except OSError:
        return None

    for fname in entries:
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(sessions_dir, fname)
        try:
            mtime = os.path.getmtime(fpath)
            with open(fpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            if data.get("workspace") != workspace:
                continue
            if (
                agent_name
                and data.get("agent_name")
                and data.get("agent_name") != agent_name
            ):
                continue
            if (
                personality
                and data.get("personality")
                and data.get("personality") != personality
            ):
                continue

            if mtime > best_mtime:
                best_mtime = mtime
                best_id = data.get("session_id")
        except Exception:
            continue

    return best_id


def list_sessions(
    workspace: Optional[str] = None,
    agent_name: Optional[str] = None,
    personality: Optional[str] = None,
) -> list[dict]:
    """
    Usage:

    - List saved sessions, optionally filtered by workspace, agent_name, or personality.

    Optional:

    - `workspace`:
        - Type: str | None
        - What: Workspace path to filter by. None returns sessions for all workspaces.
        - Default: None

    - `agent_name`:
        - Type: str | None
        - What: Agent class/display name to filter by.
        - Default: None

    - `personality`:
        - Type: str | None
        - What: Personality preset name to filter by.
        - Default: None

    Returns:

    - `sessions`:
        - Type: list[dict]
        - What: List of session data dicts sorted by updated_at descending.
    """
    sessions_dir = _sessions_dir()
    results: list[dict] = []

    try:
        entries = os.listdir(sessions_dir)
    except OSError:
        return []

    for fname in entries:
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(sessions_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            if workspace and data.get("workspace") != workspace:
                continue
            if (
                agent_name
                and data.get("agent_name")
                and data.get("agent_name") != agent_name
            ):
                continue
            if (
                personality
                and data.get("personality")
                and data.get("personality") != personality
            ):
                continue

            results.append(data)
        except Exception:
            continue

    results.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
    return results


def delete_session(session_id: str) -> bool:
    """
    Usage:

    - Delete a specific saved session by its UUID.

    Requires:

    - `session_id`:
        - Type: str
        - What: Session UUID to delete.

    Returns:

    - `success`:
        - Type: bool
        - What: True if file was deleted, False if not found or failed.
    """
    path = _session_path(session_id)
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
        return False
    except Exception:
        return False


def clear_workspace_sessions(
    workspace: str,
    agent_name: Optional[str] = None,
    personality: Optional[str] = None,
) -> int:
    """
    Usage:

    - Delete all saved sessions for a specific workspace directory.

    Requires:

    - `workspace`:
        - Type: str
        - What: Absolute path to the workspace directory.

    Optional:

    - `agent_name`:
        - Type: str | None
        - What: Optional agent name filter.
        - Default: None

    - `personality`:
        - Type: str | None
        - What: Optional personality filter.
        - Default: None

    Returns:

    - `count`:
        - Type: int
        - What: Total number of session files deleted.
    """
    targets = list_sessions(
        workspace=workspace, agent_name=agent_name, personality=personality
    )
    deleted = 0
    for s in targets:
        sid = s.get("session_id")
        if sid and delete_session(sid):
            deleted += 1
    return deleted


def clear_all_sessions() -> int:
    """
    Usage:

    - Delete all saved session files across all workspaces.

    Returns:

    - `count`:
        - Type: int
        - What: Total number of session files deleted.
    """
    sessions_dir = _sessions_dir()
    deleted = 0
    try:
        entries = os.listdir(sessions_dir)
        for fname in entries:
            if fname.endswith(".json"):
                fpath = os.path.join(sessions_dir, fname)
                try:
                    os.remove(fpath)
                    deleted += 1
                except Exception:
                    pass
    except OSError:
        pass
    return deleted


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
