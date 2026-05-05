"""
spych TUI Dashboard.

Usage:

- ``AgentDashboard`` renders a live three-section terminal interface in the
  alternate screen buffer and accepts lifecycle events from ``BaseResponder``.

- Start before launching the agent; stop in a finally block:

    dashboard = AgentDashboard(agent_name="Claude", wake_words=["claude"], responder_name="LocalClaudeCodeCLIResponder")
    dashboard.start()
    try:
        claude_code_cli(dashboard=dashboard)
    finally:
        dashboard.stop()
"""

from __future__ import annotations

import re
import shutil
import sys
import textwrap
import threading
import time
from dataclasses import dataclass
from typing import Optional

from spych.spinners import Spinner
from spych.utils import supports_unicode

_HAS_UNICODE = supports_unicode()

try:
    import select
    import termios
    import tty

    _IS_WINDOWS = False
except ImportError:
    import msvcrt

    _IS_WINDOWS = True


# -- ANSI escape codes ------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_ITALIC = "\033[3m"

_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_MAGENTA = "\033[95m"
_CYAN = "\033[96m"

_CURSOR_HIDE = "\033[?25l"
_CURSOR_SHOW = "\033[?25h"
_ALT_SCREEN_ENTER = "\033[?1049h"
_ALT_SCREEN_EXIT = "\033[?1049l"
_CURSOR_HOME = "\033[H"
_CLEAR_BELOW = "\033[J"
_CLEAR_EOL = "\033[K"

# -- Status constants -------------------------------------------------------

WAITING = "waiting"
LISTENING = "listening"
THINKING = "thinking"
SPEAKING = "speaking"
TERMINATED = "terminated"


# -- Data structures --------------------------------------------------------


@dataclass
class ToolEntry:
    """
    Usage:

    - Represents a single tool call recorded during a conversation turn.

    Requires:

    - ``name``:
        - Type: str
        - What: The tool name.

    - ``status``:
        - Type: str
        - What: Current status string (e.g. "running", "done").

    - ``is_running``:
        - Type: bool
        - What: Whether the tool is still in progress.

    - ``elapsed``:
        - Type: float | None
        - What: Seconds since the tool started, or None if not finished.

    - ``timestamp``:
        - Type: str
        - What: HH:MM:SS when the tool event was first recorded.

    Optional:

    - ``detail``:
        - Type: str | None
        - What: Short human-readable context for the tool call (e.g. file path,
          command string, search pattern). Displayed in place of the raw status.
    """

    name: str
    status: str
    is_running: bool
    elapsed: Optional[float]
    timestamp: str
    detail: Optional[str] = None


@dataclass
class ThoughtEntry:
    """
    Usage:

    - Represents an internal thought or intermediate message from the agent.

    Requires:

    - ``text``:
        - Type: str
        - What: The content of the thought.

    - ``timestamp``:
        - Type: str
        - What: HH:MM:SS when the thought was recorded.

    Optional:

    - ``is_response``:
        - Type: bool
        - What: Whether this thought should be presented as a normal agent response.
        - Default: False
    """

    text: str
    timestamp: str
    is_response: bool = False


@dataclass
class ConversationEntry:
    """
    Usage:

    - Represents one completed agent turn: user input, tool calls, and response.

    Requires:

    - ``user_input``:
        - Type: str
        - What: The transcribed user input that triggered this turn.

    - ``agent_name``:
        - Type: str
        - What: Name of the agent that produced this response.

    - ``tool_calls``:
        - Type: list[ToolEntry]
        - What: All tool events recorded during this turn.

    - ``thoughts``:
        - Type: list[ThoughtEntry]
        - What: Internal thoughts recorded during this turn.

    - ``response``:
        - Type: str
        - What: Full agent response text.

    - ``summary``:
        - Type: str
        - What: Short summary (may equal response when response is brief).

    - ``timestamp``:
        - Type: str
        - What: HH:MM:SS when the turn completed.

    - ``elapsed``:
        - Type: float
        - What: Seconds from user input to completed response.

    - ``user_duration``:
        - Type: float
        - What: Seconds for the user's listening phase.
    """

    user_input: str
    agent_name: str
    tool_calls: list[ToolEntry]
    thoughts: list[ThoughtEntry]
    response: str
    summary: str
    timestamp: str
    elapsed: float
    user_duration: float


# -- Renderer ---------------------------------------------------------------


class _DashboardRenderer:
    _HEADER_LINES = 4  # rule + title + rule + blank
    _FOOTER_LINES = 2  # blank + hint line
    _TOOL_FIXED = 3  # label + top-rule + bottom-rule (only when tools present)
    _CONV_FIXED = 0  # no label or rules in minimal mode
    _FIXED_BASE = _HEADER_LINES + _CONV_FIXED + _FOOTER_LINES  # = 6

    _SUMMARY_CHAR_LIMIT = 300

    def __init__(
        self,
        thinking_spinner: list[str],
        waiting_spinner: list[str],
        listening_spinner: list[str],
    ) -> None:
        self._prev_cols = 0
        self._prev_rows = 0
        self._thinking_spinner = thinking_spinner
        self._waiting_spinner = waiting_spinner
        self._listening_spinner = listening_spinner

        # Cache for wrapped conversation blocks:
        # (entry_id, cols) -> list[str]
        self._conv_cache: dict[tuple[int, int], list[str]] = {}
        # Cache for wrapped tool lines:
        # (tool_id, cols) -> list[str]
        self._tool_cache: dict[tuple[int, int], list[str]] = {}
        # Cache for wrapped thought entries:
        # (thought_id, cols) -> list[str]
        self._thought_cache: dict[tuple[int, int], list[str]] = {}

    _THINKING_VERBS = [
        # ... (same as before)
        "Thinking",
        "Vibing",
        "Pontificating",
        "Contemplating",
        "Deliberating",
        "Cogitating",
        "Ruminating",
        "Musing",
        "Ideating",
        "Postulating",
        "Hypothesizing",
        "Extrapolating",
        "Philosophizing",
        "Noodling",
        "Percolating",
        "Marinating",
        "Stewing",
        "Scheming",
        "Conniving",
        "Divining",
        "Spelunking",
        "Ratiocinating",
        "Cerebrating",
        "Woolgathering",
        "Daydreaming",
        "Lucubrating",
        "Excogitating",
        "Thinkulating",
        "Brainwaving",
        "Cogitronning",
        "Synapsing",
        "Thoughtcrafting",
        "Mindweaving",
        "Intellectualizing",
        "Computating",
        "Ponderizing",
        "Mentalating",
        "Brainbrewing",
    ]

    def _status_str(self, status: str) -> str:
        if status == THINKING:
            idx = int(time.time() * 10) % len(self._thinking_spinner)
            v_idx = int(time.time() / 10) % len(
                _DashboardRenderer._THINKING_VERBS
            )
            spinner = self._thinking_spinner[idx]
            verb = _DashboardRenderer._THINKING_VERBS[v_idx]
            return f"{_YELLOW}{spinner} {verb}{_RESET}"

        if status == LISTENING:
            idx = int(time.time() * 10) % len(self._listening_spinner)
            spinner = self._listening_spinner[idx]
            return f"{_BOLD}{_GREEN}{spinner} LISTENING {spinner}{_RESET}"

        if status == WAITING:
            idx = int(time.time() * 5) % len(self._waiting_spinner)
            spinner = self._waiting_spinner[idx]
            return f"{_DIM}{spinner} Waiting{_RESET}"

        indicators: dict[str, str] = {
            SPEAKING: (
                f"{_CYAN}♪ Speaking{_RESET}"
                if _HAS_UNICODE
                else f"{_CYAN}> Speaking{_RESET}"
            ),
            TERMINATED: (
                f"{_DIM}✗ Stopped{_RESET}"
                if _HAS_UNICODE
                else f"{_DIM}x Stopped{_RESET}"
            ),
        }
        return indicators.get(
            status,
            (
                f"{_DIM}○ Waiting{_RESET}"
                if _HAS_UNICODE
                else f"{_DIM}o Waiting{_RESET}"
            ),
        )

    @staticmethod
    def _truncate(text: str, width: int) -> str:
        if _DashboardRenderer._visible_len(text) <= width:
            return text

        res = ""
        cur_width = 0
        ell = "…" if _HAS_UNICODE else "..."
        # Split by ANSI escape sequences to preserve them
        parts = re.split(r"(\033\[[0-9;]*[mK])", text)
        for part in parts:
            if not part:
                continue
            if part.startswith("\033["):
                res += part
            else:
                for char in part:
                    w = _DashboardRenderer._visible_len(char)
                    if (
                        cur_width + w + _DashboardRenderer._visible_len(ell)
                        > width
                    ):
                        # Append reset to prevent color bleed if we truncated mid-color
                        return res + ell + _RESET
                    res += char
                    cur_width += w
        return res + ell + _RESET

    _ANSI_ESCAPE = re.compile(r"\033\[[0-9;]*[mK]")

    @staticmethod
    def _visible_len(text: str) -> int:
        clean = _DashboardRenderer._ANSI_ESCAPE.sub("", text)
        length = 0
        for char in clean:
            # Heuristic for double-width characters (CJK, Emojis, etc.)
            cp = ord(char)
            if cp >= 0x1100 and (
                0x1100 <= cp <= 0x115F
                or 0x2329 <= cp <= 0x232A
                or 0x2E80 <= cp <= 0xA4CF
                or 0xAC00 <= cp <= 0xD7A3
                or 0xF900 <= cp <= 0xFAFF
                or 0xFE10 <= cp <= 0xFE19
                or 0xFE30 <= cp <= 0xFE6F
                or 0xFF00 <= cp <= 0xFF60
                or 0xFFE0 <= cp <= 0xFFE6
                or cp >= 0x20000
            ):
                length += 2
            else:
                length += 1
        return length

    @staticmethod
    def _wrap_text(text: str, width: int, indent: int = 0) -> list[str]:
        if not text:
            return [""]

        # Split by existing newlines first to ensure textwrap doesn't get confused
        # and to prevent strings containing \n from entering the render loop.
        lines = text.splitlines()
        res = []
        wrapper = textwrap.TextWrapper(
            width=width,
            initial_indent=" " * indent,
            subsequent_indent=" " * indent,
            replace_whitespace=True,
            drop_whitespace=True,
        )
        for line in lines:
            if not line.strip():
                # Preserve blank lines but respect indent
                res.append(" " * indent)
                continue
            wrapped_lines = wrapper.wrap(line)
            if wrapped_lines:
                res.extend(wrapped_lines)
            else:
                # If wrap returns nothing for a non-empty line (rare),
                # at least keep the indent.
                res.append(" " * indent)
        return res if res else [""]

    @staticmethod
    def _format_line_group(
        prefix: str,
        text: str,
        cols: int,
        duration: Optional[float] = None,
    ) -> list[str]:
        prefix_len = _DashboardRenderer._visible_len(prefix)
        available_width = max(10, cols - prefix_len - 1)
        # Strip text to prevent double spaces if input starts with a space
        wrapped = _DashboardRenderer._wrap_text(text.strip(), available_width)
        res = []
        for i, line in enumerate(wrapped):
            cur_line = (prefix if i == 0 else " " * prefix_len) + line

            if duration is not None and i == len(wrapped) - 1:
                dur_str = f"({duration:.1f}s)"
                cur_visible = _DashboardRenderer._visible_len(cur_line)
                target_width = cols - 2
                spaces = target_width - cur_visible - len(dur_str)
                if spaces >= 1:
                    cur_line += " " * spaces + _DIM + dur_str + _RESET
                else:
                    # If we don't have room for spaces + dur_str on the same line,
                    # either append with a single space if it fits the target_width,
                    # or drop it to the next line.
                    if cur_visible + len(dur_str) + 1 <= target_width:
                        cur_line += " " + _DIM + dur_str + _RESET
                    elif target_width - len(dur_str) >= 0:
                        res.append(cur_line)
                        cur_line = (
                            " " * (target_width - len(dur_str))
                            + _DIM
                            + dur_str
                            + _RESET
                        )
                    else:
                        # Extremely narrow terminal, just append it
                        res.append(cur_line)
                        cur_line = _DIM + dur_str + _RESET

            if _DashboardRenderer._visible_len(cur_line) > cols - 1:
                cur_line = _DashboardRenderer._truncate(cur_line, cols - 1)
            res.append(cur_line)
        return res

    def _format_user(
        self, ts: str, name: str, text: str, duration: float, cols: int
    ) -> list[str]:
        prefix = f"  {ts}  {_CYAN}{name}:{_RESET} "
        return self._format_line_group(prefix, text, cols, duration=duration)

    def _format_agent(
        self, ts: str, name: str, text: str, duration: float, cols: int
    ) -> list[str]:
        prefix = f"  {ts}  {_BOLD}{name}:{_RESET} "
        return self._format_line_group(prefix, text, cols, duration=duration)

    def _format_thought(
        self,
        ts: str,
        text: str,
        cols: int,
        agent_name: str = "",
        is_response: bool = False,
    ) -> list[str]:
        if is_response:
            prefix = f"  {ts}  {_BOLD}{agent_name}:{_RESET} "
            return self._format_line_group(prefix, text, cols)
        else:
            name_str = f"{agent_name} " if agent_name else ""
            prefix = f"    {ts}  {_ITALIC}{_DIM}{name_str}Thought:{_RESET} "
            return self._format_line_group(prefix, text, cols)

    def _format_tool(
        self,
        ts: str,
        name: str,
        status: str,
        is_running: bool,
        elapsed: Optional[float],
        cols: int,
        show_ts: bool = False,
        detail: Optional[str] = None,
    ) -> list[str]:
        icon = (
            ("⚙" if _HAS_UNICODE else "*")
            if is_running
            else ("✓" if _HAS_UNICODE else "+")
        )
        color = _YELLOW if is_running else _GREEN
        elapsed_str = f" {_DIM}({elapsed:.2f}s){_RESET}" if elapsed else ""
        if detail:
            content = f"{name}  {_DIM}{detail}{_RESET}{elapsed_str}"
        else:
            content = f"{name}{elapsed_str}"
        if show_ts:
            ts_str = f"{_DIM}{ts}{_RESET}"
            prefix = f"  {ts_str}  {color}{icon}{_RESET}  "
        else:
            prefix = f"    {color}{icon}{_RESET}  "
        return self._format_line_group(prefix, content, cols)

    def render(
        self,
        agent_name: str,
        wake_words: list[str],
        responder_name: str,
        response_style: str,
        use_speaker: bool,
        speaker_voice: str,
        user_name: str,
        status: str,
        status_start: float,
        conversation: list[ConversationEntry],
        current_user_input: str,
        current_user_ts: str,
        current_user_duration: float,
        current_tool_calls: list[ToolEntry],
        current_thoughts: list[ThoughtEntry],
        show_all: bool,
        scroll_offset: int,
    ) -> None:
        cols, rows = shutil.get_terminal_size(fallback=(80, 24))

        resized = False
        if cols != self._prev_cols or rows != self._prev_rows:
            resized = True
            self._prev_cols = cols
            self._prev_rows = rows
            # Clear cache on resize as wrapping changes
            self._conv_cache.clear()
            self._tool_cache.clear()
            self._thought_cache.clear()

        # Periodically bound cache size (evict oldest 25% when full)
        if len(self._conv_cache) > 2000:
            keys = list(self._conv_cache.keys())
            for k in keys[:500]:
                self._conv_cache.pop(k, None)
        if len(self._tool_cache) > 2000:
            keys = list(self._tool_cache.keys())
            for k in keys[:500]:
                self._tool_cache.pop(k, None)
        if len(self._thought_cache) > 2000:
            keys = list(self._thought_cache.keys())
            for k in keys[:500]:
                self._thought_cache.pop(k, None)

        bar_width = max(10, cols - 4)
        budget = rows - 1
        lines: list[str] = []

        # Header
        h_bar = "━" if _HAS_UNICODE else "="
        lines.append(f"  {_BOLD}{h_bar * bar_width}{_RESET}")

        header_parts: list[str] = [
            f"  {_BOLD}{_CYAN}SPYCH{_RESET}",
            f"{_BOLD}{agent_name}{_RESET}",
            self._status_str(status),
        ]
        if wake_words:
            shown = wake_words[0]
            if len(wake_words) > 1:
                shown += f" +{len(wake_words) - 1}"
            header_parts.append(f"{_DIM}Wake: {shown}{_RESET}")
        if responder_name:
            header_parts.append(f"{_DIM}{responder_name}{_RESET}")
        if use_speaker and speaker_voice:
            spk = "🔊 " if _HAS_UNICODE else "Voice: "
            header_parts.append(f"{_DIM}{spk}{speaker_voice}{_RESET}")
        if show_all:
            header_parts.append(f"{_BOLD}{_YELLOW}ALL LOGS{_RESET}")

        sep = "  │  " if _HAS_UNICODE else "  |  "
        title = sep.join(header_parts)
        # Truncate title if it exceeds width
        if self._visible_len(title) > cols - 1:
            while (
                len(header_parts) > 3
                and self._visible_len(sep.join(header_parts)) > cols - 5
            ):
                header_parts.pop()
            title = sep.join(header_parts)
            if self._visible_len(title) > cols - 1:
                title = title[: cols - 5] + "..." + _RESET

        lines.append(title)
        lines.append(f"  {_BOLD}{h_bar * bar_width}{_RESET}")
        lines.append("")

        if show_all:
            # All Logs Mode
            max_log_lines = max(
                0, budget - self._HEADER_LINES - self._FOOTER_LINES
            )

            log_lines: list[str] = []
            for entry in conversation:
                entry_cache_key = (id(entry), cols)
                if entry_cache_key in self._conv_cache:
                    log_lines.extend(self._conv_cache[entry_cache_key])
                    continue

                block = []
                ts = entry.timestamp
                block.extend(
                    self._format_user(
                        ts,
                        user_name,
                        entry.user_input,
                        entry.user_duration,
                        cols,
                    )
                )

                for thought in entry.thoughts:
                    thought_cache_key = (id(thought), cols, thought.is_response)
                    if thought_cache_key not in self._thought_cache:
                        self._thought_cache[thought_cache_key] = (
                            self._format_thought(
                                thought.timestamp,
                                thought.text,
                                cols,
                                agent_name=entry.agent_name,
                                is_response=thought.is_response,
                            )
                        )
                    block.extend(self._thought_cache[thought_cache_key])

                for tool in entry.tool_calls:
                    tool_cache_key = (
                        id(tool),
                        cols,
                        tool.status,
                        tool.is_running,
                        tool.elapsed,
                        tool.detail,
                    )
                    if tool_cache_key not in self._tool_cache:
                        self._tool_cache[tool_cache_key] = self._format_tool(
                            tool.timestamp,
                            tool.name,
                            tool.status,
                            tool.is_running,
                            tool.elapsed,
                            cols,
                            detail=tool.detail,
                        )
                    block.extend(self._tool_cache[tool_cache_key])

                block.extend(
                    self._format_agent(
                        ts,
                        entry.agent_name,
                        entry.response,
                        entry.elapsed,
                        cols,
                    )
                )

                # Doubling fix: only show summary if it's different and response is long
                if (
                    len(entry.response) > self._SUMMARY_CHAR_LIMIT
                    and entry.summary != entry.response
                ):
                    block.extend(
                        self._format_line_group(
                            f"  {ts}  {_BOLD}Summary:{_RESET} ",
                            entry.summary,
                            cols,
                        )
                    )

                bar = "─" if _HAS_UNICODE else "-"
                block.append(f"  {_DIM}{bar * min(bar_width, 60)}{_RESET}")

                self._conv_cache[entry_cache_key] = block
                log_lines.extend(block)

            # In-progress turn
            if (
                current_user_input
                or status == LISTENING
                or current_tool_calls
                or current_thoughts
            ):
                if status == LISTENING:
                    elapsed = time.time() - status_start
                    ts = f"({elapsed:.1f}s)"
                    duration = elapsed
                else:
                    ts = current_user_ts
                    duration = current_user_duration

                log_lines.extend(
                    self._format_user(
                        ts,
                        user_name,
                        (
                            current_user_input
                            if status != LISTENING
                            else "Listening..."
                        ),
                        duration,
                        cols,
                    )
                )

                for thought in current_thoughts:
                    log_lines.extend(
                        self._format_thought(
                            thought.timestamp,
                            thought.text,
                            cols,
                            agent_name=agent_name,
                            is_response=thought.is_response,
                        )
                    )

                for tool in current_tool_calls:
                    log_lines.extend(
                        self._format_tool(
                            tool.timestamp,
                            tool.name,
                            tool.status,
                            tool.is_running,
                            tool.elapsed,
                            cols,
                            detail=tool.detail,
                        )
                    )

            max_scroll = max(0, len(log_lines) - max_log_lines)
            effective_scroll = min(scroll_offset, max_scroll)
            end_idx = len(log_lines) - effective_scroll
            start_idx = max(0, end_idx - max_log_lines)
            visible = log_lines[start_idx:end_idx]

            for line in visible:
                lines.append(line)
            for _ in range(max_log_lines - len(visible)):
                lines.append("")

        else:
            # Default Mode
            has_tools = bool(current_tool_calls)

            tool_content_lines = []
            if has_tools:
                for tool in current_tool_calls:
                    tool_content_lines.extend(
                        self._format_tool(
                            tool.timestamp,
                            tool.name,
                            tool.status,
                            tool.is_running,
                            tool.elapsed,
                            cols,
                            show_ts=True,
                            detail=tool.detail,
                        )
                    )

            # CAP tool panel height to roughly 1/3 of the screen to prevent overflow
            max_tool_content_height = max(0, (budget - self._FIXED_BASE) // 3)
            if len(tool_content_lines) > max_tool_content_height:
                tool_content_lines = tool_content_lines[
                    -max_tool_content_height:
                ]
                if max_tool_content_height > 0:
                    tool_content_lines[0] = (
                        f"  {_DIM}... (more tools above){_RESET}"
                    )

            tool_height = (
                (self._TOOL_FIXED + len(tool_content_lines)) if has_tools else 0
            )
            max_conv_lines = max(0, budget - self._FIXED_BASE - tool_height)

            conv_display: list[list[str]] = []
            for entry in conversation:
                # Use a different cache key for default mode because it wraps differently
                cache_key = (id(entry), cols, "default")
                if cache_key in self._conv_cache:
                    conv_display.append(self._conv_cache[cache_key])
                    continue

                block = []
                ts = entry.timestamp
                block.extend(
                    self._format_user(
                        ts,
                        user_name,
                        entry.user_input,
                        entry.user_duration,
                        cols,
                    )
                )

                # Include intermediate responses (thoughts with is_response=True)
                for thought in entry.thoughts:
                    if thought.is_response:
                        block.extend(
                            self._format_thought(
                                thought.timestamp,
                                thought.text,
                                cols,
                                agent_name=entry.agent_name,
                                is_response=True,
                            )
                        )

                resp_text = (
                    entry.summary
                    if len(entry.response) > self._SUMMARY_CHAR_LIMIT
                    and entry.summary != entry.response
                    else entry.response
                )
                block.extend(
                    self._format_agent(
                        ts, entry.agent_name, resp_text, entry.elapsed, cols
                    )
                )
                self._conv_cache[cache_key] = block
                conv_display.append(block)
            if current_user_input or status == LISTENING or current_thoughts:
                if status == LISTENING:
                    elapsed = time.time() - status_start
                    ts = f"({elapsed:.1f}s)"
                    duration = elapsed
                else:
                    ts = current_user_ts
                    duration = current_user_duration

                if current_user_input or status == LISTENING:
                    conv_display.append(
                        self._format_user(
                            ts,
                            user_name,
                            (
                                current_user_input
                                if status != LISTENING
                                else "Listening..."
                            ),
                            duration,
                            cols,
                        )
                    )

                for thought in current_thoughts:
                    conv_display.append(
                        self._format_thought(
                            thought.timestamp,
                            thought.text,
                            cols,
                            agent_name=agent_name,
                            is_response=thought.is_response,
                        )
                    )

            max_scroll = max(0, len(conv_display) - 1)
            effective_scroll = min(scroll_offset, max_scroll)

            visible_conv_lines = []
            end_block_idx = len(conv_display) - 1 - effective_scroll
            for i in range(end_block_idx, -1, -1):
                block = conv_display[i]
                if len(visible_conv_lines) + len(block) <= max_conv_lines:
                    visible_conv_lines = block + visible_conv_lines
                else:
                    if not visible_conv_lines:
                        # If a single block is too large, take as much of it as possible
                        visible_conv_lines = block[-max_conv_lines:]
                    break

            while len(visible_conv_lines) < max_conv_lines:
                visible_conv_lines.insert(0, "")

            for line in visible_conv_lines:
                lines.append(line)

            if has_tools:
                lines.append(f"  {_BOLD}Active Tools{_RESET}")
                bar = "─" if _HAS_UNICODE else "-"
                lines.append(f"  {bar * bar_width}")
                for line in tool_content_lines:
                    lines.append(line)
                lines.append(f"  {bar * bar_width}")
        lines.append("")

        sep = "│" if _HAS_UNICODE else "|"
        lines.append(
            f"  {_DIM}Ctrl+C to stop  {sep}  Ctrl+A to toggle mode  {sep}  ↑/↓ to scroll{_RESET}"
        )

        buf = ""
        if resized:
            buf += "\033[2J"  # Full clear on resize
        buf += _CURSOR_HOME
        for i, line in enumerate(lines):
            buf += _CLEAR_EOL + line
            if i < len(lines) - 1:
                buf += "\n"
        buf += _CLEAR_BELOW
        try:
            sys.stdout.write(buf)
            sys.stdout.flush()
        except UnicodeEncodeError:
            # Absolute fallback: strip non-ASCII and try again
            safe_buf = "".join(c if ord(c) < 128 else "?" for c in buf)
            sys.stdout.write(safe_buf)
            sys.stdout.flush()


# -- Dashboard controller ----------------------------------------------------


class AgentDashboard:
    """
    Usage:

    - TUI controller that renders a live three-section dashboard in the
      alternate screen buffer and receives lifecycle events from
      ``BaseResponder`` hooks.

    - Sections:
        1. Header — agent name, listening status indicator, wake words,
           response style, and speaker info.
        2. Active Tools — tool calls from the current in-progress turn
           (hidden when no tools are active).
        3. Conversation — clean per-turn view of user inputs and
           agent responses. Ctrl+A switches to a full scrollable log.

    Requires:

    - ``agent_name``:
        - Type: str
        - What: Display name for the active agent shown in the header.

    - ``wake_words``:
        - Type: list[str]
        - What: Wake words to display in the header.

    Optional:

    - ``response_style``:
        - Type: str
        - What: Response style name shown in the header (e.g. "concise").
        - Default: ""

    - ``use_speaker``:
        - Type: bool
        - What: Whether TTS is active; controls the speaker indicator.
        - Default: False

    - ``speaker_voice``:
        - Type: str
        - What: Voice name displayed next to the speaker indicator.
        - Default: ""

    - ``max_history``:
        - Type: int
        - What: Maximum number of conversation entries to retain.
        - Default: 500

    - ``stop_event``:
        - Type: threading.Event | None
        - What: External event shared with the caller for coordinated shutdown.
        - Default: None (a private event is created)

    - ``thinking_spinner``:
        - Type: list[str]
        - What: Frame sequence for the 'thinking' status.
        - Default: Spinner.ZEN

    - ``waiting_spinner``:
        - Type: list[str]
        - What: Frame sequence for the 'waiting' status.
        - Default: Spinner.CIRCLE_FILL

    - ``listening_spinner``:
        - Type: list[str]
        - What: Frame sequence for the 'listening' status.
        - Default: Spinner.EQUALIZER

    Notes:

    - Call ``start()`` to activate the TUI and ``stop()`` in a finally block
      to restore the terminal.
    - All public event methods are thread-safe.
    - ``stop()`` is idempotent; calling it twice is safe.
    """

    REFRESH_INTERVAL: float = 0.15
    _THOUGHT_TOOLS = {
        "update_topic",
        "change_topic",
        "set_topic",
        "strategic_intent",
    }

    def __init__(
        self,
        agent_name: str,
        wake_words: list[str],
        responder_name: str = "",
        response_style: str = "",
        use_speaker: bool = False,
        speaker_voice: str = "",
        user_name: str = "User",
        max_history: int = 500,
        stop_event: Optional[threading.Event] = None,
        thinking_spinner: list[str] = Spinner.ZEN,
        waiting_spinner: list[str] = Spinner.CIRCLE_FILL,
        listening_spinner: list[str] = Spinner.EQUALIZER,
    ) -> None:
        self._agent_name = agent_name
        self._wake_words = wake_words
        self._responder_name = responder_name
        self._response_style = response_style
        self._use_speaker = use_speaker
        self._speaker_voice = speaker_voice
        self._user_name = user_name
        self._max_history = max_history

        self._status: str = WAITING
        self._conversation: list[ConversationEntry] = []
        self._current_tool_calls: list[ToolEntry] = []
        self._current_thoughts: list[ThoughtEntry] = []
        self._current_user_input: str = ""
        self._current_user_ts: str = ""
        self._current_user_duration: float = 0.0
        self._current_turn_start: float = 0.0
        self._status_start: float = time.time()
        self._last_turn_id: int = 0

        self._show_all: bool = False
        self._scroll_offset: int = 0

        self._stop_event = stop_event or threading.Event()
        self._lock = threading.Lock()
        self._display_thread: Optional[threading.Thread] = None
        self._input_thread: Optional[threading.Thread] = None
        self._renderer = _DashboardRenderer(
            thinking_spinner=thinking_spinner,
            waiting_spinner=waiting_spinner,
            listening_spinner=listening_spinner,
        )
        self._started: bool = False

    # -- Public event API ---------------------------------------------------

    def set_agent(
        self,
        name: str,
        wake_words: list[str],
        responder_name: str = "",
        response_style: str = "",
        use_speaker: bool = False,
        speaker_voice: str = "",
        user_name: str = "User",
    ) -> None:
        """
        Usage:

        - Updates header metadata when a different agent activates.
          Primarily useful in multi-agent mode.

        Requires:

        - ``name``:
            - Type: str
            - What: Display name of the newly active agent.

        - ``wake_words``:
            - Type: list[str]
            - What: Wake words for the newly active agent.

        Optional:

        - ``responder_name``:
            - Type: str
            - What: Name of the responder class.
            - Default: ""

        - ``response_style``:
            - Type: str
            - Default: ""

        - ``use_speaker``:
            - Type: bool
            - Default: False

        - ``speaker_voice``:
            - Type: str
            - Default: ""

        - ``user_name``:
            - Type: str
            - Default: "User"
        """
        with self._lock:
            self._agent_name = name
            self._wake_words = wake_words
            self._responder_name = responder_name
            self._response_style = response_style
            self._use_speaker = use_speaker
            self._speaker_voice = speaker_voice
            self._user_name = user_name

    def on_status_change(
        self, status: str, turn_id: Optional[int] = None
    ) -> None:
        """
        Usage:

        - Updates the status indicator in the header.

        Requires:

        - ``status``:
            - Type: str
            - What: One of the module-level constants: WAITING, LISTENING,
              THINKING, SPEAKING, TERMINATED.

        Optional:

        - ``turn_id``:
            - Type: int | None
            - What: The ID of the turn that triggered this status change.
              If provided and older than the current turn, the change is ignored.
        """
        with self._lock:
            if turn_id is not None:
                if turn_id < self._last_turn_id:
                    return
                self._last_turn_id = turn_id

            if self._status != status:
                self._status = status
                self._status_start = time.time()

    def on_user_input(
        self,
        text: str,
        is_continuation: bool = False,
        turn_id: Optional[int] = None,
    ) -> None:
        """
        Usage:

        - Records the user's transcribed input and starts a new turn.

        Requires:

        - ``text``:
            - Type: str
            - What: Transcribed user input.

        Optional:

        - ``is_continuation``:
            - Type: bool
            - What: Whether this turn is a continuation of a previous turn
              (e.g. after an intermediate response).
            - Default: False

        - ``turn_id``:
            - Type: int | None
            - What: The ID for this turn.
        """
        with self._lock:
            if turn_id is not None:
                self._last_turn_id = turn_id

            if is_continuation and text == self._current_user_input:
                return

            self._current_user_input = text
            self._current_user_ts = time.strftime("%H:%M:%S")
            self._current_user_duration = time.time() - self._status_start
            self._current_turn_start = time.time()
            self._current_tool_calls = []
            self._current_thoughts = []

    def on_thought(self, text: str, is_response: bool = False) -> None:
        """
        Usage:

        - Records an internal thought or intermediate message.

        Requires:

        - ``text``:
            - Type: str
            - What: The content of the thought.

        Optional:

        - ``is_response``:
            - Type: bool
            - What: Whether this thought should be presented as a normal agent response.
            - Default: False
        """
        ts = time.strftime("%H:%M:%S")
        with self._lock:
            self._current_thoughts.append(
                ThoughtEntry(text=text, timestamp=ts, is_response=is_response)
            )

    def on_tool_event(
        self,
        name: str,
        status: str,
        is_running: bool = False,
        elapsed: Optional[float] = None,
        detail: Optional[str] = None,
    ) -> None:
        """
        Usage:

        - Adds or updates a tool call entry in the current turn's tool panel.
          If a running tool with the same name exists it is updated in place;
          otherwise a new entry is appended.

        Requires:

        - ``name``:
            - Type: str
            - What: The tool name.

        - ``status``:
            - Type: str
            - What: Current status string (e.g. "running", "done").

        Optional:

        - ``is_running``:
            - Type: bool
            - Default: False

        - ``elapsed``:
            - Type: float | None
            - Default: None

        - ``detail``:
            - Type: str | None
            - What: Short human-readable context shown next to the tool name
              (e.g. file path, command string, search pattern). When updating
              an in-progress entry, the original detail is preserved if None
              is passed.
            - Default: None
        """
        if name in self._THOUGHT_TOOLS:
            if detail and status == "running":
                self.on_thought(detail, is_response=False)
            return

        ts = time.strftime("%H:%M:%S")
        with self._lock:
            for i in range(len(self._current_tool_calls) - 1, -1, -1):
                t = self._current_tool_calls[i]
                if t.name == name and t.is_running:
                    self._current_tool_calls[i] = ToolEntry(
                        name=name,
                        status=status,
                        is_running=is_running,
                        elapsed=elapsed,
                        timestamp=t.timestamp,
                        detail=detail if detail is not None else t.detail,
                    )
                    return
            self._current_tool_calls.append(
                ToolEntry(
                    name=name,
                    status=status,
                    is_running=is_running,
                    elapsed=elapsed,
                    timestamp=ts,
                    detail=detail,
                )
            )

    def on_response(
        self,
        response: "AgentResponse",
        agent_name: str,
        elapsed: float,
    ) -> None:
        """
        Usage:

        - Commits the completed turn to conversation history and clears the
          in-progress state.

        Requires:

        - ``response``:
            - Type: AgentResponse
            - What: Structured response returned by the agent.

        - ``agent_name``:
            - Type: str
            - What: Name of the agent that produced the response.

        - ``elapsed``:
            - Type: float
            - What: Seconds from user input to completed response.
        """
        if response.is_intermediate_response:
            self.on_thought(response.response, is_response=True)
            return

        ts = time.strftime("%H:%M:%S")
        with self._lock:
            # Deduplicate thoughts that are part of the response
            # to avoid visual doubling in the history.
            unique_thoughts = []
            resp_text = response.response.strip()
            for t in self._current_thoughts:
                t_text = t.text.strip()
                if not t_text:
                    continue
                # If the thought is a subset of the final response, it's likely
                # a streaming artifact or an intermediate update we don't need
                # to show once we have the full response.
                # EXCEPT if it was explicitly flagged as a response (intermediate response).
                if not t.is_response and t_text in resp_text:
                    continue
                unique_thoughts.append(t)

            self._conversation.append(
                ConversationEntry(
                    user_input=self._current_user_input,
                    agent_name=agent_name,
                    tool_calls=list(self._current_tool_calls),
                    thoughts=unique_thoughts,
                    response=response.response,
                    summary=response.summary,
                    timestamp=ts,
                    elapsed=elapsed,
                    user_duration=self._current_user_duration,
                )
            )
            if len(self._conversation) > self._max_history:
                self._conversation.pop(0)
            self._current_tool_calls = []
            self._current_thoughts = []
            self._current_user_input = ""

    # -- TUI control --------------------------------------------------------

    def toggle_show_all(self) -> None:
        """Toggle between minimal and all-logs mode."""
        self._show_all = not self._show_all
        self._scroll_offset = 0

    def scroll_up(self, amount: int = 1) -> None:
        """Scroll up."""
        with self._lock:
            # Bound scroll offset to prevent excessive scrolling into the void
            # We use a generous limit of 5000 lines/entries.
            self._scroll_offset = min(self._scroll_offset + amount, 5000)

    def scroll_down(self, amount: int = 1) -> None:
        """Scroll down."""
        with self._lock:
            self._scroll_offset = max(0, self._scroll_offset - amount)

    def clear_current_turn(self) -> None:
        """
        Usage:

        - Resets the in-progress turn state (user input, tools, thoughts)
          without adding to history. Useful for clearing 'zombie' data
          after errors or empty responses.
        """
        with self._lock:
            self._current_tool_calls = []
            self._current_thoughts = []
            self._current_user_input = ""

    def start(self) -> None:
        """
        Usage:

        - Enters the alternate screen buffer, hides the cursor, and starts
          the background render and keyboard input threads.

        Notes:

        - Call ``stop()`` in a finally block to restore the terminal.
        - Safe to call again after ``stop()`` (e.g. to restart after healthchecks).
        """
        self._started = True
        self._stop_event.clear()
        sys.stdout.write(_ALT_SCREEN_ENTER + _CURSOR_HIDE)
        sys.stdout.flush()
        self._display_thread = threading.Thread(
            target=self._display_loop, daemon=True
        )
        self._input_thread = threading.Thread(
            target=self._input_loop, daemon=True
        )
        self._display_thread.start()
        self._input_thread.start()

    def stop(self) -> None:
        """
        Usage:

        - Signals threads to stop, joins them, and restores the primary screen
          buffer and cursor.

        Notes:

        - Safe to call multiple times.
        """
        if not self._started:
            return
        self._started = False
        self._stop_event.set()
        if self._display_thread is not None:
            self._display_thread.join(timeout=2.0)
        if self._input_thread is not None:
            self._input_thread.join(timeout=2.0)
        sys.stdout.write(_CURSOR_SHOW + _ALT_SCREEN_EXIT)
        sys.stdout.flush()

    # -- Internal threads ---------------------------------------------------

    def _display_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                agent_name = self._agent_name
                wake_words = list(self._wake_words)
                responder_name = self._responder_name
                response_style = self._response_style
                use_speaker = self._use_speaker
                speaker_voice = self._speaker_voice
                user_name = self._user_name
                status = self._status
                status_start = self._status_start
                conversation = list(self._conversation)
                current_user_input = self._current_user_input
                current_user_ts = self._current_user_ts
                current_user_duration = self._current_user_duration
                current_tool_calls = list(self._current_tool_calls)
                current_thoughts = list(self._current_thoughts)
                show_all = self._show_all
                scroll_offset = self._scroll_offset

            self._renderer.render(
                agent_name=agent_name,
                wake_words=wake_words,
                responder_name=responder_name,
                response_style=response_style,
                use_speaker=use_speaker,
                speaker_voice=speaker_voice,
                user_name=user_name,
                status=status,
                status_start=status_start,
                conversation=conversation,
                current_user_input=current_user_input,
                current_user_ts=current_user_ts,
                current_user_duration=current_user_duration,
                current_tool_calls=current_tool_calls,
                current_thoughts=current_thoughts,
                show_all=show_all,
                scroll_offset=scroll_offset,
            )
            self._stop_event.wait(self.REFRESH_INTERVAL)

    def _input_loop(self) -> None:
        if _IS_WINDOWS:
            self._input_loop_windows()
        else:
            self._input_loop_unix()

    def _input_loop_windows(self) -> None:
        while not self._stop_event.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch == b"\x01":
                    self.toggle_show_all()
                elif ch == b"\xe0":
                    next_ch = msvcrt.getch()
                    if next_ch == b"H":
                        self.scroll_up()
                    elif next_ch == b"P":
                        self.scroll_down()
                elif ch == b"\x00":
                    msvcrt.getch()
            time.sleep(0.05)

    def _input_loop_unix(self) -> None:
        fd = sys.stdin.fileno()
        try:
            old_settings = termios.tcgetattr(fd)
        except (termios.error, ValueError):
            return
        try:
            tty.setcbreak(fd)
            while not self._stop_event.is_set():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if ch == "\x01":
                        self.toggle_show_all()
                    elif ch == "\x1b":
                        if select.select([sys.stdin], [], [], 0.05)[0]:
                            next_ch = sys.stdin.read(1)
                            if next_ch == "[":
                                if select.select([sys.stdin], [], [], 0.05)[0]:
                                    direction = sys.stdin.read(1)
                                    if direction == "A":
                                        self.scroll_up()
                                    elif direction == "B":
                                        self.scroll_down()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
