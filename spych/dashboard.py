"""
spych TUI Dashboard.

Usage:

- ``AgentDashboard`` renders a live three-section terminal interface in the
  alternate screen buffer and accepts lifecycle events from ``BaseResponder``.

- Start before launching the agent; stop in a finally block:

    dashboard = AgentDashboard(agent_name="Claude", wake_words=["claude"])
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

try:
    import select
    import termios
    import tty

    _IS_WINDOWS = False
except ImportError:
    import msvcrt

    _IS_WINDOWS = True


# ── ANSI escape codes ──────────────────────────────────────────────────────

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_ITALIC = "\033[3m"

_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"

_CURSOR_HIDE = "\033[?25l"
_CURSOR_SHOW = "\033[?25h"
_ALT_SCREEN_ENTER = "\033[?1049h"
_ALT_SCREEN_EXIT = "\033[?1049l"
_CURSOR_HOME = "\033[H"
_CLEAR_BELOW = "\033[J"
_CLEAR_EOL = "\033[K"

# ── Status constants ───────────────────────────────────────────────────────

WAITING = "waiting"
LISTENING = "listening"
THINKING = "thinking"
SPEAKING = "speaking"
TERMINATED = "terminated"


# ── Data structures ────────────────────────────────────────────────────────


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
    """

    name: str
    status: str
    is_running: bool
    elapsed: Optional[float]
    timestamp: str


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
    """

    text: str
    timestamp: str


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


# ── Renderer ───────────────────────────────────────────────────────────────


class _DashboardRenderer:
    _HEADER_LINES = 4  # rule + title + rule + blank
    _FOOTER_LINES = 2  # blank + hint line
    _TOOL_FIXED = 3  # label + top-rule + bottom-rule (only when tools present)
    _CONV_FIXED = 3  # label + top-rule + bottom-rule
    _FIXED_BASE = _HEADER_LINES + _CONV_FIXED + _FOOTER_LINES  # = 9

    _SUMMARY_CHAR_LIMIT = 200

    def __init__(self) -> None:
        self._prev_cols = 0
        self._prev_rows = 0

    _THINKING_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    _THINKING_VERBS = [
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

    @staticmethod
    def _status_str(status: str) -> str:
        if status == THINKING:
            idx = int(time.time() * 10) % len(_DashboardRenderer._THINKING_SPINNER)
            v_idx = (
                int(time.time() / 2) % len(_DashboardRenderer._THINKING_VERBS)
            )
            spinner = _DashboardRenderer._THINKING_SPINNER[idx]
            verb = _DashboardRenderer._THINKING_VERBS[v_idx]
            return f"{_YELLOW}{spinner} {verb}{_RESET}"

        indicators: dict[str, str] = {
            LISTENING: f"{_BOLD}{_GREEN}▶▶▶ LISTENING ◀◀◀{_RESET}",
            SPEAKING: f"{_CYAN}♪ Speaking{_RESET}",
            TERMINATED: f"{_DIM}✗ Stopped{_RESET}",
            WAITING: f"{_DIM}○ Waiting{_RESET}",
        }
        return indicators.get(status, f"{_DIM}○ Waiting{_RESET}")

    @staticmethod
    def _truncate(text: str, width: int) -> str:
        if len(text) <= width:
            return text
        return text[: width - 1] + "…"

    @staticmethod
    def _visible_len(text: str) -> int:
        return len(re.sub(r"\033\[[0-9;]*[mK]", "", text))

    @staticmethod
    def _wrap_text(text: str, width: int, indent: int = 0) -> list[str]:
        if not text:
            return [""]
        wrapper = textwrap.TextWrapper(
            width=width,
            initial_indent=" " * indent,
            subsequent_indent=" " * indent,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        return wrapper.wrap(text)

    def render(
        self,
        agent_name: str,
        wake_words: list[str],
        response_style: str,
        use_speaker: bool,
        speaker_voice: str,
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

        bar_width = max(10, cols - 4)
        budget = rows - 1
        lines: list[str] = []

        # ── Header ────────────────────────────────────────────────────────
        lines.append(f"  {_BOLD}{'━' * bar_width}{_RESET}")

        header_parts: list[str] = [
            f"  {_BOLD}{_CYAN}SPYCH{_RESET}",
            f"{_BOLD}{agent_name}{_RESET}",
            self._status_str(status),
        ]
        if wake_words:
            shown = ", ".join(f'"{w}"' for w in wake_words[:2])
            if len(wake_words) > 2:
                shown += f" +{len(wake_words) - 2}"
            header_parts.append(f"{_DIM}wake: {shown}{_RESET}")
        if response_style:
            header_parts.append(f"{_DIM}{response_style}{_RESET}")
        if use_speaker and speaker_voice:
            header_parts.append(f"{_DIM}🔊 {speaker_voice}{_RESET}")
        if show_all:
            header_parts.append(f"{_BOLD}{_YELLOW}ALL LOGS{_RESET}")

        title = "  │  ".join(header_parts)
        # Truncate title if it exceeds width
        if self._visible_len(title) > cols - 1:
            while len(header_parts) > 3 and self._visible_len("  │  ".join(header_parts)) > cols - 5:
                header_parts.pop()
            title = "  │  ".join(header_parts)
            if self._visible_len(title) > cols - 1:
                title = title[:cols-5] + "..." + _RESET

        lines.append(title)
        lines.append(f"  {_BOLD}{'━' * bar_width}{_RESET}")
        lines.append("")

        if show_all:
            # ── All Logs Mode ──────────────────────────────────────────────
            max_log_lines = max(0, budget - self._HEADER_LINES - self._FOOTER_LINES)

            log_lines: list[str] = []
            for entry in conversation:
                ts = f"{_DIM}{entry.timestamp}{_RESET}"
                user_text = f"({entry.user_duration:.1f}s) {entry.user_input}"
                log_lines.extend([f"  {ts}  {_DIM}User:{_RESET}    " + l for l in self._wrap_text(user_text, cols - 25)])
                
                for thought in entry.thoughts:
                    th_ts = f"{_DIM}{thought.timestamp}{_RESET}"
                    log_lines.extend([f"    {th_ts}  {_ITALIC}{_DIM}Thought:{_RESET} " + l for l in self._wrap_text(thought.text, cols - 27)])

                for tool in entry.tool_calls:
                    icon = "⚙" if tool.is_running else "✓"
                    color = _YELLOW if tool.is_running else _GREEN
                    elapsed_str = f" ({tool.elapsed:.2f}s)" if tool.elapsed else ""
                    tool_header = f"    {color}{icon}{_RESET}  "
                    tool_content = f"{tool.name} → {tool.status}{elapsed_str}"
                    log_lines.extend([tool_header + l for l in self._wrap_text(tool_content, cols - 15)])

                for resp_line in entry.response.splitlines():
                    log_lines.extend(["  " + l for l in self._wrap_text(resp_line, cols - 5)])
                
                if (
                    len(entry.response) > self._SUMMARY_CHAR_LIMIT
                    and entry.summary != entry.response
                ):
                    log_lines.extend([f"  {_DIM}Summary:{_RESET} " + l for l in self._wrap_text(entry.summary, cols - 15)])
                
                log_lines.append(
                    f"  {_DIM}{'─' * min(bar_width, 60)}{_RESET}"
                )

            # In-progress turn
            if current_user_input or status == LISTENING or current_tool_calls or current_thoughts:
                if status == LISTENING:
                    elapsed = time.time() - status_start
                    ts = f"({elapsed:.1f}s)"
                    text = "Listening..."
                else:
                    ts = current_user_ts
                    text = f"({current_user_duration:.1f}s) {current_user_input}"

                if text:
                    log_lines.extend([f"  {_DIM}{ts}{_RESET}  {_DIM}User:{_RESET}    " + l for l in self._wrap_text(text, cols - 25)])
                
                for thought in current_thoughts:
                    th_ts = f"{_DIM}{thought.timestamp}{_RESET}"
                    log_lines.extend([f"    {th_ts}  {_ITALIC}{_DIM}Thought:{_RESET} " + l for l in self._wrap_text(thought.text, cols - 27)])

                for tool in current_tool_calls:
                    icon = "⚙" if tool.is_running else "✓"
                    color = _YELLOW if tool.is_running else _GREEN
                    elapsed_str = f" ({tool.elapsed:.2f}s)" if tool.elapsed else ""
                    tool_header = f"    {color}{icon}{_RESET}  "
                    tool_content = f"{tool.name} → {tool.status}{elapsed_str}"
                    log_lines.extend([tool_header + l for l in self._wrap_text(tool_content, cols - 15)])

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
            # ── Default Mode ───────────────────────────────────────────────
            has_tools = bool(current_tool_calls)
            
            tool_content_lines = []
            if has_tools:
                for tool in current_tool_calls:
                    icon = "⚙" if tool.is_running else "✓"
                    color = _YELLOW if tool.is_running else _GREEN
                    elapsed_str = f" ({tool.elapsed:.2f}s)" if tool.elapsed else ""
                    tool_header = f"  {_DIM}{tool.timestamp}{_RESET}  {color}{icon}{_RESET}  "
                    tool_content = f"{tool.name} → {tool.status}{elapsed_str}"
                    tool_content_lines.extend([tool_header + l for l in self._wrap_text(tool_content, cols - 25)])
            
            tool_height = (self._TOOL_FIXED + len(tool_content_lines)) if has_tools else 0
            max_conv_lines = max(0, budget - self._FIXED_BASE - tool_height)

            conv_display: list[list[str]] = [] 
            for entry in conversation:
                block = []
                ts = f"{_DIM}{entry.timestamp}{_RESET}"
                user_text = f"({entry.user_duration:.1f}s) {entry.user_input}"
                block.extend([f"  {ts}  {_DIM}User:{_RESET}    " + l for l in self._wrap_text(user_text, cols - 25)])
                
                resp_text = (
                    entry.summary
                    if len(entry.response) > self._SUMMARY_CHAR_LIMIT
                    and entry.summary != entry.response
                    else entry.response
                )
                block.extend([f"  {ts}  {_BOLD}{entry.agent_name}:{_RESET} " + l for l in self._wrap_text(resp_text, cols - 25)])
                conv_display.append(block)

            if current_user_input or status == LISTENING:
                block = []
                if status == LISTENING:
                    elapsed = time.time() - status_start
                    ts = f"({elapsed:.1f}s)"
                    text = "Listening..."
                else:
                    ts = current_user_ts
                    text = f"({current_user_duration:.1f}s) {current_user_input}"
                block.extend([f"  {_DIM}{ts}{_RESET}  {_DIM}User:{_RESET}    " + l for l in self._wrap_text(text, cols - 25)])
                conv_display.append(block)

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
                        visible_conv_lines = block[-max_conv_lines:]
                    break
            
            while len(visible_conv_lines) < max_conv_lines:
                visible_conv_lines.insert(0, "")

            lines.append(f"  {_BOLD}Recent Conversation{_RESET}")
            lines.append(f"  {'─' * bar_width}")
            for line in visible_conv_lines:
                lines.append(line)
            lines.append(f"  {'─' * bar_width}")

            if has_tools:
                lines.append(f"  {_BOLD}Active Tools{_RESET}")
                lines.append(f"  {'─' * bar_width}")
                for line in tool_content_lines:
                    lines.append(line)
                lines.append(f"  {'─' * bar_width}")

        lines.append("")
        lines.append(
            f"  {_DIM}Ctrl+C to stop  │  Ctrl+A to toggle mode  │  ↑/↓ to scroll{_RESET}"
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
        sys.stdout.write(buf)
        sys.stdout.flush()


# ── Dashboard controller ────────────────────────────────────────────────────


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
        3. Recent Conversation — clean per-turn view of user inputs and
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

    Notes:

    - Call ``start()`` to activate the TUI and ``stop()`` in a finally block
      to restore the terminal.
    - All public event methods are thread-safe.
    - ``stop()`` is idempotent; calling it twice is safe.
    """

    REFRESH_INTERVAL: float = 0.15

    def __init__(
        self,
        agent_name: str,
        wake_words: list[str],
        response_style: str = "",
        use_speaker: bool = False,
        speaker_voice: str = "",
        max_history: int = 500,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self._agent_name = agent_name
        self._wake_words = wake_words
        self._response_style = response_style
        self._use_speaker = use_speaker
        self._speaker_voice = speaker_voice
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

        self._show_all: bool = False
        self._scroll_offset: int = 0

        self._stop_event = stop_event or threading.Event()
        self._lock = threading.Lock()
        self._display_thread: Optional[threading.Thread] = None
        self._input_thread: Optional[threading.Thread] = None
        self._renderer = _DashboardRenderer()
        self._started: bool = False

    # ── Public event API ───────────────────────────────────────────────────

    def set_agent(
        self,
        name: str,
        wake_words: list[str],
        response_style: str = "",
        use_speaker: bool = False,
        speaker_voice: str = "",
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

        - ``response_style``:
            - Type: str
            - Default: ""

        - ``use_speaker``:
            - Type: bool
            - Default: False

        - ``speaker_voice``:
            - Type: str
            - Default: ""
        """
        with self._lock:
            self._agent_name = name
            self._wake_words = wake_words
            self._response_style = response_style
            self._use_speaker = use_speaker
            self._speaker_voice = speaker_voice

    def on_status_change(self, status: str) -> None:
        """
        Usage:

        - Updates the status indicator in the header.

        Requires:

        - ``status``:
            - Type: str
            - What: One of the module-level constants: WAITING, LISTENING,
              THINKING, SPEAKING, TERMINATED.
        """
        with self._lock:
            if self._status != status:
                self._status = status
                self._status_start = time.time()

    def on_user_input(self, text: str) -> None:
        """
        Usage:

        - Records the user's transcribed input and starts a new turn.

        Requires:

        - ``text``:
            - Type: str
            - What: Transcribed user input.
        """
        with self._lock:
            self._current_user_input = text
            self._current_user_ts = time.strftime("%H:%M:%S")
            self._current_user_duration = time.time() - self._status_start
            self._current_turn_start = time.time()
            self._current_tool_calls = []
            self._current_thoughts = []

    def on_thought(self, text: str) -> None:
        """
        Usage:

        - Records an internal thought or intermediate message.

        Requires:

        - ``text``:
            - Type: str
            - What: The content of the thought.
        """
        ts = time.strftime("%H:%M:%S")
        with self._lock:
            self._current_thoughts.append(ThoughtEntry(text=text, timestamp=ts))

    def on_tool_event(
        self,
        name: str,
        status: str,
        is_running: bool = False,
        elapsed: Optional[float] = None,
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
        """
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
                    )
                    return
            self._current_tool_calls.append(
                ToolEntry(
                    name=name,
                    status=status,
                    is_running=is_running,
                    elapsed=elapsed,
                    timestamp=ts,
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
        ts = time.strftime("%H:%M:%S")
        with self._lock:
            self._conversation.append(
                ConversationEntry(
                    user_input=self._current_user_input,
                    agent_name=agent_name,
                    tool_calls=list(self._current_tool_calls),
                    thoughts=list(self._current_thoughts),
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

    # ── TUI control ────────────────────────────────────────────────────────

    def toggle_show_all(self) -> None:
        """Toggle between minimal and all-logs mode."""
        self._show_all = not self._show_all
        self._scroll_offset = 0

    def scroll_up(self, amount: int = 1) -> None:
        """Scroll up."""
        self._scroll_offset += amount

    def scroll_down(self, amount: int = 1) -> None:
        """Scroll down."""
        self._scroll_offset = max(0, self._scroll_offset - amount)

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

    # ── Internal threads ───────────────────────────────────────────────────

    def _display_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                agent_name = self._agent_name
                wake_words = list(self._wake_words)
                response_style = self._response_style
                use_speaker = self._use_speaker
                speaker_voice = self._speaker_voice
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
                response_style=response_style,
                use_speaker=use_speaker,
                speaker_voice=speaker_voice,
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
