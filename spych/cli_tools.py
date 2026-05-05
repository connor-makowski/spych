import sys, time, threading, re, random, os

from spych.spinners import Spinner
from spych.utils import supports_unicode

_HAS_UNICODE = supports_unicode()

# Helper to strip ANSI escape codes before measuring string length,
# so box-drawing alignment is based on visible characters only.
_ANSI_ESCAPE_RE = re.compile(r"\033\[[0-9;]*m")


def _visible_len(text: str) -> int:
    return len(_ANSI_ESCAPE_RE.sub("", text))


# ---------------------------------------------------------------------------
# Theme system
# ---------------------------------------------------------------------------

# Escape-code constants used only inside this module to build the palettes.
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_ITALIC = "\033[3m"


class Theme:
    """
    Holds the active color palette.  Four built-in themes are provided:

    - ``"dark"``      (default) — bright colors on dark backgrounds.
    - ``"light"``     — deep, high-contrast colors on light/white backgrounds.
    - ``"solarized"`` — Solarized-Dark accent palette; comfortable for long sessions.
    - ``"mono"``      — greyscale only; no color, just bold/dim contrast.

    Every role needed by ``CliSpinner`` and ``CliPrinter`` lives here,
    including the formatting escapes (``reset``, ``bold``, ``italic``) so
    nothing outside this class references raw ANSI strings.
    """

    THEMES: dict[str, dict] = {
        # ------------------------------------------------------------------
        # dark — original palette, bright on dark
        # ------------------------------------------------------------------
        "dark": {
            # formatting
            "reset": _RESET,
            "bold": _BOLD,
            "italic": _ITALIC,
            # structural
            "chrome": "\033[90m",  # dark-gray  — borders, dividers
            "body": "\033[97m",  # bright white — primary text
            "dim": _DIM,  # secondary / de-emphasised
            # semantic
            "accent": "\033[96m",  # bright cyan  — brand accent
            "highlight": "\033[95m",  # bright magenta — speaker label
            "running": "\033[93m",  # bright yellow — in-progress
            "success": "\033[92m",  # bright green  — completed
            "error": "\033[91m",  # bright red    — failures
            # spinner
            "spinner_colors": [
                "\033[96m",  # bright cyan
                "\033[94m",  # bright blue
                "\033[95m",  # bright magenta
                "\033[96m",
            ],
        },
        # ------------------------------------------------------------------
        # light — readable on white/light backgrounds
        # ------------------------------------------------------------------
        "light": {
            "reset": _RESET,
            "bold": _BOLD,
            "italic": _ITALIC,
            "chrome": "\033[90m",  # dark-gray borders
            "body": "\033[30m",  # black text
            "dim": _DIM,
            "accent": "\033[36m",  # teal
            "highlight": "\033[35m",  # magenta/purple
            "running": "\033[33m",  # amber
            "success": "\033[32m",  # dark green
            "error": "\033[31m",  # dark red
            "spinner_colors": [
                "\033[36m",  # teal
                "\033[34m",  # blue
                "\033[35m",  # magenta
                "\033[36m",
            ],
        },
        # ------------------------------------------------------------------
        # solarized — Solarized-Dark accent colors, muted base
        # ------------------------------------------------------------------
        "solarized": {
            "reset": _RESET,
            "bold": _BOLD,
            "italic": _ITALIC,
            "chrome": "\033[38;5;240m",  # base01  — subtle borders
            "body": "\033[38;5;252m",  # base2   — primary text
            "dim": _DIM,
            "accent": "\033[38;5;37m",  # cyan    (#2aa198)
            "highlight": "\033[38;5;125m",  # magenta (#d33682)
            "running": "\033[38;5;136m",  # yellow  (#b58900)
            "success": "\033[38;5;64m",  # green   (#859900)
            "error": "\033[38;5;160m",  # red     (#dc322f)
            "spinner_colors": [
                "\033[38;5;37m",  # cyan
                "\033[38;5;33m",  # blue   (#268bd2)
                "\033[38;5;61m",  # violet (#6c71c4)
                "\033[38;5;37m",
            ],
        },
        # ------------------------------------------------------------------
        # mono — greyscale; bold/dim contrast only, no hue
        # ------------------------------------------------------------------
        "mono": {
            "reset": _RESET,
            "bold": _BOLD,
            "italic": _ITALIC,
            "chrome": "\033[90m",  # dark gray
            "body": "\033[97m",  # bright white
            "dim": _DIM,
            "accent": _BOLD,  # bold, no hue
            "highlight": _BOLD,
            "running": "\033[97m",
            "success": "\033[97m",
            "error": _BOLD,
            "spinner_colors": [
                "\033[97m",  # bright white
                "\033[37m",  # light gray
                "\033[90m",  # dark gray
                "\033[97m",
            ],
        },
    }

    VALID: tuple[str, ...] = tuple(THEMES.keys())

    def __init__(self) -> None:
        self._palette: dict = self.THEMES["dark"]

    # Convenience accessors -------------------------------------------------

    @property
    def reset(self) -> str:
        return self._palette["reset"]

    @property
    def bold(self) -> str:
        return self._palette["bold"]

    @property
    def italic(self) -> str:
        return self._palette["italic"]

    @property
    def chrome(self) -> str:
        return self._palette["chrome"]

    @property
    def body(self) -> str:
        return self._palette["body"]

    @property
    def dim(self) -> str:
        return self._palette["dim"]

    @property
    def accent(self) -> str:
        return self._palette["accent"]

    @property
    def highlight(self) -> str:
        return self._palette["highlight"]

    @property
    def running(self) -> str:
        return self._palette["running"]

    @property
    def success(self) -> str:
        return self._palette["success"]

    @property
    def error(self) -> str:
        return self._palette["error"]

    @property
    def spinner_colors(self) -> list[str]:
        return self._palette["spinner_colors"]

    # Mutation --------------------------------------------------------------

    def apply(self, name: str) -> None:
        """Switch the active theme by name."""
        if name not in self.THEMES:
            raise ValueError(
                f"Unknown theme {name!r}. Choose one of: {', '.join(self.VALID)}"
            )
        self._palette = self.THEMES[name]


# Module-level singleton consumed by CliSpinner and CliPrinter.
theme = Theme()


def set_theme(name: str) -> None:
    """
    Switch the global CLI color theme.

    Call this **once**, before any output is produced — typically right after
    argument parsing in ``cli.py``.

    Requires:

    - ``name``:
        - Type: ``str``
        - What: One of ``"dark"`` (default), ``"light"``, ``"solarized"``,
          or ``"mono"``
    """
    theme.apply(name)


# ---------------------------------------------------------------------------


class CliSpinner:
    """
    Animated terminal spinner that runs on a background thread.
    Call .start(message) and .stop() around blocking work.
    """

    DEFAULT_VERBS = [
        "thinking",
        "vibing",
        "pontificating",
        "contemplating",
        "deliberating",
        "cogitating",
        "ruminating",
        "musing",
        "ideating",
        "postulating",
        "hypothesizing",
        "extrapolating",
        "philosophizing",
        "noodling",
        "percolating",
        "marinating",
        "stewing",
        "scheming",
        "conniving",
        "divining",
        "spelunking",
        "ratiocinating",
        "cerebrating",
        "woolgathering",
        "daydreaming",
        "lucubrating",
        "excogitating",
        "thinkulating",
        "brainwaving",
        "cogitronning",
        "synapsing",
        "thoughtcrafting",
        "mindweaving",
        "intellectualizing",
        "computating",
        "ponderizing",
        "mentalating",
        "brainbrewing",
    ]

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._message = ""
        self._verb_thread: threading.Thread | None = None
        self._running = False
        self._frames = Spinner.BRAILLE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        message: str | None = None,
        spinner: list[str] | None = None,
    ) -> None:
        """
        Start the spinner.

        Optional:

        - ``message``:
            - Type: str | None
            - What: Text displayed beside the spinner frame.
            - Default: None (keeps previous message)

        - ``spinner``:
            - Type: list[str] | None
            - What: Spinner Frames to use
            - Default: None (Braille)
        """
        if self._thread and self._thread.is_alive():
            self.stop()
        self._running = True
        self._stop_event.clear()
        if spinner is not None:
            self._frames = spinner
        if message:
            self._message = message
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def start_with_verbs(
        self,
        name: str,
        verbs: list[str] | None = None,
        interval: float = 10.0,
        spinner: list[str] | None = None,
    ) -> None:
        """
        Start the spinner with a cycling verb message: "<name> is <verb>".
        The verb rotates through `verbs` every `interval` seconds.

        Requires:

        - `name`:
            - Type: str
            - What: The subject displayed before the verb (e.g. "Claude")

        Optional:

        - `verbs`:
            - Type: list[str] | None
            - What: Verbs to cycle through. Defaults to CliSpinner.DEFAULT_VERBS
            - Default: None

        - `interval`:
            - Type: float
            - What: Seconds between each verb swap
            - Default: 10.0

        - `spinner`:
            - Type: list[str] | None
            - What: Frame list or None to not change.
            - Default: None
        """
        verbs = verbs if verbs is not None else self.DEFAULT_VERBS

        def _get_random_message():
            idx = random.randrange(len(verbs))
            return f"{name} is {verbs[idx]}"

        def _get_random_spinner():
            options = [
                Spinner.ARC,
                Spinner.CIRCLE_ARCS,
                Spinner.CIRCLE_FILL,
                Spinner.LINE,
                Spinner.DOT_PULSE,
                Spinner.BOUNCE,
                Spinner.BLOCK,
            ]
            idx = random.randrange(len(options))
            return options[idx]

        self.start(_get_random_message(), spinner=_get_random_spinner())

        def _verb_cycle() -> None:
            while not self._stop_event.wait(timeout=interval):
                # Update message and spinner frames directly instead of
                # calling start(), which would call stop() and try to
                # join this thread — causing "cannot join current thread".
                self._message = _get_random_message()
                self._frames = _get_random_spinner()

        self._verb_thread = threading.Thread(target=_verb_cycle, daemon=True)
        self._verb_thread.start()

    def stop(self, final_message: str | None = None) -> None:
        was_running = self._running
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join()
        self._thread = None
        # Don't join verb_thread if we're being called from within it —
        # that would raise "cannot join current thread".
        if (
            self._verb_thread
            and self._verb_thread.is_alive()
            and self._verb_thread is not threading.current_thread()
        ):
            self._verb_thread.join()
        self._verb_thread = None
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()
        if final_message:
            print(final_message)
        return was_running

    def _spin(self) -> None:
        frame_idx = 0
        color_idx = 0
        dot_count = 0
        frames = self._frames  # snapshot so swaps don't mid-spin
        while not self._stop_event.is_set():
            frame = frames[frame_idx % len(frames)]
            colors = theme.spinner_colors
            color = colors[color_idx % len(colors)]
            dots = "." * (dot_count % 4)

            visible_content = f"  {frame}  {self._message}{dots:<3}"
            padding = max(0, 60 - _visible_len(visible_content)) * " "
            line = (
                f"\r  {color}{theme.bold}{frame}{theme.reset}  "
                f"{theme.body}{self._message}{theme.chrome}{dots:<3}{theme.reset}"
                f"{padding}"
            )
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except UnicodeEncodeError:
                # Fallback: strip non-ASCII characters (like braille) from the line
                # but keep the rest of the message and formatting.
                safe_line = "".join(c if ord(c) < 128 else " " for c in line)
                sys.stdout.write(safe_line)
                sys.stdout.flush()

            time.sleep(0.12)
            frame_idx += 1
            if frame_idx % 5 == 0:
                dot_count += 1
            if frame_idx % 20 == 0:
                color_idx += 1

            # Re-read frames each tick so verb-cycle updates take effect
            frames = self._frames


class NullSpinner:
    """
    Usage:

    - Drop-in replacement for ``CliSpinner`` that suppresses all terminal
      output. Used automatically by ``BaseResponder`` when an
      ``AgentDashboard`` is active, so the spinner never corrupts the
      alternate screen buffer.

    Notes:

    - All methods are no-ops. ``stop()`` returns ``False`` to match the
      ``CliSpinner.stop()`` return value contract.
    """

    def start(
        self, message: str | None = None, spinner: list[str] | None = None
    ) -> None:
        pass

    def start_with_verbs(
        self,
        name: str,
        verbs: list[str] | None = None,
        interval: float = 10.0,
        spinner: list[str] | None = None,
    ) -> None:
        pass

    def stop(self, final_message: str | None = None) -> bool:
        return False


class CliPrinter:
    @staticmethod
    def divider(
        char: str = "─", width: int = 60, color: str | None = None
    ) -> None:
        if not _HAS_UNICODE and char == "─":
            char = "-"
        color = color if color is not None else theme.accent
        try:
            print(f"{color}{char * width}{theme.reset}")
        except UnicodeEncodeError:
            # Fallback to standard hyphen if the chosen char fails
            print(f"{color}{'-' * width}{theme.reset}")

    @staticmethod
    def empty_line() -> None:
        """Create an empty line for spacing."""
        print()

    @staticmethod
    def header(label: str) -> None:
        inner = (
            f"  {theme.accent}{theme.bold}Spych{theme.reset}"
            f": {theme.body}{label}{theme.reset}"
        )
        pad = max(0, 58 - _visible_len(inner))
        
        # Safe characters for box drawing
        tl, tr, bl, br, h, v = ("┌", "┐", "└", "┘", "─", "│") if _HAS_UNICODE else ("+", "+", "+", "+", "-", "|")
        
        try:
            print(
                f"\n{theme.chrome}{tl}{h * 58}{tr}{theme.reset}\n"
                f"{theme.chrome}{v}{theme.reset}{inner}{theme.reset}"
                f"{' ' * pad}{theme.chrome}{v}{theme.reset}\n"
                f"{theme.chrome}{bl}{h * 58}{br}{theme.reset}"
            )
        except UnicodeEncodeError:
            # Absolute ASCII fallback
            print(
                f"\n{theme.chrome}+{'=' * 58}+{theme.reset}\n"
                f"{theme.chrome}|{theme.reset}{inner}{theme.reset}"
                f"{' ' * pad}{theme.chrome}|{theme.reset}\n"
                f"{theme.chrome}+{'=' * 58}+{theme.reset}"
            )

    @staticmethod
    def kwarg_inputs(**kwargs) -> None:
        for key, value in kwargs.items():
            print(
                f"  {theme.accent}{key}{theme.reset}: {theme.body}{value}{theme.reset}"
            )

    @staticmethod
    def label(tag: str, text: str, color: str | None = None) -> None:
        color = color if color is not None else theme.accent
        print(
            f"  {color}{theme.bold}{tag}{theme.reset} {theme.body}{text}{theme.reset}"
        )

    @staticmethod
    def tool_event(
        tool_name: str,
        status: str,
        is_running: bool = False,
        elapsed: float | None = None,
        detail: str | None = None,
    ) -> None:
        icon = ("⚙" if _HAS_UNICODE else "*") if is_running else ("✓" if _HAS_UNICODE else "+")
        color = theme.running if is_running else theme.success
        elapsed_str = (
            f" {theme.chrome}({elapsed:.2f}s){theme.reset}" if elapsed else ""
        )
        detail_str = f"  {theme.chrome}{detail}{theme.reset}" if detail else ""
        print(
            f"  {color}{icon}{theme.reset}  {theme.dim}tool:{theme.reset} "
            f"{theme.italic}{tool_name}{theme.reset}{detail_str}{elapsed_str}"
        )

    @staticmethod
    def info(message: str, color: str | None = None) -> None:
        """
        Usage:

        - Print a single informational line. Useful from inside respond() to
          surface status updates without touching the spinner directly.

        Requires:

        - `message`:
            - Type: str
            - What: The message to print

        Optional:

        - `color`:
            - Type: str (ANSI escape code)
            - What: ANSI color code for the message. Defaults to theme accent.
            - Default: None
        """
        color = color if color is not None else theme.accent
        icon = "i"
        print(
            f"  {color}{theme.bold}{icon}{theme.reset}  {theme.body}{message}{theme.reset}"
        )

    @staticmethod
    def typewrite(text: str, delay: float = 0.008) -> None:
        """Print text with a subtle typewriter effect."""
        for ch in text:
            try:
                sys.stdout.write(ch)
            except UnicodeEncodeError:
                sys.stdout.write(" ")
            sys.stdout.flush()
            time.sleep(delay)
        print()

    @staticmethod
    def print_response(name: str, text: str) -> None:
        """Render the final response with light formatting."""
        print(f"  {theme.highlight}{theme.bold}{name}:{theme.reset}")
        print()
        # Use a safe way to print the response text to avoid UnicodeEncodeError on Windows
        try:
            print(text)
        except UnicodeEncodeError:
            # Fallback for the whole block of text
            print(text.encode("ascii", "replace").decode("ascii"))

    @staticmethod
    def print_summary(text: str) -> None:
        """Render a condensed summary line below the full response."""
        try:
            print(
                f"\n  {theme.dim}Summary:{theme.reset} {theme.body}{text}{theme.reset}"
            )
        except UnicodeEncodeError:
            safe_text = text.encode("ascii", "replace").decode("ascii")
            print(
                f"\n  {theme.dim}Summary:{theme.reset} {theme.body}{safe_text}{theme.reset}"
            )

    @staticmethod
    def print_status(name: str, success: bool, elapsed: float) -> None:
        icon = ("✓" if _HAS_UNICODE else "+") if success else ("✗" if _HAS_UNICODE else "x")
        color = theme.success if success else theme.error
        print(
            f"\n  {color}{icon}{theme.reset} {theme.dim}{name} {elapsed:.2f}s{theme.reset}"
        )
