import sys, time, threading, re, random

# Helper to strip ANSI escape codes before measuring string length,
# so box-drawing alignment is based on visible characters only.
_ANSI_ESCAPE_RE = re.compile(r"\033\[[0-9;]*m")


def _visible_len(text: str) -> int:
    return len(_ANSI_ESCAPE_RE.sub("", text))


class CliColor:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"

    # Foreground
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    ORANGE = "\033[38;5;208m"

    @staticmethod
    def fg(hex_or_256: int) -> str:
        return f"\033[38;5;{hex_or_256}m"


class CliSpinner:
    """
    Animated terminal spinner that runs on a background thread.
    Call .start(message) and .stop() around blocking work.
    """

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    COLORS = [CliColor.CYAN, CliColor.BLUE, CliColor.MAGENTA, CliColor.CYAN]

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

    def start(self, message: str | None = None) -> None:
        # Stop any existing spinner before starting a new one,
        # preventing leaked threads from double start() calls.
        if self._thread and self._thread.is_alive():
            self.stop()
        self._running = True
        self._stop_event.clear()
        if message:
            self._message = message
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def start_with_verbs(
        self,
        name: str,
        verbs: list[str] | None = None,
        interval: float = 10.0,
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
        """
        verbs = verbs if verbs is not None else self.DEFAULT_VERBS

        def _get_random_message():
            idx = random.randrange(len(verbs))
            return f"{name} is {verbs[idx]}"

        self.start(_get_random_message())

        def _verb_cycle() -> None:
            while not self._stop_event.wait(timeout=interval):
                self.update(_get_random_message())

        self._verb_thread = threading.Thread(target=_verb_cycle, daemon=True)
        self._verb_thread.start()

    def update(self, message: str) -> None:
        self._message = message

    def stop(self, final_message: str | None = None) -> None:
        was_running = self._running
        self._running = False
        self._stop_event.set()
        # Guard join() so it's only called when the thread actually ran.
        if self._thread and self._thread.is_alive():
            self._thread.join()
        self._thread = None
        if self._verb_thread and self._verb_thread.is_alive():
            self._verb_thread.join()
        self._verb_thread = None
        # Clear the spinner line
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()
        if final_message:
            print(final_message)
        return was_running

    def _spin(self) -> None:
        frame_idx = 0
        color_idx = 0
        dot_count = 0
        while not self._stop_event.is_set():
            frame = self.FRAMES[frame_idx % len(self.FRAMES)]
            color = self.COLORS[color_idx % len(self.COLORS)]
            dots = "." * (dot_count % 4)

            visible_content = f"  {frame}  {self._message}{dots:<3}"
            padding = max(0, 60 - _visible_len(visible_content)) * " "
            line = (
                f"\r  {color}{CliColor.BOLD}{frame}{CliColor.RESET}  "
                f"{CliColor.WHITE}{self._message}{CliColor.GRAY}{dots:<3}{CliColor.RESET}"
                f"{padding}"
            )
            sys.stdout.write(line)
            sys.stdout.flush()

            time.sleep(0.08)
            frame_idx += 1
            if frame_idx % 5 == 0:
                dot_count += 1
            if frame_idx % 20 == 0:
                color_idx += 1


class CliPrinter:
    @staticmethod
    def divider(
        char: str = "─", width: int = 60, color: str = CliColor.GRAY
    ) -> None:
        print(f"{color}{char * width}{CliColor.RESET}")

    @staticmethod
    def empty_line() -> None:
        """Create an empty line for spacing."""
        print()

    @staticmethod
    def header(label: str) -> None:
        inner = f"  {CliColor.CYAN}{CliColor.BOLD}Spych{CliColor.RESET}: {CliColor.WHITE}{label}{CliColor.RESET}"
        pad = max(0, 58 - _visible_len(inner))
        print(
            f"\n{CliColor.GRAY}┌{'─' * 58}┐{CliColor.RESET}\n"
            f"{CliColor.GRAY}│{CliColor.RESET}{inner}{CliColor.RESET}"
            f"{' ' * pad}{CliColor.GRAY}│{CliColor.RESET}\n"
            f"{CliColor.GRAY}└{'─' * 58}┘{CliColor.RESET}"
        )

    @staticmethod
    def kwarg_inputs(**kwargs) -> None:
        for key, value in kwargs.items():
            print(
                f"  {CliColor.GRAY}{key}{CliColor.RESET}: {CliColor.WHITE}{value}{CliColor.RESET}"
            )

    @staticmethod
    def label(tag: str, text: str, color: str = CliColor.CYAN) -> None:
        print(
            f"  {color}{CliColor.BOLD}{tag}{CliColor.RESET} {CliColor.WHITE}{text}{CliColor.RESET}"
        )

    @staticmethod
    def tool_event(
        tool_name: str,
        status: str,
        is_running: bool = False,
        elapsed: float | None = None,
    ) -> None:
        icon = "⚙" if is_running else "✓"
        color = CliColor.YELLOW if is_running else CliColor.GREEN
        elapsed_str = (
            f" {CliColor.GRAY}({elapsed:.2f}s){CliColor.RESET}"
            if elapsed
            else ""
        )
        print(
            f"  {color}{icon}{CliColor.RESET}  {CliColor.DIM}tool:{CliColor.RESET} {CliColor.ITALIC}{tool_name}{CliColor.RESET} -> {CliColor.GRAY}{status}{elapsed_str}"
        )

    @staticmethod
    def info(message: str, color: str = CliColor.CYAN) -> None:
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
            - Type: str (CliColor constant)
            - What: ANSI color code for the message
            - Default: CliColor.CYAN
        """
        print(
            f"  {color}{CliColor.BOLD}i{CliColor.RESET}  {CliColor.WHITE}{message}{CliColor.RESET}"
        )

    @staticmethod
    def typewrite(text: str, delay: float = 0.008) -> None:
        """Print text with a subtle typewriter effect."""
        for ch in text:
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(delay)
        print()

    @staticmethod
    def print_response(name: str, text: str) -> None:
        """Render the final response with light formatting."""
        print(f"  {CliColor.MAGENTA}{CliColor.BOLD}{name}:{CliColor.RESET}")
        print()
        print(text)

    @staticmethod
    def print_status(name: str, success: bool, elapsed: float) -> None:
        icon = "✓" if success else "✗"
        color = CliColor.GREEN if success else CliColor.RED
        print(
            f"\n  {color}{icon}{CliColor.RESET} {CliColor.DIM}{name} {elapsed:.2f}s{CliColor.RESET}"
        )
