from spych.utils import Notify
from spych.cli import CliColor, CliSpinner, CliPrinter
from typing import Optional
import time


class BaseResponder(Notify):
    def __init__(
        self,
        spych_object: "Spych",
        listen_duration: int | float = 5,
        name: Optional[str] = None,
    ) -> None:
        """
        Usage:

        - Base class for all responders. Handles the listen-transcribe-respond cycle,
          provides a consistent interface for subclasses to implement, and includes
          a rich terminal UI and animated spinner

        - Subclasses only need to implement `respond(user_input: str) -> str`.
          All CLI chrome (spinner, dividers, timing, response box) is handled here.

        - Public helper methods are available inside `respond()` for common
          UI needs without importing CLI internals:

            - `self.spinner.start(message)`          — restart spinner after a pause
            - `self.spinner.update(message)`         — change the spinner label
            - `self.spinner.stop()`                  — stop spinner (e.g. before printing)
            - `self.print_info(message, color)`      — print a styled info line

        Requires:

        - `spych_object`:
            - Type: Spych
            - What: An initialized Spych instance used to record and transcribe audio
              after the wake word is detected

        Optional:

        - `listen_duration`:
            - Type: int | float
            - What: The number of seconds to listen for after the wake word is detected
            - Default: 5

        - `name`:
            - Type: str
            - What: A custom name for the responder to use in printed messages
            - Default: The class name of the responder (e.g., "Ollama")

        Notes:

        - Subclasses must implement the `respond` method
        - The `__call__` method orchestrates the full listen -> transcribe -> respond
          cycle; subclasses only need to handle the response logic
        """
        self.spych_object = spych_object
        self.listen_duration = max(
            listen_duration, 3
        )  # enforce a minimum listen duration of 3 seconds
        self.name = name if name else self.__class__.__name__
        self.spinner = CliSpinner()
        self._start_time: float = 0.0

    # ------------------------------------------------------------------ #
    #  Public helper API — safe to call from inside respond()            #
    # ------------------------------------------------------------------ #

    def wait_for_next_wake_word(self, divider: bool = True) -> None:
        """
        Usage:

        - Call this to print a divider and reset the spinner after each complete cycle,
        to indicate that the responder is waiting for the next wake word. This is called
        by default at the end of `__call__`, but you can also call it manually if you want
        to reset the UI state at any point (e.g. after an error).

        Optional:

        - `divider`:
            - Type: bool
            - What: Whether to print a divider line before restarting the spinner
            - Default: True
        """
        if divider:
            CliPrinter.divider("─", 60, CliColor.GRAY)
        self.spinner.start(f"Waiting for wake word")

    def tool_event(
        self,
        tool_name: str,
        status: str,
        is_running: bool = False,
        elapsed: float | None = None,
    ) -> None:
        was_running = self.spinner.stop()
        CliPrinter.tool_event(
            tool_name, status, is_running=is_running, elapsed=elapsed
        )
        if was_running:
            self.spinner.start()

    def print_info(self, message: str, color: str = CliColor.CYAN) -> None:
        """
        Usage:

        - Print a styled informational line from inside `respond()`.
          The spinner is paused automatically and resumed afterwards so
          the output line is never overwritten.

        Requires:

        - `message`:
            - Type: str
            - What: The message to display

        Optional:

        - `color`:
            - Type: str (CliColor constant)
            - What: ANSI color for the info icon
            - Default: CliColor.CYAN
        """
        was_running = self.spinner.stop()
        CliPrinter.info(message, color)
        if was_running:
            self.spinner.start()

    # ------------------------------------------------------------------ #
    #  Extension hooks — override in subclasses for custom behaviour      #
    # ------------------------------------------------------------------ #

    def respond(self, user_input: str) -> str:
        """
        Usage:

        - Called with the transcribed (and optionally clarified) user input.
          Must return a response string. All CLI chrome is handled by the base
          class; this method only needs to produce the response.

        - Use the public helper methods for UI feedback inside this method:

            - `self.spinner.start(message)`          — restart spinner after a pause
            - `self.spinner.update(message)`         — change the spinner label
            - `self.spinner.stop()`                  — stop spinner (e.g. before printing)
            - `self.print_info(message, color)`      — print a styled info line

        Requires:

        - `user_input`:
            - Type: str
            - What: The transcribed text from the user's audio input

        Returns:

        - `response`:
            - Type: str
            - What: The response string to print to the terminal
        """
        raise NotImplementedError(
            "Subclasses must implement the `respond` method"
        )

    def on_before_respond(self, user_input: str) -> None:
        """
        Usage:

        - Optional lifecycle hook called immediately before `respond()`.
          Override for setup, logging, or any pre-flight work.
          The spinner is already running when this is called.

        Requires:

        - `user_input`:
            - Type: str
            - What: The enriched transcribed input (after optional clarification)
        """

    def on_after_respond(self, user_input: str, response: str) -> None:
        """
        Usage:

        - Optional lifecycle hook called immediately after `respond()` returns,
          before the response box is printed. Override for logging, analytics,
          caching, or post-processing.

        Requires:

        - `user_input`:
            - Type: str
            - What: The enriched transcribed input passed to `respond()`

        - `response`:
            - Type: str
            - What: The raw string returned by `respond()`
        """

    # ------------------------------------------------------------------ #
    #  Orchestration — not intended for override; use the hooks above     #
    # ------------------------------------------------------------------ #

    def on_listen_start(self) -> None:
        # Start a spinner immediately to indicate we're processing the wake event and listening;
        # this also gives a visual cue that the wake was detected successfully and the responder is active
        self.spinner.update(
            f"{CliColor.BOLD}{CliColor.MAGENTA}{self.name}{CliColor.RESET} {CliColor.GREEN}is listening for {self.listen_duration}s{CliColor.RESET}"
        )

    def on_user_input(self, user_input: str) -> None:
        CliPrinter.label("User:", user_input)
        self._start_time = time.time()
        self.spinner.start_with_verbs(self.name, interval=15)

    def on_response(self, response: str) -> None:
        elapsed = time.time() - self._start_time
        self.spinner.stop()
        if response:
            CliPrinter.print_response(self.name, response)
            CliPrinter.print_status(self.name, success=True, elapsed=elapsed)
        else:
            CliPrinter.print_status(self.name, success=False, elapsed=elapsed)

    def on_terminate(self) -> None:
        self.spinner.stop()
        CliPrinter.info(f"{self.name} has been terminated.")

    def on_listen_end(self) -> None:
        self.spinner.stop()

    def __call__(self) -> str:
        """
        Usage:

        - Executes one full wake-triggered cycle: listens, transcribes, responds,
          and prints.

        Returns:

        - `response`:
            - Type: str
            - What: The response string returned by `respond`
        """
        self.on_listen_start()
        user_input = self.spych_object.listen(duration=self.listen_duration)
        self.on_listen_end()
        if not user_input:
            self.wait_for_next_wake_word(divider=False)
            return ""
        self.on_user_input(user_input)
        try:
            self.on_before_respond(user_input)
            response = self.respond(user_input)
            self.on_after_respond(user_input, response)
        except Exception as exc:
            self.spinner.stop()
            print(f"  {CliColor.RED}✗  Error: {exc}{CliColor.RESET}\n")
            return ""
        self.on_response(response)
        self.wait_for_next_wake_word()
        return response

    def ready_message(self, **kwargs) -> None:
        """
        Formats and prints the ready message when the responder is initialized, showing the wake words and terminate words.

        Requires:

        - `wake_words`:
            - Type: list[str]
            - What: List of wake words that trigger the responder

        - `terminate_words`:
            - Type: list[str]
            - What: List of words that can be spoken to terminate the responder
        """
        CliPrinter.header(self.name)
        CliPrinter.kwarg_inputs(**kwargs)
        CliPrinter.empty_line()
        self.wait_for_next_wake_word()
