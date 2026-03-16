from spych.utils import Notify
from spych.cli_tools import CliSpinner, CliPrinter, theme
from spych.spinners import Spinner
from typing import Optional
import time


class BaseResponder(Notify):
    def __init__(
        self,
        spych_object: "Spych",
        listen_duration: int | float = 0,
        name: Optional[str] = None,
        spinner: Optional[CliSpinner] = None,
    ) -> None:
        """
        Usage:

        - Base class for all responders. Handles the listen-transcribe-respond cycle,
          provides a consistent interface for subclasses to implement, and includes
          a rich terminal UI and animated spinner.

        - Subclasses only need to implement `respond(user_input: str) -> str`.
          All CLI chrome (spinner, dividers, timing, response box) is handled here.

        - Public helper methods are available inside `respond()` for common
          UI needs without importing CLI internals:

            - `self.spinner.start(message)`          — restart spinner after a pause
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
            - Default: 0 (listen indefinitely until silence is detected)

        - `name`:
            - Type: str
            - What: A custom name for the responder to use in printed messages
            - Default: The class name of the responder (e.g., "Ollama")

        - `spinner`:
            - Type: CliSpinner | None
            - What: An externally-owned spinner to share across multiple responders.
              When provided (e.g. by SpychOrchestrator), all responders drive the same
              spinner so their output never interleaves. When None, a private spinner is
              created, preserving the original single-responder behaviour.
            - Default: None

        Notes:

        - Subclasses must implement the `respond` method.
        - The `__call__` method orchestrates the full voice listen -> transcribe -> respond
          cycle; use `text_input` for the typed equivalent.
        """
        self.spych_object = spych_object
        self.listen_duration = listen_duration
        self.name = name if name else self.__class__.__name__
        # Accept an injected shared spinner or create a private one.
        self.spinner: CliSpinner = (
            spinner if spinner is not None else CliSpinner()
        )
        self._start_time: float = 0.0

    # ------------------------------------------------------------------ #
    #  Public helper API — safe to call from inside respond()             #
    # ------------------------------------------------------------------ #

    def wait_for_next_wake_word(self, divider: bool = True) -> None:
        """
        Usage:

        - Print a divider and reset the spinner after each complete cycle to indicate
          that the responder is waiting for the next wake word. Called automatically at
          the end of `__call__` and `text_input`, but can also be called manually.

        Optional:

        - `divider`:
            - Type: bool
            - What: Whether to print a divider line before restarting the spinner
            - Default: True
        """
        if divider:
            CliPrinter.divider()
        self.spinner.start("Waiting for wake word", spinner=Spinner.BRAILLE)

    def tool_event(
        self,
        tool_name: str,
        status: str,
        is_running: bool = False,
        elapsed: float | None = None,
    ) -> None:
        """
        Usage:

        - Print a tool event message with the spinner paused and resumed automatically.
          This method stops the spinner before printing the tool event and restarts it
          afterwards to prevent interference with the output.

        Requires:

        - `tool_name`:
            - Type: str
            - What: The name of the tool being invoked

        - `status`:
            - Type: str
            - What: The status of the tool (e.g., "starting", "running", "completed", "failed")

        Optional:

        - `is_running`:
            - Type: bool
            - What: Whether the tool is currently running
            - Default: False

        - `elapsed`:
            - Type: float | None
            - What: Elapsed time in seconds since the tool started (if available)
            - Default: None
        """
        was_running = self.spinner.stop()
        CliPrinter.tool_event(
            tool_name, status, is_running=is_running, elapsed=elapsed
        )
        if was_running:
            self.spinner.start()

    def print_info(self, message: str, color: str | None = None) -> None:
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
            - Type: str | None
            - What: ANSI escape code for the info icon. Defaults to the theme accent.
            - Default: None
        """
        was_running = self.spinner.stop()
        CliPrinter.info(message, color)
        if was_running:
            self.spinner.start()

    def print_response(self, name: str, message: str) -> None:
        """
        Usage:

        - Print a styled response message with the spinner paused and resumed automatically.
          This method stops the spinner before printing the response and restarts it
          afterwards to prevent interference with the output.

        Requires:

        - `name`:
            - Type: str
            - What: The name of the responder

        - `message`:
            - Type: str
            - What: The response message to display
        """
        was_running = self.spinner.stop()
        CliPrinter.print_response(name, message)
        if was_running:
            self.spinner.start()

    # ------------------------------------------------------------------ #
    #  Extension hooks — override in subclasses for custom behaviour      #
    # ------------------------------------------------------------------ #

    def healthcheck(self) -> bool:
        """
        Usage:

        - Optional method that can be overridden to perform a health check of the responder's
          dependencies (e.g., API connectivity, model availability). If implemented, this method
          should return True if the responder is healthy and ready to respond, or False if there
          is an issue that would prevent it from functioning properly.

        """
        return True

    def respond(self, user_input: str) -> str:
        """
        Usage:

        - Called with the transcribed (or typed) user input. Must return a response
          string. All CLI chrome is handled by the base class; this method only needs
          to produce the response.

        - Use the public helper methods for UI feedback inside this method:

            - `self.spinner.start(message)`          — restart spinner after a pause
            - `self.spinner.stop()`                  — stop spinner (e.g. before printing)
            - `self.print_info(message, color)`      — print a styled info line

        Requires:

        - `user_input`:
            - Type: str
            - What: The transcribed or typed text from the user

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
        """
        Usage:

        - Update the spinner message to indicate that the responder is listening for audio.

        Notes:

        - This method is called automatically at the start of each listen cycle.
        - It updates the spinner label to show the responder name and listening duration.
        """
        listen_string = (
            f" for {self.listen_duration}s" if self.listen_duration > 0 else ""
        )
        self.spinner.start(
            f"{theme.bold}{theme.highlight}{self.name}{theme.reset} "
            f"{theme.success}is listening{listen_string}{theme.reset}",
            spinner=Spinner.EQUALIZER,
        )

    def on_user_input(self, user_input: str) -> None:
        """
        Usage:

        - Print the user input and start the spinner with verbs to indicate processing.

        Requires:

        - `user_input`:
            - Type: str
            - What: The transcribed text from the user

        Notes:

        - This method is called automatically when user input is received.
        - It prints the user input label and starts the spinner with verb animation.
        - The start time is recorded for timing purposes.
        """
        CliPrinter.label("User:", user_input)
        self._start_time = time.time()
        self.spinner.start_with_verbs(
            self.name, interval=15, spinner=Spinner.ZEN
        )

    def on_response(self, response: str) -> None:
        """
        Usage:

        - Print the final response and status information.

        Requires:

        - `response`:
            - Type: str
            - What: The response string returned by the respond method

        Notes:

        - This method is called automatically after processing is complete.
        - It stops the spinner, prints the response, and shows success/failure status with elapsed time.
        """
        elapsed = time.time() - self._start_time
        self.spinner.stop()
        if response:
            CliPrinter.print_response(self.name, response)
            CliPrinter.print_status(self.name, success=True, elapsed=elapsed)
        else:
            CliPrinter.print_status(self.name, success=False, elapsed=elapsed)

    def on_terminate(self) -> None:
        """
        Usage:

        - Print a termination message and stop the spinner.

        Notes:

        - This method is called when the responder is terminated.
        - It stops the spinner and prints an informative message about the termination.
        """
        self.spinner.stop()
        CliPrinter.info(f"{self.name} has been terminated.")

    def on_listen_end(self) -> None:
        """
        Usage:

        - Stop the spinner after listening is complete.

        Notes:

        - This method is called automatically at the end of each listen cycle.
        - It stops the spinner to indicate that listening has finished.
        """
        self.spinner.start(spinner=Spinner.BRAILLE)
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
            print(f"  {theme.error}✗  Error: {exc}{theme.reset}\n")
            return ""
        self.on_response(response)
        self.wait_for_next_wake_word()
        return response

    def ready_message(self, show_wait_for_wake: bool = True, **kwargs) -> None:
        """
        Formats and prints the ready message when the responder is initialized,
        showing the wake words and terminate words.

        Optional:

        - `show_wait_for_wake`:
            - Type: bool
            - What: Whether to include "Waiting for wake word" in the ready message
            - Default: True

        - `**kwargs`:
            - Type: dict
            - What: Additional keyword arguments to display in the ready message

        """
        CliPrinter.header(self.name)
        CliPrinter.kwarg_inputs(**kwargs)
        CliPrinter.empty_line()
        if show_wait_for_wake:
            self.wait_for_next_wake_word()
