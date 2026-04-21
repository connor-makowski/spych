from spych.utils import Notify
from spych.cli_tools import CliSpinner, CliPrinter, theme
from spych.spinners import Spinner
from spych.speaker import Speaker, SPEAKER_STYLES
from typing import Optional
import threading
import time


class BaseResponder(Notify):
    def __init__(
        self,
        spych_object: "Spych",
        listen_duration: int | float = 0,
        name: Optional[str] = None,
        spinner: Optional[CliSpinner] = None,
        use_speaker: bool = False,
        speaker_voice: str = "af_heart",
        speaker_style: Optional[str] = None,
        follow_up_listen_duration: int | float = 0,
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

        - `use_speaker`:
            - Type: bool
            - What: Whether to speak responses aloud via kokoro TTS after printing them
            - Default: False

        - `speaker_voice`:
            - Type: str
            - What: A kokoro voice ID used for all spoken responses
            - Default: "af_heart"
            - Note: American English voices use prefix `am_` or `af_`; British English
              use `bm_` or `bf_`. See spych.speaker.Speaker for the full voice list.

        - `speaker_style`:
            - Type: str | None
            - What: A style preset that re-prompts the same agent to reformat the
              response before speaking. Available presets are keys of
              `spych.speaker.SPEAKER_STYLES` (e.g. "military", "five_year_old", "fast").
              When None the raw response text is spoken without reformatting.
            - Default: None

        - `follow_up_listen_duration`:
            - Type: int | float
            - What: How long to listen for a follow-up answer after the agent speaks
              a response that contains a question. Uses the same duration semantics as
              `listen_duration`: 0 uses Silero VAD to auto-detect the end of speech;
              a positive number records for exactly that many seconds.
            - Default: 0 (VAD auto-detect)
            - Note: Only used when `use_speaker` is True and the spoken summary
              contains a question mark.

        Notes:

        - Subclasses must implement the `respond` method.
        - The `__call__` method orchestrates the full voice listen -> transcribe -> respond
          cycle; use `text_input` for the typed equivalent.
        - When `use_speaker` is True, `summarize_for_speech()` is called in a background
          thread after each response so audio playback does not block the main loop.
        - When the spoken summary contains a question, `spoken_follow_up_loop()` listens for
          a user answer and re-runs the respond cycle inline, chaining follow-ups until
          no question remains or the user triggers the wake word to interrupt.
        """
        self.spych_object = spych_object
        self.listen_duration = listen_duration
        self.name = name if name else self.__class__.__name__
        # Accept an injected shared spinner or create a private one.
        self.spinner: CliSpinner = (
            spinner if spinner is not None else CliSpinner()
        )
        self._start_time: float = 0.0

        self._current_user_input: str = ""
        self.use_speaker = use_speaker
        self.speaker_voice = speaker_voice
        self.speaker_style = speaker_style
        self.follow_up_listen_duration = follow_up_listen_duration
        self._speaker = None
        self._speech_text: str = ""
        self._speech_complete: threading.Event = threading.Event()
        self._speech_complete.set()
        if use_speaker:
            self._speaker = Speaker(speaker_voice)

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
    #  Speaker integration                                                #
    # ------------------------------------------------------------------ #

    def summarize_for_speech(self, response: str) -> str:
        """
        Usage:

        - Produces the text that will be spoken aloud by the Speaker. The default
          implementation re-prompts the same agent with a style meta-prompt so
          that the spoken output is reformatted according to `speaker_style`.
          When `speaker_style` is None the response is summarized for spoken
          delivery without a specific style constraint.

        - Override in subclasses for custom speech generation behaviour, such as
          injecting a final prompt into a CLI session via a temp file instead of
          spawning a new subprocess call.

        Requires:

        - `response`:
            - Type: str
            - What: The full response string returned by `respond()`

        Returns:

        - `speech_text`:
            - Type: str
            - What: The text to pass to `Speaker.speak()`

        Notes:

        - This method is called from a background thread; avoid mutating shared
          state without a lock when overriding.
        - For `OllamaResponder` and other history-based responders, the default
          implementation calls `self.respond()` which appends the meta-prompt to
          the conversation history. Override `summarize_for_speech()` in those
          subclasses to avoid history pollution if desired.
        """
        if len(response) < 500:
            return response
        style_prompt = SPEAKER_STYLES.get(self.speaker_style, "")
        summary_command = f"""
        Summarize the content below for clear and natural delivery. 
            - If reasonable, return the original text.  
            - Do not indicate that it is a summary or that it is for voice transcription.
            - Important: Only return the summary, nothing else.

        Additional Style Instructions:
        {style_prompt}

        Begin summary:
        {response}
        """
        summary_response = self.respond(summary_command)
        if len(summary_response) > 1000:
            # self.print_info(f"{theme.warning}Warning: summarize_for_speech response is very long ({len(summary_response)} characters). Truncating to 1000 characters.{theme.reset}")
            summary_response = summary_response[:1000]
        return summary_response

    def speak_to_user(self, response: str) -> None:
        try:
            text = self.summarize_for_speech(response)
            self._speech_text = text
            if not self._speaker._interrupted.is_set():
                self._speaker.speak(text)
        except Exception:
            self._speech_text = ""
        finally:
            self._speech_complete.set()

    def spoken_follow_up_loop(self) -> None:
        """
        Usage:

        - Called at the end of each `__call__` cycle when `use_speaker` is True.
          Waits for the background speech thread to finish, then checks whether the
          spoken summary contained a question. If so, enters a follow-up listen →
          respond loop that continues until one of the following:

            1. The spoken summary has no question.
            2. The user does not respond (VAD returns empty string).
            3. The speaker is interrupted by the wake word.

        Notes:

        - The follow-up cycle re-uses `on_user_input`, `respond`, `on_response`, and
          `on_before_respond` / `on_after_respond`, so all hooks fire normally.
        - Each follow-up response is itself summarized and spoken; if that summary also
          contains a question the loop continues, enabling natural multi-turn dialogue
          without requiring the user to repeat the wake word.
        - Interrupting via the wake word (which calls `Speaker.interrupt()`) causes
          `_speech_complete` to be set and `_interrupted` to be flagged, so the loop
          exits cleanly and the new `__call__` cycle takes over.
        """
        while True:
            # Wait for the speech thread to finish (or for an interrupt to arrive).
            self._speech_complete.wait()
            if self._speaker._interrupted.is_set():
                self.wait_for_next_wake_word(divider=False)
                return
            if "?" not in self._speech_text:
                self.wait_for_next_wake_word(divider=False)
                return
            # A question was spoken — start listening for the user's answer.
            self.spinner.start(
                f"{theme.bold}{theme.highlight}{self.name}{theme.reset} "
                f"{theme.success}is listening for follow-up{theme.reset}",
                spinner=Spinner.EQUALIZER,
            )
            follow_up_input = self.spych_object.listen(
                duration=self.follow_up_listen_duration
            )
            self.on_listen_end()
            if not follow_up_input or self._speaker._interrupted.is_set():
                self.wait_for_next_wake_word(divider=False)
                return
            self.on_user_input(follow_up_input)
            try:
                self.on_before_respond(follow_up_input)
                follow_up_response = self.respond(follow_up_input)
                self.on_after_respond(follow_up_input, follow_up_response)
            except Exception as exc:
                self.spinner.stop()
                print(f"  {theme.error}✗  Error: {exc}{theme.reset}\n")
                self.wait_for_next_wake_word(divider=False)
                return
            self.on_response(follow_up_response)
            CliPrinter.divider()
            # Loop: check whether the new spoken response also contains a question.

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
        self._current_user_input = user_input
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
            if self.use_speaker and self._speaker is not None:
                self._speech_text = ""
                self._speech_complete.clear()
                self._speaker._interrupted.clear()
                threading.Thread(
                    target=self.speak_to_user,
                    args=(response,),
                    daemon=True,
                ).start()
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
        if self.use_speaker and self._speaker is not None:
            self._speaker.interrupt()
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
        if self.use_speaker and self._speaker is not None:
            CliPrinter.divider()
            self.spoken_follow_up_loop()
        else:
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
