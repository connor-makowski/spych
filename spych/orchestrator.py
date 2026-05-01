from __future__ import annotations

from typing import Optional, TypedDict

from spych.cli_tools import theme, CliPrinter, CliSpinner, NullSpinner
from spych.wake import SpychWake
from spych.spinners import Spinner

from spych.dashboard import AgentDashboard


class OrchestratorEntry(TypedDict, total=False):
    responder: "BaseResponder"  # required
    wake_words: list[str]  # required
    terminate_words: list[str]  # optional — defaults to ["terminate"]


class SpychOrchestrator:
    def __init__(
        self,
        entries: list[OrchestratorEntry],
        spych_wake_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Usage:

        - Runs one or more responders under a single SpychWake and a single shared
          spinner. Each entry binds a responder to its own wake words.

        Requires:

        - `entries`:
            - Type: list[OrchestratorEntry]
            - What: Each dict registers one responder. Required keys:
                - "responder"      : BaseResponder instance
                - "wake_words"     : list[str] — words that trigger this responder
              Optional keys:
                - "terminate_words": list[str] — words that stop the entire orchestrator
                                     Default: ["terminate"]
            - Note: Wake words must be unique across all entries.
              Terminate words are merged — any terminate word from any entry stops
              the whole orchestrator.
            - Example:
                [
                    {
                        "responder": claude_responder,
                        "wake_words": ["claude", "cloud"],
                        "terminate_words": ["terminate"],
                    },
                    {
                        "responder": ollama_responder,
                        "wake_words": ["llama", "ollama"],
                    },
                ]

        Optional:

        - `spych_wake_kwargs`:
            - Type: dict | None
            - What: Extra kwargs forwarded to SpychWake (e.g. whisper_model,
              wake_listener_count). Keys present here override the defaults.
            - Default: None
        """
        if not entries:
            raise ValueError("SpychOrchestrator requires at least one entry.")

        self.entries = self.build_entries(entries)
        # Inherit the dashboard from the first responder if one was injected.
        self._dashboard: Optional[AgentDashboard] = getattr(
            self.entries[0]["responder"], "dashboard", None
        ) if self.entries else None
        self.spinner = self.build_spinner()
        self.wake_word_map, self.terminate_words = self.build_wake_word_map()
        self.spych_wake = self.build_spych_wake(spych_wake_kwargs)

    # ------------------------------------------------------------------ #
    #  Initialisation helpers                                              #
    # ------------------------------------------------------------------ #

    def build_entries(
        self, raw_entries: list[OrchestratorEntry]
    ) -> list[OrchestratorEntry]:
        """
        Usage:

        - Validates and normalises the raw entries list passed to __init__,
          ensuring required keys are present and filling in defaults.

        Requires:

        - `raw_entries`:
            - Type: list[OrchestratorEntry]
            - What: The unvalidated entries passed by the caller

        Returns:

        - `entries`:
            - Type: list[OrchestratorEntry]
            - What: Validated and normalised entries with defaults applied
        """
        entries: list[OrchestratorEntry] = []
        for raw in raw_entries:
            if "responder" not in raw:
                raise ValueError("Each entry must contain a 'responder' key.")
            if "wake_words" not in raw or not raw["wake_words"]:
                raise ValueError(
                    "Each entry must contain a non-empty 'wake_words' key."
                )
            entries.append(
                {
                    "responder": raw["responder"],
                    "wake_words": list(raw["wake_words"]),
                    "terminate_words": list(
                        raw.get("terminate_words", ["terminate"])
                    ),
                }
            )
        return entries

    def build_spinner(self) -> CliSpinner:
        """
        Usage:

        - Creates a single shared CliSpinner and injects it into every registered
          responder so all UI output goes through one instance and lines never
          interleave.

        Returns:

        - `spinner`:
            - Type: CliSpinner
            - What: The shared spinner instance assigned to all responders
        """
        if self._dashboard is not None:
            return NullSpinner()
        spinner = CliSpinner()
        for entry in self.entries:
            entry["responder"].spinner = spinner
        return spinner

    def build_wake_word_map(
        self,
    ) -> tuple[dict[str, "BaseResponder"], list[str]]:
        """
        Usage:

        - Iterates over all entries to build a flat wake-word-to-responder map
          and a deduplicated list of terminate words. Each responder's __call__
          is wrapped so the most recently triggered responder is always tracked.

        - Raises if any wake word is registered by more than one responder.

        Returns:

        - `(wake_word_map, terminate_words)`:
            - Type: tuple[dict[str, BaseResponder], list[str]]
            - What: The resolved wake word map and merged terminate word list
        """
        terminate_words: list[str] = []
        wake_word_map: dict[str, "BaseResponder"] = {}

        for entry in self.entries:
            for word in entry["wake_words"]:
                w = word.lower()
                if w in wake_word_map:
                    raise ValueError(
                        f"Wake word '{w}' is registered by more than one responder."
                    )
                wake_word_map[w] = entry["responder"]
            for word in entry["terminate_words"]:
                w = word.lower()
                if w not in terminate_words:
                    terminate_words.append(w)

        def make_tracked_call(resp: "BaseResponder", entry_wake_words: list[str]):
            original = resp.__call__

            def tracked():
                self.last_responder = resp
                if self._dashboard is not None:
                    self._dashboard.set_agent(
                        name=resp.name,
                        wake_words=entry_wake_words,
                        response_style=getattr(resp, "response_style", ""),
                        use_speaker=getattr(resp, "use_speaker", False),
                        speaker_voice=getattr(resp, "speaker_voice", ""),
                        user_name=getattr(resp, "user_name", "User"),
                    )
                original()

            return tracked

        # Build a per-entry wake-word map to preserve each entry's wake_words for set_agent.
        entry_wake_words_by_resp: dict[str, list[str]] = {}
        for entry in self.entries:
            for word in entry["wake_words"]:
                entry_wake_words_by_resp[word.lower()] = entry["wake_words"]

        tracked_wake_map: dict[str, callable] = {
            word: make_tracked_call(resp, entry_wake_words_by_resp.get(word, []))
            for word, resp in wake_word_map.items()
        }

        self.last_responder: "BaseResponder" = self.entries[0]["responder"]
        return tracked_wake_map, terminate_words

    def build_spych_wake(self, spych_wake_kwargs: Optional[dict]) -> SpychWake:
        """
        Usage:

        - Constructs the SpychWake instance using the merged wake word map and
          terminate word list built by build_wake_word_map().

        Requires:

        - `spych_wake_kwargs`:
            - Type: dict | None
            - What: Extra kwargs forwarded to SpychWake. whisper_model defaults
              to "base.en" if not provided.

        Returns:

        - `spych_wake`:
            - Type: SpychWake
            - What: The configured SpychWake instance ready to be started
        """
        kwargs = spych_wake_kwargs or {}
        kwargs.setdefault("whisper_model", "base.en")
        return SpychWake(
            wake_word_map=self.wake_word_map,
            terminate_words=self.terminate_words,
            on_terminate=self.on_terminate,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    #  Lifecycle hooks                                                     #
    # ------------------------------------------------------------------ #

    def on_terminate(self) -> None:
        """
        Usage:

        - Called by SpychWake when any terminate word is detected. Calls
          on_terminate() on every registered responder.
        """
        for entry in self.entries:
            try:
                entry["responder"].on_terminate()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """
        Usage:

        - Prints the ready header for every registered responder, then starts
          the SpychWake voice listener. Blocks until a terminate word is spoken
          or ctrl+c is pressed.
        """
        for entry in self.entries:
            if not entry["responder"].healthcheck():
                CliPrinter.info(
                    f"{entry['responder'].name} healthcheck failed.",
                    color=theme.error,
                )
                return
            entry["responder"].ready_message(
                show_wait_for_wake=False,
                wake_words=entry["wake_words"],
                terminate_words=entry["terminate_words"],
                responder=entry["responder"].__class__.__name__,
            )

        if self._dashboard is not None:
            self._dashboard.start()
        else:
            CliPrinter.divider("─", 60, theme.accent)
        self.spinner.start("Waiting for wake word", spinner=Spinner.BRAILLE)

        try:
            self.spych_wake.start()
        except KeyboardInterrupt:
            self.on_terminate()
