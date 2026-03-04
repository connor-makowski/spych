from spych.wake import SpychWake
from spych.core import Spych
from spych.responders import BaseResponder
from typing import Optional, Any
import subprocess, json


class LocalGeminiCLIResponder(BaseResponder):
    def __init__(
        self,
        spych_object: "Spych",
        continue_conversation: bool = True,
        listen_duration: int | float = 5,
        name: Optional[str] = "Gemini",
    ) -> None:
        """
        Usage:

        - A responder that pipes transcribed audio into the Gemini CLI (`gemini -p`)
          and returns the final response, waiting for all tool calls to complete

        Requires:

        - `spych_object`:
            - Type: Spych
            - What: An initialized Spych instance used to record and transcribe audio

        Optional:

        - `continue_conversation`:
            - Type: bool
            - What: Whether to pass `--continue` to reuse the most recent session
            - Default: True

        - `listen_duration`:
            - Type: int | float
            - What: The number of seconds to listen for after the wake word is detected
            - Default: 5

        - `name`:
            - Type: str
            - What: A custom name for the responder to use in printed messages
            - Default: "Gemini"

        Notes:

        - Uses `--output-format json` so the subprocess blocks until Gemini has finished
          all tool calls and returns only the final result, not intermediate tool call XML
        - Gemini CLI must be installed and authenticated before use
        """
        super().__init__(
            spych_object=spych_object,
            listen_duration=listen_duration,
            name=name,
        )
        self.continue_conversation = continue_conversation
        self.first_call = True

    def respond(self, user_input: str) -> str:
        """
        Usage:

        - Pipes the transcribed user input into `gemini -p` and returns the final response
          after all tool calls have completed

        Requires:

        - `user_input`:
            - Type: str
            - What: The transcribed text from the user's audio input

        Returns:

        - `response`:
            - Type: str
            - What: The final response string from Gemini CLI
        """
        cmd = ["gemini", "-p", user_input, "--output-format", "json"]
        if self.continue_conversation and not self.first_call:
            cmd.append("--resume")
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.first_call = False
        try:
            return json.loads(result.stdout).get("response", "").strip()
        except json.JSONDecodeError:
            return result.stdout.strip()


def gemini_cli(
    wake_words: list[str] = ["gemini"],
    terminate_words: list[str] = ["terminate"],
    listen_duration: int | float = 5,
    continue_conversation: bool = True,
    spych_kwargs: Optional[dict[str, Any]] = None,
    spych_wake_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    """
    Usage:

    - Starts a wake word listener that pipes detected speech into the Gemini CLI

    Optional:

    - `wake_words`:
        - Type: list[str]
        - What: A list of wake words that each trigger the Gemini CLI responder
        - Default: ["gemini"]
        - Note: All wake words in this list map to the same LocalGeminiCLIResponder
          instance, sharing conversation history across triggers

    - `terminate_words`:
        - Type: list[str]
        - What: A list of terminate words that each trigger the termination of the Gemini CLI responder
        - Default: ["terminate"]
        - Note: All terminate words in this list map to the same LocalGeminiCLIResponder
          instance, sharing conversation history across triggers

    - `listen_duration`:
        - Type: int | float
        - What: The number of seconds to listen for after the wake word is detected
        - Default: 5

    - `continue_conversation`:
        - Type: bool
        - What: Whether to pass `--resume` to reuse the most recent session in Gemini CLI
        - Default: True

    - `spych_kwargs`:
        - Type: dict
        - What: Additional keyword arguments to pass to the Spych constructor
        - Default: None

    - `spych_wake_kwargs`:
        - Type: dict
        - What: Additional keyword arguments to pass to the SpychWake constructor
        - Default: None
    """
    # Set default spych_kwargs if not provided
    if spych_kwargs is None:
        spych_kwargs = {}

    # Set default spych_wake_kwargs if not provided
    if spych_wake_kwargs is None:
        spych_wake_kwargs = {}

    # Merge kwargs with defaults
    spych_kwargs = {"whisper_model": "base.en", **spych_kwargs}

    spych_wake_kwargs = {
        "whisper_model": "base.en",
        "terminate_words": terminate_words,
        **spych_wake_kwargs,
    }

    spych_object = Spych(**spych_kwargs)
    responder = LocalGeminiCLIResponder(
        spych_object,
        continue_conversation=continue_conversation,
        listen_duration=listen_duration,
    )
    wake_word_map = {word: responder for word in wake_words}
    wake_object = SpychWake(wake_word_map=wake_word_map, **spych_wake_kwargs)
    wake_object.start()
