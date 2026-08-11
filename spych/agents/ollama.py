from spych.core import Spych
from spych.orchestrator import SpychOrchestrator
from spych.responders import BaseResponder, AgentResponse
from spych.cli_tools import CliPrinter, theme
from typing import Optional
import requests


class OllamaResponder(BaseResponder):
    def __init__(
        self,
        spych_object: "Spych",
        model: str,
        history_length: int = 10,
        host: str = "http://localhost:11434",
        listen_duration: int | float | str = 0,
        name: Optional[str] = None,
        use_speaker: bool = True,
        speaker_voice: str = "af_heart",
        response_style: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Usage:

        - A responder that sends transcribed audio to a locally running Ollama instance
          and returns the model's response.

        Requires:

        - `spych_object`:
            - Type: Spych
            - What: An initialized Spych instance used to record and transcribe audio

        - `model`:
            - Type: str
            - What: The Ollama model name to use for generating responses
            - Example: "llama3.2:latest"
            - Note: Run `ollama list` in your terminal to see available models

        Optional:

        - `history_length`:
            - Type: int
            - What: The number of previous interactions to include in each request for
              conversational context
            - Default: 10
            - Note: Each interaction counts as one user message and one assistant message;
              the actual history buffer is `history_length * 2` entries

        - `host`:
            - Type: str
            - What: The base URL of the running Ollama instance
            - Default: "http://localhost:11434"

        - `listen_duration`:
            - Type: int | float | str
            - What: How long to listen for after the wake word is detected
            - Default: 0 (Auto detect)
            - Options:
                - int | float : Record for exactly this many seconds
                - "auto" or 0 : Use Silero VAD to detect a complete utterance and
                                stop automatically when the speaker finishes

        - `name`:
            - Type: str
            - What: A custom name for the responder to use in printed messages
            - Default: "Ollama"

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

        - `response_style`:
            - Type: str | None
            - What: Style preset or custom instruction shaping how the LLM formats its
              summary. Named presets: concise, friendly, military, five_year_old, fast,
              pirate, news_anchor, haiku, shakespearean, robot, caveman, yoda, jarvis.
              Any other string is used verbatim as a custom instruction.
            - Default: None
        """
        name = name or "Ollama"
        super().__init__(
            spych_object=spych_object,
            listen_duration=listen_duration,
            name=name,
            use_speaker=use_speaker,
            speaker_voice=speaker_voice,
            response_style=response_style,
            **kwargs,
        )
        self.model = model
        self.history_length = history_length
        self.host = host
        self.history = []

    def healthcheck(self) -> bool:
        """
        Usage:

        - Checks if the Ollama instance is reachable and responding to requests.

        Returns:

        - `is_healthy`:
            - Type: bool
            - What: True if the Ollama instance responded successfully, False otherwise
        """
        try:
            # List Ollama models that have been pulled to check if the host is responsive
            response = requests.get(f"{self.host}/api/tags", timeout=3)
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            if self.model not in model_names:
                CliPrinter.info(
                    f"Ollama is reachable at {self.host}, but the specified model '{self.model}' was not found.",
                    color=theme.error,
                )
                CliPrinter.info(
                    f"Available models: {', '.join(model_names)}",
                    color=theme.error,
                )
                CliPrinter.info(
                    f"Run `ollama pull {self.model}` in your terminal to download the model and try again.",
                    color=theme.error,
                )
                return False
            return True
        except requests.RequestException:
            # CliPrinter.info(f"{entry["responder"].name} healthcheck failed.", color=theme.error)
            CliPrinter.info(
                f"Failed to connect to Ollama at {self.host}. Check if Ollama is running and the host URL is correct.",
                color=theme.error,
            )
            return False

    def respond(
        self, user_input: str, is_continuation: bool = False
    ) -> AgentResponse:
        """
        Usage:

        - Sends the transcribed user input to Ollama and returns a structured
          response. Maintains a rolling conversation history across calls.

        Requires:

        - `user_input`:
            - Type: str
            - What: The transcribed text from the user's audio input

        Returns:

        - `response`:
            - Type: AgentResponse
            - What: Parsed structured response from the Ollama model
        """
        prompt = user_input
        if is_continuation:
            prompt = "Please continue."
        else:
            self.history.append({"role": "user", "content": user_input})

        prompt_history = "\n".join(
            f"{e['role'].capitalize()}: {e['content']}" for e in self.history
        )
        prompt = f"""
        Your conversation history:

        {prompt_history}

        Now here is your current prompt:

        {self.format_prompt(prompt)}
        """

        output = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
            },
        )

        agent_response = self.parse_output(output.json().get("response", ""))
        self.history.append(
            {"role": "assistant", "content": agent_response.response}
        )
        self.history = self.history[-self.history_length * 2 :]
        return agent_response


def ollama(
    model: str,
    wake_words: list[str] = ["llama", "ollama", "lama"],
    terminate_words: list[str] = ["terminate"],
    listen_duration: int | float | str = 0,
    history_length: int = 10,
    host: str = "http://localhost:11434",
    name: Optional[str] = None,
    use_speaker: bool = True,
    speaker_voice: str = "af_heart",
    response_style: Optional[str] = None,
    spych_kwargs: dict[str, any] | None = None,
    spych_wake_kwargs: dict[str, any] | None = None,
    start: bool = True,
    **kwargs,
) -> Optional[SpychOrchestrator]:
    """
    Usage:

    - Starts a wake word listener that pipes detected speech into a locally running
      Ollama instance

    Requires:

    - `model`:
        - Type: str
        - What: The Ollama model name to use for generating responses
        - Example: "llama3.2:latest"
        - Note: Run `ollama list` in your terminal to see available models

    Optional:

    - `wake_words`:
        - Type: list[str]
        - What: A list of wake words that each trigger the Ollama responder
        - Default: ["llama", "ollama", "lama"]
        - Note: All wake words in this list map to the same OllamaResponder instance,
          sharing conversation history across triggers

    - `terminate_words`:
        - Type: list[str]
        - What: A list of terminate words that each trigger the termination of the Ollama responder
        - Default: ["terminate"]
        - Note: All terminate words in this list map to the same OllamaResponder instance,
            sharing conversation history across triggers

    - `listen_duration`:
        - Type: int | float
        - What: The number of seconds to listen for after the wake word is detected
        - Default: 0 (Auto detect)

    - `history_length`:
        - Type: int
        - What: The number of previous interactions to include in each request for conversational context
        - Default: 10
        - Note: Each interaction counts as one user message and one assistant message; the actual history
            buffer is `history_length * 2` entries

    - `host`:
        - Type: str
        - What: The base URL of the running Ollama instance
        - Default: "http://localhost:11434"

    - `name`:
        - Type: str
        - What: A custom display name for the responder shown in printed messages
        - Default: None (uses "Ollama")

    - `use_speaker`:
        - Type: bool
        - What: Whether to speak responses aloud via kokoro TTS
        - Default: False

    - `speaker_voice`:
        - Type: str
        - What: Kokoro voice ID for spoken responses
        - Default: "af_heart"

    - `response_style`:
        - Type: str | None
        - What: Style preset for reformatting spoken output (e.g. "military", "fast")
        - Default: None

    - `spych_kwargs`:
        - Type: dict
        - What: Additional keyword arguments to pass to the Spych constructor
        - Default: None

    - `spych_wake_kwargs`:
        - Type: str
        - What: Additional keyword arguments to pass to SpychWake via SpychOrchestrator
        - Default: None
    """
    spych_kwargs = {"whisper_model": "base.en", **(spych_kwargs or {})}
    spych_object = Spych(**spych_kwargs)

    responder = OllamaResponder(
        spych_object=spych_object,
        model=model,
        listen_duration=listen_duration,
        history_length=history_length,
        host=host,
        name=name,
        use_speaker=use_speaker,
        speaker_voice=speaker_voice,
        response_style=response_style,
        **kwargs,
    )

    orchestrator = SpychOrchestrator(
        entries=[
            {
                "responder": responder,
                "wake_words": wake_words,
                "terminate_words": terminate_words,
            }
        ],
        spych_wake_kwargs=spych_wake_kwargs,
    )
    if start:
        orchestrator.start()
    return orchestrator
