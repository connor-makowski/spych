from spych.core import Spych
from spych.wake import SpychWake
from spych.responders import BaseResponder
import requests


class OllamaResponder(BaseResponder):
    def __init__(
        self,
        spych_object: "Spych",
        model: str,
        history_length: int = 10,
        host: str = "http://localhost:11434",
        listen_duration: int | float = 5,
        name: str | None = "Ollama",
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
            - Type: int | float
            - What: The number of seconds to listen for after the wake word is detected
            - Default: 5

        - `name`:
            - Type: str
            - What: A custom name for the responder to use in printed messages
            - Default: "Ollama"
        """
        super().__init__(
            spych_object=spych_object,
            listen_duration=listen_duration,
            name=name,
        )
        self.model = model
        self.history_length = history_length
        self.host = host
        self.history = []

    def respond(self, user_input: str) -> str:
        """
        Usage:

        - Sends the transcribed user input to Ollama and returns the model's response.
        - Maintains a rolling conversation history across calls.

        Requires:

        - `user_input`:
            - Type: str
            - What: The transcribed text from the user's audio input

        Returns:

        - `response`:
            - Type: str
            - What: The response string from the Ollama model
        """
        self.history.append({"role": "user", "content": user_input})
        prompt = (
            "\n".join(
                f"{e['role'].capitalize()}: {e['content']}"
                for e in self.history
            )
            + "\nAssistant:"
        )

        output = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
        )

        response = output.json().get("response", "").strip()
        self.history.append({"role": "assistant", "content": response})
        self.history = self.history[-self.history_length * 2 :]
        return response


def ollama(
    model: str,
    wake_words: list[str] = ["llama", "ollama", "lama"],
    terminate_words: list[str] = ["terminate"],
    listen_duration: int | float = 5,
    history_length: int = 10,
    host: str = "http://localhost:11434",
    spych_kwargs: dict[str, any] | None = None,
    spych_wake_kwargs: dict[str, any] | None = None,
) -> None:
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
        - Default: 5

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

    - `spych_kwargs`:
        - Type: dict
        - What: Additional keyword arguments to pass to the Spych constructor
        - Default: None

    - `spych_wake_kwargs`:
        - Type: str
        - What: Additional keyword arguments to pass to the SpychWake constructor
        - Default: None
    """
    # Spych Object
    spych_kwargs = {"whisper_model": "base.en", **(spych_kwargs or {})}
    spych_object = Spych(**spych_kwargs)

    # Responder Object
    responder = OllamaResponder(
        spych_object=spych_object,
        model=model,
        listen_duration=listen_duration,
        history_length=history_length,
        host=host,
    )

    # SpychWake Object
    spych_wake_kwargs = {
        "whisper_model": "base.en",
        "on_terminate": responder.on_terminate,
        "wake_word_map": {word: responder for word in wake_words},
        "terminate_words": terminate_words,
        **(spych_wake_kwargs or {}),
    }
    spych_wake_object = SpychWake(**spych_wake_kwargs)

    # Fire ready message and start wake listener
    responder.ready_message(
        wake_words=wake_words, terminate_words=terminate_words
    )
    spych_wake_object.start()
