from spych import SpychWake, Spych
from spych.responders import LocalClaudeCodeCLIResponder, OllamaResponder


def get_spych_object(whisper_device):
    """
    Usage:

    - Initializes and returns a Spych transcription object

    Requires:

    - `whisper_device`:
        - Type: str
        - What: The device to run the whisper model on
        - Note: Use "cuda" for GPU acceleration if available, otherwise "cpu"

    Returns:

    - `spych_object`:
        - Type: Spych
        - What: An initialized Spych instance using the base.en model
    """
    return Spych(
        whisper_model="base.en",
        whisper_device=whisper_device,
        whisper_compute_type="int8",
    )


def get_wake_object(wake_word_map, whisper_device, terminate_words=None):
    """
    Usage:

    - Initializes and returns a SpychWake detection object

    Requires:

    - `wake_word_map`:
        - Type: dict[str, callable]
        - What: A dictionary mapping wake words to their corresponding callback functions

    - `whisper_device`:
        - Type: str
        - What: The device to run the whisper model on
        - Note: Use "cuda" for GPU acceleration if available, otherwise "cpu"

    Optional:

    - `terminate_words`:
        - Type: list[str]
        - What: A list of words that, if detected in the wake listener's transcription,
          will immediately terminate the entire SpychWake system
        - Note: Use with caution, as any false positive on a terminate word will stop
          the wake system until it is manually restarted
        - default: None (disabled)

    Returns:

    - `wake_object`:
        - Type: SpychWake
        - What: An initialized SpychWake instance using the tiny.en model
    """
    return SpychWake(
        wake_word_map=wake_word_map,
        terminate_words=terminate_words,
        whisper_model="tiny.en",
        whisper_device=whisper_device,
        whisper_compute_type="int8",
    )


def claude_code_cli(whisper_device="cpu", wake_words=["claude"], terminate_words=["terminate"], listen_duration=5, continue_conversation=True):
    """
    Usage:

    - Starts a wake word listener that pipes detected speech into the Claude Code CLI

    Optional:

    - `whisper_device`:
        - Type: str
        - What: The device to run the whisper models on
        - Default: "cpu"
        - Note: Use "cuda" for GPU acceleration if available

    - `wake_words`:
        - Type: list[str]
        - What: A list of wake words that each trigger the Claude Code CLI responder
        - Default: ["claude"]
        - Note: All wake words in this list map to the same LocalClaudeCodeCLIResponder
          instance, sharing conversation history across triggers

    - `terminate_words`:
        - Type: list[str]
        - What: A list of terminate words that each trigger the termination of the Claude Code CLI responder
        - Default: ["terminate"]
        - Note: All terminate words in this list map to the same LocalClaudeCodeCLIResponder
          instance, sharing conversation history across triggers
        
    - `listen_duration`:
        - Type: int | float
        - What: The number of seconds to listen for after the wake word is detected
        - Default: 5

    - `continue_conversation`:
        - Type: bool
        - What: Whether to pass `--continue` to reuse the most recent session in Claude CLI
        - Default: True
    """
    spych_object = get_spych_object(whisper_device=whisper_device)
    responder = LocalClaudeCodeCLIResponder(spych_object, continue_conversation=continue_conversation, listen_duration=listen_duration)
    wake_word_map = {word: responder for word in wake_words}
    wake_object = get_wake_object(
        wake_word_map=wake_word_map,
        whisper_device=whisper_device,
        terminate_words=terminate_words,
    )
    wake_object.start()


def ollama(model="llama3.2:latest", whisper_device="cpu", wake_words=["llama", "ollama"], terminate_words=["terminate"], listen_duration=5, history_length=10, host="http://localhost:11434"):
    """
    Usage:

    - Starts a wake word listener that pipes detected speech into a locally running
      Ollama instance

    Optional:

    - `model`:
        - Type: str
        - What: The Ollama model name to use for generating responses
        - Default: "llama3.2:latest"
        - Note: Run `ollama list` in your terminal to see available models

    - `whisper_device`:
        - Type: str
        - What: The device to run the whisper models on
        - Default: "cpu"
        - Note: Use "cuda" for GPU acceleration if available

    - `wake_words`:
        - Type: list[str]
        - What: A list of wake words that each trigger the Ollama responder
        - Default: ["speech"]
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
    """
    spych_object = get_spych_object(whisper_device=whisper_device)
    responder = OllamaResponder(spych_object=spych_object, model=model, listen_duration=listen_duration, history_length=history_length, host=host)
    wake_object = get_wake_object(
        wake_word_map={word: responder for word in wake_words},
        whisper_device=whisper_device,
        terminate_words=terminate_words,
    )
    wake_object.start()