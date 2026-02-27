import subprocess, requests, json
from spych.utils import Notify


class BaseResponder(Notify):
    def __init__(self, spych_object, listen_duration=5):
        """
        Usage:

        - Base class for all responders. Handles the listen-transcribe-respond cycle
          and provides a consistent interface for subclasses to implement

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

        Notes:

        - Subclasses must implement the `respond` method
        - The `__call__` method orchestrates the full listen -> transcribe -> respond cycle
          and handles printing; subclasses only need to handle the response logic
        """
        self.spych_object = spych_object
        self.listen_duration = listen_duration
        self.name = self.__class__.__name__

    def respond(self, user_input):
        """
        Usage:

        - Called with the transcribed user input and expected to return a response string
        - Must be implemented by all subclasses

        Requires:

        - `user_input`:
            - Type: str
            - What: The transcribed text from the user's audio input

        Returns:

        - `response`:
            - Type: str
            - What: The response string to print and return to the caller
        """
        raise NotImplementedError(
            "Subclasses must implement the `respond` method"
        )

    def on_listen_start(self):
        """
        Usage:

        - Returns a string message to print when starting to listen for user input
        - Can be overridden by subclasses for custom messages
        - prints a default message indicating how long it will listen for instructions

        Returns:

        - None (default behavior does nothing)
        """
        print(f"Waiting {self.listen_duration}s for instructions...")
        return None

    def on_user_input(self, user_input):
        """
        Usage:

        - Hook method that is called with the transcribed user input before generating a response
        - Can be overridden by subclasses to perform actions based on the user input
        - prints the user input by default for visibility, but can be customized or overridden as needed

        Requires:

        - `user_input`:
            - Type: str
            - What: The transcribed user input

        Returns:

        - `None` (default behavior does nothing)
        """
        print("You: ", user_input)

    def on_response(self, response):
        """
        Usage:

        - Hook method that is called with the generated response before printing
        - Can be overridden by subclasses to perform actions based on the response
        - prints the response by default for visibility, but can be customized or overridden as needed

        Requires:

        - `response`:
            - Type: str
            - What: The generated response string

        Returns:
        - `None` (default behavior does nothing)
        """
        print(f"{self.name}: ", response)

    def on_listen_end(self):
        """
        Usage:

        - Returns a string message to print when finished listening for user input
        - Can be overridden by subclasses for custom messages

        Returns:

        - `message`:
            - Type: str
            - What: The message to print when finished listening
        """
        return print("=" * 20)

    def __call__(self):
        """
        Usage:

        - Executes one full wake-triggered cycle: listens, transcribes, responds, and prints

        Returns:

        - `response`:
            - Type: str
            - What: The response string returned by `respond`
        """
        self.on_listen_start()
        user_input = self.spych_object.listen(duration=self.listen_duration)
        self.on_user_input(user_input)
        response = self.respond(user_input)
        self.on_response(response)
        self.on_listen_end()
        return response


class OllamaResponder(BaseResponder):
    def __init__(
        self,
        spych_object,
        model="llama3.2:latest",
        history_length=10,
        host="http://localhost:11434",
        listen_duration=5,
    ):
        """
        Usage:

        - A responder that sends transcribed audio to a locally running Ollama instance
          and returns the model's response

        Requires:

        - `spych_object`:
            - Type: Spych
            - What: An initialized Spych instance used to record and transcribe audio

        Optional:

        - `model`:
            - Type: str
            - What: The Ollama model name to use for generating responses
            - Default: "llama3.2:latest"
            - Note: Run `ollama list` in your terminal to see available models

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
        """
        super().__init__(
            spych_object=spych_object, listen_duration=listen_duration
        )
        self.name = "Ollama"
        self.model = model
        self.history_length = history_length
        self.host = host
        self.history = []

    def respond(self, user_input):
        """
        Usage:

        - Sends the transcribed user input to Ollama and returns the model's response
        - Maintains a rolling conversation history across calls

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
                [
                    f"{e['role'].capitalize()}: {e['content']}"
                    for e in self.history
                ]
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


class LocalClaudeCodeCLIResponder(BaseResponder):
    def __init__(
        self, spych_object, continue_conversation=True, listen_duration=5
    ):
        """
        Usage:

        - A responder that pipes transcribed audio into the Claude Code CLI (`claude -p`)
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

        Notes:

        - Uses `--output-format json` so the subprocess blocks until Claude has finished
          all tool calls and returns only the final result, not intermediate tool call XML
        - Claude Code must be installed and authenticated before use
        """
        super().__init__(
            spych_object=spych_object, listen_duration=listen_duration
        )
        self.name = "Claude Code"
        self.continue_conversation = continue_conversation
        self.first_call = True

    def respond(self, user_input):
        """
        Usage:

        - Pipes the transcribed user input into `claude -p` and returns the final response
          after all tool calls have completed

        Requires:

        - `user_input`:
            - Type: str
            - What: The transcribed text from the user's audio input

        Returns:

        - `response`:
            - Type: str
            - What: The final response string from Claude Code
        """
        cmd = ["claude", "-p", user_input, "--output-format", "json"]
        if self.continue_conversation and not self.first_call:
            cmd.append("--continue")
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.first_call = False
        try:
            return json.loads(result.stdout).get("result", "").strip()
        except json.JSONDecodeError:
            return result.stdout.strip()
