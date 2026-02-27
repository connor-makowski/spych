from spych import SpychWake
from spych import Spych
import requests

wake_object = SpychWake(
    wake_word="speech",
    whisper_model="tiny.en",  # any faster-whisper compatible model
    whisper_device="cuda",    # "cpu" or "cuda"
    whisper_compute_type="int8",
)

spych_object = Spych(
    whisper_model="base.en",  # any faster-whisper compatible model
    whisper_device="cuda",    # "cpu" or "cuda"
    whisper_compute_type="int8"
)

class AgentResponder:
    def __init__(self, model="llama3.2:latest", history_length=10):
        self.history = []
        self.model = model
        self.history_length = history_length

    def ask_ollama(self, prompt):
        self.history.append({"role": "user", "content": prompt})
        prompt = "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in self.history]) + "\nAssistant:"
        output = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
        )
        response = output.json().get("response", "").strip()
        self.history.append({"role": "assistant", "content": response})
        self.history = self.history[-self.history_length*2:]  # Keep only the last N interactions
        return response

    def __call__(self):
        print("Listening to you for 5 seconds...")
        user_input = spych_object.listen(duration=5)
        print(f"You: {user_input}")
        agent_output = self.ask_ollama(user_input)
        print(f"Agent: {agent_output}")
        print("=" * 20)


print("=" * 20)
print("Starting wake listener. Say 'spych' (pronounced speech) to trigger the on_wake function.")
print("=" * 20)
wake_object.start(on_wake_fn=AgentResponder())