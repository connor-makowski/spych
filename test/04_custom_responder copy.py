from spych import Spych, SpychOrchestrator
from spych.responders import BaseResponder


class MyResponder(BaseResponder):
    def respond(self, user_input: str) -> str:
        return f"'{self.name}' heard: {user_input}"


SpychOrchestrator(
    entries=[
        {
            "responder": MyResponder(
                spych_object=Spych(whisper_model="base.en"),
                listen_duration=5,
                name="TestResponder",
            ),
            "wake_words": ["test"],
            "terminate_words": ["terminate"],
        }
    ]
).start()
