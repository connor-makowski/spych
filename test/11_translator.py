from spych import Spych, SpychOrchestrator
from spych.agents import OllamaResponder


class Spanish(OllamaResponder):
    def respond(self, user_input: str) -> str:
        user_input = f"Translate the following text to Spanish and return only the translated text: '{user_input}'"
        response = super().respond(user_input)
        return response


class German(OllamaResponder):
    def respond(self, user_input: str) -> str:
        user_input = f"Translate the following text to German and return only the translated text: '{user_input}'"
        response = super().respond(user_input)
        return response


SpychOrchestrator(
    entries=[
        {
            "responder": Spanish(
                spych_object=Spych(whisper_model="base.en"),
                listen_duration=5,
                name="SpanishTranslator",
                model="llama3.2:latest",
            ),
            "wake_words": ["spanish"],
            "terminate_words": ["terminate"],
        },
        {
            "responder": German(
                spych_object=Spych(whisper_model="base.en"),
                listen_duration=5,
                name="GermanTranslator",
                model="llama3.2:latest",
            ),
            "wake_words": ["german"],
            "terminate_words": ["terminate"],
        },
    ]
).start()
