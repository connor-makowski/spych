from spych import Spych, SpychOrchestrator
from spych.agents import OllamaResponder


class Spanish(OllamaResponder):
    def respond(self, user_input: str):
        prompt = f"Translate the following text to Spanish: '{user_input}'"
        return super().respond(prompt)


class German(OllamaResponder):
    def respond(self, user_input: str):
        prompt = f"Translate the following text to German: '{user_input}'"
        return super().respond(prompt)


def test_multi_agent_orchestrator_setup():
    spych_object = Spych(whisper_model="tiny.en")

    spanish_agent = Spanish(
        spych_object=spych_object,
        name="SpanishTranslator",
        model="llama3.2:latest",
        use_speaker=False,
    )
    german_agent = German(
        spych_object=spych_object,
        name="GermanTranslator",
        model="llama3.2:latest",
        use_speaker=False,
    )

    orchestrator = SpychOrchestrator(
        entries=[
            {
                "responder": spanish_agent,
                "wake_words": ["spanish"],
                "terminate_words": ["terminate"],
            },
            {
                "responder": german_agent,
                "wake_words": ["german"],
                "terminate_words": ["terminate"],
            },
        ]
    )

    assert len(orchestrator.entries) == 2
    assert "spanish" in orchestrator.wake_word_map
    assert "german" in orchestrator.wake_word_map
