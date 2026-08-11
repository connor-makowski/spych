from spych.agents.ollama import OllamaResponder, ollama
from spych.core import Spych


def test_ollama_responder():
    spych_obj = Spych(whisper_model="tiny.en")
    responder = OllamaResponder(
        spych_object=spych_obj,
        model="llama3.2:latest",
        use_speaker=False,
    )
    assert responder.model == "llama3.2:latest"

    formatted = responder.format_prompt("Hello world")
    assert "Hello world" in formatted
    assert "requires_user_feedback" in formatted

    # Test factory instantiation
    orchestrator = ollama(
        wake_words=["hey llama"],
        model="llama3.2:latest",
        use_speaker=False,
        spych_kwargs={"whisper_model": "tiny.en"},
        start=False,
    )
    assert orchestrator is not None
    assert len(orchestrator.entries) == 1
    assert "hey llama" in orchestrator.entries[0]["wake_words"]
