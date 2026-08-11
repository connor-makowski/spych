from spych.agents.agy import LocalAntigravityCLIResponder, antigravity_cli
from spych.core import Spych


def test_antigravity_cli_responder():
    spych_obj = Spych(whisper_model="tiny.en")
    responder = LocalAntigravityCLIResponder(
        spych_object=spych_obj,
        continue_conversation=True,
        show_tool_events=False,
        use_speaker=False,
    )
    assert responder.name == "Antigravity"
    assert responder.continue_conversation is True

    formatted = responder.format_prompt("Explain gravity")
    assert "Explain gravity" in formatted

    orchestrator = antigravity_cli(
        wake_words=["hey antigravity"],
        use_speaker=False,
        spych_kwargs={"whisper_model": "tiny.en"},
        start=False,
    )
    assert orchestrator is not None
    assert len(orchestrator.entries) == 1
    assert "hey antigravity" in orchestrator.entries[0]["wake_words"]
