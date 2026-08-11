from spych.agents.opencode import LocalOpenCodeCLIResponder, opencode_cli
from spych.core import Spych


def test_opencode_cli_responder():
    spych_obj = Spych(whisper_model="tiny.en")
    responder = LocalOpenCodeCLIResponder(
        spych_object=spych_obj,
        continue_conversation=True,
        show_tool_events=False,
        use_speaker=False,
    )
    assert responder.name == "OpenCode"
    assert responder.continue_conversation is True

    formatted = responder.format_prompt("Refactor code")
    assert "Refactor code" in formatted

    orchestrator = opencode_cli(
        wake_words=["hey opencode"],
        use_speaker=False,
        spych_kwargs={"whisper_model": "tiny.en"},
        start=False,
    )
    assert orchestrator is not None
    assert len(orchestrator.entries) == 1
    assert "hey opencode" in orchestrator.entries[0]["wake_words"]
