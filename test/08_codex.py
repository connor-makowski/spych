from spych.agents.codex import LocalCodexCLIResponder, codex_cli
from spych.core import Spych


def test_codex_cli_responder():
    spych_obj = Spych(whisper_model="tiny.en")
    responder = LocalCodexCLIResponder(
        spych_object=spych_obj,
        continue_conversation=True,
        show_tool_events=False,
        use_speaker=False,
    )
    assert responder.name == "Codex"
    assert responder.continue_conversation is True

    formatted = responder.format_prompt("Write fibonacci")
    assert "Write fibonacci" in formatted

    orchestrator = codex_cli(
        wake_words=["hey codex"],
        use_speaker=False,
        spych_kwargs={"whisper_model": "tiny.en"},
        start=False,
    )
    assert orchestrator is not None
    assert len(orchestrator.entries) == 1
    assert "hey codex" in orchestrator.entries[0]["wake_words"]
