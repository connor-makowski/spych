from spych.agents.claude import LocalClaudeCodeCLIResponder, claude_code_cli
from spych.core import Spych


def test_claude_cli_responder():
    spych_obj = Spych(whisper_model="tiny.en")
    responder = LocalClaudeCodeCLIResponder(
        spych_object=spych_obj,
        continue_conversation=True,
        show_tool_events=False,
        use_speaker=False,
    )
    assert responder.name == "Claude"
    assert responder.continue_conversation is True

    formatted = responder.format_prompt("Fix the bug")
    assert "Fix the bug" in formatted

    orchestrator = claude_code_cli(
        wake_words=["hey claude"],
        use_speaker=False,
        spych_kwargs={"whisper_model": "tiny.en"},
        start=False,
    )
    assert orchestrator is not None
    assert len(orchestrator.entries) == 1
    assert "hey claude" in orchestrator.entries[0]["wake_words"]
