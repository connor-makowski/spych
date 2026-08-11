import pytest
from spych import Spych
from spych.speaker import Speaker
from spych.responders import BaseResponder, AgentResponse


class EchoResponder(BaseResponder):
    def respond(self, user_input: str) -> AgentResponse:
        return AgentResponse(
            response=f"Echo: {user_input}",
            summary=f"Heard: {user_input}",
            requires_user_feedback=False,
        )


# Instantiating a real backend loads (and on a cold cache downloads) model
# weights, which blows well past the 30s global pytest timeout.
@pytest.mark.timeout(600)
def test_speaker_and_echo_responder():
    try:
        speaker = Speaker()
        assert speaker is not None
    except (ImportError, NotImplementedError):
        pass

    try:
        kokoro = Speaker(voice="af_heart", backend="kokoro")
    except (ImportError, NotImplementedError):
        pass

    try:
        chatterbox = Speaker(voice="af_heart", backend="chatterbox")
    except (ImportError, NotImplementedError):
        pass

    spych_object = Spych(whisper_model="tiny.en")
    responder = EchoResponder(
        spych_object=spych_object,
        use_speaker=False,
        speaker_voice="am_michael",
    )

    # Verify parse_output handles valid JSON
    json_text = '{"response": "The sky is blue.", "summary": "Sky is blue.", "requires_user_feedback": false}'
    parsed = responder.parse_output(json_text)
    assert parsed.response == "The sky is blue."
    assert parsed.summary == "Sky is blue."
    assert parsed.requires_user_feedback is False

    # Verify parse_output falls back gracefully on invalid JSON
    fallback = responder.parse_output("not json at all")
    assert fallback.response == "not json at all"
    assert fallback.requires_user_feedback is False

    # Verify EchoResponder returns correct AgentResponse fields
    result = responder.respond("hello world")
    assert result.response == "Echo: hello world"
    assert result.summary == "Heard: hello world"
    assert result.requires_user_feedback is False
