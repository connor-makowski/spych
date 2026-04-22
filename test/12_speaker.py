from spych import Spych
from spych.speaker import Speaker
from spych.responders import BaseResponder, AgentResponse

# -- Basic TTS -----------------------------------------------------------

speaker = Speaker()
speaker.speak("Hello! I am your default voice.")

michael = Speaker(voice="am_michael")
michael.speak("And I am your American male voice.")

# Test explicit backend selection (assuming both are installed or skip on failure)
try:
    kokoro = Speaker(voice="af_heart", backend="kokoro")
    kokoro.speak("I am explicitly using the Kokoro backend.")
except (ImportError, NotImplementedError):
    print("Skipping Kokoro explicit test (not installed)")

try:
    chatterbox = Speaker(voice="af_heart", backend="chatterbox")
    chatterbox.speak("I am explicitly using the Chatterbox backend.")
except (ImportError, NotImplementedError):
    print("Skipping Chatterbox explicit test (not installed)")

# Test fallback from an unavailable explicit backend
# (Providing an invalid name should trigger priority fallback)
fallback_speaker = Speaker(voice="af_heart", backend="invalid_backend")
fallback_speaker.speak("I fell back to an available backend because 'invalid_backend' was requested.")

# -- parse_output demo ---------------------------------------------------
# Verify JSON parsing and fallback behavior without an LLM.


class EchoResponder(BaseResponder):
    def respond(self, user_input: str) -> AgentResponse:
        return AgentResponse(
            response=f"Echo: {user_input}",
            summary=f"Heard: {user_input}",
            requires_user_feedback=False,
        )


spych_object = Spych(whisper_model="base.en")
responder = EchoResponder(
    spych_object=spych_object,
    use_speaker=True,
    speaker_voice="am_michael",
    speaker_backend="kokoro", # Test passing backend to responder
)

# Verify responder initialized its speaker with the requested backend
if responder.speaker:
    print(f"Responder speaker backend: {type(responder.speaker.backend).__name__}")

# Verify parse_output handles valid JSON
json_text = '{"response": "The sky is blue.", "summary": "Sky is blue.", "requires_user_feedback": false}'
parsed = responder.parse_output(json_text)
assert parsed.response == "The sky is blue.", f"Unexpected response: {parsed.response!r}"
assert parsed.summary == "Sky is blue.", f"Unexpected summary: {parsed.summary!r}"
assert parsed.requires_user_feedback is False, f"Unexpected feedback flag: {parsed.requires_user_feedback!r}"

# Verify parse_output falls back gracefully on invalid JSON
fallback = responder.parse_output("not json at all")
assert fallback.response == "not json at all", f"Unexpected fallback: {fallback.response!r}"
assert fallback.requires_user_feedback is False

# Verify EchoResponder returns correct AgentResponse fields
result = responder.respond("hello world")
assert result.response == "Echo: hello world", f"Unexpected: {result.response!r}"
assert result.summary == "Heard: hello world", f"Unexpected: {result.summary!r}"
assert result.requires_user_feedback is False

michael.speak(result.summary)

print("test/12_speaker.py: all assertions passed")
