from spych import Speaker, SPEAKER_STYLES, Spych
from spych.responders import BaseResponder

# -- Basic TTS -----------------------------------------------------------

speaker = Speaker()
speaker.speak("Hello! I am your default voice.")

michael = Speaker(voice="am_michael")
michael.speak("And I am your American male voice.")

# emma = Speaker(voice="bf_emma")
# emma.speak("And I am your British female voice.")

# george = Speaker(voice="bm_george")
# george.speak("And I am your British male voice.")

# -- Available styles ----------------------------------------------------

# print("Available speaker styles:")
# for style, prompt in SPEAKER_STYLES.items():
#     print(f"  {style}: {prompt[:60]}...")

# -- summarize_for_speech override demo ----------------------------------
# Demonstrates that the hook is overridable without an LLM.


class EchoResponder(BaseResponder):
    def respond(self, user_input: str) -> str:
        return f"Echo: {user_input}"

    def summarize_for_speech(self, user_input: str, response: str) -> str:
        return f"Heard: {user_input}"


spych_object = Spych(whisper_model="base.en")
responder = EchoResponder(
    spych_object=spych_object,
    use_speaker=True,
    speaker_voice="am_michael",
)

# Manually invoke summarize_for_speech to verify the override
speech_text = responder.summarize_for_speech("hello world", "Echo: hello world")
print(f"summarize_for_speech returned: {speech_text!r}")
assert speech_text == "Heard: hello world", f"Unexpected: {speech_text!r}"

michael.speak(speech_text)

print("test/12_speaker.py: all assertions passed")
