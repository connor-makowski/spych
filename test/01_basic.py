from spych import SpychWake
from spych import Spych

wake_object = SpychWake(
    wake_word="speech",
    whisper_model="tiny.en",
    whisper_device="cuda",
    whisper_compute_type="int8",
)
# wake_object.verbose = True  # Enable verbose notifications for testing

spych_object = Spych(
    whisper_model="base.en", whisper_device="cuda", whisper_compute_type="int8"
)


def on_wake():
    print("Wake word heard! listening for 5 seconds and transcribing...")
    output = spych_object.listen(duration=5)
    print(f"Transcription: {output}")
    print("")


print("Starting wake listener. Say 'speech' to trigger the on_wake function.")
wake_object.start(on_wake_fn=on_wake)
