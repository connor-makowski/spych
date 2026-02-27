from spych import SpychWake, Spych

spych_object = Spych(whisper_model="base.en")


def on_wake():
    print("Wake word heard! Listening for 5 seconds and transcribing...")
    output = spych_object.listen(duration=5)
    print(f"Transcription: {output}")
    print("")


wake_object = SpychWake(
    wake_word_map={"speech": on_wake},
    whisper_model="tiny.en",
    terminate_words=["terminate"],
)

wake_object.verbose = True  # Enable verbose notifications for testing

print(
    "Starting wake listener. Say 'Spych' (pronounced Speech) to trigger the on_wake function."
)
wake_object.start()
