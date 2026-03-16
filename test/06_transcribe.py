from spych.live import SpychLive

live = SpychLive(
    output_format="srt",  # "txt", "srt", or "both"
    output_path="06_test",
    show_timestamps=True,
    stop_key="q",
    terminate_words=["stop recording"],
)
live.start()
