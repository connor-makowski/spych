from spych.live import SpychLive

live = SpychLive(
    output_format="both",        # "txt", "srt", or "both"
    output_path="my_transcript",
    show_timestamps=True,
    stop_key="q",
    terminate_words=["stop recording"],
)
live.start()