import os
from spych.live import SpychLive


def test_spych_live_init():
    output_prefix = "test_06_output"
    live = SpychLive(
        output_format="both",
        output_path=output_prefix,
        show_timestamps=True,
        stop_key="q",
        terminate_words=["stop recording"],
    )

    assert live.output_format == "both"
    assert live.output_path == output_prefix
    assert live.show_timestamps is True
    assert "stop recording" in live.terminate_words

    # Cleanup any generated test artifacts if present
    for ext in [".txt", ".srt"]:
        path = f"{output_prefix}{ext}"
        if os.path.exists(path):
            os.remove(path)
