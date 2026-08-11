from spych import Spych, SpychWake


def test_basic_wake_and_spych_init():
    spych_object = Spych(whisper_model="tiny.en")
    assert spych_object.wake_model is not None

    heard = []

    def on_wake():
        heard.append("triggered")

    wake_object = SpychWake(
        wake_word_map={"speech": on_wake, "hey computer": on_wake},
        whisper_model="tiny.en",
        terminate_words=["terminate"],
        wake_listener_count=2,
    )
    wake_object.verbose = True

    keys = wake_object.wake_word_map.keys()
    assert "speech" in keys
    assert "hey computer" in keys
    assert "terminate" in keys
    assert wake_object.kill is False

    # Test waking callback execution directly
    wake_object.wake("speech")
    assert len(heard) == 1
    assert heard[0] == "triggered"
