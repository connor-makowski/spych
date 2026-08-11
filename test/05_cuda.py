import pytest
import torch
from spych import Spych, SpychWake


def test_cuda_init_and_config():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available in this environment")

    spych_object = Spych(
        whisper_model="tiny.en",
        whisper_device="cuda",
        whisper_compute_type="float16",
    )
    assert spych_object.whisper_device == "cuda"

    wake_object = SpychWake(
        wake_word_map={"speech": lambda: None},
        whisper_model="tiny.en",
        whisper_device="cuda",
        terminate_words=["terminate"],
    )
    assert wake_object.whisper_device == "cuda"
