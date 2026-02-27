from faster_whisper import WhisperModel
from spych.utils import Notify, record, get_clean_audio_buffer
from typing import Optional, Union


class Spych(Notify):
    def __init__(
        self,
        whisper_model: str = "base",
        whisper_device: str = "cpu",
        whisper_compute_type: str = "int8",
    ) -> None:
        """
        Usage:

        - Initializes a transcription object using faster-whisper for fully offline
          speech-to-text

        Optional:

        - `whisper_model`:
            - Type: str
            - What: The faster-whisper model name to use for transcription
            - Default: "base"
            - Note: Larger models (small, medium, large) provide better accuracy at
              the cost of speed; smaller models (tiny, base) are faster but less accurate

        - `whisper_device`:
            - Type: str
            - What: The device to run the whisper model on
            - Default: "cpu"
            - Note: Use "cuda" for GPU acceleration if available

        - `whisper_compute_type`:
            - Type: str
            - What: The compute type to use for the whisper model
            - Default: "int8"
            - Note: "int8" offers a good balance of speed and accuracy on both CPU and GPU
        """
        self.wake_model = WhisperModel(
            whisper_model,
            device=whisper_device,
            compute_type=whisper_compute_type,
        )

    def listen(self, duration: Union[int, float] = 5, device_index: int = -1) -> str:
        """
        Usage:

        - Records audio from the microphone for a specified duration and returns
          the transcription as a string

        Optional:

        - `duration`:
            - Type: int | float
            - What: The number of seconds to record
            - Default: 5

        - `device_index`:
            - Type: int
            - What: The microphone device index to record from
            - Default: -1
            - Note: Use `-1` to select the system default input device

        Returns:

        - `transcription`:
            - Type: str
            - What: The transcribed text from the recorded audio
            - Note: Multiple segments are joined with a single space
        """
        buffer = record(device_index=device_index, duration=duration)
        audio_buffer = get_clean_audio_buffer(buffer)
        segments, _ = self.wake_model.transcribe(audio_buffer, beam_size=2)
        return " ".join([segment.text for segment in segments])
