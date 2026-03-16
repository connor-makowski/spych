import traceback, sys
from pvrecorder import PvRecorder
import numpy as np
from typing import Union
from silero_vad import load_silero_vad
import torch


def get_clean_audio_buffer(buffer: list[int]) -> np.ndarray:
    """
    Usage:

    - Converts a raw int16 PCM buffer into a normalized float32 numpy array
      suitable for use with faster-whisper

    Requires:

    - `buffer`:
        - Type: list[int]
        - What: A flat list of raw int16 PCM samples, as returned by `Recorder.record`

    Returns:

    - `audio_buffer`:
        - Type: np.ndarray
        - What: A float32 numpy array with values normalized to the range [-1.0, 1.0]
        - Note: This format is required by faster-whisper's `transcribe` method
    """
    return np.array(buffer, dtype=np.int16).astype(np.float32) / 32768.0


class Recorder:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 512,
    ) -> None:
        """
        Usage:

        - Initializes a recorder that wraps PvRecorder for fixed-duration recording
          and Silero VAD for utterance-gated recording. The VAD model is loaded once
          at first call and reused across all `record_vad` calls.

        Optional:

        - `sample_rate`:
            - Type: int
            - What: The sample rate in Hz used for both recording and VAD inference
            - Default: 16000
            - Note: Should match the sample rate expected by your downstream model

        - `frame_length`:
            - Type: int
            - What: The number of samples per frame read from PvRecorder
            - Default: 512
            - Note: Also used as Silero VAD's native 16kHz window (~32ms per frame)
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_ms = self.frame_length / self.sample_rate * 1000

    def ensure_vad(self) -> None:
        if not hasattr(self, "_vad_model"):
            self._vad_model = load_silero_vad()
            self._vad_model.eval()

    def record(
        self,
        device_index: int,
        duration: Union[int, float],
    ) -> list[int]:
        """
        Usage:

        - Records audio from a microphone for a fixed duration and returns a raw
          PCM buffer.

        Requires:

        - `device_index`:
            - Type: int
            - What: The index of the microphone device to record from
            - Note: Use `-1` to select the system default input device

        - `duration`:
            - Type: int | float
            - What: The number of seconds to record

        Returns:

        - `buffer`:
            - Type: list[int]
            - What: A flat list of raw int16 PCM samples at the configured sample rate
        """
        recorder = PvRecorder(
            device_index=device_index, frame_length=self.frame_length
        )
        try:
            recorder.start()
            frames = int(self.sample_rate * duration / self.frame_length)
            buffer = []
            for _ in range(frames):
                buffer.extend(recorder.read())
        except Exception as e:
            raise e
        finally:
            recorder.stop()
            recorder.delete()
        return buffer

    def record_vad(
        self,
        device_index: int,
        speech_threshold: float,
        silence_threshold: float,
        silence_frames_threshold: int,
        speech_pad_frames: int,
        max_speech_duration_s: float,
        stop_event=None,
    ) -> list[int]:
        """
        Usage:

        - Records from the microphone using the cached Silero VAD model to detect
          a single complete utterance and returns the raw PCM buffer. Blocks until
          speech is detected and then silence confirms the utterance is complete.

        Requires:

        - `device_index`:
            - Type: int
            - What: PvRecorder device index; -1 uses system default

        - `speech_threshold`:
            - Type: float
            - What: Silero probability above which a frame is considered speech onset

        - `silence_threshold`:
            - Type: float
            - What: Silero probability below which a frame is considered silence
              during an active speech segment; must be less than `speech_threshold`
              to create a hysteresis band

        - `silence_frames_threshold`:
            - Type: int
            - What: Consecutive below-threshold frames required to confirm end of utterance

        - `speech_pad_frames`:
            - Type: int
            - What: Pre-roll frames captured before onset confirmation; also the
              number of consecutive voiced frames required to confirm speech onset

        - `max_speech_duration_s`:
            - Type: float
            - What: Hard cap on utterance duration in seconds; forces a return
              even if the speaker never pauses

        Returns:

        - `buffer`:
            - Type: list[int]
            - What: Flat list of raw int16 PCM samples representing the captured utterance
        """
        self.ensure_vad()
        frame_ms = self.frame_length / self.sample_rate * 1000
        max_speech_frames = int(max_speech_duration_s * 1000 / frame_ms)

        recorder = PvRecorder(
            device_index=device_index, frame_length=self.frame_length
        )

        speech_buffer: list[list[int]] = []
        pre_roll: list[list[int]] = []
        in_speech: bool = False
        voiced_frame_count: int = 0
        silent_frame_count: int = 0

        try:
            recorder.start()
            while stop_event is None or not stop_event.is_set():
                frame = recorder.read()
                tensor = (
                    torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
                    / 32768.0
                )

                with torch.no_grad():
                    speech_prob = self._vad_model(
                        tensor, self.sample_rate
                    ).item()

                if not in_speech:
                    pre_roll.append(frame)
                    if len(pre_roll) > speech_pad_frames:
                        pre_roll.pop(0)

                    if speech_prob >= speech_threshold:
                        voiced_frame_count += 1
                        if voiced_frame_count >= speech_pad_frames:
                            in_speech = True
                            silent_frame_count = 0
                            speech_buffer = list(pre_roll)
                    else:
                        voiced_frame_count = 0
                else:
                    speech_buffer.append(frame)
                    if speech_prob < silence_threshold:
                        silent_frame_count += 1
                    else:
                        silent_frame_count = 0

                    if (
                        silent_frame_count >= silence_frames_threshold
                        or len(speech_buffer) >= max_speech_frames
                    ):
                        break
        finally:
            recorder.stop()
            recorder.delete()

        return [sample for frame in speech_buffer for sample in frame]


class Notify:
    def notify(
        self,
        message: str,
        notification_type: str = "warning",
        depth: int = 0,
        force: bool = False,
    ) -> None:
        """
        Usage:

        - Creates a class based notification message

        Requires:

        - `message`:
            - Type: str
            - What: The message to warn users with
            - Note: Messages with `{class_name}` and `{method_name}` in them are formatted appropriately

        Optional:

        - `notification_type`:
            - Type: str
            - What: The type of notification to send (warning, verbose or exception)
            - Default: "warning"
            - Note:
                - "warning" prints a warning message
                - "verbose" prints a verbose message only if `self.verbose=True`
                - "exception" raises an exception with the message

        - `depth`:
            - Type: int
            - What: The depth of the nth call below the top of the method stack
            - Note: Depth starts at 0 (indicating the current method in the stack)
            - Default: 0

        - `force`:
            - Type: bool
            - What: If True, forces the message to print regardless of warning or verbose settings
            - Default: False

        Notes:

        - If `self.warning_stack=True`, prints the stack trace alongside warning messages
        - If `self.warnings=False`, suppresses all warning messages
        - If `self.verbose=True`, enables verbose messages
        """
        notification_types = {
            "warning": "WARNING",
            "verbose": "",
            "exception": "EXCEPTION",
        }
        message = f"{self.__class__.__name__}.{sys._getframe(depth).f_back.f_code.co_name} {notification_types.get(notification_type, '')}: {message}"
        if notification_type == "exception":
            raise Exception(message)
        elif notification_type == "warning":
            if self.__dict__.get("warnings", True) or force:
                if self.__dict__.get("warning_stack", False):
                    traceback.print_stack(limit=10)
                print(message)
        elif notification_type == "verbose" or force:
            if self.__dict__.get("verbose", False):
                print(message)
        else:
            raise Exception(
                f"Invalid notification type. Must be one of: {list(notification_types.keys())}"
            )
