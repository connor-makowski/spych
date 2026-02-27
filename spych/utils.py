import traceback, sys
from pvrecorder import PvRecorder
import numpy as np


def record(device_index, duration, sample_rate=16000, frame_length=512):
    """
    Usage:

    - Records audio from a microphone and returns a raw PCM buffer

    Requires:

    - `device_index`:
        - Type: int
        - What: The index of the microphone device to record from
        - Note: Use `-1` to select the system default input device

    - `duration`:
        - Type: int | float
        - What: The number of seconds to record

    Optional:

    - `sample_rate`:
        - Type: int
        - What: The sample rate in Hz to record at
        - Default: 16000
        - Note: Should match the sample rate expected by your downstream model

    - `frame_length`:
        - Type: int
        - What: The number of samples per frame read from the recorder
        - Default: 512

    Returns:

    - `buffer`:
        - Type: list[int]
        - What: A flat list of raw int16 PCM samples at the specified sample rate
    """
    recorder = PvRecorder(device_index=device_index, frame_length=frame_length)
    try:
        recorder.start()
        frames = int(sample_rate * duration / frame_length)
        buffer = []
        for _ in range(frames):
            buffer.extend(recorder.read())
    except Exception as e:
        raise e
    finally:
        recorder.stop()
        recorder.delete()
    return buffer


def get_clean_audio_buffer(buffer):
    """
    Usage:

    - Converts a raw int16 PCM buffer into a normalized float32 numpy array
      suitable for use with faster-whisper

    Requires:

    - `buffer`:
        - Type: list[int]
        - What: A flat list of raw int16 PCM samples, as returned by `record`

    Returns:

    - `audio_buffer`:
        - Type: np.ndarray
        - What: A float32 numpy array with values normalized to the range [-1.0, 1.0]
        - Note: This format is required by faster-whisper's `transcribe` method
    """
    return np.array(buffer, dtype=np.int16).astype(np.float32) / 32768.0


class Notify:
    def notify(
        self, message, notification_type="warning", depth=0, force=False
    ):
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
