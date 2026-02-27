import threading, time
from faster_whisper import WhisperModel
from spych.utils import Notify, record, get_clean_audio_buffer


class SpychWakeListener(Notify):
    def __init__(self, spych_wake_object):
        """
        Usage:

        - Initializes a single wake word listener thread worker

        Requires:

        - `spych_wake_object`:
            - Type: SpychWake
            - What: The parent SpychWake instance that owns this listener
            - Note: Used to access shared state such as `locked`, `wake_word`, and `device_index`
        """
        self.spych_wake_object = spych_wake_object
        self.locked = False
        self.kill = False

    def stop(self):
        """
        Usage:

        - Signals this listener to stop at the next available checkpoint
        - Note: Does not immediately halt execution; the listener will exit cleanly
          after its current operation completes
        """
        self.kill = True

    def should_stop(self):
        """
        Usage:

        - Checks whether this listener should stop processing and exit early
        - Resets `kill` and `locked` state if stopping is required

        Returns:

        - `should_stop`:
            - Type: bool
            - What: True if the listener should stop, False if it should continue
            - Note: Returns True if `self.kill` is set or the parent `SpychWake` is locked
        """
        if self.kill or self.spych_wake_object.locked:
            self.kill = False
            self.locked = False
            return True
        return False

    def __call__(self):
        """
        Usage:

        - Executes one full listen-and-detect cycle when this listener is invoked as a thread target
        - Records audio, transcribes it, and triggers a wake event if the wake word is detected

        Notes:

        - Skips execution silently if this listener is already locked (i.e. mid-cycle)
        - Checks `should_stop` at each major step to allow early exit without blocking
        - Uses `beam_size=2` for fast transcription appropriate for short wake word clips
        - The `initial_prompt` biases the model toward the wake word to reduce false negatives
        """
        if self.locked:
            self.notify(
                "Listener is locked, skipping...", notification_type="verbose"
            )
            return
        if self.should_stop():
            return
        self.locked = True
        buffer = record(
            device_index=self.spych_wake_object.device_index,
            duration=self.spych_wake_object.wake_listener_time,
        )
        if self.should_stop():
            return
        audio_buffer = get_clean_audio_buffer(buffer)
        if self.should_stop():
            return
        segments, _ = self.spych_wake_object.wake_model.transcribe(
            audio_buffer,
            beam_size=2,
            initial_prompt=f"You are listening for a wake word. Only respond with the wake word '{self.spych_wake_object.wake_word}' if you hear it. Otherwise, respond with an empty string.",
        )
        for segment in segments:
            if self.should_stop():
                return
            if self.spych_wake_object.wake_word in segment.text.lower():
                self.spych_wake_object.wake()
                break
        self.locked = False
        self.kill = False


class SpychWake(Notify):
    def __init__(
        self,
        wake_word,
        wake_listener_count=3,
        wake_listener_time=2,
        wake_listener_max_processing_time=0.5,
        device_index=-1,
        whisper_model="tiny",
        whisper_device="cpu",
        whisper_compute_type="int8",
    ):
        """
        Usage:

        - Initializes a wake word detection system using overlapping listener threads
          and faster-whisper for offline transcription

        Requires:

        - `wake_word`:
            - Type: str
            - What: The word to listen for
            - Note: Stored and matched in lowercase

        Optional:

        - `wake_listener_count`:
            - Type: int
            - What: The number of concurrent listener threads to run
            - Default: 3
            - Note: More listeners reduce the chance of missing the wake word between
              recording windows; at least 3 is recommended for continuous coverage

        - `wake_listener_time`:
            - Type: int | float
            - What: The duration in seconds each listener records per cycle
            - Default: 2

        - `wake_listener_max_processing_time`:
            - Type: int | float
            - What: The estimated maximum time in seconds for transcription to complete
            - Default: 0.5
            - Note: Used alongside `wake_listener_time` and `wake_listener_count` to
              calculate the stagger delay between thread launches

        - `device_index`:
            - Type: int
            - What: The microphone device index to record from
            - Default: -1
            - Note: Use `-1` to select the system default input device

        - `whisper_model`:
            - Type: str
            - What: The faster-whisper model name to use for wake word transcription
            - Default: "tiny"
            - Note: Smaller models (tiny, base) are recommended here for low latency

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
        self.wake_word = wake_word.lower()
        self.wake_listener_count = wake_listener_count
        self.wake_listener_time = wake_listener_time
        self.wake_listener_max_processing_time = (
            wake_listener_max_processing_time
        )
        self.device_index = device_index
        self.locked = False
        self.kill = False
        self.wake_model = WhisperModel(
            whisper_model,
            device=whisper_device,
            compute_type=whisper_compute_type,
        )
        self.wake_listeners = [
            SpychWakeListener(self) for _ in range(self.wake_listener_count)
        ]

    def start(self, on_wake_fn):
        """
        Usage:

        - Starts the wake word detection loop using overlapping listener threads
        - Blocks until a KeyboardInterrupt is received or `stop()` is called

        Requires:

        - `on_wake_fn`:
            - Type: callable
            - What: A no-argument callable that is executed each time the wake word is detected
            - Note: Execution is serialized — subsequent detections are ignored until
              `on_wake_fn` returns

        Notes:

        - Listener threads are staggered by `(wake_listener_time + wake_listener_max_processing_time)
          / wake_listener_count` seconds to ensure continuous audio coverage
        - New threads are only launched when the system is not locked (i.e. not currently
          processing a wake event)
        """
        self.notify(
            f"Listening for wake word: '{self.wake_word}'...",
            notification_type="verbose",
        )
        self.on_wake_fn = on_wake_fn
        try:
            while True:
                for listener in self.wake_listeners:
                    if self.kill:
                        self.kill = False
                        return
                    if not self.locked:
                        threading.Thread(target=listener).start()
                    time.sleep(
                        (
                            self.wake_listener_time
                            + self.wake_listener_max_processing_time
                        )
                        / self.wake_listener_count
                    )
        except KeyboardInterrupt:
            self.notify("Stopping.", notification_type="verbose")

    def stop_listeners(self):
        """
        Usage:

        - Signals all listener threads to stop at their next available checkpoint
        - Note: Does not block; listeners will exit cleanly after their current operation
        """
        for listener in self.wake_listeners:
            listener.stop()

    def stop(self):
        """
        Usage:

        - Stops all listener threads and exits the `start` loop
        - Note: Combines `stop_listeners` with setting the kill flag on the main loop
        """
        self.stop_listeners()
        self.kill = True

    def wake(self):
        """
        Usage:

        - Called internally when the wake word is detected
        - Stops all listeners, locks the system, executes `on_wake_fn`, then unlocks

        Notes:

        - If the system is already locked when `wake` is called, the call is a no-op
          to prevent concurrent wake executions
        - Any exception raised by `on_wake_fn` is caught and re-raised as a spych exception
        - The system is always unlocked in the `finally` block, even if `on_wake_fn` raises
        """
        self.stop_listeners()
        if self.locked:
            return
        self.locked = True
        try:
            self.on_wake_fn()
        except Exception as e:
            self.notify(
                f"Error in on_wake_fn: {e}", notification_type="exception"
            )
        finally:
            self.locked = False
