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
            - Note: Used to access shared state such as `locked`, `wake_word_map`,
              and `device_index`
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
        - Records audio, transcribes it, and triggers a wake event if any wake word is detected

        Notes:

        - Skips execution silently if this listener is already locked (i.e. mid-cycle)
        - Checks `should_stop` at each major step to allow early exit without blocking
        - Uses `beam_size=2` for fast transcription appropriate for short wake word clips
        - The `initial_prompt` biases the model toward all registered wake words to reduce
          false negatives
        - If multiple wake words are present in a single segment, the first match wins
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
        wake_words = list(self.spych_wake_object.wake_word_map.keys())
        segments, _ = self.spych_wake_object.wake_model.transcribe(
            audio_buffer,
            beam_size=2,
            initial_prompt=f"You are listening for wake words: {str(wake_words)}. Do not transcribe anything else. You should be quite sure you hear these words before you return them.",
        )
        for segment in segments:
            if self.should_stop():
                return
            text = segment.text.lower()
            for wake_word in wake_words:
                if wake_word in text:
                    self.spych_wake_object.wake(wake_word)
                    self.locked = False
                    self.kill = False
                    return
        self.locked = False
        self.kill = False


class SpychWake(Notify):
    def __init__(
        self,
        wake_word_map,
        terminate_words=None,
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
        - Supports multiple wake words, each mapped to a different callback function

        Requires:

        - `wake_word_map`:
            - Type: dict[str, callable]
            - What: A dictionary mapping wake words to their corresponding no-argument
              callback functions
            - Note: Keys are stored and matched in lowercase
            - Example:
                {
                    "jarvis": on_jarvis_wake,
                    "computer": on_computer_wake,
                }

        Optional:

        - `terminate_words`:
            - Type: list[str]
            - What: A list of words that, if detected in the wake listener's transcription,
              will immediately terminate the entire SpychWake system
            - Note: Use with caution, as any false positive on a terminate word will stop
              the wake system until it is manually restarted
            - default: None (disabled)

        - `wake_listener_count`:
            - Type: int
            - What: The number of concurrent listener threads to run
            - Default: 3
            - Note: More listeners reduce the chance of missing a wake word between
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
        self.wake_word_map = {k.lower(): v for k, v in wake_word_map.items()}
        # Handle Terminating Words
        self.terminate_words = [w.lower() for w in terminate_words] if terminate_words else []
        for word in self.terminate_words:
            if word in self.wake_word_map:
                raise ValueError(f"Terminate word '{word}' cannot also be a wake word.")
            self.wake_word_map[word] = self.stop
        self.wake_listener_count = wake_listener_count
        self.wake_listener_time = wake_listener_time
        self.wake_listener_max_processing_time = wake_listener_max_processing_time
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

    def start(self):
        """
        Usage:

        - Starts the wake word detection loop using overlapping listener threads
        - Blocks until a KeyboardInterrupt is received or `stop()` is called

        Notes:

        - Callbacks are defined in `wake_word_map` at init time rather than passed to `start`
        - Listener threads are staggered by `(wake_listener_time + wake_listener_max_processing_time)
          / wake_listener_count` seconds to ensure continuous audio coverage
        - New threads are only launched when the system is not locked (i.e. not currently
          processing a wake event)
        """
        self.notify(
            f"Listening for wake words: {list(self.wake_word_map.keys())}...",
            notification_type="verbose",
        )
        try:
            while True:
                for listener in self.wake_listeners:
                    if self.kill:
                        self.kill = False
                        return
                    if not self.locked:
                        threading.Thread(target=listener).start()
                    time.sleep(
                        (self.wake_listener_time + self.wake_listener_max_processing_time)
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

    def wake(self, wake_word):
        """
        Usage:

        - Called internally when a wake word is detected
        - Stops all listeners, locks the system, executes the mapped callback for the
          detected wake word, then unlocks

        Requires:

        - `wake_word`:
            - Type: str
            - What: The detected wake word, used to look up the correct callback in
              `wake_word_map`

        Notes:

        - If the system is already locked when `wake` is called, the call is a no-op
          to prevent concurrent wake executions
        - Any exception raised by the callback is caught and re-raised as a spych exception
        - The system is always unlocked in the `finally` block, even if the callback raises
        """
        self.stop_listeners()
        if self.locked:
            return
        self.locked = True
        try:
            self.wake_word_map[wake_word]()
        except Exception as e:
            self.notify(
                f"Error in on_wake_fn for '{wake_word}': {e}",
                notification_type="exception",
            )
        finally:
            self.locked = False