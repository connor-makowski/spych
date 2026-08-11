import threading, time
from queue import Queue, Empty
from typing import Optional
from faster_whisper import WhisperModel
from spych.utils import (
    Notify,
    Recorder,
    get_clean_audio_buffer,
    resolve_whisper_device,
)


class _WakeCapture(Notify):
    """
    Usage:

    - Owns the single persistent microphone handle for wake-word spotting.
    - Loops calling `Recorder.record_vad()`, which blocks until Silero VAD
      confirms real speech has started and then ended, and pushes each
      isolated utterance buffer onto the shared audio queue for a
      `_WakeTranscriber` worker to pick up.
    - Intended to run as the target of a single dedicated thread.

    Notes:

    - Pauses (without holding the mic open for decoding) whenever the parent
      `SpychWake` is `locked` (i.e. a wake callback is currently running),
      so capture doesn't compete with an in-flight response.
    - No decoding happens here — this thread only ever produces audio
      buffers; all Whisper calls happen in `_WakeTranscriber` workers.
    """

    def __init__(self, spych_wake_object: "SpychWake"):
        self.spych_wake_object = spych_wake_object

    def run(self) -> None:
        w = self.spych_wake_object
        while not w.stop_event.is_set():
            if w.locked:
                time.sleep(0.05)
                continue
            buffer = w.recorder.record_vad(
                device_index=w.device_index,
                speech_threshold=w.vad_speech_threshold,
                silence_threshold=w.vad_silence_threshold,
                silence_frames_threshold=w.vad_silence_frames_threshold,
                speech_pad_frames=w.vad_speech_pad_frames,
                max_speech_duration_s=w.wake_listener_time,
                stop_event=w.stop_event,
            )
            if w.stop_event.is_set() or not buffer:
                continue
            w.audio_queue.put(buffer)


class _WakeTranscriber(Notify):
    """
    Usage:

    - Pulls completed utterance buffers off the shared audio queue and runs
      the wake-spotting Whisper model on just that isolated utterance.
    - Matches the transcribed text against the registered wake words and
      triggers `SpychWake.wake()` on the first match.
    - Intended to run as the target of one of `wake_listener_count`
      dedicated worker threads.

    Notes:

    - Uses `beam_size=1` (greedy decoding) — fast, and sufficient for a
      substring match rather than a high-fidelity transcription.
    - The `initial_prompt` biases the model toward all registered wake words
      to reduce false negatives.
    - If multiple wake words are present in a single segment, the first
      match wins.
    - Skips processing (but still drains the queue) while the parent
      `SpychWake` is `locked`, so no decoding competes with an in-flight
      response.
    """

    def __init__(self, spych_wake_object: "SpychWake"):
        self.spych_wake_object = spych_wake_object

    def run(self) -> None:
        w = self.spych_wake_object
        while not w.stop_event.is_set():
            try:
                buffer = w.audio_queue.get(timeout=0.5)
            except Empty:
                continue
            if w.stop_event.is_set() or w.locked:
                continue
            self._check_buffer(buffer)

    def _check_buffer(self, buffer: list) -> None:
        w = self.spych_wake_object
        audio_buffer = get_clean_audio_buffer(buffer)
        wake_words = list(w.wake_word_map.keys())
        wake_string = "[" + ", ".join(wake_words) + "]"
        segments, _ = w.wake_model.transcribe(
            audio_buffer,
            beam_size=1,
            initial_prompt=f"""Here are some wake words: {wake_string}. Only return what you understood was said, but place extra weight on those words if there is a tie.""",
        )
        for segment in segments:
            # Skip segments with high no_speech_prob to reduce false positives on silence/background noise;
            # the threshold can be adjusted based on testing and environment
            if segment.no_speech_prob > w.no_speech_threshold:
                continue
            if w.stop_event.is_set() or w.locked:
                return
            text = segment.text.lower()
            for wake_word in wake_words:
                if wake_word in text:
                    w.wake(wake_word)
                    return


class SpychWake(Notify):
    def __init__(
        self,
        wake_word_map,
        terminate_words=None,
        wake_listener_count=2,
        wake_listener_time=4,
        wake_listener_max_processing_time=0.5,
        device_index=-1,
        whisper_model="small.en",
        whisper_device="auto",
        whisper_compute_type="int8",
        no_speech_threshold=0.25,
        vad_speech_threshold=0.5,
        vad_silence_threshold=0.35,
        vad_silence_frames_threshold=15,
        vad_speech_pad_frames=5,
        on_terminate=None,
        wake_model_instance: Optional[WhisperModel] = None,
    ):
        """
        Usage:

        - Initializes a wake word detection system using a single persistent,
          VAD-gated capture thread and a small pool of transcription worker
          threads, using faster-whisper for offline transcription
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
            - What: The number of parallel transcription worker threads pulling
              completed utterances off the internal audio queue and running Whisper
            - Default: 2
            - Note: Audio capture uses a single persistent VAD-gated thread rather
              than one thread per listener; this count only affects how many
              utterances can be transcribed concurrently, which rarely needs to
              exceed 2 since utterances are naturally spaced out by speech

        - `wake_listener_time`:
            - Type: int | float
            - What: Hard cap in seconds on a single wake-word utterance capture
              (passed to `Recorder.record_vad` as `max_speech_duration_s`); bounds
              worst-case latency if the speaker doesn't pause
            - Default: 4

        - `wake_listener_max_processing_time`:
            - Type: int | float
            - What: Deprecated and ignored. Retained only so existing constructor
              calls keep working
            - Default: 0.5
            - Note: Capture and transcription are decoupled via a shared queue, so
              no stagger timing calculation is needed anymore

        - `device_index`:
            - Type: int
            - What: The microphone device index to record from
            - Default: -1
            - Note: Use `-1` to select the system default input device

        - `whisper_model`:
            - Type: str
            - What: The faster-whisper model name to use for wake word transcription
            - Default: "small.en"
            - Note: Smaller models (tiny, base) are recommended here for low latency

        - `whisper_device`:
            - Type: str
            - What: The device to run the whisper model on
            - Default: "auto"
            - Options: "auto", "cpu", "cuda"
            - Note: "auto" selects "cuda" when Python <=3.13 and a CUDA device is
              available, otherwise falls back to "cpu"

        - `whisper_compute_type`:
            - Type: str
            - What: The compute type to use for the whisper model
            - Default: "int8"
            - Note: "int8" offers a good balance of speed and accuracy on both CPU and GPU

        - `no_speech_threshold`:
            - Type: float
            - What: The threshold for the `no_speech_prob` returned by faster-whisper
            - Default: 0.25
            - Note: Segments with a `no_speech_prob` above this threshold will be ignored to reduce false positives from silence or background noise

        - `vad_speech_threshold`:
            - Type: float
            - What: Silero probability above which a frame is considered speech onset
            - Default: 0.5

        - `vad_silence_threshold`:
            - Type: float
            - What: Silero probability below which a frame is considered silence
              during an active utterance; must be less than `vad_speech_threshold`
              to create a hysteresis band
            - Default: 0.35

        - `vad_silence_frames_threshold`:
            - Type: int
            - What: Consecutive below-threshold frames required to confirm the
              utterance has ended and hand it off for transcription
            - Default: 15  (~480ms at 32ms/frame)
            - Note: Lower values reduce detection latency but risk cutting off
              multi-word wake phrases on a natural mid-phrase pause

        - `vad_speech_pad_frames`:
            - Type: int
            - What: Pre-roll frames captured before onset confirmation; also the
              number of consecutive voiced frames required to confirm speech onset
            - Default: 5  (~160ms)

        - `on_terminate`:
            - Type: callable
            - What: A no-argument callback function to execute when a terminate word is detected
            - Default: None (disabled)
            - Note: If provided, this callback will be executed before the system is stopped when a terminate word is detected

        - `wake_model_instance`:
            - Type: faster_whisper.WhisperModel | None
            - What: An already-loaded WhisperModel to reuse for wake-word spotting
              instead of constructing a new one
            - Default: None (constructs a new WhisperModel from `whisper_model`,
              `whisper_device`, and `whisper_compute_type`)
            - Note: When provided, `whisper_model`, `whisper_device`, and
              `whisper_compute_type` are ignored for model construction purposes.
              Intended for callers (e.g. SpychOrchestrator) that already loaded an
              identically-configured model for command transcription and want to
              avoid loading a second copy into memory.
        """
        self.recorder = Recorder()
        self.wake_word_map = {k.lower(): v for k, v in wake_word_map.items()}
        # Handle Terminating Words
        self.terminate_words = (
            [w.lower() for w in terminate_words] if terminate_words else []
        )
        for word in self.terminate_words:
            if word in self.wake_word_map:
                raise ValueError(
                    f"Terminate word '{word}' cannot also be a wake word."
                )
            self.wake_word_map[word] = self.stop
        self.no_speech_threshold = no_speech_threshold
        self.vad_speech_threshold = vad_speech_threshold
        self.vad_silence_threshold = vad_silence_threshold
        self.vad_silence_frames_threshold = vad_silence_frames_threshold
        self.vad_speech_pad_frames = vad_speech_pad_frames
        self.on_terminate = on_terminate
        self.wake_listener_count = wake_listener_count
        self.wake_listener_time = wake_listener_time
        self.wake_listener_max_processing_time = (
            wake_listener_max_processing_time
        )
        self.device_index = device_index
        self.locked = False
        self.kill = False
        self.stop_event = threading.Event()
        self.audio_queue: Queue = Queue()
        self.whisper_device = resolve_whisper_device(whisper_device)
        if wake_model_instance is not None:
            self.wake_model = wake_model_instance
        else:
            self.wake_model = WhisperModel(
                whisper_model,
                device=self.whisper_device,
                compute_type=whisper_compute_type,
            )
        self._capture_thread: Optional[threading.Thread] = None
        self._transcriber_threads: list[threading.Thread] = []

    def start(self):
        """
        Usage:

        - Starts the wake word detection system: one persistent VAD-gated capture
          thread plus `wake_listener_count` transcription worker threads
        - Blocks until a KeyboardInterrupt is received or `stop()` is called

        Notes:

        - Callbacks are defined in `wake_word_map` at init time rather than passed to `start`
        - Threads are tracked and joined on shutdown instead of being fire-and-forget
        """
        self.stop_event.clear()
        self.kill = False
        self._capture_thread = threading.Thread(
            target=_WakeCapture(self).run, daemon=True
        )
        self._transcriber_threads = [
            threading.Thread(target=_WakeTranscriber(self).run, daemon=True)
            for _ in range(self.wake_listener_count)
        ]
        self._capture_thread.start()
        for thread in self._transcriber_threads:
            thread.start()

        try:
            while not self.stop_event.is_set() and not self.kill:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
        finally:
            self.stop_event.set()
            self._capture_thread.join(timeout=2)
            for thread in self._transcriber_threads:
                thread.join(timeout=2)

    def stop_listeners(self):
        """
        Usage:

        - Signals the capture thread and all transcription worker threads to stop
          at their next available checkpoint
        - Note: Does not block; threads will exit cleanly after their current operation
        """
        self.stop_event.set()

    def stop(self):
        """
        Usage:

        - Stops all wake-word threads and exits the `start` loop
        - Note: Combines `stop_listeners` with setting the kill flag on the main loop
        """
        self.stop_listeners()
        self.kill = True
        if self.on_terminate:
            try:
                self.on_terminate()
            except Exception as e:
                self.notify(
                    f"Error in on_terminate callback: {e}",
                    notification_type="exception",
                )

    def wake(self, wake_word):
        """
        Usage:

        - Called internally when a wake word is detected
        - Locks the system, executes the mapped callback for the detected wake
          word, then unlocks

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
