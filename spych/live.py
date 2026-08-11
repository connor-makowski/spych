import threading
import time
import signal
from datetime import timedelta
from queue import Queue, Empty
from typing import Optional, Callable

from faster_whisper import WhisperModel

from spych.utils import (
    Notify,
    get_clean_audio_buffer,
    load_whisper_model,
    Recorder,
    resolve_whisper_device,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_timestamp_srt(seconds: float) -> str:
    """Convert a float second offset into an SRT-compatible timestamp string."""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    millis = int((td.total_seconds() - total_seconds) * 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def format_timestamp_txt(seconds: float) -> str:
    """Convert a float second offset into a human-readable [HH:MM:SS] string."""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"[{hours:02}:{minutes:02}:{secs:02}]"


# ---------------------------------------------------------------------------
# Data container for a completed transcription segment
# ---------------------------------------------------------------------------


class Segment:
    """Internal container for a fully transcribed speech segment."""

    __slots__ = ("text", "start_time", "end_time", "index")

    def __init__(
        self, text: str, start_time: float, end_time: float, index: int
    ):
        self.text = text.strip()
        self.start_time = start_time
        self.end_time = end_time
        self.index = index


# ---------------------------------------------------------------------------
# VAD-gated recording thread
# ---------------------------------------------------------------------------


class VADRecorder(Notify):
    """
    Continuously reads frames from the microphone, runs each through the
    Silero VAD model, and flushes complete speech utterances to a queue.

    Design decisions:

    - Silero VAD expects 512-sample chunks at 16kHz (32ms per frame). This
      is its native window size for 16kHz audio and produces the most reliable
      probability scores. PvRecorder is configured to match exactly.
    - Rather than a binary is_speech flag, Silero returns a speech probability
      (0.0–1.0). We apply `speech_threshold` (onset) and `silence_threshold`
      (offset) as separate values, creating a hysteresis band that prevents
      flickering on borderline frames. Onset requires a higher confidence than
      offset, which is the conventional approach for VAD state machines.
    - The model manages its own hidden state internally across calls. In the
      current silero-vad API (v5+), hidden state is no longer passed explicitly
      — the model object is stateful between `.forward()` calls, giving it
      temporal context across the session without external tensor management.
    - `max_speech_duration_s` acts as a hard flush cap for unbroken monologues,
      bounding memory growth for 1hr+ sessions.
    - Pre-roll audio (`speech_pad_frames` frames captured before onset
      confirmation) prevents clipped leading consonants.
    - The model is run on CPU regardless of the main Whisper device. At one
      512-sample inference per 32ms, the load is negligible and avoids CUDA
      stream contention with the transcription thread.
    """

    def __init__(
        self,
        audio_queue: Queue,
        stop_event: threading.Event,
        device_index: int = -1,
        speech_threshold: float = 0.5,
        silence_threshold: float = 0.35,
        silence_frames_threshold: int = 20,
        speech_pad_frames: int = 5,
        max_speech_duration_s: float = 30.0,
    ):
        """
        Requires:

        - `audio_queue`:
            - Type: Queue
            - What: Thread-safe queue receiving (audio_array, start_time, end_time) tuples

        - `stop_event`:
            - Type: threading.Event
            - What: When set, the recording loop exits cleanly after the current frame

        Optional:

        - `device_index`:
            - Type: int
            - What: PvRecorder microphone device index; -1 uses system default
            - Default: -1

        - `speech_threshold`:
            - Type: float (0.0–1.0)
            - What: Silero probability above which a frame is considered speech onset
            - Default: 0.5
            - Note: Higher values reduce false positives in noisy environments but
              may miss soft or distant speech

        - `silence_threshold`:
            - Type: float (0.0–1.0)
            - What: Silero probability below which a frame is considered silence during
              an active speech segment; lower than `speech_threshold` to create hysteresis
            - Default: 0.35
            - Note: Must be less than `speech_threshold`; the gap between the two
              defines the hysteresis band that prevents rapid on/off toggling

        - `silence_frames_threshold`:
            - Type: int
            - What: Consecutive below-silence-threshold frames required to close a
              speech segment and flush it to the queue
            - Default: 20  (~640ms at 32ms/frame)
            - Note: Lower values reduce output latency but may split sentences on
              natural mid-speech pauses; increase for slower or more deliberate speech

        - `speech_pad_frames`:
            - Type: int
            - What: Frames held in pre-roll before onset confirmation; also the
              number of consecutive speech frames required to confirm onset
            - Default: 5  (~160ms)

        - `max_speech_duration_s`:
            - Type: float
            - What: Hard cap on a single speech segment in seconds; forces a flush
              even if the speaker has not paused, bounding memory growth
            - Default: 30.0
        """
        self.audio_queue = audio_queue
        self.stop_event = stop_event
        self.device_index = device_index
        self.speech_threshold = speech_threshold
        self.silence_threshold = silence_threshold
        self.silence_frames_threshold = silence_frames_threshold
        self.speech_pad_frames = speech_pad_frames
        self.max_speech_duration_s = max_speech_duration_s
        self.recorder = Recorder()
        self.max_speech_frames = int(
            max_speech_duration_s * 1000 / self.recorder.frame_ms
        )

    def run(self, session_start_time: float):
        """
        Blocking recording loop. Intended to be run inside a dedicated thread.

        Requires:

        - `session_start_time`:
            - Type: float
            - What: Unix timestamp of session start, used to compute relative
              segment timestamps

        Notes:

        - record_vad() from utils handles a single complete utterance capture.
          This loop calls it repeatedly, tracking session-relative timestamps
          and checking stop_event between utterances so the session can be
          cleanly terminated at any utterance boundary.
        - Silero model loading and frame inference are fully encapsulated in
          record_vad(); this method only handles orchestration.
        """
        try:
            while not self.stop_event.is_set():
                start_wall = time.time()
                frames = self.recorder.record_vad(
                    device_index=self.device_index,
                    speech_threshold=self.speech_threshold,
                    silence_threshold=self.silence_threshold,
                    silence_frames_threshold=self.silence_frames_threshold,
                    speech_pad_frames=self.speech_pad_frames,
                    max_speech_duration_s=self.max_speech_duration_s,
                    stop_event=self.stop_event,
                )
                if self.stop_event.is_set():
                    break
                end_wall = time.time()
                start_time = start_wall - session_start_time
                end_time = end_wall - session_start_time
                self.flush(frames, start_time, end_time)

        finally:
            pass

    def flush(
        self,
        buffer: list[int],
        start_time: float,
        end_time: float,
    ):
        """Convert flat PCM buffer to float32 numpy array and push to audio_queue."""
        audio = get_clean_audio_buffer(buffer)
        self.audio_queue.put((audio, start_time, end_time))


# ---------------------------------------------------------------------------
# Transcription worker thread
# ---------------------------------------------------------------------------


class Transcriber(Notify):
    """
    Pulls (audio, start_time, end_time) tuples from audio_queue, runs
    faster-whisper inference, and pushes Segment objects to segment_queue.

    Design decisions:

    - A single transcription thread serializes model calls, avoiding the
      overhead and complexity of multiple model instances competing for CPU/GPU.
    - `initial_prompt` carries the last N words of confirmed transcript into
      each new transcription call. This significantly improves accuracy on
      names, domain terms, and mid-sentence context.
    - The sentinel value `None` pushed to audio_queue signals the transcriber
      to flush and exit cleanly.
    """

    def __init__(
        self,
        audio_queue: Queue,
        segment_queue: Queue,
        model: WhisperModel,
        stop_event: threading.Event,
        no_speech_threshold: float = 0.4,
        context_words: int = 32,
    ):
        """
        Requires:

        - `audio_queue`: Queue of (np.ndarray, float, float) tuples from VADRecorder
        - `segment_queue`: Queue of Segment objects consumed by Writer
        - `model`: A pre-initialized WhisperModel instance
        - `stop_event`: Shared stop signal

        Optional:

        - `no_speech_threshold`:
            - Type: float
            - What: Segments with no_speech_prob above this are discarded
            - Default: 0.4

        - `context_words`:
            - Type: int
            - What: Number of trailing words from prior transcript passed as
              initial_prompt to each transcription call
            - Default: 32
        """
        self.audio_queue = audio_queue
        self.segment_queue = segment_queue
        self.model = model
        self.stop_event = stop_event
        self.no_speech_threshold = no_speech_threshold
        self.context_words = context_words
        self.context_buffer: list[str] = []
        self.segment_index: int = 0

    def run(self):
        """Blocking transcription loop. Intended to be run in a dedicated thread."""
        while True:
            try:
                item = self.audio_queue.get(timeout=0.5)
            except Empty:
                if self.stop_event.is_set():
                    break
                continue

            if item is None:
                break

            audio, start_time, end_time = item
            initial_prompt = " ".join(
                self.context_buffer[-self.context_words :]
            )

            segments, _ = self.model.transcribe(
                audio,
                beam_size=5,
                initial_prompt=initial_prompt if initial_prompt else None,
            )

            words = []
            for seg in segments:
                if seg.no_speech_prob > self.no_speech_threshold:
                    continue
                words.append(seg.text.strip())

            if not words:
                continue

            text = " ".join(words)
            self.context_buffer.extend(text.split())
            # Keep context buffer bounded
            if len(self.context_buffer) > 256:
                self.context_buffer = self.context_buffer[-128:]

            self.segment_index += 1
            self.segment_queue.put(
                Segment(text, start_time, end_time, self.segment_index)
            )


# ---------------------------------------------------------------------------
# Writer / output thread
# ---------------------------------------------------------------------------


class Writer(Notify):
    """
    Consumes Segment objects from segment_queue and writes them to disk
    and/or the terminal.

    Design decisions:

    - File handles are opened once at session start and kept alive for the
      duration rather than opened per-write. This avoids repeated fsync
      overhead and is safe for 1hr+ sessions.
    - Both output formats are written in the same pass to avoid duplicating
      queue consumption logic.
    - `None` sentinel in segment_queue signals a clean shutdown.
    - SRT index is 1-based per the SRT spec.
    """

    def __init__(
        self,
        segment_queue: Queue,
        stop_event: threading.Event,
        output_format: str = "both",
        output_path: str = "transcript",
        show_timestamps: bool = True,
    ):
        """
        Requires:

        - `segment_queue`: Queue of Segment objects from Transcriber
        - `stop_event`: Shared stop signal

        Optional:

        - `output_format`:
            - Type: str
            - What: Output format(s) to write
            - Default: "both"
            - Options: "txt", "srt", "both"

        - `output_path`:
            - Type: str
            - What: Base file path (without extension); extensions are appended
              automatically
            - Default: "transcript"

        - `show_timestamps`:
            - Type: bool
            - What: If True, prepends relative timestamps to terminal and TXT output
            - Default: True
        """
        self.segment_queue = segment_queue
        self.stop_event = stop_event
        self.output_format = output_format
        self.output_path = output_path
        self.show_timestamps = show_timestamps
        self.txt_file = None
        self.srt_file = None

    def run(self):
        """Blocking writer loop. Intended to be run in a dedicated thread."""
        try:
            if self.output_format in ("txt", "both"):
                self.txt_file = open(
                    f"{self.output_path}.txt", "w", encoding="utf-8"
                )
            if self.output_format in ("srt", "both"):
                self.srt_file = open(
                    f"{self.output_path}.srt", "w", encoding="utf-8"
                )

            while True:
                try:
                    segment = self.segment_queue.get(timeout=0.5)
                except Empty:
                    if self.stop_event.is_set():
                        break
                    continue

                if segment is None:
                    break

                self.write_segment(segment)

        finally:
            if self.txt_file:
                self.txt_file.flush()
                self.txt_file.close()
            if self.srt_file:
                self.srt_file.flush()
                self.srt_file.close()

    def write_segment(self, segment: Segment):
        """Write one segment to all active outputs and flush to terminal."""
        # --- Terminal ---
        if self.show_timestamps:
            ts = format_timestamp_txt(segment.start_time)
            terminal_line = f"{ts} {segment.text}"
        else:
            terminal_line = segment.text
        print(terminal_line, flush=True)

        # --- TXT file ---
        if self.txt_file:
            self.txt_file.write(terminal_line + "\n")
            self.txt_file.flush()

        # --- SRT file ---
        if self.srt_file:
            srt_block = (
                f"{segment.index}\n"
                f"{format_timestamp_srt(segment.start_time)} --> "
                f"{format_timestamp_srt(segment.end_time)}\n"
                f"{segment.text}\n\n"
            )
            self.srt_file.write(srt_block)
            self.srt_file.flush()


# ---------------------------------------------------------------------------
# Keystroke listener
# ---------------------------------------------------------------------------


class KeystrokeListener(Notify):
    """
    Runs a blocking readline loop on a background thread watching for a
    user-defined stop key sequence. On match, sets the shared stop_event.

    Design decisions:

    - `input()` is used rather than raw terminal manipulation to remain
      cross-platform (Windows, macOS, Linux) without requiring `termios`
      or `curses`. The tradeoff is that the user must press Enter after
      the keystroke.
    - Runs on a daemon thread so it is automatically cleaned up if the
      main process exits unexpectedly.
    """

    def __init__(self, stop_event: threading.Event, stop_key: str = "q"):
        """
        Requires:

        - `stop_event`: threading.Event to set when the stop key is detected

        Optional:

        - `stop_key`:
            - Type: str
            - What: The string the user must type (followed by Enter) to stop
            - Default: "q"
        """
        self.stop_event = stop_event
        self.stop_key = stop_key.lower().strip()

    def run(self):
        """Blocking input loop. Run on a daemon thread."""
        while not self.stop_event.is_set():
            try:
                line = input()
                if line.strip().lower() == self.stop_key:
                    print(
                        f"\n[spych] Stop key '{self.stop_key}' received. "
                        "Finishing current segment and shutting down...",
                        flush=True,
                    )
                    self.stop_event.set()
                    break
            except EOFError:
                # stdin closed (e.g. piped input) — treat as stop
                self.stop_event.set()
                break


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class SpychLive(Notify):
    def __init__(
        self,
        output_format: str = "srt",
        output_path: str = "transcript",
        show_timestamps: bool = True,
        stop_key: str = "q",
        terminate_words: Optional[list[str]] = None,
        on_terminate: Optional[Callable] = None,
        device_index: int = -1,
        whisper_model: str = "base.en",
        whisper_device: str = "auto",
        whisper_compute_type: str = "int8",
        no_speech_threshold: float = 0.4,
        speech_threshold: float = 0.5,
        silence_threshold: float = 0.35,
        silence_frames_threshold: int = 20,
        speech_pad_frames: int = 5,
        max_speech_duration_s: float = 30.0,
        context_words: int = 32,
    ):
        """
        Usage:

        - Initializes a live transcription session using VAD-gated audio
          segmentation and faster-whisper inference
        - Transcribes continuously until stopped by the user via keystroke
          or terminate word; the session cannot be stopped by any other means

        Optional:

        - `output_format`:
            - Type: str
            - What: The file format(s) to write transcription output to
            - Default: "srt"
            - Options: "txt", "srt", "both"

        - `output_path`:
            - Type: str
            - What: Base output file path without extension
            - Default: "transcript"
            - Note: Extensions are appended automatically (.txt, .srt)

        - `show_timestamps`:
            - Type: bool
            - What: If True, prepends relative [HH:MM:SS] timestamps to each
              line in terminal and TXT output
            - Default: True

        - `stop_key`:
            - Type: str
            - What: The key (followed by Enter) the user types to stop recording
            - Default: "q"
            - Note: This is the only non-word mechanism to end the session

        - `terminate_words`:
            - Type: list[str] | None
            - What: Words that, if detected in the transcript, immediately stop
              the session
            - Default: None (disabled)
            - Note: Terminate words are detected after transcription, not in
              real time during recording; expect ~1–3s detection latency

        - `on_terminate`:
            - Type: callable | None
            - What: Optional no-argument callback executed when a terminate word
              triggers a stop
            - Default: None

        - `device_index`:
            - Type: int
            - What: Microphone device index; -1 uses the system default
            - Default: -1

        - `whisper_model`:
            - Type: str
            - What: The faster-whisper model name
            - Default: "base.en"
            - Note: Use "small.en" on larger setups or for improved accuracy; "tiny.en"
              for low-latency CPU setups

        - `whisper_device`:
            - Type: str
            - What: Device for whisper inference
            - Default: "auto"
            - Options: "auto", "cpu", "cuda"
            - Note: "auto" selects "cuda" when Python <=3.13 and a CUDA device is
              available, otherwise falls back to "cpu"

        - `whisper_compute_type`:
            - Type: str
            - What: Compute precision for the whisper model
            - Default: "int8"
            - Options: "int8", "float16", "float32"

        - `no_speech_threshold`:
            - Type: float
            - What: Whisper segments with no_speech_prob above this are discarded
            - Default: 0.4

        - `speech_threshold`:
            - Type: float (0.0–1.0)
            - What: Silero probability above which a frame is considered speech onset
            - Default: 0.5
            - Note: Raise to reduce false positives in noisy environments

        - `silence_threshold`:
            - Type: float (0.0–1.0)
            - What: Silero probability below which a frame is considered silence
              during an active speech segment; lower than `speech_threshold` to
              create hysteresis and prevent rapid toggling
            - Default: 0.35

        - `silence_frames_threshold`:
            - Type: int
            - What: Consecutive silent 30ms frames required to close a speech segment
            - Default: 20  (~600ms)
            - Note: Lower values reduce latency but may split mid-sentence pauses;
              higher values allow more natural phrasing at the cost of latency

        - `speech_pad_frames`:
            - Type: int
            - What: Consecutive voiced frames required to confirm speech onset; also
              controls pre-roll buffer depth
            - Default: 5  (~150ms)

        - `max_speech_duration_s`:
            - Type: float
            - What: Hard cap on a single speech segment in seconds; forces a flush
              even if the speaker never pauses
            - Default: 30.0

        - `context_words`:
            - Type: int
            - What: Number of trailing transcript words passed as initial_prompt
              to each whisper call for improved contextual accuracy
            - Default: 32
        """
        self.output_format = output_format
        self.output_path = output_path
        self.show_timestamps = show_timestamps
        self.stop_key = stop_key
        self.terminate_words = (
            [w.lower() for w in terminate_words] if terminate_words else []
        )
        self.on_terminate = on_terminate
        self.device_index = device_index
        self.no_speech_threshold = no_speech_threshold
        self.speech_threshold = speech_threshold
        self.silence_threshold = silence_threshold
        self.silence_frames_threshold = silence_frames_threshold
        self.speech_pad_frames = speech_pad_frames
        self.max_speech_duration_s = max_speech_duration_s
        self.context_words = context_words

        self.model = load_whisper_model(
            whisper_model,
            device=resolve_whisper_device(whisper_device),
            compute_type=whisper_compute_type,
        )

        # Shared stop signal — the only way to set this is via stop_key or
        # terminate_words; there is no automatic timeout or external trigger
        self.stop_event = threading.Event()

        self.audio_queue: Queue = Queue()
        self.segment_queue: Queue = Queue()

    def start(self):
        """
        Usage:

        - Starts the live transcription session and blocks until the user
          stops it via the configured stop key or a terminate word
        - Prints a startup message indicating how to stop the session

        Notes:

        - Thread startup order: keystroke listener → recorder → transcriber → writer
          This ensures consumers are ready before producers start pushing data
        - On stop, all threads are given time to flush and exit cleanly before
          the method returns; no audio or transcription is dropped
        - SIGINT (Ctrl+C) is caught and redirected to the same graceful stop
          path rather than raising KeyboardInterrupt mid-thread
        """
        original_sigint = signal.getsignal(signal.SIGINT)

        def handle_sigint(sig, frame):
            print(
                "\n[spych] Interrupt received. "
                "Finishing current segment and shutting down...",
                flush=True,
            )
            self.stop_event.set()

        signal.signal(signal.SIGINT, handle_sigint)

        stop_instructions = [f"Press '{self.stop_key}' + Enter"]
        if self.terminate_words:
            words_display = ", ".join(f'"{w}"' for w in self.terminate_words)
            stop_instructions.append(f"say {words_display}")
        print(
            f"[spych] Live transcription started. "
            f"To stop: {' or '.join(stop_instructions)}.",
            flush=True,
        )

        # --- Keystroke listener (daemon so it doesn't block process exit) ---
        ks_listener = KeystrokeListener(self.stop_event, self.stop_key)
        ks_thread = threading.Thread(target=ks_listener.run, daemon=True)
        ks_thread.start()

        session_start = time.time()

        # --- VAD recorder ---
        recorder = VADRecorder(
            audio_queue=self.audio_queue,
            stop_event=self.stop_event,
            device_index=self.device_index,
            speech_threshold=self.speech_threshold,
            silence_threshold=self.silence_threshold,
            silence_frames_threshold=self.silence_frames_threshold,
            speech_pad_frames=self.speech_pad_frames,
            max_speech_duration_s=self.max_speech_duration_s,
        )
        rec_thread = threading.Thread(
            target=recorder.run, args=(session_start,), daemon=False
        )

        # --- Transcriber ---
        transcriber = Transcriber(
            audio_queue=self.audio_queue,
            segment_queue=self.segment_queue,
            model=self.model,
            stop_event=self.stop_event,
            no_speech_threshold=self.no_speech_threshold,
            context_words=self.context_words,
        )
        trans_thread = threading.Thread(
            target=self.transcribe_and_check,
            args=(transcriber,),
            daemon=False,
        )

        # --- Writer ---
        writer = Writer(
            segment_queue=self.segment_queue,
            stop_event=self.stop_event,
            output_format=self.output_format,
            output_path=self.output_path,
            show_timestamps=self.show_timestamps,
        )
        write_thread = threading.Thread(target=writer.run, daemon=False)

        # Start consumers before producers
        write_thread.start()
        trans_thread.start()
        rec_thread.start()

        # Block on recorder — it drives the session lifecycle
        rec_thread.join()

        # Signal transcriber to drain remaining audio and exit
        self.audio_queue.put(None)
        trans_thread.join()

        # Signal writer to drain remaining segments and exit
        self.segment_queue.put(None)
        write_thread.join()

        signal.signal(signal.SIGINT, original_sigint)
        print(
            f"[spych] Session complete. "
            f"Output saved to: {self.output_path}.*",
            flush=True,
        )

    def transcribe_and_check(self, transcriber: Transcriber):
        """
        Runs transcriber.run() in this thread and intercepts every segment
        put onto segment_queue to check for terminate words.

        All transcription logic lives in Transcriber.run(). Terminate-word
        checking is injected here via a thin queue.put wrapper to keep
        Transcriber free of session-management concerns.
        """
        original_put = self.segment_queue.put

        def checked_put(segment):
            original_put(segment)
            if not self.terminate_words or not isinstance(segment, Segment):
                return
            text_lower = segment.text.lower()
            for word in self.terminate_words:
                if word in text_lower:
                    print(
                        f'\n[spych] Terminate word "{word}" detected. '
                        "Finishing and shutting down...",
                        flush=True,
                    )
                    if self.on_terminate:
                        try:
                            self.on_terminate()
                        except Exception as e:
                            self.notify(
                                f"Error in on_terminate callback: {e}",
                                notification_type="exception",
                            )
                    self.stop_event.set()
                    return

        self.segment_queue.put = checked_put
        try:
            transcriber.run()
        finally:
            self.segment_queue.put = original_put
