import json
import re
import threading
import time
import signal
import requests
from queue import Queue, Empty
from typing import Optional

from faster_whisper import WhisperModel

from spych.utils import Notify, resolve_whisper_device
from spych.live import VADRecorder, KeystrokeListener, format_timestamp_srt, format_timestamp_txt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LANGUAGE_NAMES: dict[str, str] = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}


def _select_whisper_model(model: str, lang_a: str, lang_b: str) -> str:
    """Strip `.en` suffix if provided"""
    if model.endswith(".en"):
        return model[:-3]
    return model


def _parse_translation_json(
    raw: str, lang_a: str, lang_b: str
) -> Optional[tuple[str, str, str]]:
    """
    Usage:

    - Parses a JSON object from an Ollama response string.
    - Strips markdown code fences before parsing.
    - Expects keys "input_language" and "output_content".
    - Clamps input_language to the known pair and derives output_language.
    - Returns (input_language, output_language, content) or None on failure.

    Requires:

    - `raw`:
        - Type: str
        - What: Raw response string from Ollama, may include markdown fences.

    - `lang_a`:
        - Type: str
        - What: BCP-47 code of the first language in the pair.

    - `lang_b`:
        - Type: str
        - What: BCP-47 code of the second language in the pair.

    Returns:

    - `result`:
        - Type: Optional[tuple[str, str, str]]
        - What: (input_language, output_language, translated_text), or None on failure.
    """
    text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        input_language = str(data.get("input_language", "")).strip()
        output_content = str(data.get("output_content", "")).strip()
        if input_language and output_content:
            if input_language not in (lang_a, lang_b):
                input_language = lang_a
            output_language = lang_b if input_language == lang_a else lang_a
            return input_language, output_language, output_content
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return None


def _detect_and_translate(
    text: str,
    lang_a: str,
    lang_b: str,
    host: str,
    model: str,
) -> Optional[tuple[str, str, str]]:
    """
    Usage:

    - Asks Ollama to detect which of two languages the text is in, then translate
      it to the other language.
    - Returns (input_language, output_language, translated_text) or None on failure.

    Requires:

    - `text`:
        - Type: str
        - What: The transcribed text to translate.

    - `lang_a`:
        - Type: str
        - What: BCP-47 code of the first language in the pair (e.g. "en").

    - `lang_b`:
        - Type: str
        - What: BCP-47 code of the second language in the pair (e.g. "es").

    - `host`:
        - Type: str
        - What: Ollama HTTP base URL (e.g. "http://localhost:11434").

    - `model`:
        - Type: str
        - What: Ollama model name to use for translation (e.g. "llama3.2").

    Returns:

    - `result`:
        - Type: Optional[tuple[str, str, str]]
        - What: (input_language, output_language, translated_text), or None on any error.
    """
    name_a = _LANGUAGE_NAMES.get(lang_a, lang_a)
    name_b = _LANGUAGE_NAMES.get(lang_b, lang_b)
    prompt = (
        f"You are translating between two people having a conversation. "
        f"One speaks {name_a} (code: {lang_a}) and the other speaks {name_b} (code: {lang_b}). "
        f"First identify the input language, then translate the text to the other language."
        f"The text to be translated might be in either language. Make sure to respond in the other language.\n\n"
        f"Translate the following:\n\n {text}\n\n"
    )
    schema = {
        "type": "object",
        "properties": {
            "input_language": {"type": "string", "enum": [lang_a, lang_b]},
            'output_language': {"type": "string", "enum": [lang_a, lang_b]},
            "output_content": {"type": "string"},
        },
        "required": ["input_language", "output_language", "output_content"],
    }
    try:
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "format": schema},
            timeout=10,
        )
        raw = resp.json().get("response", "")
        return _parse_translation_json(raw, lang_a, lang_b)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data container for a completed translation segment
# ---------------------------------------------------------------------------


class TranslationSegment:
    """Internal container for a fully transcribed and translated speech segment."""

    __slots__ = (
        "text",
        "translated_text",
        "input_language",
        "output_language",
        "start_time",
        "end_time",
        "index",
    )

    def __init__(
        self,
        text: str,
        translated_text: str,
        input_language: str,
        output_language: str,
        start_time: float,
        end_time: float,
        index: int,
    ):
        self.text = text.strip()
        self.translated_text = translated_text.strip()
        self.input_language = input_language
        self.output_language = output_language
        self.start_time = start_time
        self.end_time = end_time
        self.index = index


# ---------------------------------------------------------------------------
# Transcription + translation worker thread
# ---------------------------------------------------------------------------


class TranslatingTranscriber(Notify):
    """
    Pulls (audio, start_time, end_time) tuples from audio_queue, runs
    faster-whisper inference with a language-hint initial_prompt, detects
    which language was spoken via Ollama, translates to the other, and
    pushes TranslationSegment objects to segment_queue.

    Ollama failures are soft errors: the segment is emitted with
    translated_text = "[translation unavailable]" so the session continues.
    """

    def __init__(
        self,
        audio_queue: Queue,
        segment_queue: Queue,
        model: WhisperModel,
        stop_event: threading.Event,
        lang_a: str,
        lang_b: str,
        ollama_host: str,
        ollama_translation_model: str,
        no_speech_threshold: float = 0.4,
        show_timestamps: bool = True,
    ):
        """
        Requires:

        - `audio_queue`: Queue of (np.ndarray, float, float) tuples from VADRecorder
        - `segment_queue`: Queue of TranslationSegment objects consumed by TranslationWriter
        - `model`: A pre-initialized WhisperModel instance
        - `stop_event`: Shared stop signal
        - `lang_a`: BCP-47 code of the first language in the pair (e.g. "en")
        - `lang_b`: BCP-47 code of the second language in the pair (e.g. "es")
        - `ollama_host`: Ollama HTTP base URL
        - `ollama_translation_model`: Ollama model name for translation

        Optional:

        - `no_speech_threshold`:
            - Type: float
            - What: Segments with no_speech_prob above this are discarded
            - Default: 0.4

        - `show_timestamps`:
            - Type: bool
            - What: If True, prepends relative timestamps to terminal output
            - Default: True
        """
        self.audio_queue = audio_queue
        self.segment_queue = segment_queue
        self.model = model
        self.stop_event = stop_event
        self.lang_a = lang_a
        self.lang_b = lang_b
        self.ollama_host = ollama_host
        self.ollama_translation_model = ollama_translation_model
        self.no_speech_threshold = no_speech_threshold
        self.show_timestamps = show_timestamps
        self.segment_index: int = 0

    def run(self):
        """Blocking transcription + translation loop. Intended to be run in a dedicated thread."""
        name_a = _LANGUAGE_NAMES.get(self.lang_a, self.lang_a)
        name_b = _LANGUAGE_NAMES.get(self.lang_b, self.lang_b)
        initial_prompt = f"Expect only audio in {name_a} or {name_b}. Do not transcribe from other languages. Transcribe only the input audio."

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

            segments, _ = self.model.transcribe(
                audio,
                initial_prompt=initial_prompt,
            )

            words = []
            for seg in segments:
                if seg.no_speech_prob > self.no_speech_threshold:
                    continue
                words.append(seg.text.strip())

            if not words:
                continue

            text = " ".join(words)

            if self.stop_event.is_set():
                continue

            result = _detect_and_translate(
                text=text,
                lang_a=self.lang_a,
                lang_b=self.lang_b,
                host=self.ollama_host,
                model=self.ollama_translation_model,
            )

            if result is None:
                input_language = self.lang_a
                output_language = self.lang_b
                translated_text = "[translation unavailable]"
            else:
                input_language, output_language, translated_text = result

            self.segment_index += 1
            segment = TranslationSegment(
                text=text,
                translated_text=translated_text,
                input_language=input_language,
                output_language=output_language,
                start_time=start_time,
                end_time=end_time,
                index=self.segment_index,
            )

            if self.show_timestamps:
                ts = format_timestamp_txt(segment.start_time)
                src_line = f"{ts}({segment.input_language}) {segment.text}"
                tgt_line = f"{ts}({segment.output_language}) {segment.translated_text}"
            else:
                src_line = f"({segment.input_language}) {segment.text}"
                tgt_line = f"({segment.output_language}) {segment.translated_text}"
            print(src_line, flush=True)
            print(tgt_line, flush=True)

            self.segment_queue.put(segment)


# ---------------------------------------------------------------------------
# VAD recorder that pauses while TTS is speaking
# ---------------------------------------------------------------------------


class PauseableVADRecorder(VADRecorder):
    """
    VADRecorder that holds off starting a new recording window while TTS is
    speaking. Before each call to record_vad(), it blocks until speaking_event
    is cleared, preventing the microphone from picking up the speaker's output.
    """

    def __init__(self, *args, speaking_event: threading.Event, **kwargs):
        super().__init__(*args, **kwargs)
        self.speaking_event = speaking_event

    def run(self, session_start_time: float):
        _TTS_TIMEOUT_S = 30.0
        # Watchdog: if record_vad hasn't returned in this many seconds, the
        # PvRecorder.read() call is likely blocked (e.g. PulseAudio suspended
        # the source). Abandon that cycle and start fresh.
        _RECORD_WATCHDOG_S = self.max_speech_duration_s + 15.0
        # Recycle the PvRecorder every 5 s of silence so the device never
        # drifts into a stale state between utterances.
        _INACTIVITY_TIMEOUT_S = 5.0
        try:
            while not self.stop_event.is_set():
                # Block while TTS is playing so the mic doesn't hear the speaker.
                # Safety timeout: force-clear speaking_event if TTS hangs > 30 s.
                wait_start = None
                while self.speaking_event.is_set() and not self.stop_event.is_set():
                    if wait_start is None:
                        wait_start = time.time()
                    elif time.time() - wait_start > _TTS_TIMEOUT_S:
                        self.speaking_event.clear()
                        break
                    time.sleep(0.05)
                if self.stop_event.is_set():
                    break

                # Combined abort: fires when the session stops OR TTS starts.
                abort_event = threading.Event()

                def _watch(abort_event: threading.Event = abort_event) -> None:
                    while (
                        not self.stop_event.is_set()
                        and not self.speaking_event.is_set()
                    ):
                        time.sleep(0.02)
                    abort_event.set()

                threading.Thread(target=_watch, daemon=True).start()

                # Run record_vad in a daemon thread. PvRecorder.read() is a
                # blocking C call — if PulseAudio suspends the audio source the
                # read never returns and no Python event can unblock it. The
                # watchdog detects this and abandons the cycle so a fresh
                # PvRecorder is opened on the next iteration.
                _result: list = [None]
                _done = threading.Event()

                def _record(
                    abort_event: threading.Event = abort_event,
                    _result: list = _result,
                    _done: threading.Event = _done,
                ) -> None:
                    try:
                        _result[0] = self.recorder.record_vad(
                            device_index=self.device_index,
                            speech_threshold=self.speech_threshold,
                            silence_threshold=self.silence_threshold,
                            silence_frames_threshold=self.silence_frames_threshold,
                            speech_pad_frames=self.speech_pad_frames,
                            max_speech_duration_s=self.max_speech_duration_s,
                            inactivity_timeout=_INACTIVITY_TIMEOUT_S,
                            stop_event=abort_event,
                        )
                    except Exception:
                        _result[0] = []
                    finally:
                        _done.set()

                start_wall = time.time()
                threading.Thread(target=_record, daemon=True).start()

                completed = _done.wait(timeout=_RECORD_WATCHDOG_S)
                if not completed:
                    # PvRecorder.read() is hung — abandon and let the daemon
                    # thread die at process exit. Signal the watcher to stop.
                    abort_event.set()
                    continue

                frames = _result[0] or []

                if self.stop_event.is_set():
                    break
                # TTS fired during recording or no speech — discard.
                if self.speaking_event.is_set() or not frames:
                    continue
                end_wall = time.time()
                start_time = start_wall - session_start_time
                end_time = end_wall - session_start_time
                self.flush(frames, start_time, end_time)
        finally:
            pass


# ---------------------------------------------------------------------------
# Writer / output thread
# ---------------------------------------------------------------------------


class TranslationWriter(Notify):
    """
    Consumes TranslationSegment objects from segment_queue and writes bilingual
    output to disk and the terminal.

    Each segment produces two lines: one for the source language and one for
    the target language, each prefixed with a timestamp and language hint.

    Optionally speaks the translated text via Speaker.
    """

    def __init__(
        self,
        segment_queue: Queue,
        stop_event: threading.Event,
        lang_a: str,
        lang_b: str,
        speaking_event: threading.Event,
        output_format: str = "",
        output_path: str = "transcript",
        show_timestamps: bool = True,
        use_speaker: bool = True,
        speaker_voice: str = "",
    ):
        """
        Requires:

        - `segment_queue`: Queue of TranslationSegment objects from TranslatingTranscriber
        - `stop_event`: Shared stop signal
        - `lang_a`: BCP-47 code of the first language in the pair (e.g. "en")
        - `lang_b`: BCP-47 code of the second language in the pair (e.g. "es")
        - `speaking_event`: Shared event set while TTS is playing; signals PauseableVADRecorder to hold off

        Optional:

        - `output_format`:
            - Type: str
            - What: Output format(s) to write; empty string disables file output
            - Default: "" (no file output)
            - Options: "txt", "srt", "both"

        - `output_path`:
            - Type: str
            - What: Base file path (without extension)
            - Default: "transcript"

        - `show_timestamps`:
            - Type: bool
            - What: If True, prepends relative timestamps to terminal and TXT output
            - Default: True

        - `use_speaker`:
            - Type: bool
            - What: If True, speaks the translated text via TTS after each segment
            - Default: True

        - `speaker_voice`:
            - Type: str
            - What: Wave voice name for zero-shot cloning; empty string uses the
              model's built-in default voice
            - Default: ""
        """
        self.segment_queue = segment_queue
        self.stop_event = stop_event
        self.lang_a = lang_a
        self.lang_b = lang_b
        self.speaking_event = speaking_event
        self.output_format = output_format
        self.output_path = output_path
        self.show_timestamps = show_timestamps
        self.use_speaker = use_speaker
        self.speaker_voice = speaker_voice
        self.txt_file = None
        self.srt_file = None
        self.speakers: dict[str, object] = {}

    def run(self):
        """Blocking writer loop. Intended to be run in a dedicated thread."""
        if self.use_speaker:
            from spych.speaker.speaker import Speaker

            for lang_id in (self.lang_a, self.lang_b):
                try:
                    self.speakers[lang_id] = Speaker(
                        voice=self.speaker_voice,
                        backend="chatterbox_multilingual",
                        language_id=lang_id,
                    )
                except Exception as e:
                    print(
                        f"[spych] TTS for {lang_id} unavailable, "
                        f"continuing without speaker for that language: {e}",
                        flush=True,
                    )

        try:
            if self.output_format and self.output_format in ("txt", "both"):
                self.txt_file = open(
                    f"{self.output_path}.txt", "w", encoding="utf-8"
                )
            if self.output_format and self.output_format in ("srt", "both"):
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
            for speaker in self.speakers.values():
                speaker.interrupt()
                speaker.wait_for_speak()
            if self.txt_file:
                self.txt_file.flush()
                self.txt_file.close()
            if self.srt_file:
                self.srt_file.flush()
                self.srt_file.close()
            if self.speakers:
                import pygame
                try:
                    pygame.mixer.quit()
                except Exception:
                    pass

    def write_segment(self, segment: TranslationSegment):
        """Write one bilingual segment to file outputs and queue TTS."""
        if self.txt_file:
            if self.show_timestamps:
                ts = format_timestamp_txt(segment.start_time)
                src_line = f"{ts}({segment.input_language}) {segment.text}"
                tgt_line = f"{ts}({segment.output_language}) {segment.translated_text}"
            else:
                src_line = f"({segment.input_language}) {segment.text}"
                tgt_line = f"({segment.output_language}) {segment.translated_text}"
            self.txt_file.write(src_line + "\n")
            self.txt_file.write(tgt_line + "\n")
            self.txt_file.flush()

        if self.srt_file:
            srt_block = (
                f"{segment.index}\n"
                f"{format_timestamp_srt(segment.start_time)} --> "
                f"{format_timestamp_srt(segment.end_time)}\n"
                f"[{segment.input_language}] {segment.text}\n"
                f"[{segment.output_language}] {segment.translated_text}\n\n"
            )
            self.srt_file.write(srt_block)
            self.srt_file.flush()

        speaker = self.speakers.get(segment.output_language)
        if speaker and segment.translated_text != "[translation unavailable]":
            if self.stop_event.is_set():
                return
            # Serialize TTS: both speakers share pygame.mixer.music, so we must
            # wait for any in-progress playback to finish before starting the next.
            # Poll with stop_event so Ctrl+C can interrupt this wait.
            for s in self.speakers.values():
                while s.is_speaking() and not self.stop_event.is_set():
                    time.sleep(0.05)
                if self.stop_event.is_set():
                    s.interrupt()
            if self.stop_event.is_set():
                return
            self.speaking_event.set()

            def _on_complete():
                self.speaking_event.clear()

            speaker.speak_async(segment.translated_text, on_complete=_on_complete)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class SpychLiveTranslation(Notify):
    def __init__(
        self,
        lang_a: str,
        lang_b: str,
        output_format: str = "",
        output_path: str = "transcript",
        show_timestamps: bool = True,
        stop_key: str = "q",
        terminate_words: Optional[list[str]] = None,
        device_index: int = -1,
        whisper_model: str = "base",
        whisper_device: str = "auto",
        whisper_compute_type: str = "int8",
        no_speech_threshold: float = 0.4,
        speech_threshold: float = 0.5,
        silence_threshold: float = 0.35,
        silence_frames_threshold: int = 20,
        speech_pad_frames: int = 5,
        max_speech_duration_s: float = 30.0,
        ollama_host: str = "http://localhost:11434",
        ollama_translation_model: str = "llama3.2",
        use_speaker: bool = True,
        speaker_voice: str = "",
    ):
        """
        Usage:

        - Initializes a bidirectional live translation session. Either participant
          may speak in either language; Whisper transcribes and Ollama detects
          which language was spoken then translates to the other.
        - Runs continuously until stopped by keystroke, terminate word, or Ctrl+C.

        Requires:

        - `lang_a`:
            - Type: str
            - What: BCP-47 code of the first language in the pair (e.g. "en")

        - `lang_b`:
            - Type: str
            - What: BCP-47 code of the second language in the pair (e.g. "es")

        Optional:

        - `output_format`:
            - Type: str
            - What: Output format(s) to write; empty string disables file output
            - Default: "" (no file output)
            - Options: "txt", "srt", "both"

        - `output_path`:
            - Type: str
            - What: Base output file path without extension
            - Default: "transcript"

        - `show_timestamps`:
            - Type: bool
            - What: If True, prepends relative [HH:MM:SS] timestamps to each line
            - Default: True

        - `stop_key`:
            - Type: str
            - What: The key (followed by Enter) the user types to stop recording
            - Default: "q"

        - `terminate_words`:
            - Type: list[str] | None
            - What: Words that, if detected in the transcript, immediately stop the session
            - Default: None

        - `device_index`:
            - Type: int
            - What: Microphone device index; -1 uses the system default
            - Default: -1

        - `whisper_model`:
            - Type: str
            - What: faster-whisper model name; `.en` suffix is stripped automatically
              when either language is not English
            - Default: "base"

        - `whisper_device`:
            - Type: str
            - What: Device for whisper inference
            - Default: "auto"
            - Options: "auto", "cpu", "cuda"
            - Note: "auto" selects "cuda" when Python <=3.13 and a CUDA device is
              available, otherwise falls back to "cpu". "cuda" requires
              nvidia-cublas-cu12 and nvidia-cudnn-cu12 (pip).

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

        - `silence_threshold`:
            - Type: float (0.0–1.0)
            - What: Silero probability below which a frame is considered silence
            - Default: 0.35

        - `silence_frames_threshold`:
            - Type: int
            - What: Consecutive silent frames required to close a speech segment
            - Default: 20

        - `speech_pad_frames`:
            - Type: int
            - What: Pre-roll frames and onset confirmation count
            - Default: 5

        - `max_speech_duration_s`:
            - Type: float
            - What: Hard cap on a single speech segment in seconds
            - Default: 30.0

        - `ollama_host`:
            - Type: str
            - What: Ollama HTTP base URL for translation requests
            - Default: "http://localhost:11434"

        - `ollama_translation_model`:
            - Type: str
            - What: Ollama model name used for translation
            - Default: "llama3.2"

        - `use_speaker`:
            - Type: bool
            - What: If True, speaks each translated segment aloud via TTS
            - Default: True

        - `speaker_voice`:
            - Type: str
            - What: Wave voice name for zero-shot cloning; empty string uses the
              model's built-in default voice
            - Default: ""
        """
        self.lang_a = lang_a
        self.lang_b = lang_b
        self.output_format = output_format
        self.output_path = output_path
        self.show_timestamps = show_timestamps
        self.stop_key = stop_key
        self.terminate_words = (
            [w.lower() for w in terminate_words] if terminate_words else []
        )
        self.device_index = device_index
        self.no_speech_threshold = no_speech_threshold
        self.speech_threshold = speech_threshold
        self.silence_threshold = silence_threshold
        self.silence_frames_threshold = silence_frames_threshold
        self.speech_pad_frames = speech_pad_frames
        self.max_speech_duration_s = max_speech_duration_s
        self.ollama_host = ollama_host
        self.ollama_translation_model = ollama_translation_model
        self.use_speaker = use_speaker
        self.speaker_voice = speaker_voice

        resolved_model = _select_whisper_model(whisper_model, lang_a, lang_b)
        self.model = WhisperModel(
            resolved_model,
            device=resolve_whisper_device(whisper_device),
            compute_type=whisper_compute_type,
        )

        self.stop_event = threading.Event()
        self.speaking_event = threading.Event()
        self.audio_queue: Queue = Queue()
        self.segment_queue: Queue = Queue()

    def start(self):
        """
        Usage:

        - Starts the live transcription + translation session and blocks until
          the user stops it via the configured stop key or a terminate word
        - Prints a startup message indicating how to stop the session

        Notes:

        - Thread startup order: keystroke listener → recorder → transcriber → writer
        - SIGINT (Ctrl+C) is caught and redirected to the same graceful stop path
        """
        original_sigint = signal.getsignal(signal.SIGINT)

        def handle_sigint(sig, frame):
            print(
                "\n[spych] Interrupt received. "
                "Finishing current segment and shutting down...",
                flush=True,
            )
            self.stop_event.set()
            signal.signal(signal.SIGINT, original_sigint)

        signal.signal(signal.SIGINT, handle_sigint)

        stop_instructions = [f"Press '{self.stop_key}' + Enter"]
        if self.terminate_words:
            words_display = ", ".join(f'"{w}"' for w in self.terminate_words)
            stop_instructions.append(f"say {words_display}")
        print(
            f"[spych] Live translation started "
            f"({self.lang_a} ↔ {self.lang_b}). "
            f"To stop: {' or '.join(stop_instructions)}.",
            flush=True,
        )

        ks_listener = KeystrokeListener(self.stop_event, self.stop_key)
        ks_thread = threading.Thread(target=ks_listener.run, daemon=True)
        ks_thread.start()

        session_start = time.time()

        recorder = PauseableVADRecorder(
            audio_queue=self.audio_queue,
            stop_event=self.stop_event,
            device_index=self.device_index,
            speech_threshold=self.speech_threshold,
            silence_threshold=self.silence_threshold,
            silence_frames_threshold=self.silence_frames_threshold,
            speech_pad_frames=self.speech_pad_frames,
            max_speech_duration_s=self.max_speech_duration_s,
            speaking_event=self.speaking_event,
        )
        rec_thread = threading.Thread(
            target=recorder.run, args=(session_start,), daemon=False
        )

        transcriber = TranslatingTranscriber(
            audio_queue=self.audio_queue,
            segment_queue=self.segment_queue,
            model=self.model,
            stop_event=self.stop_event,
            lang_a=self.lang_a,
            lang_b=self.lang_b,
            ollama_host=self.ollama_host,
            ollama_translation_model=self.ollama_translation_model,
            no_speech_threshold=self.no_speech_threshold,
            show_timestamps=self.show_timestamps,
        )
        trans_thread = threading.Thread(
            target=self.transcribe_and_check,
            args=(transcriber,),
            daemon=False,
        )

        writer = TranslationWriter(
            segment_queue=self.segment_queue,
            stop_event=self.stop_event,
            lang_a=self.lang_a,
            lang_b=self.lang_b,
            speaking_event=self.speaking_event,
            output_format=self.output_format,
            output_path=self.output_path,
            show_timestamps=self.show_timestamps,
            use_speaker=self.use_speaker,
            speaker_voice=self.speaker_voice,
        )
        write_thread = threading.Thread(target=writer.run, daemon=False)

        write_thread.start()
        trans_thread.start()
        rec_thread.start()

        rec_thread.join()

        self.audio_queue.put(None)
        trans_thread.join()

        self.segment_queue.put(None)
        write_thread.join()

        signal.signal(signal.SIGINT, original_sigint)
        if self.output_format:
            print(
                f"[spych] Session complete. Output saved to: {self.output_path}.*",
                flush=True,
            )
        else:
            print("[spych] Session complete.", flush=True)

    def transcribe_and_check(self, transcriber: TranslatingTranscriber):
        """
        Runs transcriber.run() and intercepts every segment put onto segment_queue
        to check for terminate words.
        """
        original_put = self.segment_queue.put

        def checked_put(segment):
            original_put(segment)
            if not self.terminate_words or not isinstance(segment, TranslationSegment):
                return
            text_lower = segment.text.lower()
            for word in self.terminate_words:
                if word in text_lower:
                    print(
                        f'\n[spych] Terminate word "{word}" detected. '
                        "Finishing and shutting down...",
                        flush=True,
                    )
                    self.stop_event.set()
                    return

        self.segment_queue.put = checked_put
        try:
            transcriber.run()
        finally:
            self.segment_queue.put = original_put
