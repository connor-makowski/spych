import json
import traceback, sys, os, wave, shutil, threading, queue, subprocess
from pvrecorder import PvRecorder
import numpy as np
from typing import Any, Union, Optional
from silero_vad import load_silero_vad
import torch


def resolve_cmd(name: str) -> str:
    """
    Usage:

    - Resolves a CLI command name to its full executable path using shutil.which.
      On Windows, Node.js CLI tools are installed as .cmd wrappers (e.g. gemini.cmd).
      subprocess.Popen with shell=False cannot find them by bare name, but shutil.which
      respects PATHEXT and returns the full path including the extension.

    Requires:

    - `name`:
        - Type: str
        - What: The CLI command name to resolve (e.g. "gemini", "claude")

    Returns:

    - `path`:
        - Type: str
        - What: The full path to the executable if found, otherwise the original name
    """
    return shutil.which(name) or name


def get_cache_dir(folder="voices") -> str:
    """Returns the path to the project's voice cache directory."""
    if os.name == "nt":  # Windows
        base_dir = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    elif sys.platform == "darwin":  # macOS
        base_dir = os.path.expanduser("~/Library/Caches")
    else:  # Linux/Unix
        base_dir = os.environ.get(
            "XDG_CACHE_HOME", os.path.expanduser("~/.cache")
        )

    path = os.path.join(base_dir, "spych", folder)
    os.makedirs(path, exist_ok=True)
    return path


def get_setting(key: str, default: Any = None) -> Any:
    """Returns a setting from the cache."""
    path = os.path.join(get_cache_dir("settings"), "settings.json")
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        try:
            settings = json.load(f)
        except json.JSONDecodeError:
            return default
    return settings.get(key, default)


def set_setting(key: str, value: Any) -> None:
    """Sets a setting in the cache."""
    folder = get_cache_dir("settings")
    path = os.path.join(folder, "settings.json")
    settings = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                settings = json.load(f)
            except json.JSONDecodeError:
                settings = {}
    settings[key] = value
    with open(path, "w") as f:
        json.dump(settings, f, indent=4)


def get_user(name: str) -> Optional[dict]:
    """Returns a user profile from the cache."""
    path = os.path.join(get_cache_dir("users"), f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None


def get_all_users() -> list[str]:
    """Returns a list of all user profile names."""
    folder = get_cache_dir("users")
    return [f[:-5] for f in os.listdir(folder) if f.endswith(".json")]


def set_user(name: str, data: dict) -> None:
    """Sets a user profile in the cache."""
    folder = get_cache_dir("users")
    path = os.path.join(folder, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def get_default_user() -> Optional[str]:
    """Returns the default user name."""
    return get_setting("default_user")


def set_default_user(name: Optional[str]) -> None:
    """Sets the default user name."""
    set_setting("default_user", name)


def save_wav(path: str, buffer: list[int], sample_rate: int = 16000) -> None:
    """Saves a raw PCM buffer to a .wav file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(np.array(buffer, dtype=np.int16).tobytes())


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
        inactivity_timeout: Optional[float] = None,
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

        Optional:

        - `inactivity_timeout`:
            - Type: float | None
            - What: Seconds to wait for speech onset before returning an empty buffer
            - Default: None (wait indefinitely)

        Returns:

        - `buffer`:
            - Type: list[int]
            - What: Flat list of raw int16 PCM samples representing the captured utterance
        """
        self.ensure_vad()
        frame_ms = self.frame_length / self.sample_rate * 1000
        max_speech_frames = int(max_speech_duration_s * 1000 / frame_ms)
        inactivity_frames = (
            int(inactivity_timeout * 1000 / frame_ms)
            if inactivity_timeout
            else None
        )

        recorder = PvRecorder(
            device_index=device_index, frame_length=self.frame_length
        )

        speech_buffer: list[list[int]] = []
        pre_roll: list[list[int]] = []
        in_speech: bool = False
        voiced_frame_count: int = 0
        silent_frame_count: int = 0
        total_frames: int = 0

        try:
            recorder.start()
            while stop_event is None or not stop_event.is_set():
                frame = recorder.read()
                total_frames += 1
                tensor = (
                    torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
                    / 32768.0
                )

                with torch.no_grad():
                    speech_prob = self._vad_model(
                        tensor, self.sample_rate
                    ).item()

                if not in_speech:
                    if (
                        inactivity_frames is not None
                        and total_frames >= inactivity_frames
                    ):
                        break

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


def get_response_style(style: Optional[str]) -> str:
    """
    Usage:

    - Maps a high-level style descriptor to a prompt suffix that instructs the
      model to stylize its response accordingly.

    Requires:

    - `style`:
        - Type: str
        - What: A high-level style descriptor (e.g. "concise", "detailed", "humorous")

    Returns:

    - `style_prompt`:
        - Type: str
        - What: A prompt suffix that can be appended to the user input to elicit
          the desired response style from the model
    """
    styles: dict[str, str] = {
        "assistant": "Respond as a helpful and precise assistant. Be concise and informative.",
        "concise": "Respond with a focus on key points and being direct.",
        "friendly": "Use a friendly and approachable tone. Use simple language and be concise.",
        "military": "Respond in military brevity style. Be concise and direct, using short sentences and clear language.",
        "five_year_old": "Explain this like I'm 5 years old. Use simple words, be short and friendly.",
        "fast": "Keep your responses as fast as reasonably possible. Be direct and concise.",
        "pirate": "Style your responses in pirate speak. Keep it short and colorful. Arrr!",
        "news_anchor": "Your speaking style should match how a professional TV news anchor would say it.",
        "haiku": "Respond in the form of a haiku (5-7-5 syllables). Be concise and poetic.",
        "shakespearean": "Respond in Shakespearean English. Brief and poetic.",
        "robot": "Respond as a robot speaking. Monotone, literal, and short.",
        "caveman": "Respond in the style of a caveman. Use very simple language, short sentences, and be direct.",
        "yoda": "Respond in the style of Yoda from Star Wars. Use inverted sentence structure and be concise.",
        "jarvis": (
            "Respond as J.A.R.V.I.S. (Just A Rather Very Intelligent System), "
            "Tony Stark's AI assistant from Iron Man. Be precise, efficient, and "
            "professionally deferential with understated dry wit. Address the user "
            "as 'sir' or 'ma'am' (sir by default unless their gender is specified)."
            "Keep responses brief and to the point — never verbose."
        ),
    }
    if not style:
        return ""
    return styles.get(style.lower(), style)


PERSONALITIES: dict[str, dict] = {
    "assistant": {
        "name": "Assistant",
        "wake_words": ["assistant", "helper", "computer"],
        "speaker_voice": "af_heart",
        "use_speaker": True,
        "response_style": "assistant",
    },
    "friend": {
        "name": "Friend",
        "wake_words": ["friend", "buddy", "pal"],
        "speaker_voice": "af_amy",
        "use_speaker": True,
        "response_style": "friendly",
    },
    "jarvis": {
        "name": "JARVIS",
        "wake_words": ["jarvis", "jarves", "jargus", "jervis"],
        "speaker_voice": "bm_george",
        "use_speaker": True,
        "response_style": "jarvis",
    },
    "pirate": {
        "name": "Blackbeard",
        "wake_words": ["blackbeard", "pirate", "ahoy"],
        "speaker_voice": "am_michael",
        "use_speaker": True,
        "response_style": "pirate",
    },

    "news_anchor": {
        "name": "Bella the News Anchor",
        "wake_words": ["bella", "news anchor", "anchor"],
        "speaker_voice": "af_bella",
        "use_speaker": True,
        "response_style": "news_anchor",
    },
    "robot": {
        "name": "Rob the Robot",
        "wake_words": ["rob", "robot"],
        "speaker_voice": "am_adam",
        "use_speaker": True,
        "response_style": "robot",
    },
    "caveman": {
        "name": "Ur the Caveman",
        "wake_words": ["er", "ur", "caveman", "cave man"],
        "speaker_voice": "am_onyx",
        "use_speaker": True,
        "response_style": "caveman",
    },
}


def get_personality(name: str) -> dict:
    """
    Usage:

    - Returns a personality preset dict containing default kwargs (name,
      wake_words, speaker_voice, use_speaker, response_style) for the named
      personality. Raises ValueError for unknown names.

    Requires:

    - `name`:
        - Type: str
        - What: The personality preset name (e.g. "jarvis")

    Returns:

    - `preset`:
        - Type: dict
        - What: A dict of agent kwargs to apply as defaults.

    Notes:

    - Raises ValueError if the name is not found in PERSONALITIES.
    """
    key = name.lower()
    if key not in PERSONALITIES:
        valid = ", ".join(sorted(PERSONALITIES))
        raise ValueError(f"Unknown personality {name!r}. Valid options: {valid}")
    return dict(PERSONALITIES[key])


class StreamSubprocess:
    """
    Usage:

    - A context manager/iterator wrapper for subprocess.Popen that drains
      stderr in the background and provides a thread-safe iterator over
      stdout via a queue. This prevents deadlocks on Windows where a full
      stderr pipe can block the process.

    Example:

    ```python
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in StreamSubprocess(proc):
        print(line)
    ```
    """

    def __init__(self, proc: subprocess.Popen) -> None:
        self.proc = proc
        self.q = queue.Queue()
        self._sentinel = object()

        self._stdout_thread = threading.Thread(
            target=self._reader, args=(self.proc.stdout,), daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._drain, args=(self.proc.stderr,), daemon=True
        )

        self._stdout_thread.start()
        self._stderr_thread.start()

    def _reader(self, pipe):
        try:
            for line in pipe:
                self.q.put(line)
        except Exception:
            pass
        finally:
            self.q.put(self._sentinel)

    def _drain(self, pipe):
        try:
            for _ in pipe:
                pass
        except Exception:
            pass
        finally:
            if pipe:
                pipe.close()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.q.get()
        if item is self._sentinel:
            raise StopIteration
        return item

    def get(self, timeout: Optional[float] = None):
        """
        Get the next line from stdout with an optional timeout.
        Returns None if the process has exited or the timeout is reached.
        """
        try:
            item = self.q.get(timeout=timeout)
            if item is self._sentinel:
                return None
            return item
        except queue.Empty:
            return None
