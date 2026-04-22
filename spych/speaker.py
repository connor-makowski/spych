import os
import warnings

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import glob
import io
import threading
import time
import wave
import numpy as np
import torch
import pygame
import requests
from huggingface_hub import constants as _hf_constants
from spych.utils import get_voice_cache_dir


class BaseBackend:
    def __init__(self, speaker: Speaker, voice: str):
        self.speaker = speaker
        self.voice = voice

    def speak(self, text: str):
        raise NotImplementedError
    
    def resolve_voice_path(self, voice: str) -> str | None:
        """Resolves a voice name or path to a .wav file path, checked once at init."""
        if not voice:
            return None
        if os.path.isfile(voice):
            return voice

        voice_filename = voice if voice.endswith(".wav") else f"{voice}.wav"
        # Check if in cached voices directory
        voice_cache_path = os.path.join(get_voice_cache_dir(), voice_filename)
        if os.path.isfile(voice_cache_path):
            return voice_cache_path

        try:
            url = f"https://raw.githubusercontent.com/connor-makowski/spych/main/voices/{voice_filename}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(voice_cache_path, "wb") as f:
                    f.write(response.content)
                return voice_cache_path
        except Exception:
            pass
        return None


class ChatterboxBackend(BaseBackend):
    def __init__(self, speaker: Speaker, voice: str):
        super().__init__(speaker, voice)
        from chatterbox.tts import ChatterboxTTS

        self.REPO_ID = "resemble-ai/chatterbox-english"
        self.audio_prompt_path = self.resolve_voice_path(voice)
        self.load_model()
        if self.audio_prompt_path:
            self.model.prepare_conditionals(self.audio_prompt_path)
        self.sample_rate = self.model.sr

    def load_model(self) -> None:
        from chatterbox.tts import ChatterboxTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_cache = os.path.join(
            _hf_constants.HF_HUB_CACHE,
            "models--" + self.REPO_ID.replace("/", "--"),
        )
        is_cached = os.path.exists(model_cache) and bool(
            glob.glob(os.path.join(model_cache, "snapshots", "*"))
        )

        prev_HF_HUB_OFFLINE = _hf_constants.HF_HUB_OFFLINE
        if is_cached:
            _hf_constants.HF_HUB_OFFLINE = True
        try:
            self.model = ChatterboxTTS.from_pretrained(device=device)
        finally:
            _hf_constants.HF_HUB_OFFLINE = prev_HF_HUB_OFFLINE

    def speak(self, text: str):
        wav = self.model.generate(text)
        if self.speaker.interrupted.is_set():
            return
        audio = wav.cpu().numpy().flatten()
        self.speaker._play_pcm(audio, self.sample_rate)


class KokoroBackend(BaseBackend):
    def __init__(self, speaker: Speaker, voice: str):
        super().__init__(speaker, voice)
        from kokoro import KPipeline

        self.REPO_ID = "hexgrad/Kokoro-82M"
        if not self.voice:
            self.voice = "af_heart"
        self.load_pipeline()
        self.sample_rate = 24000

    def load_pipeline(self) -> None:
        from kokoro import KPipeline

        lang_code = "b" if self.voice.startswith(("bm_", "bf_")) else "a"
        model_cache = os.path.join(
            _hf_constants.HF_HUB_CACHE,
            "models--" + self.REPO_ID.replace("/", "--"),
        )
        voice_pattern = os.path.join(
            model_cache, "**", f"voices/{self.voice}.pt"
        )
        is_cached = bool(glob.glob(voice_pattern, recursive=True))

        prev_HF_HUB_OFFLINE = _hf_constants.HF_HUB_OFFLINE
        if is_cached:
            _hf_constants.HF_HUB_OFFLINE = True
        try:
            self.pipeline = KPipeline(lang_code=lang_code, repo_id=self.REPO_ID)
            self.pipeline.load_voice(self.voice)
        finally:
            _hf_constants.HF_HUB_OFFLINE = prev_HF_HUB_OFFLINE

    def speak(self, text: str):
        for _, _, audio in self.pipeline(text, voice=self.voice):
            if self.speaker.interrupted.is_set():
                return
            self.speaker._play_pcm(audio.numpy(), self.sample_rate)


class DummyBackend(BaseBackend):
    def speak(self, text: str):
        print(f"[Speaker (OFFLINE)]: {text}")

class Speaker:
    def __init__(self, voice: str = "") -> None:
        """
        Usage:

        - Initializes a Speaker that converts text to speech.
          Attempts to use Chatterbox (high quality, slower) first.
          Falls back to Kokoro (lightweight) if Chatterbox is not available.
          Falls back to a silent (print-only) backend if neither is available.

        Optional:

        - `voice`:
            - Type: str
            - What: A voice name (e.g., "af_heart") or a path to a .wav file.
            - Default: "" (uses the backend's default built-in voice)
        """
        pygame.mixer.init()
        self.voice = voice
        self.interrupted = threading.Event()
        self.speaking_complete = threading.Event()
        self.speaking_complete.set()

        self.backend = None

        if not self.backend:
            try:
                self.backend = ChatterboxBackend(self, voice)
            except (ImportError, Exception):
                pass
        if not self.backend:
            try:
                self.backend = KokoroBackend(self, voice)
            except (ImportError, Exception):
                pass
        if not self.backend:
            self.backend = DummyBackend(self, voice)

    def _play_pcm(self, audio: np.ndarray, sample_rate: int) -> None:
        """Helper to play a numpy PCM array via pygame."""
        pcm = (audio.flatten() * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())

        buf.seek(0)
        pygame.mixer.music.load(buf)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            if self.interrupted.is_set():
                pygame.mixer.music.stop()
                return
            time.sleep(0.05)

    def speak(self, text: str) -> None:
        """
        Usage:

        - Converts text to speech and plays it through the system audio output.
          Blocks until playback is complete.
        """
        self.interrupted.clear()
        self.backend.speak(text)

    def speak_async(self, text: str) -> None:
        """
        Usage:

        - Converts text to speech and plays it through the system audio output in
          a background thread. Does not block the main thread.
        """

        def run_speak():
            try:
                if not self.interrupted.is_set():
                    self.speak(text)
            except Exception:
                pass
            finally:
                self.speaking_complete.set()

        self.speaking_complete.clear()
        self.interrupted.clear()
        threading.Thread(target=run_speak, daemon=True).start()

    def wait_for_speak(self) -> None:
        """Blocks until the current speak call is complete."""
        self.speaking_complete.wait()

    def is_speaking(self) -> bool:
        """Returns True if a speak call is in progress."""
        return not self.speaking_complete.is_set()

    def interrupt(self) -> None:
        """Stops any in-progress speak call immediately."""
        self.interrupted.set()
        pygame.mixer.music.stop()
