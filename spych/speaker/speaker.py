import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import io
import threading
import time
import wave
import numpy as np
import pygame
from spych.speaker.backends import get_backend

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

        self.backend = get_backend(speaker=self, voice=voice)

    def play_pcm_array(self, audio: np.ndarray, sample_rate: int) -> None:
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
