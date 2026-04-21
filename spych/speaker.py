import glob
import io
import os
import threading
import time
import warnings
import wave
import numpy as np

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import pygame
from huggingface_hub import constants as _hf_constants
from kokoro import KPipeline


class Speaker:
    def __init__(self, voice: str = "af_heart") -> None:
        """
        Usage:

        - Initializes a Speaker that converts text to speech using kokoro neural
          TTS. The kokoro model and voice are loaded eagerly at construction time
          so that all subsequent speak() calls run fully offline with no
          initialization overhead. The kokoro model (~82MB) is downloaded on
          first use and cached locally; all subsequent runs are fully offline.
          A single KModel instance is shared across all language pipelines on
          this Speaker instance.

        Optional:

        - `voice`:
            - Type: str
            - What: A kokoro voice ID to use for all speak() calls
            - Default: "af_heart"
            - Note: American English voices use prefix `am_` or `af_`; British
              English use `bm_` or `bf_`. The language is detected automatically
              from the prefix.

        Notes:

        - Available voices and grades:
          https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md

        - Recommended English voices:

          American English (prefix `am_` / `af_`, lang detected automatically):

          | Voice      | Gender | Grade |
          |------------|--------|-------|
          | af_heart   | F      | A     |
          | af_bella   | F      | A-    |
          | af_nicole  | F      | B-    |
          | af_aoede   | F      | C+    |
          | af_kore    | F      | C+    |
          | af_sarah   | F      | C+    |
          | am_michael | M      | C+    |
          | am_fenrir  | M      | C+    |
          | am_puck    | M      | C+    |

          British English (prefix `bm_` / `bf_`, lang detected automatically):

          | Voice       | Gender | Grade |
          |-------------|--------|-------|
          | bf_emma     | F      | B-    |
          | bf_isabella | F      | C     |
          | bm_george   | M      | C     |
        """
        pygame.mixer.init()
        self.REPO_ID = "hexgrad/Kokoro-82M"
        self.voice = voice
        self.model = None
        self._interrupted = threading.Event()
        self.load_pipeline(voice)

    def load_pipeline(self, voice: str) -> None:
        lang_code = "b" if voice.startswith(("bm_", "bf_")) else "a"
        model_cache = os.path.join(
            _hf_constants.HF_HUB_CACHE,
            "models--" + self.REPO_ID.replace("/", "--"),
        )
        voice_pattern = os.path.join(model_cache, "**", f"voices/{voice}.pt")
        is_cached = bool(glob.glob(voice_pattern, recursive=True))
        prev_HF_HUB_OFFLINE = _hf_constants.HF_HUB_OFFLINE
        if is_cached:
            _hf_constants.HF_HUB_OFFLINE = True
        try:
            self.pipeline = KPipeline(lang_code=lang_code, repo_id=self.REPO_ID)
            self.pipeline.load_voice(voice)
        finally:
            _hf_constants.HF_HUB_OFFLINE = prev_HF_HUB_OFFLINE

    def speak(self, text: str) -> None:
        """
        Usage:

        - Converts text to speech and plays it through the system audio output.
          Blocks until playback is complete. The model and voice are already
          loaded from __init__, so this call is fully offline and incurs no
          initialization overhead.

        Requires:

        - `text`:
            - Type: str
            - What: The text to convert to speech and play aloud

        Notes:

        - Playback is blocking — this method returns only after all audio has
          finished playing or `interrupt()` is called.
        - Calling `interrupt()` stops playback immediately and causes this
          method to return early.
        """
        self._interrupted.clear()
        for _, _, audio in self.pipeline(text, voice=self.voice):
            if self._interrupted.is_set():
                return
            pcm = (audio.numpy() * 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(pcm.tobytes())
            buf.seek(0)
            pygame.mixer.music.load(buf)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if self._interrupted.is_set():
                    pygame.mixer.music.stop()
                    return
                time.sleep(0.05)

    def interrupt(self) -> None:
        """
        Usage:

        - Stops any in-progress `speak()` call immediately. Safe to call from
          any thread at any time. If `speak()` is not running, this is a no-op
          — the flag is cleared at the start of the next `speak()` call.
        """
        self._interrupted.set()
        pygame.mixer.music.stop()
