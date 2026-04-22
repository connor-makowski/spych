import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import glob
import torch
import requests
from huggingface_hub import constants as _hf_constants
from spych.utils import get_cache_dir

class BaseBackend:
    def __init__(self, speaker: "Speaker", voice: str):
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
        voice_cache_path = os.path.join(get_cache_dir(folder='voices'), voice_filename)
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
    def __init__(self, speaker: "Speaker", voice: str):
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
        self.speaker.play_pcm_array(audio, self.sample_rate)


class KokoroBackend(BaseBackend):
    def __init__(self, speaker: "Speaker", voice: str):
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
            self.speaker.play_pcm_array(audio.numpy(), self.sample_rate)


def get_backend(speaker: "Speaker", voice: str) -> BaseBackend | None:
    """Helper to get the first available backend, checked once at init."""
    try:
        return ChatterboxBackend(speaker, voice)
    except (ImportError, Exception):
        pass
    try:
        return KokoroBackend(speaker, voice)
    except (ImportError, Exception):
        pass
    raise NotImplementedError("No TTS backend available. To use a speaker, please install one with `pip install spych[chatterbox]` or `pip install spych[kokoro]`.")