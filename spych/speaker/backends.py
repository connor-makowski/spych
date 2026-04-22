import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import glob
import torch
from huggingface_hub import constants as _hf_constants
from spych.utils import get_cache_dir
from spych.voice_manager import get_wave_voice, get_pt_voice

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

KOKORO_VOICES = [
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "em_santa",
    "ff_siwis",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex",
    "pm_santa",
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
]


def _get_cached_wave_voices() -> list:
    cache_wave_dir = os.path.join(get_cache_dir(folder="voices"), "wave")
    if not os.path.exists(cache_wave_dir):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(cache_wave_dir)
        if f.endswith(".wav")
    )


class BaseBackend:
    def __init__(self, speaker: "Speaker", voice: str):
        self.speaker = speaker
        self.voice = voice

    def speak(self, text: str):
        raise NotImplementedError


class ChatterboxBackend(BaseBackend):
    def __init__(self, speaker: "Speaker", voice: str):
        super().__init__(speaker, voice)
        from spych.speaker.chatterbox import SpychChatterboxTTS

        self.ChatterboxTTS = SpychChatterboxTTS

        if not voice:
            local_voices = _get_cached_wave_voices()
            local_hint = (
                f"  Locally cached voices: {', '.join(local_voices)}\n"
                if local_voices
                else "  No locally cached voices found.\n"
            )
            raise ValueError(
                "[spych] Chatterbox requires a voice sample (.wav) for voice cloning.\n"
                "  Browse included voices: https://github.com/connor-makowski/spych/tree/main/voices/wave\n"
                + local_hint
                + "  Record a custom voice: from spych.voice_manager import profile_my_voice; profile_my_voice('my_voice')\n"
                "  Then pass to Speaker: Speaker(voice='my_voice')"
            )

        try:
            self.audio_prompt_path = get_wave_voice(voice)
        except FileNotFoundError:
            local_voices = _get_cached_wave_voices()
            local_hint = (
                f"  Locally cached voices: {', '.join(local_voices)}\n"
                if local_voices
                else "  No locally cached voices found.\n"
            )
            raise ValueError(
                f"[spych] Chatterbox voice '{voice}' not found.\n"
                "  Browse included voices: https://github.com/connor-makowski/spych/tree/main/voices/wave\n"
                + local_hint
                + "  Record a custom voice: from spych.voice_manager import profile_my_voice; profile_my_voice('my_voice')\n"
                "  Then pass to Speaker: Speaker(voice='my_voice')"
            )

        self.load_model()
        self.model.prepare_conditionals(self.audio_prompt_path)
        self.sample_rate = self.model.sr

    def load_model(self) -> None:
        self.model = self.ChatterboxTTS.from_pretrained(device=torch_device)

    def speak(self, text: str):
        audio = self.model.generate(text)
        if self.speaker.interrupted.is_set():
            return
        self.speaker.play_pcm_array(audio.flatten(), self.sample_rate)


class KokoroBackend(BaseBackend):
    REPO_ID = "hexgrad/Kokoro-82M"

    def __init__(self, speaker: "Speaker", voice: str):
        super().__init__(speaker, voice)
        from kokoro import KPipeline

        self.KPipeline = KPipeline

        if not self.voice:
            self.voice = "af_heart"

        if self.voice not in KOKORO_VOICES:
            raise ValueError(
                f"[spych] Kokoro voice '{self.voice}' not recognized.\n"
                "  Browse available voices: https://github.com/connor-makowski/spych/tree/main/voices/pt\n"
                f"  Available voices: {', '.join(sorted(KOKORO_VOICES))}"
            )

        self.voice_path = get_pt_voice(self.voice)
        self.load_pipeline()
        self.sample_rate = 24000

    def load_pipeline(self) -> None:
        lang_code = "b" if self.voice.startswith(("bm_", "bf_")) else "a"

        model_cache_dir = get_cache_dir(folder="models")
        model_cache = os.path.join(
            model_cache_dir,
            "models--" + self.REPO_ID.replace("/", "--"),
        )
        is_cached = os.path.exists(model_cache) and bool(
            glob.glob(os.path.join(model_cache, "snapshots", "*"))
        )

        prev_HF_HUB_CACHE = _hf_constants.HF_HUB_CACHE
        prev_HF_HUB_OFFLINE = _hf_constants.HF_HUB_OFFLINE
        _hf_constants.HF_HUB_CACHE = model_cache_dir
        if is_cached:
            _hf_constants.HF_HUB_OFFLINE = True
        try:
            self.pipeline = self.KPipeline(lang_code=lang_code, repo_id=self.REPO_ID)
            self.pipeline.load_voice(self.voice_path)
        finally:
            _hf_constants.HF_HUB_CACHE = prev_HF_HUB_CACHE
            _hf_constants.HF_HUB_OFFLINE = prev_HF_HUB_OFFLINE

    def speak(self, text: str):
        for _, _, audio in self.pipeline(text, voice=self.voice_path):
            if self.speaker.interrupted.is_set():
                return
            self.speaker.play_pcm_array(audio.numpy(), self.sample_rate)


def get_backend(speaker: "Speaker", voice: str) -> BaseBackend | None:
    """Helper to get the first available backend, checked once at init."""

    # try:
    return ChatterboxBackend(speaker, voice)
    # except (ImportError, Exception):
    #     pass
    try:
        return KokoroBackend(speaker, voice)
    except (ImportError, Exception):
        pass
    raise NotImplementedError(
        "No TTS backend available. To use a speaker, please install one with `pip install spych[chatterbox]` or `pip install spych[kokoro]`."
    )
