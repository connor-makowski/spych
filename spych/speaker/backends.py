import logging
import os
import warnings
from huggingface_hub import hf_hub_download
from spych.utils import get_cache_dir
from spych.voice_manager import get_wave_voice, get_pt_voice

# Suppress Torch and HuggingFace Hub warnings that clutter the CLI
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    """
    Usage:

    - Abstract base class for TTS backends.
    - Subclass and implement `speak(text)` to add a new backend.
    """

    def __init__(self, speaker: "Speaker", voice: str):
        self.speaker = speaker
        self.voice = voice

    def speak(self, text: str):
        raise NotImplementedError


class ChatterboxBackend(BaseBackend):
    """
    Usage:

    - TTS backend using Chatterbox Turbo for high-quality zero-shot voice cloning.
    - Requires a voice name (e.g. "af_heart") or path to a .wav file for cloning.
    - Wave voice files are auto-downloaded from the spych voices repo on first use
      and cached to the spych voices cache (~/.cache/spych/voices/wave/).
    - Browse included voices: https://github.com/connor-makowski/spych/tree/main/voices/wave
    - Model weights are downloaded from HuggingFace on first use and cached to
      ~/.cache/spych/models/.
    - Record a custom voice with: spych profile_my_voice --name my_voice
    """

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
        self.model = self.ChatterboxTTS.from_pretrained(device=device)

    def speak(self, text: str):
        audio = self.model.generate(text)
        if self.speaker.interrupted.is_set():
            return
        self.speaker.play_pcm_array(audio.flatten(), self.sample_rate)


class KokoroBackend(BaseBackend):
    """
    Usage:

    - Lightweight TTS backend using the Kokoro neural TTS model (~82 MB).
    - Requires a Kokoro voice name; see KOKORO_VOICES for the full list.
    - Voice .pt files are auto-downloaded from the spych voices repo on first use
      and cached to the spych voices cache (~/.cache/spych/voices/pt/).
    - Browse available voices: https://github.com/connor-makowski/spych/tree/main/voices/pt
    - Model weights are downloaded from HuggingFace on first use and cached to
      ~/.cache/spych/models/kokoro-82M/.

    Notes:

    - Not compatible with Python 3.14+.
    """

    REPO_ID = "hexgrad/Kokoro-82M"

    def __init__(self, speaker: "Speaker", voice: str):
        super().__init__(speaker, voice)
        from kokoro import KPipeline, KModel

        self.KPipeline = KPipeline
        self.KModel = KModel

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
        kokoro_dir = os.path.join(model_cache_dir, "kokoro")
        os.makedirs(kokoro_dir, exist_ok=True)

        for fname in ["config.json", "kokoro-v1_0.pth"]:
            if not os.path.isfile(os.path.join(kokoro_dir, fname)):
                hf_hub_download(
                    repo_id=self.REPO_ID, filename=fname, local_dir=kokoro_dir
                )

        kmodel = (
            self.KModel(
                repo_id=self.REPO_ID,
                config=os.path.join(kokoro_dir, "config.json"),
                model=os.path.join(kokoro_dir, "kokoro-v1_0.pth"),
            )
            .to(device)
            .eval()
        )

        self.pipeline = self.KPipeline(
            lang_code=lang_code, repo_id=self.REPO_ID, model=kmodel
        )
        self.pipeline.load_voice(self.voice_path)

    def speak(self, text: str):
        for _, _, audio in self.pipeline(text, voice=self.voice_path):
            if self.speaker.interrupted.is_set():
                return
            self.speaker.play_pcm_array(audio.numpy(), self.sample_rate)


class ChatterboxMultilingualBackend(BaseBackend):
    """
    Usage:

    - TTS backend using the Chatterbox multilingual model for speech synthesis
      in any of the supported languages.
    - When voice is empty, the model's built-in default voice (conds.pt) is used.
    - When voice is provided, zero-shot voice cloning is applied using the
      named wave voice or a path to a .wav file.
    - Model weights are downloaded from HuggingFace on first use and cached to
      ~/.cache/spych/models/chatterbox-multilingual/.
    """

    def __init__(self, speaker: "Speaker", voice: str, language_id: str = "en"):
        super().__init__(speaker, voice)
        from spych.speaker.chatterbox_multilingual import SpychChatterboxMultilingualTTS

        self.language_id = language_id
        self.model = SpychChatterboxMultilingualTTS.from_pretrained(device=device)
        self.sample_rate = self.model.sr

        if voice:
            try:
                audio_prompt_path = get_wave_voice(voice)
            except FileNotFoundError:
                local_voices = _get_cached_wave_voices()
                local_hint = (
                    f"  Locally cached voices: {', '.join(local_voices)}\n"
                    if local_voices
                    else "  No locally cached voices found.\n"
                )
                raise ValueError(
                    f"[spych] ChatterboxMultilingual voice '{voice}' not found.\n"
                    "  Browse included voices: https://github.com/connor-makowski/spych/tree/main/voices/wave\n"
                    + local_hint
                )
            self.model.prepare_conditionals(audio_prompt_path)

    def speak(self, text: str):
        audio = self.model.generate(text, language_id=self.language_id)
        if self.speaker.interrupted.is_set():
            return
        self.speaker.play_pcm_array(audio.flatten(), self.sample_rate)


def get_backend(
    speaker: "Speaker", voice: str, backend_name: str = "", language_id: str = ""
) -> BaseBackend:
    """
    Usage:

    - Returns the first available TTS backend, tried in priority order:
      Explicit Choice → Chatterbox Turbo → Kokoro.
    - If backend_name is provided, that backend is tried first.
    - If unavailable or not installed, falls back to the others.
    - Raises NotImplementedError if no requested or fallback backend is available.

    Requires:

    - `speaker`:
        - Type: Speaker
        - What: The owning Speaker instance (passed through to the backend).
    - `voice`:
        - Type: str
        - What: Voice name or .wav file path passed to the backend.

    Optional:

    - `backend_name`:
        - Type: str
        - What: Explicit backend to use ("chatterbox", "kokoro", or "chatterbox_multilingual").
          "chatterbox_multilingual" requires language_id and does not fall back.
        - Default: "" (priority order: Chatterbox → Kokoro)

    - `language_id`:
        - Type: str
        - What: BCP-47 language code passed to ChatterboxMultilingualBackend when
          backend_name is "chatterbox_multilingual" (e.g. "es", "en", "fr").
        - Default: "" (resolved to "en" for the multilingual backend)

    Returns:

    - `backend`:
        - Type: BaseBackend
        - What: An initialized backend ready to call `speak(text)`.
    """
    # Multilingual backend: no fallback — requested explicitly for a specific language
    if backend_name.lower() == "chatterbox_multilingual":
        return ChatterboxMultilingualBackend(speaker, voice, language_id=language_id or "en")

    backends = [ChatterboxBackend, KokoroBackend]
    if backend_name.lower() == "chatterbox":
        backends = [ChatterboxBackend, KokoroBackend]
    elif backend_name.lower() == "kokoro":
        backends = [KokoroBackend, ChatterboxBackend]

    errors = []
    for backend_class in backends:
        try:
            return backend_class(speaker, voice)
        except ImportError:
            continue
        except Exception as e:
            errors.append(f"{backend_class.__name__}: {str(e)}")
            continue

    if errors:
        error_msg = "\n".join([f"  - {e}" for e in errors])
        print(
            f"\033[91m[spych] Failed to initialize TTS backends:\n{error_msg}\033[0m"
        )

    raise NotImplementedError(
        "No TTS backend available. Install one with `pip install spych[chatterbox]` or `pip install spych[kokoro]`."
    )
