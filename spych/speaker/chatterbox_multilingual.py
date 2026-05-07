import contextlib
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from spych.utils import get_cache_dir
from chatterbox.models.s3gen import S3GEN_SR, S3Gen
from chatterbox.models.s3tokenizer import S3_SR, S3_TOKEN_RATE, drop_invalid_tokens
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.tokenizers import MTLTokenizer
from chatterbox.models.voice_encoder import VoiceEncoder

logger = logging.getLogger(__name__)

REPO_ID = "ResembleAI/chatterbox"

SUPPORTED_LANGUAGES = {
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

_MODEL_FILES = [
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
]


@contextlib.contextmanager
def _quiet():
    """Suppress tqdm progress bars, stray stdout, and noisy chatterbox loggers during inference."""
    from unittest.mock import patch

    noop_tqdm = lambda iterable, *args, **kwargs: iterable

    _noisy_loggers = [
        "chatterbox.models.t3.inference.alignment_stream_analyzer",
        "chatterbox.models.t3.t3",
        "chatterbox.models.s3gen.flow_matching",
        "transformers",
    ]
    original_levels = {}
    for name in _noisy_loggers:
        lg = logging.getLogger(name)
        original_levels[name] = lg.level
        lg.setLevel(logging.ERROR)

    try:
        with (
            patch("chatterbox.models.t3.t3.tqdm", noop_tqdm),
            patch("chatterbox.models.s3gen.flow_matching.tqdm", noop_tqdm),
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stdout(devnull),
        ):
            yield
    finally:
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)


def punc_norm(text: str) -> str:
    if len(text) == 0:
        return "You need to add some text for me to talk."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", '"'),
        ("”", '"'),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old, new in punc_to_replace:
        text = text.replace(old, new)
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."
    return text


@dataclass
class Conditionals:
    t3: T3Cond
    gen: dict

    def to(self, device: str) -> "Conditionals":
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path) -> None:
        torch.save(dict(t3=self.t3.__dict__, gen=self.gen), fpath)

    @classmethod
    def load(cls, fpath: Path, map_location: str = "cpu") -> "Conditionals":
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs["t3"]), kwargs["gen"])


class SpychChatterboxMultilingualTTS:
    """
    Usage:

    - Multilingual Chatterbox TTS model. Synthesizes speech in any of the
      supported languages with optional zero-shot voice cloning.
    - When no reference audio is provided, the model's built-in default voice
      (conds.pt) is used automatically.
    - Typical usage:

        model = SpychChatterboxMultilingualTTS.from_pretrained(device="cpu")
        audio = model.generate("Hola, ¿cómo estás?", language_id="es")

    Notes:

    - Model weights are downloaded once to ~/.cache/spych/models/chatterbox-multilingual/
      and reused on subsequent runs.
    - No watermarking is applied; the raw decoded waveform is returned as numpy float32.
    - See SUPPORTED_LANGUAGES for the full list of available language codes.
    """

    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str,
        conds: Optional[Conditionals] = None,
    ):
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds

    @classmethod
    def from_local(
        cls, cache_dir: Path, device: str
    ) -> "SpychChatterboxMultilingualTTS":
        """
        Usage:

        - Loads a SpychChatterboxMultilingualTTS model from a local directory.

        Requires:

        - `cache_dir`:
            - Type: Path | str
            - What: Directory containing the model files (ve.pt,
              t3_mtl23ls_v2.safetensors, s3gen.pt,
              grapheme_mtl_merged_expanded_v1.json, and optionally conds.pt).
        - `device`:
            - Type: str
            - What: Torch device to load the model onto (e.g. "cpu", "cuda").

        Returns:

        - `model`:
            - Type: SpychChatterboxMultilingualTTS
            - What: Fully initialized model ready for prepare_conditionals() and generate().
        """
        cache_dir = Path(cache_dir)
        map_location = torch.device("cpu") if device in ("cpu", "mps") else None

        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(
                cache_dir / "ve.pt", map_location=map_location, weights_only=True
            )
        )
        ve.to(device).eval()

        t3 = T3(T3Config.multilingual())
        t3_state = load_file(cache_dir / "t3_mtl23ls_v2.safetensors")
        if "model" in t3_state:
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(
                cache_dir / "s3gen.pt", map_location=map_location, weights_only=True
            )
        )
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(
            str(cache_dir / "grapheme_mtl_merged_expanded_v1.json")
        )

        conds = None
        builtin_voice = cache_dir / "conds.pt"
        if builtin_voice.exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(
                device
            )

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device: str) -> "SpychChatterboxMultilingualTTS":
        """
        Usage:

        - Downloads the Chatterbox multilingual model files from HuggingFace
          (first run only) and loads the model. All files are cached to
          ~/.cache/spych/models/chatterbox-multilingual/ and reused on
          subsequent runs.

        Requires:

        - `device`:
            - Type: str
            - What: Torch device string (e.g. "cpu", "cuda").

        Returns:

        - `model`:
            - Type: SpychChatterboxMultilingualTTS
            - What: Fully initialized model ready for prepare_conditionals() and generate().
        """
        model_cache_dir = os.path.join(
            get_cache_dir(folder="models"), "chatterbox-multilingual"
        )
        os.makedirs(model_cache_dir, exist_ok=True)

        for fname in _MODEL_FILES:
            fpath = os.path.join(model_cache_dir, fname)
            if not os.path.isfile(fpath):
                try:
                    hf_hub_download(
                        repo_id=REPO_ID,
                        filename=fname,
                        local_dir=model_cache_dir,
                    )
                except Exception:
                    if fname not in (
                        "ve.pt",
                        "t3_mtl23ls_v2.safetensors",
                        "s3gen.pt",
                        "grapheme_mtl_merged_expanded_v1.json",
                    ):
                        continue
                    raise

        return cls.from_local(model_cache_dir, device)

    def prepare_conditionals(
        self, wav_fpath: str, exaggeration: float = 0.5
    ) -> None:
        """
        Usage:

        - Loads a reference .wav file and computes speaker conditioning vectors
          used by generate() for zero-shot voice cloning.
        - Must be called before generate() when using a custom voice.

        Requires:

        - `wav_fpath`:
            - Type: str
            - What: Path to a .wav reference audio file.

        Optional:

        - `exaggeration`:
            - Type: float
            - What: Emotion exaggeration factor.
            - Default: 0.5
        """
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(
            s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR
        ).astype(np.float32)

        s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(
            s3gen_ref_wav, S3GEN_SR, device=self.device
        )

        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward(
                [ref_16k_wav[: self.ENC_COND_LEN]], max_len=plen
            )
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(
                self.device
            )

        ve_embed = torch.from_numpy(
            self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR)
        )
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text: str,
        language_id: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
    ) -> np.ndarray:
        """
        Usage:

        - Synthesizes speech from text in the given language.
        - Returns raw waveform as a numpy float32 array at S3GEN_SR (24000 Hz).
        - No watermarking is applied.

        Requires:

        - `text`:
            - Type: str
            - What: Text to synthesize.

        - `language_id`:
            - Type: str
            - What: BCP-47 language code for synthesis (e.g. "es", "en", "fr").
              Must be a key in SUPPORTED_LANGUAGES.

        Optional:

        - `exaggeration`:
            - Type: float
            - What: Emotion exaggeration factor.
            - Default: 0.5
        - `cfg_weight`:
            - Type: float
            - What: Classifier-free guidance weight.
            - Default: 0.5
        - `temperature`:
            - Type: float
            - What: Sampling temperature.
            - Default: 0.8
        - `repetition_penalty`:
            - Type: float
            - What: Repetition penalty for token sampling.
            - Default: 1.2
        - `min_p`:
            - Type: float
            - What: Min-p sampling cutoff.
            - Default: 0.05
        - `top_p`:
            - Type: float
            - What: Nucleus sampling probability cutoff.
            - Default: 1.0

        Returns:

        - `audio`:
            - Type: np.ndarray
            - What: Synthesized waveform, float32, shape (n_samples,), at 24000 Hz.
        """
        assert self.conds is not None, (
            "No conditioning loaded. Call prepare_conditionals() first or ensure "
            "conds.pt exists in the model cache."
        )

        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        with _quiet():
            text = punc_norm(text)
            lang = language_id.lower() if language_id else None
            text_tokens = self.tokenizer.text_to_tokens(text, language_id=lang).to(
                self.device
            )
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

            sot = self.t3.hp.start_text_token
            eot = self.t3.hp.stop_text_token
            text_tokens = F.pad(text_tokens, (1, 0), value=sot)
            text_tokens = F.pad(text_tokens, (0, 1), value=eot)

            with torch.inference_mode():
                speech_tokens = self.t3.inference(
                    t3_cond=self.conds.t3,
                    text_tokens=text_tokens,
                    max_new_tokens=1000,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                )
                speech_tokens = speech_tokens[0]
                speech_tokens = drop_invalid_tokens(speech_tokens)
                speech_tokens = speech_tokens.to(self.device)

                wav, _ = self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=self.conds.gen,
                )
                wav = wav.squeeze(0).detach().cpu().numpy()

                n_tokens = int(speech_tokens.shape[-1])
                st_len = max(1, n_tokens - 1)
                wav = wav[: st_len * (S3GEN_SR // S3_TOKEN_RATE)]

        return wav.astype(np.float32)
