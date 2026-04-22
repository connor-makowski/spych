import contextlib
import os
import math
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

import librosa
import numpy as np
import pyloudnorm as ln
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from spych.utils import get_cache_dir
from chatterbox.models.s3gen import S3GEN_SR, S3Gen
from chatterbox.models.s3gen.const import S3GEN_SIL
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.voice_encoder import VoiceEncoder

logger = logging.getLogger(__name__)

REPO_ID = "ResembleAI/chatterbox-turbo"

@contextlib.contextmanager
def _quiet():
    """Suppresses tqdm progress bars and stray print statements during inference."""
    from unittest.mock import patch
    noop_tqdm = lambda iterable, *args, **kwargs: iterable
    with (
        patch("chatterbox.models.t3.t3.tqdm", noop_tqdm),
        patch("chatterbox.models.s3gen.flow_matching.tqdm", noop_tqdm),
        open(os.devnull, "w") as devnull,
        contextlib.redirect_stdout(devnull),
    ):
        yield


def punc_norm(text: str) -> str:
    if len(text) == 0:
        return "You need to add some text for me to talk."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    punc_to_replace = [
        ("…", ", "),
        (":", ","),
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
    if not any(text.endswith(p) for p in {".", "!", "?", "-", ","}):
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


class SpychChatterboxTTS:
    """
    Usage:

    - Chatterbox Turbo TTS model. Generates speech as a numpy float32 array.
    - No watermarking is applied; the raw decoded waveform is returned directly.
    - Typical usage:

        model = SpychChatterboxTTS.from_pretrained(device="cpu")
        model.prepare_conditionals("path/to/voice.wav")
        audio = model.generate("Hello world.")   # np.ndarray, float32

    Notes:

    - Model weights are downloaded once to ~/.cache/spych/models/ and reused.
    - `from_pretrained` fetches only the files required for Chatterbox Turbo
      (ve.safetensors, t3_turbo_v1.safetensors, s3gen_meanflow.safetensors,
      tokenizer files, conds.pt).
    - Reference audio for `prepare_conditionals` must be longer than 5 seconds.
    """

    ENC_COND_LEN = 15 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: AutoTokenizer,
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
    def from_local(cls, cache_dir: Path, device: str) -> "SpychChatterboxTTS":
        """
        Usage:

        - Loads a SpychChatterboxTTS model from a local directory.

        Requires:

        - `cache_dir`:
            - Type: Path | str
            - What: Directory containing the model files (ve.safetensors,
              t3_turbo_v1.safetensors, s3gen_meanflow.safetensors, tokenizer
              files, and optionally conds.pt).
        - `device`:
            - Type: str
            - What: Torch device to load the model onto (e.g. "cpu", "cuda").

        Returns:

        - `model`:
            - Type: SpychChatterboxTTS
            - What: Fully initialized model ready for `prepare_conditionals` and `generate`.
        """
        cache_dir = Path(cache_dir)

        map_location = torch.device("cpu") if device in ("cpu", "mps") else None

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(cache_dir / "ve.safetensors"))
        ve.to(device).eval()

        hp = T3Config(text_tokens_dict_size=50276)
        hp.llama_config_name = "GPT2_medium"
        hp.speech_tokens_dict_size = 6563
        hp.input_pos_emb = None
        hp.speech_cond_prompt_len = 375
        hp.use_perceiver_resampler = False
        hp.emotion_adv = False

        t3 = T3(hp)
        t3_state = load_file(cache_dir / "t3_turbo_v1.safetensors")
        if "model" in t3_state:
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        del t3.tfmr.wte
        t3.to(device).eval()

        s3gen = S3Gen(meanflow=True)
        s3gen.load_state_dict(
            load_file(cache_dir / "s3gen_meanflow.safetensors"), strict=True
        )
        s3gen.to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if len(tokenizer) != 50276:
            logger.warning("Tokenizer len %d != 50276", len(tokenizer))

        conds = None
        builtin_voice = cache_dir / "conds.pt"
        if builtin_voice.exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(
                device
            )

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device: str) -> "SpychChatterboxTTS":
        """
        Usage:

        - Downloads the Chatterbox Turbo model files from HuggingFace (first run
          only) and loads the model. All files are cached to
          ~/.cache/spych/models/ and reused on subsequent runs.

        Requires:

        - `device`:
            - Type: str
            - What: Torch device string (e.g. "cpu", "cuda").

        Returns:

        - `model`:
            - Type: SpychChatterboxTTS
            - What: Fully initialized model ready for `prepare_conditionals` and `generate`.
        """
        model_cache_dir = os.path.join(get_cache_dir(folder="models"), "chatterbox-turbo")
        for fname in [
            "ve.safetensors",
            "t3_turbo_v1.safetensors",
            "s3gen_meanflow.safetensors",
            "tokenizer_config.json",
            "added_tokens.json",
            "merges.txt",
            "vocab.json",
            "special_tokens_map.json"
        ]:
            if not os.path.isfile(os.path.join(model_cache_dir, fname)):
                logger.info(f"Model file '{fname}' not found in cache. Downloading from Hugging Face Hub...")
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=fname,
                    local_dir=model_cache_dir,
                )
        return cls.from_local(model_cache_dir, device)

    def norm_loudness(self, wav: np.ndarray, sr: int, target_lufs: float = -27) -> np.ndarray:
        try:
            meter = ln.Meter(sr)
            loudness = meter.integrated_loudness(wav)
            gain_db = target_lufs - loudness
            gain_linear = 10.0 ** (gain_db / 20.0)
            if math.isfinite(gain_linear) and gain_linear > 0.0:
                wav = wav * gain_linear
        except Exception as e:
            logger.warning("norm_loudness skipped: %s", e)
        return wav

    def prepare_conditionals(
        self,
        wav_fpath: str,
        exaggeration: float = 0.5,
        norm_loudness: bool = True,
    ) -> None:
        """
        Usage:

        - Loads a reference .wav file and computes the speaker conditioning vectors
          used by `generate` for zero-shot voice cloning.
        - Must be called once before `generate` (unless `audio_prompt_path` is
          passed directly to `generate`).

        Requires:

        - `wav_fpath`:
            - Type: str
            - What: Path to a .wav reference audio file. Must be longer than 5 seconds.

        Optional:

        - `exaggeration`:
            - Type: float
            - What: Emotion exaggeration factor.
            - Default: 0.5
        - `norm_loudness`:
            - Type: bool
            - What: Normalize loudness of the reference audio before processing.
            - Default: True
        """
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        assert len(s3gen_ref_wav) / _sr > 5.0, "Audio prompt must be longer than 5 seconds!"

        if norm_loudness:
            s3gen_ref_wav = self.norm_loudness(s3gen_ref_wav, _sr)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR).astype(np.float32)

        s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

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
        repetition_penalty: float = 1.2,
        min_p: float = 0.0,
        top_p: float = 0.95,
        exaggeration: float = 0.0,
        cfg_weight: float = 0.0,
        temperature: float = 0.8,
        top_k: int = 1000,
        norm_loudness: bool = True,
    ) -> np.ndarray:
        """
        Usage:

        - Synthesizes speech from text using the Chatterbox Turbo model.
        - Returns raw waveform as a numpy float32 array (no watermarking).
        - `prepare_conditionals` must have been called first.

        Requires:

        - `text`:
            - Type: str
            - What: Text to synthesize.

        Optional:

        - `repetition_penalty`:
            - Type: float
            - What: Repetition penalty for token sampling.
            - Default: 1.2
        - `top_p`:
            - Type: float
            - What: Nucleus sampling probability cutoff.
            - Default: 0.95
        - `temperature`:
            - Type: float
            - What: Sampling temperature.
            - Default: 0.8
        - `top_k`:
            - Type: int
            - What: Top-k sampling cutoff.
            - Default: 1000
        - `norm_loudness`:
            - Type: bool
            - What: Normalize loudness of any inline audio prompt.
            - Default: True

        Returns:

        - `audio`:
            - Type: np.ndarray
            - What: Synthesized waveform, float32, shape (n_samples,), at 24000 Hz.

        Notes:

        - `cfg_weight`, `min_p`, and `exaggeration` are not supported by the turbo
          model and will be ignored with a warning if non-zero.
        - Progress bars and inference prints from the underlying model are suppressed.
        """
        with _quiet():

            if cfg_weight > 0.0 or exaggeration > 0.0 or min_p > 0.0:
                logger.warning(
                    "cfg_weight, min_p, and exaggeration are not supported by turbo and will be ignored."
                )

            text = punc_norm(text)
            text_tokens = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            )
            text_tokens = text_tokens.input_ids.to(self.device)

            speech_tokens = self.t3.inference_turbo(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            speech_tokens = speech_tokens[speech_tokens < 6561]
            speech_tokens = speech_tokens.to(self.device)
            silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(self.device)
            speech_tokens = torch.cat([speech_tokens, silence])

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
                n_cfm_timesteps=2,
            )
            output = wav.squeeze(0).detach().cpu().numpy()
        return output
