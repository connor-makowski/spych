import os
import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pyloudnorm as ln
import torch
from huggingface_hub import snapshot_download
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
    def from_local(cls, ckpt_dir: Path, device: str) -> "SpychChatterboxTTS":
        ckpt_dir = Path(ckpt_dir)

        map_location = torch.device("cpu") if device in ("cpu", "mps") else None

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve.to(device).eval()

        hp = T3Config(text_tokens_dict_size=50276)
        hp.llama_config_name = "GPT2_medium"
        hp.speech_tokens_dict_size = 6563
        hp.input_pos_emb = None
        hp.speech_cond_prompt_len = 375
        hp.use_perceiver_resampler = False
        hp.emotion_adv = False

        t3 = T3(hp)
        t3_state = load_file(ckpt_dir / "t3_turbo_v1.safetensors")
        if "model" in t3_state:
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        del t3.tfmr.wte
        t3.to(device).eval()

        s3gen = S3Gen(meanflow=True)
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen_meanflow.safetensors"), strict=True
        )
        s3gen.to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if len(tokenizer) != 50276:
            logger.warning("Tokenizer len %d != 50276", len(tokenizer))

        conds = None
        builtin_voice = ckpt_dir / "conds.pt"
        if builtin_voice.exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(
                device
            )

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device: str) -> "SpychChatterboxTTS":
        model_cache_dir = get_cache_dir(folder="models")
        local_path = snapshot_download(
            repo_id=REPO_ID,
            cache_dir=model_cache_dir,
            token=os.getenv("HF_TOKEN") or None,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
        )
        return cls.from_local(local_path, device)

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
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        assert len(s3gen_ref_wav) / _sr > 5.0, "Audio prompt must be longer than 5 seconds!"

        if norm_loudness:
            s3gen_ref_wav = self.norm_loudness(s3gen_ref_wav, _sr)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

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
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.0,
        cfg_weight: float = 0.0,
        temperature: float = 0.8,
        top_k: int = 1000,
        norm_loudness: bool = True,
    ) -> np.ndarray:
        if audio_prompt_path:
            self.prepare_conditionals(
                audio_prompt_path, exaggeration=exaggeration, norm_loudness=norm_loudness
            )
        else:
            assert (
                self.conds is not None
            ), "Please `prepare_conditionals` first or specify `audio_prompt_path`"

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
        return wav.squeeze(0).detach().cpu().numpy()
