import os
import time
import requests
from spych.utils import Recorder, save_wav, get_cache_dir


def list_cached_wave_voices() -> list:
    """
    Usage:

    - Returns a sorted list of wave voice names that are already present in the
      local voice cache (~/.cache/spych/voices/wave/).

    Returns:

    - `voice_names`:
        - Type: list
        - What: Sorted list of voice name strings (without the .wav extension),
          or an empty list if the cache directory does not exist.
    """
    cache_wave_dir = os.path.join(get_cache_dir(folder="voices"), "wave")
    if not os.path.exists(cache_wave_dir):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(cache_wave_dir)
        if f.endswith(".wav")
    )


def get_wave_voice(voice: str) -> str:
    """
    Usage:

    - Returns the local path to a wave voice .wav file, downloading it from the
      spych voices repo if it is not already in the cache.
    - Used by ChatterboxBackend to resolve voice names to file paths.

    Requires:

    - `voice`:
        - Type: str
        - What: Voice name (e.g. "af_heart") or an existing file path to a .wav file.
          If a valid file path is given it is returned as-is without downloading.

    Returns:

    - `local_path`:
        - Type: str
        - What: Absolute path to the cached .wav file.

    Notes:

    - Raises FileNotFoundError if the voice is not found locally and cannot be
      downloaded from the repo.
    - Browse available voices: https://github.com/connor-makowski/spych/tree/main/voices/wave
    """
    if os.path.isfile(voice):
        return voice  # If it's already a valid file path, just return it
    cache_dir = get_cache_dir(folder="voices")
    wave_dir = os.path.join(cache_dir, "wave")
    os.makedirs(wave_dir, exist_ok=True)
    local_path = os.path.join(wave_dir, f"{voice}.wav")

    if not os.path.isfile(local_path):
        print(f"[spych] Downloading voice '{voice}' from wave voices...")
        url = f"https://raw.githubusercontent.com/connor-makowski/spych/main/voices/wave/{voice}.wav"
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(response.content)
            print(f"[spych] Voice '{voice}' downloaded and saved to cache.")
        except Exception as e:
            print(
                f"\033[91m[spych] Failed to download voice '{voice}': {e}\033[0m"
            )
            raise FileNotFoundError(
                f"Voice '{voice}' not found in cache and failed to download from {url}."
            )
    return local_path


def get_pt_voice(voice: str) -> str:
    """
    Usage:

    - Returns the local path to a Kokoro voice .pt file, downloading it from the
      spych voices repo if it is not already in the cache.
    - Used by KokoroBackend to resolve voice names to file paths.

    Requires:

    - `voice`:
        - Type: str
        - What: Kokoro voice name (e.g. "af_heart") or an existing file path to a .pt file.
          If a valid file path is given it is returned as-is without downloading.

    Returns:

    - `local_path`:
        - Type: str
        - What: Absolute path to the cached .pt file.

    Notes:

    - Raises FileNotFoundError if the voice is not found locally and cannot be
      downloaded from the repo.
    - Browse available voices: https://github.com/connor-makowski/spych/tree/main/voices/pt
    """
    if os.path.isfile(voice):
        return voice  # If it's already a valid file path, just return it
    cache_dir = get_cache_dir(folder="voices")
    pt_dir = os.path.join(cache_dir, "pt")
    os.makedirs(pt_dir, exist_ok=True)
    local_path = os.path.join(pt_dir, f"{voice}.pt")

    if not os.path.isfile(local_path):
        print(f"[spych] Downloading voice '{voice}' from pt voices...")
        url = f"https://raw.githubusercontent.com/connor-makowski/spych/main/voices/pt/{voice}.pt"
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(response.content)
            print(f"[spych] Voice '{voice}' downloaded and saved to cache.")
        except Exception as e:
            print(
                f"\033[91m[spych] Failed to download voice '{voice}': {e}\033[0m"
            )
            raise FileNotFoundError(
                f"Voice '{voice}' not found in cache and failed to download from {url}."
            )
    return local_path


def get_model(name: str) -> str:
    """
    Usage:

    - Returns the local path to a model file, downloading it from the spych
      voices/model folder if it is not already in the cache.

    Requires:

    - `name`:
        - Type: str
        - What: Model filename (e.g. "my_model.bin") or an existing file path.
          If a valid file path is given it is returned as-is without downloading.

    Returns:

    - `local_path`:
        - Type: str
        - What: Absolute path to the cached model file.

    Notes:

    - Raises FileNotFoundError if the model is not found locally and cannot be
      downloaded from the repo.
    """
    if os.path.isfile(name):
        return name  # If it's already a valid file path, just return it
    cache_dir = get_cache_dir(folder="voices")
    model_dir = os.path.join(cache_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    local_path = os.path.join(model_dir, f"{name}")
    if not os.path.isfile(local_path):
        print(f"[spych] Downloading model '{name}' from voices/model...")
        url = f"https://raw.githubusercontent.com/connor-makowski/spych/main/voices/model/{name}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(response.content)
            print(f"[spych] Model '{name}' downloaded and saved to cache.")
        except Exception as e:
            print(
                f"\033[91m[spych] Failed to download model '{name}': {e}\033[0m"
            )
            raise FileNotFoundError(
                f"Model '{name}' not found in cache and failed to download from {url}."
            )
    return local_path


def profile_my_voice(
    name: str, device_index: int = -1, alternate_output_file: str | None = None
) -> None:
    """
    Prompts the user to record a short voice sample, then saves it to the
    voice cache directory for use with zero-shot cloning.
    """
    cache_dir = get_cache_dir(folder="voices")
    wave_dir = os.path.join(cache_dir, "wave")
    os.makedirs(wave_dir, exist_ok=True)
    output_path = os.path.join(wave_dir, f"{name}.wav")

    print(f"\n[spych] Preparing to record voice profile for: {name}")
    print("[spych] Please prepare to read the following passage aloud:")
    print("--------------------------------------------------")
    print(
        "The wild dogs are howling in the woods and even if this \n"
        "fort will not shelter our shamans, it isn't doomed to repeat \n"
        "their history of past, so let's just eat like free kings if only \n"
        "for tonight"
    )
    print("--------------------------------------------------")

    # Simple countdown
    for i in range(3, 0, -1):
        print(f"[spych] Starting in {i}...", end="\r", flush=True)
        time.sleep(1)

    # Green light indicator
    print("\033[92m" + "[spych] GO! Speak now..." + "\033[0m")

    recorder = Recorder()
    # Using VAD to record a single utterance
    # We'll use slightly more generous silence thresholds to ensure we capture everything
    buffer = recorder.record_vad(
        device_index=device_index,
        speech_threshold=0.5,
        silence_threshold=0.35,
        silence_frames_threshold=30,  # ~1 second of silence to end
        speech_pad_frames=10,  # ~320ms pre-roll
        max_speech_duration_s=15.0,  # Max 15 seconds
    )

    if not buffer:
        print(
            "\033[91m"
            + "[spych] No speech detected. Profile creation failed."
            + "\033[0m"
        )
        return

    save_wav(output_path, buffer)
    print(
        f"\033[92m"
        + f"[spych] Success! Voice profile saved to: {output_path}"
        + "\033[0m"
    )
    if alternate_output_file:
        save_wav(alternate_output_file, buffer)
        print(
            f"\033[92m"
            + f"[spych] Also saved a copy to: {alternate_output_file}"
            + "\033[0m"
        )
    print(f"[spych] You can now use this voice with: --speaker-voice {name}")
