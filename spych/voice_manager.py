import os
import time
import requests
from spych.utils import Recorder, save_wav, get_cache_dir

DEFAULT_VOICES_URL = "https://raw.githubusercontent.com/connor-makowski/spych/main/voices/{voice}.wav"
KOKORO_VOICES = [
    "af_heart", "af_bella", "af_nicole", 
    "af_aoede", "af_kore", "af_sarah",
    "am_michael", "am_fenrir", "am_puck",
    "bf_emma", "bf_isabella", "bm_george"
]


def profile_my_voice(name: str, device_index: int = -1, alternate_output_file: str | None = None) -> None:
    """
    Prompts the user to record a short voice sample, then saves it to the
    voice cache directory for use with zero-shot cloning.
    """
    cache_dir = get_cache_dir(folder='voices')
    output_path = os.path.join(cache_dir, f"{name}.wav")

    print(f"\n[spych] Preparing to record voice profile for: {name}")
    print("[spych] Please prepare to read the following passage aloud:")
    print("--------------------------------------------------")
    print(
        "The wild dogs are howling in the woods and even if this \n"\
        "fort will not shelter our shamans, it isn't doomed to repeat \n"\
        "their history of past, so let's just eat like free kings if only \n"\
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
        speech_pad_frames=10,         # ~320ms pre-roll
        max_speech_duration_s=15.0,    # Max 15 seconds
    )

    if not buffer:
        print("\033[91m" + "[spych] No speech detected. Profile creation failed." + "\033[0m")
        return

    save_wav(output_path, buffer)
    print(f"\033[92m" + f"[spych] Success! Voice profile saved to: {output_path}" + "\033[0m")
    if alternate_output_file:
        save_wav(alternate_output_file, buffer)
        print(f"\033[92m" + f"[spych] Also saved a copy to: {alternate_output_file}" + "\033[0m")
    print(f"[spych] You can now use this voice with: --speaker-voice {name}")


def sync_default_voices() -> None:
    """
    Attempts to download default voices from the GitHub repo to the local
    voice cache directory. If a local 'voices/' directory exists in the project
    root, it copies from there first.
    """
    cache_dir = get_cache_dir(folder='voices')
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_voices_dir = os.path.join(project_root, "voices")

    print(f"[spych] Syncing default voices to: {cache_dir}")

    for voice in KOKORO_VOICES:
        target_path = os.path.join(cache_dir, f"{voice}.wav")
        if os.path.exists(target_path):
            continue

        # 1. Try local copy if available
        local_path = os.path.join(local_voices_dir, f"{voice}.wav")
        if os.path.exists(local_path):
            print(f"[spych] Copying {voice} from local project...")
            import shutil
            shutil.copy(local_path, target_path)
            continue

        # 2. Try download
        url = DEFAULT_VOICES_URL.format(voice=voice)
        print(f"[spych] Downloading {voice} from repository...")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(target_path, "wb") as f:
                    f.write(response.content)
            else:
                print(f"[spych] Failed to download {voice}: HTTP {response.status_code}")
        except Exception as e:
            print(f"[spych] Error downloading {voice}: {e}")

    print("[spych] Voice sync complete.")
