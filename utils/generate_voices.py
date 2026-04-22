import os
import wave
import numpy as np
from kokoro import KPipeline

# Original Kokoro voices to preserve as defaults
# Source: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
VOICES = [
    "af_heart",
    "af_bella",
    "af_nicole",
    "af_aoede",
    "af_kore",
    "af_sarah",
    "am_michael",
    "am_fenrir",
    "am_puck",
    "bf_emma",
    "bf_isabella",
    "bm_george",
]

SAMPLE_TEXT = (
    "The wild dogs are howling in the woods and even if this fort will not shelter our shamans, it isn't doomed to repeat their history of past, so let's just eat like free kings if only for tonight",
)


def generate_samples():
    if not os.path.exists("voices"):
        os.makedirs("voices")

    print("Initializing Kokoro pipelines...")
    # American English ('a') and British English ('b')
    pipeline_a = KPipeline(lang_code="a")
    pipeline_b = KPipeline(lang_code="b")

    for voice in VOICES:
        lang = "b" if voice.startswith(("bm_", "bf_")) else "a"
        pipeline = pipeline_b if lang == "b" else pipeline_a
        print(f"Generating sample for: {voice} ({lang})...")

        output_path = f"voices/{voice}.wav"

        # Generate audio
        # Note: pipeline yields (graphemes, phonemes, audio)
        all_audio = []
        for _, _, audio in pipeline(SAMPLE_TEXT, voice=voice):
            all_audio.append(audio.numpy())

        if not all_audio:
            print(f"Warning: No audio generated for {voice}")
            continue

        # Concatenate parts if multiple were yielded
        combined_audio = np.concatenate(all_audio)

        # Convert to 16-bit PCM
        pcm = (combined_audio * 32767).astype(np.int16)

        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm.tobytes())

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    try:
        generate_samples()
    except ImportError:
        print("Error: 'kokoro' not found. Please install it with 'pip install kokoro' to use this script.")
    except Exception as e:
        print(f"An error occurred: {e}")
