import os
import wave
import numpy as np
from kokoro import KPipeline

SAMPLE_TEXT = (
    "The wild dogs are howling in the woods and even if this fort will not shelter our shamans, it isn't doomed to repeat their history of past, so let's just eat like free kings if only for tonight"
)


def generate_samples():
    pt_dir = "voices/pt"
    wave_dir = "voices/wave"

    if not os.path.exists(pt_dir):
        print(f"Error: Directory '{pt_dir}' not found.")
        return

    if not os.path.exists(wave_dir):
        os.makedirs(wave_dir)

    # Find all .pt files and use their base names as voice IDs
    voices = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(pt_dir)
        if f.endswith(".pt")
    ])

    if not voices:
        print(f"No .pt voice files found in {pt_dir}")
        return

    print(f"Found {len(voices)} voices to generate.")

    # Cache pipelines by language code to avoid redundant initialization
    # lang_code is the first character of the voice name (a, b, e, f, h, i, j, p, z)
    pipelines = {}

    for voice in voices:
        lang_code = voice[0]
        if lang_code not in pipelines:
            print(f"Initializing Kokoro pipeline for language: {lang_code}...")
            try:
                pipelines[lang_code] = KPipeline(lang_code=lang_code)
            except Exception as e:
                print(f"Failed to initialize pipeline for '{lang_code}': {e}")
                continue

        pipeline = pipelines[lang_code]
        print(f"Generating sample for: {voice}...")

        output_path = os.path.join(wave_dir, f"{voice}.wav")

        try:
            all_audio = []
            # Note: pipeline yields (graphemes, phonemes, audio)
            for _, _, audio in pipeline(SAMPLE_TEXT, voice=voice):
                if audio is not None:
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
        except Exception as e:
            print(f"Error generating {voice}: {e}")


if __name__ == "__main__":
    try:
        generate_samples()
    except ImportError:
        print("Error: 'kokoro' not found. Please install it with 'pip install kokoro' to use this script.")
    except Exception as e:
        print(f"An error occurred: {e}")
