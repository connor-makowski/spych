
from faster_whisper import WhisperModel
from spych.audio import Audio

# Run on GPU with FP16
model = WhisperModel("tiny", device="cuda", compute_type="float16")
audio = Audio()

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

buffer = audio.parse_audio("test/audio.wav")


segments_buffer, info_buffer = model.transcribe(audio.parse_audio("test/audio.wav"))
segments, info = model.transcribe("test/audio.wav")

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments_buffer:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))