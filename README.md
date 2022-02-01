Spych
==========
Pronounced: Speech

Python wrapper for easily accessing the [DeepSpeech](https://github.com/mozilla/DeepSpeech/) python package via python (without the DeepSpeech CLI)


Documentation for Spych Functions
--------
https://connor-makowski.github.io/spych/core.html

https://connor-makowski.github.io/spych/wake.html

Key Features
--------

- Simplified access to pretrained DeepSpeech models for offline and free speech transcription


Setup
----------

Make sure you have Python 3.6.x (or higher) installed on your system. You can download it [here](https://www.python.org/downloads/).

### Installation

1. Install SoX
- On Debian/Ubuntu
  ```
  sudo apt install sox
  ```
- On Mac (via homebrew)
  ```
  brew install sox
  ```
- On windows (Recommend WSL)

2. Install Spych
```
pip install spych
```

3. Get DeepSpeech Model and Score files:
```
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
```

# Examples

## Transcribe Existing Audio File
- Note: A `.wav` file at the same sample rate as your selected DeepSpeech models is processed the fastest
```py
from spych import spych

spych_obj=spych(model_file='deepspeech-0.9.3-models.pbmm', scorer_file='deepspeech-0.9.3-models.scorer')

# Convert the audio file to text
print('Transcription:')
print(spych_obj.stt(audio_file='test.wav'))
```

## Record and Transcribe
```py
from spych import spych

spych_obj=spych(model_file='deepspeech-0.9.3-models.pbmm', scorer_file='deepspeech-0.9.3-models.scorer')

# Record using your default microphone for 3 seconds
print('Recording...')
my_audio_buffer=spych_obj.record(duration=3)
print('Recording Finished')

# Convert the audio buffer to text
print('You said:')
print(spych_obj.stt(my_audio_buffer))
```

## Process a Function After Hearing a Wake Word (Example Wake Word: `computer`)
```py
from spych import spych, spych_wake

model_file='deepspeech-0.9.3-models.pbmm'
scorer_file='deepspeech-0.9.3-models.scorer'

spych_object=spych(model_file=model_file, scorer_file=scorer_file)

def my_function():
    print("Listening...")
    audio_buffer=spych_object.record(duration=3)
    print("You said:",spych_object.stt(audio_buffer=audio_buffer))

listener=spych_wake(model_file=model_file, scorer_file=scorer_file, on_wake_fn=my_function, wake_word="computer")

listener.start()
```
