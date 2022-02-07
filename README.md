Spych
==========
Pronounced: Speech

Python wrapper for easily accessing the [DeepSpeech](https://github.com/mozilla/DeepSpeech/) python package via python (without the DeepSpeech CLI)


Documentation for Spych Functions
--------
Spych - https://connor-makowski.github.io/spych/core.html

Spych Wake - https://connor-makowski.github.io/spych/wake.html

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
```sh
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
```

# Examples

## Transcribe Existing Audio File

- [Spych Docs](https://connor-makowski.github.io/spych/core.html)

```py
from spych import spych

spych_obj=spych(model_file='deepspeech-0.9.3-models.pbmm', scorer_file='deepspeech-0.9.3-models.scorer')

# Convert the audio file to text
print('Transcription:')
print(spych_obj.stt(audio_file='test.wav'))
```
- Note: A `.wav` file at the same sample rate as your selected DeepSpeech models is processed the fastest

## Record and Transcribe

- [Spych Docs](https://connor-makowski.github.io/spych/core.html)

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

- [Spych Wake Docs](https://connor-makowski.github.io/spych/wake.html)
- [Spych Docs](https://connor-makowski.github.io/spych/core.html)

```py
from spych import spych, spych_wake

model_file='deepspeech-0.9.3-models.pbmm'
scorer_file='deepspeech-0.9.3-models.scorer'

spych_object=spych(model_file=model_file, scorer_file=scorer_file)

def my_function():
    print("Listening...")
    audio_buffer=spych_object.record(duration=3)M
    print("You said:",spych_object.stt(audio_buffer=audio_buffer))

listener=spych_wake(model_file=model_file, scorer_file=scorer_file, on_wake_fn=my_function, wake_word="computer")

listener.start()
```

## Modifying the DeepSpeech Model

- Initialized `spych` objects contain a fully functional `DeepSpeech.Model` object inside of them
- You can modify this for each `spych` object any time after initialization
- `DeepSpeech.Model` options are documented [here](https://deepspeech.readthedocs.io/en/latest/Python-API.html#)

Example:
```py
spych_obj=spych(model_file='deepspeech-0.9.3-models.pbmm')

spych_obj.model.enableExternalScorer('deepspeech-0.9.3-models.scorer')
spych_obj.model.addHotWord('activate',10.0)
```


# Rasberry Pi 4 Setup

1. Install system requirements
```
sudo apt install sox git python3-pip python3-scipy python3-numpy python3-pyaudio libatlas3-base
```

2. Install python requirements
```
pip3 install deepspeech
pip3 install spych --no-deps
```

3. Get the DeepSpeech model and score files (note Pi must user .tflite model file)
```sh
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.tflite
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
```

4. Use the examples above substituting the original model file name for the `.tflite` one
