Spych
==========
Python wrapper for easily accessing the [DeepSpeech](https://github.com/mozilla/DeepSpeech/) python package via python (without the DeepSpeech CLI)


Documentation for Spych Functions
--------
https://connor-makowski.github.io/spych/core.html

Key Features
--------

- Simplified access to pretrained DeepSpeech models for offline and free speech transcription


Setup
----------

Make sure you have Python 3.6.x (or higher) installed on your system. You can download it [here](https://www.python.org/downloads/).

### Installation

```
pip install spych
```

# Getting Started

## Getting DeepSpeech Models
```sh
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
```

## Basic Usage
```py
from spych import spych

model=spych(model_file='model/deepspeech-0.9.3-models.pbmm', scorer_file='model/deepspeech-0.9.3-models.scorer')

print(model.compute(audio_file='test.wav'))
```

## Recording on Ubuntu
```sh
arecord test.wav -r 16000
```
