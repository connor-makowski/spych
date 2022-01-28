# Oringinal code modified from client.py
# https://github.com/mozilla/DeepSpeech/blob/master/native_client/python/client.py

from spych.utils import error
from deepspeech import Model, version

import numpy as np
import shlex, subprocess, sys, wave, json

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

class spych(error):
    def __init__(self, model_file, scorer_file=None):
        """
        Initialize a spych class

        Required:
            `model_file`:
                Type: str
                What: The location of your deepspeech model

        Optional:
            `scorer_file`:
                Type: str
                What: The location of your deepspeech scorer
        """
        self.model_file=model_file
        self.scorer_file=scorer_file
        self.model = Model(self.model_file)
        if self.scorer_file:
            self.model.enableExternalScorer(self.scorer_file)
        self.desired_sample_rate=self.model.sampleRate()

    def sox_convert_sample_rate(self, audio_file):
        """
        Helper function to attempt converting your audio file using SoX

        Required:
            `audio_file`:
                Type: str
                What: The location of your target audio file to transcribe
        """
        sox_cmd = f'sox {quote(audio_file)} --type raw --bits 16 --channels 1 --rate {self.desired_sample_rate} --encoding signed-integer --endian little --compression 0.0 --no-dither - '
        try:
            output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
        except OSError as e:
            raise OSError(e.errno, f'SoX not found, use matching sample rate ({self.desired_sample_rate}) files or install SoX: {e.strerror}')
        return np.frombuffer(output, np.int16)

    def parse_audio(self, audio_file):
        """
        Helper function to parse your raw audio file to match that of the deepspeech model

        Required:
            `audio_file`:
                Type: str
                What: The location of your target audio file to transcribe
        """
        with wave.open(audio_file, 'rb') as audio_raw:
            audio_sample_rate = audio_raw.getframerate()
            if audio_sample_rate != self.desired_sample_rate:
                self.warn(f"Selected audio sample rate ({audio_sample_rate}) is different from the desired rate ({self.desired_sample_rate}). This may cause unexpected results. Attempting SoX conversion.")
                audio = self.sox_convert_sample_rate(audio_file)
            else:
                audio = np.frombuffer(audio_raw.readframes(audio_raw.getnframes()), np.int16)
        return audio

    def get_transcript_dict(self, transcript):
        """
        Helper function to parse a transcription dictionary from deepspeech Model returned meta data

        Required:
            `transcript`:
                Type: CandidateTranscript (from deepspeech)
                What: The candidate transcript to parse
        """
        string=''.join(i.text for i in transcript.tokens)
        return {
            'confidence':transcript.confidence,
            'string':string,
            'words':string.split(" ")
        }

    def compute_full(self, audio_file, num_candidates=3, return_meta=False):
        """
        Compute a full list of potential transcripts on a provided audio file and return the results

        Required:
            `audio_file`:
                Type: str
                What: The location of your target audio file to transcribe

        Optional:
            `num_candidates`:
                Type: int
                What: The number of potential transcript candidates to return (the most confident/likely result appears first)
            `return_meta`:
                Type: bool
                What: A flag to indicate if the acompaning result metadata should returned as well as the standard output
        """
        audio = self.parse_audio(audio_file)
        output_meta=self.model.sttWithMetadata(audio, num_candidates)
        output={'output':[self.get_transcript_dict(transcript) for transcript in output_meta.transcripts]}
        if return_meta:
            output['meta']=output_meta
        return output

    def compute(self, audio_file):
        """
        Compute the most likely potential transcript and return it as a python string

        Required:
            `audio_file`:
                Type: str
                What: The location of your target audio file to transcribe
        """
        try:
            return self.compute_full(audio_file=audio_file,num_candidates=1).get('output')[0].get('string')
        except:
            return None
