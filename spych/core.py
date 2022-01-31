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

            - `model_file`:
                - Type: str
                - What: The location of your deepspeech model

        Optional:

            - `scorer_file`:
                - Type: str
                - What: The location of your deepspeech scorer
                - Default: None
        """
        self.model_file=model_file
        self.scorer_file=scorer_file
        self.model = Model(self.model_file)
        if self.scorer_file:
            self.model.enableExternalScorer(self.scorer_file)
        self.desired_sample_rate=self.model.sampleRate()

    def execute_cmd(self, cmd, capture_output=True):
        """
        Execute a subprocess cmd on the terminal / command line

        Required:

            - `cmd`:
                - Type: str
                - What: The command to execute
        """
        try:
            output = subprocess.run(shlex.split(cmd), capture_output=capture_output)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Execution of {cmd} returned non-zero status: {e.stderr}')
        except OSError as e:
            raise OSError(e.errno, f'Execution of {cmd} returned OS Error: {e.strerror}')
        return output

    def format_audio(self, audio_file):
        """
        Attempt auto formatting your audio file using SoX to match that of the DeepSpeech Model

        Required:

            - `audio_file`:
                - Type: str
                - What: The location of your target audio file to transcribe
        """
        converted_audio_file=audio_file+'.spych_converted.wav'
        sox_cmd = f'sox {quote(audio_file)} --rate {self.desired_sample_rate} --no-dither {quote(converted_audio_file)}'
        self.execute_cmd(sox_cmd)
        return converted_audio_file

    def parse_audio(self, audio_file):
        """
        Helper function to parse your raw audio file to match audio structures for the DeepSpeech Model

        Required:

            - `audio_file`:
                - Type: str
                - What: The location of your target audio file to transcribe
        """
        if ".wav" not in audio_file:
            self.warn(f"Selected audio file is not in `.wav` format. Attempting SoX conversion.")
            audio_file = self.format_audio(audio_file)
        with wave.open(audio_file, 'rb') as audio_raw:
            audio_sample_rate = audio_raw.getframerate()
            if audio_sample_rate != self.desired_sample_rate:
                self.warn(f"Selected audio sample rate ({audio_sample_rate}) is different from the desired rate ({self.desired_sample_rate}). Attempting SoX conversion.")
                audio = self.parse_audio(self.format_audio(audio_file))
            else:
                audio = np.frombuffer(audio_raw.readframes(audio_raw.getnframes()), np.int16)
        return audio

    def record(self, output_audio_file, duration=5):
        """
        Record an audio file for a set duration using SoX

        Required:

            - `output_audio_file`:
                - Type: str
                - What: The location to output the collected recording

        Optional:

            - `duration`:
                - Type: int
                - What: The duration of time to record in seconds
                - Default: 5
        """
        sox_cmd = f'sox -d --channels 1 --rate {self.desired_sample_rate} --no-dither {quote(output_audio_file)} trim 0 {duration}'
        self.execute_cmd(sox_cmd)
        return output_audio_file

    def play(self, audio_file):
        """
        Play an audio file using SoX

        Required:

            - `audio_file`:
                - Type: str
                - What: The location of your target audio file to transcribe
        """
        sox_cmd = f'sox {quote(audio_file)} -d'
        self.execute_cmd(sox_cmd)

    def get_word_timings(self, transcript):
        """
        Helper function to parse word timings and duration from a transcription metadata object

        Required:

            - `transcript`:
                - Type: CandidateTranscript (from deepspeech)
                - What: The candidate transcript to parse
        """
        if len(transcript.tokens)==0:
            return []
        word_data=[]
        word_tokens=[]
        for token in transcript.tokens:
            word_tokens.append(token)
            if token.text==" ":
                word_data.append(word_tokens)
                word_tokens=[]
        word_data.append(word_tokens)
        output=[]
        for word_tokens in word_data:
            try:
                start=round(word_tokens[0].start_time,3)
                end=round(word_tokens[-1].start_time,3)
                output.append({
                    'start':start,
                    'end':end,
                    'duration':round(end-start,3)
                })
            except:
                pass
        return output

    def get_transcript_dict(self, transcript, return_text=True, return_confidence=True, return_words=True, return_word_timings=True, return_meta=False):
        """
        Helper function to parse a clean dictionary from a transcription metadata object

        Required:

            - `transcript`:
                - Type: CandidateTranscript (from deepspeech)
                - What: The candidate transcript to parse

        Optional:

            - `return_text`:
                - Type: bool
                - What: Flag to indicate if the predicted text should be returned
                - Default: True
            - `return_confidence`:
                - Type: bool
                - What: Flag to indicate if the confidence level for this text should be returned
                - Default: True
            - `return_words`:
                - Type: bool
                - What: Flag to indicate if a words list (from the predicted text) should be returned
                - Default: True
            - `return_word_timings`:
                - Type: bool
                - What: Flag to indicate if the predicted timings (start, end and duration) for each word should be returned
                - Default: True
            - `return_meta`:
                - Type: bool
                - What: Flag to indicate if the transcript metadata object from DeepSpeech should be returned
                - Default: False
        """
        string=''.join(i.text for i in transcript.tokens)
        output={}
        if return_text:
            output['text']=string
        if return_confidence:
            output['confidence']=transcript.confidence
        if return_words:
            output['words']=string.split(" ")
        if return_word_timings:
            output['words_timings']=self.get_word_timings(transcript)
        if return_meta:
            output['meta']=transcript
        return output

    def stt(self, audio_file):
        """
        Compute speech-to-text transcription for a provided audio file

        Required:

            - `audio_file`:
                - Type: str
                - What: The location of your target audio file to transcribe
        """
        audio = self.parse_audio(audio_file)
        return self.model.stt(audio)

    def stt_expanded(self, audio_file, num_candidates=1, **kwargs):
        """
        Compute speech-to-text with extra data for N predicted candidates given an audio file

        Required:

            - `audio_file`:
                - Type: str
                - What: The location of your target audio file to transcribe

        Optional:

            - `num_candidates`:
                - Type: int
                - What: The number of potential transcript candidates to return
                - Default: 1
                - Note: The most confident/likely result appears first
            - `return_text`:
                - Type: bool
                - What: Flag to indicate if the predicted text should be returned
                - Default: True
            - `return_confidence`:
                - Type: bool
                - What: Flag to indicate if the confidence level for this text should be returned
                - Default: True
            - `return_words`:
                - Type: bool
                - What: Flag to indicate if a words list (from the predicted text) should be returned
                - Default: True
            - `return_word_timings`:
                - Type: bool
                - What: Flag to indicate if the predicted timings (start, end and duration) for each word should be returned
                - Default: True
            - `return_meta`:
                - Type: bool
                - What: Flag to indicate if the transcript metadata object from DeepSpeech should be returned
                - Default: False
        """
        audio = self.parse_audio(audio_file)
        output_meta=self.model.sttWithMetadata(audio, num_candidates)
        return [self.get_transcript_dict(transcript, **kwargs) for transcript in output_meta.transcripts]

    def audio_stt_expanded(self, audio, num_candidates=1, **kwargs):
        """
        Internal helper function for more efficient conversion processes (no need to save data)

        Required:

            - `audio`:
                - Type: Audio stream data
                - What: Formatted audio data to match DeepSpeech Model

        Optional:

            - `num_candidates`:
                - Type: int
                - What: The number of potential transcript candidates to return
                - Default: 1
                - Note: The most confident/likely result appears first
            - `return_text`:
                - Type: bool
                - What: Flag to indicate if the predicted text should be returned
                - Default: True
            - `return_confidence`:
                - Type: bool
                - What: Flag to indicate if the confidence level for this text should be returned
                - Default: True
            - `return_words`:
                - Type: bool
                - What: Flag to indicate if a words list (from the predicted text) should be returned
                - Default: True
            - `return_word_timings`:
                - Type: bool
                - What: Flag to indicate if the predicted timings (start, end and duration) for each word should be returned
                - Default: True
            - `return_meta`:
                - Type: bool
                - What: Flag to indicate if the transcript metadata object from DeepSpeech should be returned
                - Default: False
        """
        output_meta=self.model.sttWithMetadata(audio, num_candidates)
        return [self.get_transcript_dict(transcript, **kwargs) for transcript in output_meta.transcripts]
