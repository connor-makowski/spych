# Oringinal code modified from client.py
# https://github.com/mozilla/DeepSpeech/blob/master/native_client/python/client.py

from spych.utils import error
from deepspeech import Model, version

import numpy as np
import shlex, subprocess, sys, wave, json

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

    def execute_cmd(self, cmd, capture_output=True, check=True):
        """
        Execute a subprocess cmd on the terminal / command line

        Required:

            - `cmd`:
                - Type: str
                - What: The command to execute
        """
        try:
            output = subprocess.run(shlex.split(cmd), check=check, capture_output=capture_output)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Execution of {cmd} returned non-zero status: {e.stderr}')
        except OSError as e:
            raise OSError(e.errno, f'Execution of {cmd} returned OS Error: {e.strerror}')
        return output

    def parse_audio_sox(self, audio_file):
        """
        Attempt auto formatting your audio file using SoX to match that of the DeepSpeech Model

        Required:

            - `audio_file`:
                - Type: str
                - What: The location of your target audio file to transcribe

        Returns:

            - `audio_buffer`:
                - Type: np.array
                - What: A 16-bit, mono raw audio signal at the appropriate sample rate serialized as a numpy array
                - Note: Exactly matches the DeepSpeech Model
        """
        sox_cmd = f'sox {shlex.quote(audio_file)} --type raw --bits 16 --channels 1 --rate {self.desired_sample_rate} --encoding signed-integer --endian little --compression 0.0 --no-dither - '
        output = self.execute_cmd(sox_cmd)
        return np.frombuffer(output.stdout, np.int16)

    def parse_audio(self, audio_file):
        """
        Helper function to parse your raw audio file to match audio structures for the DeepSpeech Model

        Required:

            - `audio_file`:
                - Type: str
                - What: The location of your target audio file to transcribe
                - Note: `.wav` files with the model specified sample rate are handled without external packages. Everything else gets converted using SoX (if possible)

        Returns:

            - `audio_buffer`:
                - Type: np.array
                - What: A 16-bit, mono raw audio signal at the appropriate sample rate serialized as a numpy array
                - Note: Exactly matches the DeepSpeech Model
        """
        if ".wav" not in audio_file:
            self.warn(f"Selected audio file is not in `.wav` format. Attempting SoX conversion.")
            return self.parse_audio_sox(audio_file=audio_file)
        with wave.open(audio_file, 'rb') as audio_raw:
            audio_sample_rate = audio_raw.getframerate()
            if audio_sample_rate != self.desired_sample_rate:
                self.warn(f"Selected audio sample rate ({audio_sample_rate}) is different from the desired rate ({self.desired_sample_rate}). Attempting SoX conversion.")
                return self.parse_audio_sox(audio_file=audio_file)
            else:
                return np.frombuffer(audio_raw.readframes(audio_raw.getnframes()), np.int16)

    def record(self, output_audio_file=None, duration=3):
        """
        Record an audio file for a set duration using SoX

        Optional:

            - `output_audio_file`:
                - Type: str
                - What: The location to output the collected recording
                - Default: None
                - Note: Must be a `.wav` file
                - Note: If specified, the audio file location is returned
                - Note: If not specified or None, an audio buffer is returned

            - `duration`:
                - Type: int
                - What: The duration of time to record in seconds
                - Default: 3

        Returns:

            - `audio_file`:
                - Type: str
                - What: The provided `output_audio_file` given at the invokation of this function
                - Note: Returned only if `output_audio_file` is specified

            - OR

            - `audio_buffer`:
                - Type: np.array
                - What: A 16-bit, mono raw audio signal at the appropriate sample rate serialized as a numpy array
                - Note: Exactly matches the DeepSpeech Model
                - Note: Returned only if `output_audio_file` is not specified

        """
        if output_audio_file:
            sox_cmd = f'sox -d --channels 1 --rate {self.desired_sample_rate} --no-dither {shlex.quote(output_audio_file)} trim 0 {duration}'
            output=self.execute_cmd(sox_cmd)
            return output_audio_file
        else:
            sox_cmd = f'sox -d --type raw --bits 16 --channels 1 --rate {self.desired_sample_rate} --encoding signed-integer --endian little --compression 0.0 --no-dither - trim 0 {duration}'
            output = self.execute_cmd(sox_cmd)
            return np.frombuffer(output.stdout, np.int16)

    def play(self, audio_file):
        """
        Play an audio file using SoX

        Required:

            - `audio_file`:
                - Type: str
                - What: The location of your target audio file to transcribe
        """
        sox_cmd = f'sox {shlex.quote(audio_file)} -d'
        self.execute_cmd(sox_cmd)

    def get_word_timings(self, transcript):
        """
        Helper function to parse word timings and duration from a transcription metadata object

        Required:

            - `transcript`:
                - Type: CandidateTranscript (from deepspeech)
                - What: The candidate transcript to parse

        Returns:

            - `timings`:
                - Type: dict
                - What: A dictionary of the `start_time`, `end_time` and `duration` for each word in this transcript where those values are provided in seconds
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

    def get_transcript_dict(self, transcript, return_text=True, return_confidence=False, return_words=False, return_word_timings=False, return_meta=False):
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
                - Default: False
            - `return_words`:
                - Type: bool
                - What: Flag to indicate if a words list (from the predicted text) should be returned
                - Default: False
            - `return_word_timings`:
                - Type: bool
                - What: Flag to indicate if the predicted timings (start, end and duration) for each word should be returned
                - Default: False
            - `return_meta`:
                - Type: bool
                - What: Flag to indicate if the transcript metadata object from DeepSpeech should be returned
                - Default: False

        Returns:

            - `transcript_dictionary`:
                - Type: dict
                - What: Dictionary of serialized transcription items specified by optional inputs
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

    def stt(self, audio_buffer=None, audio_file=None, ):
        """
        Compute speech-to-text transcription for a provided audio file

        Required:

            - `audio_buffer`:
                - Type: np.array
                - What: A 16-bit, mono raw audio signal at the appropriate sample rate serialized as a numpy array
                - Note: Exactly matches the DeepSpeech Model

            - OR

            - `audio_file`:
                - Type: str
                - What: The location of your target audio file to transcribe

        Returns:

            - `text`:
                - Type: str
                - What: The transcribed text
        """
        if audio_file:
            audio_buffer = self.parse_audio(audio_file)
        if audio_buffer is None:
            self.exception("You must specify a valid audio_file or audio_buffer")
        return self.model.stt(audio_buffer)

    def stt_expanded(self, audio_file=None, audio_buffer=None, num_candidates=1, **kwargs):
        """
        Compute speech-to-text with extra data for N predicted candidates given an audio file

        Required:

            - `audio_buffer`:
                - Type: np.array
                - What: A 16-bit, mono raw audio signal at the appropriate sample rate serialized as a numpy array
                - Note: Exactly matches the DeepSpeech Model

            - OR

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
                - Default: False
            - `return_words`:
                - Type: bool
                - What: Flag to indicate if a words list (from the predicted text) should be returned
                - Default: False
            - `return_word_timings`:
                - Type: bool
                - What: Flag to indicate if the predicted timings (start, end and duration) for each word should be returned
                - Default: False
            - `return_meta`:
                - Type: bool
                - What: Flag to indicate if the transcript metadata object from DeepSpeech should be returned
                - Default: False

        Returns:

            - `transcriptions`:
                - Type: list of dictionaries
                - What: A list of dictionaries with various transcription output data
        """
        if audio_file:
            audio_buffer = self.parse_audio(audio_file)
        if audio_buffer is None:
            self.exception("You must specify a valid audio_file or audio_buffer")
        output_meta=self.model.sttWithMetadata(audio_buffer, num_candidates)
        return [self.get_transcript_dict(transcript, **kwargs) for transcript in output_meta.transcripts]

    def stt_list(self, audio_file=None, audio_buffer=None, num_candidates=3):
        """
        Compute speech-to-text with extra data for N predicted candidates given an audio file

        Required:

            - `audio_buffer`:
                - Type: np.array
                - What: A 16-bit, mono raw audio signal at the appropriate sample rate serialized as a numpy array
                - Note: Exactly matches the DeepSpeech Model

            - OR

            - `audio_file`:
                - Type: str
                - What: The location of your target audio file to transcribe

        Optional:

            - `num_candidates`:
                - Type: int
                - What: The number of potential transcript candidates to return
                - Default: 1
                - Note: The most confident/likely result appears first

        Returns:

            - `transcriptions`:
                - Type: list of strs
                - What: A list of potential transcriptions in decending order of confidence
        """
        expanded_list=self.stt_expanded(audio_file=audio_file, audio_buffer=audio_buffer, num_candidates=num_candidates, return_text=True)
        return [i['text'] for i in expanded_list]
