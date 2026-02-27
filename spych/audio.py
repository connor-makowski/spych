import numpy as np
import shlex, subprocess
import wave
from spych.utils import Notify

class Audio(Notify):
    def __init__(self, desired_sample_rate=16000, verbose=False):
        self.desired_sample_rate=desired_sample_rate

    def __execute_cmd__(self, cmd, capture_output=True, check=True):
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
    
    # Recorder Utilities
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
            assert output_audio_file.endswith('.wav'), "Output audio file must be a .wav file"
            sox_cmd = f'sox -d --channels 1 --rate {self.desired_sample_rate} --no-dither {shlex.quote(output_audio_file)} trim 0 {duration}'
            output=self.__execute_cmd__(sox_cmd)
            return output_audio_file
        else:
            sox_cmd = f'sox -d --type raw --bits 16 --channels 1 --rate {self.desired_sample_rate} --encoding signed-integer --endian little --compression 0.0 --no-dither - trim 0 {duration}'
            output = self.__execute_cmd__(sox_cmd)
            # Return the raw audio buffer in a format that whisper can process (16-bit, mono, raw audio signal at the appropriate sample rate serialized as a numpy array)
            return np.frombuffer(output.stdout, np.int16)

    def play(self, audio_file):
        """
        Play an audio file using SoX

        Required:

            - `audio_file`:
                - Type: str
                - What: The location of your target audio file to transcribe
                - Note: Must be a `.wav` file
        """
        assert audio_file.endswith('.wav'), "Audio file must be a .wav file"
        self.__execute_cmd__(f'sox {shlex.quote(audio_file)} -d')

    # Parser Utilities
    def __parse_audio_sox__(self, audio_file):
        """
        Attempt auto formatting your audio file to a 16-bit, mono raw audio signal at the appropriate sample rate serialized as a numpy array using SoX

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
        output = self.__execute_cmd__(sox_cmd)
        return np.frombuffer(output.stdout, np.int16)

    def parse_audio(self, audio_file):
        """
        Helper function to parse your raw audio file to match audio structures expected by the openai whisper model. If your audio file is not in the correct format, it will attempt to use SoX to convert it to the appropriate format.

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
            self.notify(f"Selected audio file is not in `.wav` format. Attempting SoX conversion.")
            try:
                return self.__parse_audio_sox__(audio_file=audio_file)
            except:
                self.notify(f"Audio parsing failed. Please ensure SoX is installed and your audio file is a valid format.", notification_type="exception")
        with wave.open(audio_file, 'rb') as audio_raw:
            audio_sample_rate = audio_raw.getframerate()
            if audio_sample_rate != self.desired_sample_rate:
                self.notify(f"Selected audio sample rate ({audio_sample_rate}) is different from the desired rate ({self.desired_sample_rate}). Attempting SoX conversion.")
                return self.__parse_audio_sox__(audio_file=audio_file)
            else:
                return np.frombuffer(audio_raw.readframes(audio_raw.getnframes()), np.int16)