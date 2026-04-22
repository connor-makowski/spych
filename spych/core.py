from faster_whisper import WhisperModel
from spych.utils import Notify, Recorder, get_clean_audio_buffer
from typing import Union, Optional


class Spych(Notify):
    def __init__(
        self,
        whisper_model: str = "base.en",
        whisper_device: str = "cpu",
        whisper_compute_type: str = "int8",
        no_speech_threshold: float = 0.4,
        vad_speech_threshold: float = 0.5,
        vad_silence_threshold: float = 0.35,
        vad_silence_frames_threshold: int = 40,
        vad_speech_pad_frames: int = 10,
        vad_max_speech_duration_s: float = 30.0,
    ) -> None:
        """
        Usage:

        - Initializes a transcription object using faster-whisper for fully offline
          speech-to-text

        Optional:

        - `whisper_model`:
            - Type: str
            - What: The faster-whisper model name to use for transcription
            - Default: "base.en"
            - Note: Larger models (small, medium, large) provide better accuracy at
              the cost of speed; smaller models (tiny, base) are faster but less accurate

        - `whisper_device`:
            - Type: str
            - What: The device to run the whisper model on
            - Default: "cpu"
            - Options: "cpu", "cuda"
            - Note: Use "cuda" for GPU acceleration if available

        - `whisper_compute_type`:
            - Type: str
            - What: The compute type to use for the whisper model
            - Default: "int8"
            - Options: "int8", "float16", "float32"
            - Note: "int8" offers a good balance of speed and accuracy on both CPU and GPU

        - `no_speech_threshold`:
            - Type: float
            - What: The threshold for the `no_speech_prob` returned by faster-whisper
            - Default: 0.4
            - Note: Segments with a `no_speech_prob` above this threshold will be
              ignored to reduce false positives from silence or background noise

        - `vad_speech_threshold`:
            - Type: float (0.0–1.0)
            - What: Silero probability above which a frame is considered speech onset
              when `listen` is called with `duration="auto"` or `duration=0`
            - Default: 0.5
            - Note: Raise to reduce false positives in noisy environments

        - `vad_silence_threshold`:
            - Type: float (0.0–1.0)
            - What: Silero probability below which a frame is considered silence
              during an active utterance; must be less than `vad_speech_threshold`
              to create a hysteresis band that prevents rapid toggling
            - Default: 0.35

        - `vad_silence_frames_threshold`:
            - Type: int
            - What: Consecutive silent frames (~32ms each) required to confirm the
              utterance has ended and return the buffer
            - Default: 40  (~1280ms)

        - `vad_speech_pad_frames`:
            - Type: int
            - What: Pre-roll frames captured before onset confirmation; also the
              number of consecutive voiced frames required to confirm speech onset
            - Default: 10  (~320ms)

        - `vad_max_speech_duration_s`:
            - Type: float
            - What: Hard cap on a single VAD-captured utterance in seconds; forces
              a return even if the speaker never pauses
            - Default: 30.0
        """
        self.wake_model = WhisperModel(
            whisper_model,
            device=whisper_device,
            compute_type=whisper_compute_type,
        )
        self.no_speech_threshold = no_speech_threshold
        self.vad_speech_threshold = vad_speech_threshold
        self.vad_silence_threshold = vad_silence_threshold
        self.vad_silence_frames_threshold = vad_silence_frames_threshold
        self.vad_speech_pad_frames = vad_speech_pad_frames
        self.vad_max_speech_duration_s = vad_max_speech_duration_s
        self.recorder = Recorder()

    def listen(
        self,
        duration: Union[int, float, str] = 0,
        device_index: int = -1,
        inactivity_timeout: Optional[float] = 8.0,
    ) -> str:
        """
        Usage:

        - Records audio from the microphone and returns the transcription as a string.
          Supports both fixed-duration recording and VAD-gated recording that
          automatically stops at the end of a natural utterance.

        Optional:

        - `duration`:
            - Type: int | float | str
            - What: Controls how long to record
            - Default: 0
            - Options:
                - int | float : Record for exactly this many seconds
                - "auto" or 0 : Use Silero VAD to detect a complete utterance and
                                stop automatically when the speaker finishes;
                                honours the `vad_*` parameters set at init time
            - Note: "auto" and 0 are equivalent; either signals VAD-gated recording

        - `device_index`:
            - Type: int
            - What: The microphone device index to record from
            - Default: -1
            - Note: Use `-1` to select the system default input device

        - `inactivity_timeout`:
            - Type: float | None
            - What: Seconds to wait for speech onset before returning an empty string
              when using VAD-gated recording (`duration="auto"` or `0`)
            - Default: 8.0 (wait for 8 seconds of inactivity)

        Returns:

        - `transcription`:
            - Type: str
            - What: The transcribed text from the recorded audio
            - Note: Multiple segments are joined with a single space
        """
        if duration == "auto" or duration == 0:
            buffer = self.recorder.record_vad(
                device_index=device_index,
                speech_threshold=self.vad_speech_threshold,
                silence_threshold=self.vad_silence_threshold,
                silence_frames_threshold=self.vad_silence_frames_threshold,
                speech_pad_frames=self.vad_speech_pad_frames,
                max_speech_duration_s=self.vad_max_speech_duration_s,
                inactivity_timeout=inactivity_timeout,
            )
        else:
            buffer = self.recorder.record(
                device_index=device_index, duration=duration
            )

        audio_buffer = get_clean_audio_buffer(buffer)
        segments, _ = self.wake_model.transcribe(audio_buffer, beam_size=2)
        return " ".join(
            [
                segment.text
                for segment in segments
                if segment.no_speech_prob <= self.no_speech_threshold
            ]
        )
