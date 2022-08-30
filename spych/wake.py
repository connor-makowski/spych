from spych.core import spych
import time, threading


class wake_listener:
    """
    An internal class to be used as a thunkified listener function for threading purposes
    """

    def __init__(self, spych_wake_obj):
        """
        Internal function to initialize a wake_listener class

        Required:

            - `spych_wake_obj`:
                - Type: `spych_wake` class
                - What: An invoked spych_wake class to use for listening

        """
        self.spych_wake_obj = spych_wake_obj
        self.spych_object = spych_wake_obj.spych_object
        self.locked = False

    def __call__(self):
        """
        Internal function to allow for multiple thunkified (delayed call functional inputs)
        function access while only providing inputs once
        """
        if self.locked:
            return
        self.locked = True
        if self.spych_wake_obj.locked:
            self.locked = False
            return
        audio_buffer = self.spych_object.record(duration=self.spych_wake_obj.listen_time)
        if self.spych_wake_obj.locked:
            self.locked = False
            return
        if self.spych_wake_obj.candidates_per_listener > 1:
            transcriptions = self.spych_object.stt_list(
                audio_buffer=audio_buffer,
                num_candidates=self.spych_wake_obj.candidates_per_listener,
            )
            words = " ".join(transcriptions).split(" ")
        else:
            transcription = self.spych_object.stt(audio_buffer=audio_buffer)
            words = transcription.split(" ")
        if self.spych_wake_obj.wake_word in words:
            if self.spych_wake_obj.locked:
                self.locked = False
                return
            self.spych_wake_obj.locked = True
            self.spych_wake_obj.on_wake_fn()
            self.spych_wake_obj.locked = False
        self.locked = False


class spych_wake:
    """
    A spcial class to triger a wake function after hearing a wake word
    """

    def __init__(
        self,
        on_wake_fn,
        wake_word,
        spych_object=None,
        model_file=None,
        scorer_file=None,
        listeners=3,
        listen_time=2,
        candidates_per_listener=3,
    ):
        """
        Initialize a spych_wake class

        Required:

            - `on_wake_fn`:
                - Type: callable class or function
                - What: A no input callable class or function that is executed when the wake word is said
            - `wake_word`:
                - Type: str
                - What: The word that triggers the on_wake_fn function


            - `model_file`:
                - Type: str
                - What: The location of your deepspeech model file
                - Note: If provided, this class will automatically initialize a new spych_object given this `model_file`
                - Note: If `model_file` and `scorer_file` are both provided, then the `wake_word` is added as a hot word
            - OR
            - `spych_object`:
                - Type: spych object
                - What: An initialized spych object to use
                - Note: This is only used if a `model_file` is not specified

        Optional:

            - `scorer_file`:
                - Type: str
                - What: The location of your deepspeech scorer
                - Default: None
                - Note: Only used if `model_file` is specified
            - `listeners`:
                - Type: int
                - What: The amount of concurrent threads to listen for the wake word with
                - Default: 3
                - Note: To allow for continuous listening, at least three should be used
            - `listen_time`:
                - Type: int
                - What: The amount of time each listener will listen for the wake word
                - Default: 2
            - `candidates_per_listener`:
                - Type: int
                - What: The number of candidate transcripts to check for the wake word

        """

        self.on_wake_fn = on_wake_fn
        self.wake_word = wake_word
        self.listeners = listeners
        self.listen_time = listen_time
        self.candidates_per_listener = candidates_per_listener

        if model_file is None:
            if spych_object is None:
                self.exception("A spych_object or model_file must be supplied")
            self.spych_object = spych_object
        else:
            self.spych_object = spych(model_file=model_file, scorer_file=scorer_file)
            if scorer_file:
                self.spych_object.model.addHotWord(self.wake_word, 10.0)

        self.thunks = [wake_listener(spych_wake_obj=self) for i in range(self.listeners)]

        self.locked = False

    def start(self):
        """
        Start the spych_wake runtime to listen for the wake word
        """
        while True:
            for thunk in self.thunks:
                thread = threading.Thread(target=thunk)
                thread.start()
                time.sleep((self.listen_time + 1) / self.listeners)
