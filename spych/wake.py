from spych.core import spych
import time, threading

class wake_listener:
    """
    An internal class to be used as a thunkified listener function for threading purposes
    """
    def __init__(self, spych_wake_obj, spych_object):
        """
        Internal function to initialize a wake_listener class

        Required:

            - `spych_wake_obj`:
                - Type: `spych_wake` class
                - What: An invoked spych_wake class to use for listening
            - `spych_obj`:
                - Type: `spych` class
                - What: An invoked spych class to use for generating transcripts

        """
        self.spych_wake_obj=spych_wake_obj
        self.spych_object=spych_object
        self.locked=False

    def __call__(self):
        """
        Internal function to allow for multiple thunkified (delayed call functional inputs)
        function access while only providing inputs once
        """
        if self.locked:
            return
        self.locked=True
        if self.spych_wake_obj.locked:
            self.locked=False
            return
        audio_buffer=self.spych_object.record(duration=self.spych_wake_obj.listen_time)
        if self.spych_wake_obj.locked:
            self.locked=False
            return
        transcriptions=self.spych_object.stt_list(audio_buffer=audio_buffer, num_candidates=self.spych_wake_obj.candidates_per_listener)
        words=" ".join(transcriptions).split(" ")
        if self.spych_wake_obj.wake_word in words:
            if self.spych_wake_obj.locked:
                self.locked=False
                return
            self.spych_wake_obj.locked=True
            self.spych_wake_obj.on_wake_fn()
            self.spych_wake_obj.locked=False
        self.locked=False

class spych_wake:
    """
    A spcial class to triger a wake function after hearing a wake word
    """
    def __init__(self, model_file, on_wake_fn, wake_word, scorer_file=None, listeners=3, listen_time=2, candidates_per_listener=3):
        """
        Initialize a spych_wake class

        Required:

            - `model_file`:
                - Type: str
                - What: The location of your deepspeech model
            - `on_wake_fn`:
                - Type: callable class or function
                - What: A no input callable class or function that is executed when the wake word is said
            - `wake_word`:
                - Type: str
                - What: The word that triggers the on_wake_fn function

        Optional:

            - `scorer_file`:
                - Type: str
                - What: The location of your deepspeech scorer
                - Default: None
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
        self.model_file=model_file
        self.scorer_file=scorer_file
        self.on_wake_fn=on_wake_fn
        self.wake_word=wake_word
        self.listeners=listeners
        self.listen_time=listen_time
        self.candidates_per_listener=candidates_per_listener

        self.locked=False

    def start(self):
        """
        Start the spych_wake runtime to listen for the wake word
        """
        thunks=[]
        for i in range(self.listeners):
            spych_object=spych(model_file=self.model_file, scorer_file=self.scorer_file)
            thunks.append(wake_listener(spych_wake_obj=self, spych_object=spych_object))
        while True:
            for thunk in thunks:
                thread=threading.Thread(target=thunk)
                thread.start()
                time.sleep((self.listen_time+1)/self.listeners)
