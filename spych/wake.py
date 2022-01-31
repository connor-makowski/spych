from spych.core import spych
from pamda import pamda
import time, threading

class wake_listener:
    def __init__(self, spych_wake_obj, spych_object, file):
        self.spych_wake_obj=spych_wake_obj
        self.spych_object=spych_object
        self.file=file
        self.locked=False

    def __call__(self):
        if self.locked:
            return
        self.locked=True
        if self.spych_wake_obj.locked:
            self.locked=False
            return
        wake=self.spych_object.record(output_audio_file=self.file, duration=self.spych_wake_obj.listen_time)
        if self.spych_wake_obj.locked:
            self.locked=False
            return
        transcriptions=self.spych_object.stt_expanded(audio_file=wake, num_candidates=self.spych_wake_obj.candidates_per_listener)
        word_data=[i['words'] for i in transcriptions]
        words=[i for sub_list in word_data for i in sub_list]
        if self.spych_wake_obj.wake_word in words:
            if self.spych_wake_obj.locked:
                self.locked=False
                return
            self.spych_wake_obj.locked=True
            self.spych_wake_obj.on_wake_fn()
            self.spych_wake_obj.locked=False
        self.locked=False

class spych_wake:
    def __init__(self, model_file, scorer_file, on_wake_fn, wake_word, listeners=3, listen_time=2, candidates_per_listener=3):
        self.model_file=model_file
        self.scorer_file=scorer_file
        self.on_wake_fn=on_wake_fn
        self.wake_word=wake_word
        self.listeners=listeners
        self.listen_time=listen_time
        self.candidates_per_listener=candidates_per_listener

        self.locked=False

    def asyncify(self, fn):
        thread=threading.Thread(target=fn)
        thread.start()

    def start(self):
        thunks=[]
        for i in range(self.listeners):
            spych_object=spych(model_file=self.model_file, scorer_file=self.scorer_file)
            thunks.append(wake_listener(spych_wake_obj=self, spych_object=spych_object, file=f'test_{i}.wav'))
        while True:
            for thunk in thunks:
                self.asyncify(thunk)
                time.sleep((self.listen_time+1)/self.listeners)
