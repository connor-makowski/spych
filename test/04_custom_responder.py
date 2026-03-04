from spych.responders import BaseResponder
from spych import Spych, SpychWake


class MyResponder(BaseResponder):
    def respond(self, user_input: str) -> str:
        return f"'{self.name}' heard: {user_input}"


my_responder = MyResponder(
    spych_object=Spych(whisper_model="base.en"),
    listen_duration=5,
    name="TestResponder",
)

wake_object = SpychWake(
    wake_word_map={"test": my_responder},
    whisper_model="tiny.en",
    terminate_words=["terminate"],
)
my_responder.ready_message(wake_words=["test"], terminate_words=["terminate"])
wake_object.start()
