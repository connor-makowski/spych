from spych.responders import BaseResponder
from spych import Spych, SpychWake

class MyResponder(BaseResponder):
    # A custom init function to set up any necessary variables or connections for your responder
    def __init__(self, spych_object):
        super().__init__(spych_object)
        # You can also customize the responder's name
        self.name = "Custom Responder"
    
    def respond(self, user_input: str) -> str:
        # Your custom logic here
        return "I am a custom responder and I heard: " + user_input

spych_object = Spych(whisper_model="base.en")
wake_object = SpychWake(
    wake_word_map={"test": MyResponder(spych_object)},
    whisper_model="tiny.en",
    terminate_words=["terminate"]
)
print("Starting wake listener. Say 'test' to trigger the MyResponder function.")
wake_object.start()