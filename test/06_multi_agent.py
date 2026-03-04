# NOTE: THIS TEST FAILS CURRENTLY

from spych import SpychWake, Spych
from spych.agents import OllamaResponder, LocalClaudeCodeCLIResponder

spych = Spych(whisper_model="base.en", whisper_device="cpu")

ollama = OllamaResponder(
    spych_object=spych,
    model="llama3.2:latest",
)
claude = LocalClaudeCodeCLIResponder(
    spych_object=spych,
)

wake = SpychWake(
    wake_word_map={
        "llama": ollama,
        "claude": claude,
    },
    whisper_model="tiny.en",
    terminate_words=["terminate"],
)
ollama.ready_message(wake_words=["llama"], terminate_words=["terminate"])
ollama.spinner.stop()
claude.ready_message(wake_words=["claude"], terminate_words=["terminate"])

wake.start()
