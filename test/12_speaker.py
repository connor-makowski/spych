from spych.speaker import Speaker

speaker = Speaker()
speaker.speak("Hello! I am your default voice.")

michael = Speaker(voice="am_michael")
michael.speak("And I am your American male voice.")

emma = Speaker(voice="bf_emma")
emma.speak("And I am your British female voice.")

george = Speaker(voice="bm_george")
george.speak("And I am your British male voice.")

isabella = Speaker(voice="bf_isabella")
isabella.speak("And I am your British female voice with a C grade.")