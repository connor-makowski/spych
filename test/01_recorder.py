from spych.audio import Audio

x = Audio()

x.record(output_audio_file='test.wav', duration=3)
x.play('test.wav')