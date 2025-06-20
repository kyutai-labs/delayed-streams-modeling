import sounddevice as sd
import numpy as np

print("Recording 3 seconds...")
audio = sd.rec(3 * 24000, samplerate=24000, channels=1, dtype='float32')
sd.wait()
print("Playing back...")
sd.play(audio, samplerate=24000)
sd.wait()
