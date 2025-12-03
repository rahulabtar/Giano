
import time
import mido
from protocols import VoiceCommand, AudioProtocol
print(mido.get_output_names())
audio = AudioProtocol()
while True:
    for i in range (9):
        print("Turning Note On " + str(60 + i))
        audio.note_on(60 + i, velocity=100)
        time.sleep(6)
        print("Turning Note Off " + str(60 + i))
        audio.note_off(60 + i)
        time.sleep(3)
