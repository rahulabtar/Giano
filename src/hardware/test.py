
import time
import mido
from protocols import voice_command, AudioProtocol
print(mido.get_output_names())
audio = AudioProtocol()
while True:
    for i in range (9):
        print("SEND INSTR COMPLETE " + str(i))
        audio.play_voice_command(voice_command(i))
        time.sleep(2)
