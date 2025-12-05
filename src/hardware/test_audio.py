
import time
import mido
from protocols import VoiceCommand
from serial_manager import AudioBoardManager
print(mido.get_output_names())
audio = AudioBoardManager()
while True:
    for i in range (9):
        print("Playing sine wave")
        audio.play_voice_command(VoiceCommand.CALIB_HARD_PRESS)
        time.sleep(1)
        print("Playing soft press")
        audio.play_voice_command(VoiceCommand.CALIB_SOFT_PRESS)
        time.sleep(1)
        print("Playing velo no press")
        audio.play_voice_command(VoiceCommand.CALIB_VELO_NO_PRESS)
        time.sleep(12)
        print("CALIBRATION SUCCESS")
        audio.play_voice_command(VoiceCommand.CALIBRATION_SUCCESS)
        time.sleep(1)


        # print("Turning Note On " + str(60 + i))
        # audio.note_on(60 + i, velocity=100)
        # time.sleep(0.5)
        # audio.note_on(65 + i, velocity=100)
        # time.sleep(0.5)
        # audio.note_on(70 + i, velocity=100)
        # time.sleep(3)
        # print("Turning Note Off " + str(60 + i))
        # audio.note_off(60 + i)
        # time.sleep(0.125)
        # audio.note_off(65 + i)
        # time.sleep(0.125)
        # audio.note_off(70 + i)
        # time.sleep(3)
        # print("Playing instruction notes")
        # audio.note_on(19 + i, velocity=100)
        # time.sleep(1)
        # audio.note_off(19 + i)
        # time.sleep(10)
