from serial_manager import LeftGloveSerialManager, RightGloveSerialManager
from protocols import PlayingMode
from src.core.constants import LEFT_PORT, RIGHT_PORT, SERIAL_BAUD_RATE
import time

glove_1 = LeftGloveSerialManager(baud_rate=SERIAL_BAUD_RATE)
glove_2 = LeftGloveSerialManager(baud_rate=SERIAL_BAUD_RATE)

time.sleep(1)

correct_1, hand_1, glove_1 = glove_1.connect()
correct_2, hand_2, glove_2 = glove_2.connect()
print(f"Glove 1: {glove_1}")
print(f"Glove 2: {glove_2}")
print(f"Hand 1: {hand_1}")
print(f"Hand 2: {hand_2}")
print(f"Correct 1: {correct_1}")
print(f"Correct 2: {correct_2}")




while True:
    time.sleep(1)
    glove_1.get_responses()
    glove_2.get_responses()

glove_2.start()

while True:
    time.sleep(1)
    glove_1.get_responses()
    glove_2.get_responses()