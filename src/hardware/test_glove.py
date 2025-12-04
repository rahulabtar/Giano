from serial_manager import LeftGloveSerialManager, RightGloveSerialManager, AudioProtocol
from protocols import PlayingMode, Hand
from src.core.constants import LEFT_PORT, RIGHT_PORT, SERIAL_BAUD_RATE
import time

def teensy_connect():
  audio_board = AudioProtocol()
  result = audio_board.connect(exclude_ports=[LEFT_PORT, RIGHT_PORT])

  glove_1 = LeftGloveSerialManager(baud_rate=SERIAL_BAUD_RATE)
  glove_2 = LeftGloveSerialManager(baud_rate=SERIAL_BAUD_RATE)

  glove_1.connect(num_retries=10)
  glove_2.connect(num_retries=10)

  if glove_1.hand == Hand.LEFT:
    glove_left = glove_1
    glove_right = glove_2
  else:
    glove_left = glove_2
    glove_right = glove_1

  return glove_left, glove_right

gloves_connect()

# while True:
#     time.sleep(1)
#     glove_1.get_responses()
#     glove_2.get_responses()

# glove_2.start()