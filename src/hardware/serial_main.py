from serial_manager import LeftGloveSerialManager, RightGloveSerialManager, AudioBoardManager
from teensy_connector import teensy_connect
from protocols import PlayingMode, Hand, VoiceCommand
from src.core.constants import LEFT_PORT, RIGHT_PORT, SERIAL_BAUD_RATE
import time
import logging

num_gloves = 1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# MAY NEED THREADSAFETY


def teensy_connect():
  audio_board = AudioBoardManager()

  glove_1 = LeftGloveSerialManager(baud_rate=SERIAL_BAUD_RATE)
  glove_2 = LeftGloveSerialManager(baud_rate=SERIAL_BAUD_RATE)

  result_1, hand_1, glove_1 = glove_1.connect(num_retries=10)
  result_2, hand_2, glove_2 = glove_2.connect(num_retries=10)

  if not result_1 or not result_2:
    raise ValueError("Failed to connect to gloves")

  if hand_1 == Hand.LEFT and hand_2 == Hand.RIGHT:
    glove_left = glove_1
    glove_right = glove_2
  elif hand_1 == Hand.RIGHT and hand_2 == Hand.LEFT:
    glove_left = glove_2
    glove_right = glove_1
  else:
    raise ValueError("Invalid hands")

  return glove_left, glove_right, audio_board


if __name__ == "__main__":
  left_glove, right_glove, audio_board = teensy_connect()

  print("swag is swag")

  # entering calibation process
  time.sleep(1)
  while True:
    command = left_glove.receive_byte()
    if command == VoiceCommand.FLUSH.value:
      logger.info("Flush command received")
      continue
    if command is not None:
      print(f"command is {command}")

      if (command == PlayingMode.LEARNING_MODE.value):
        left_glove._play_mode = PlayingMode.LEARNING_MODE
        logger.info("Learning mode")
      elif (command == PlayingMode.FREEPLAY_MODE.value):
        left_glove._play_mode = PlayingMode.FREEPLAY_MODE
        logger.info("Freeplay mode")
      else:
        command_enum = VoiceCommand(command)
        audio_board.play_voice_command(command_enum)
      time.sleep(0.1)
    else:
      print("No command received")
      time.sleep(0.1)




  

  


#gloves_connect()

# while True:
#     time.sleep(1)
#     glove_1.get_responses()
#     glove_2.get_responses()

# glove_2.start()