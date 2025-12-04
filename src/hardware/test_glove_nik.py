from serial_manager import LeftGloveSerialManager, RightGloveSerialManager, AudioBoardManager
from protocols import PlayingMode, Hand, VoiceCommand
from src.core.constants import LEFT_PORT, RIGHT_PORT, SERIAL_BAUD_RATE
import time
import logging

num_gloves = 1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# MAY NEED THREADSAFETY

def teensy_connect():
    # Clear in case trying again
    #it's alwasy gonna be left_glove


    # result = audio_board.connect(exclude_ports=[LEFT_PORT, RIGHT_PORT]) no connect method for audioboard

    # Initialize and connect to the glove

    right_glove = None
    left_glove = None
    audio_board = None

    left_glove = LeftGloveSerialManager(baud_rate=SERIAL_BAUD_RATE, port = 'COM10')
    audio_board = AudioBoardManager()
    print("Initialized the glove.")

    

    success, hand, left_glove = left_glove.connect(num_retries=10)
    
 

    if hand == Hand.RIGHT:
      right_glove = left_glove
      left_glove = None


    if not success:
      # Try to connect to the glove
      print("Could not connect to serial port")
      return False

    # Check if the glove is the correct hand (if applicable)
    if num_gloves == 2 and right_glove.hand == Hand.LEFT:
        return False  # Example condition for hand mismatch
    
    return left_glove, right_glove, audio_board



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