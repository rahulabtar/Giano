from .serial_manager import LeftGloveSerialManager, RightGloveSerialManager, AudioBoardManager
from .protocols import PlayingMode, Hand, VoiceCommand
from src.core.constants import LEFT_PORT, RIGHT_PORT, SERIAL_BAUD_RATE
import time
import logging

# NOTE: this needs to be 


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# MAY NEED THREADSAFETY



def teensy_connect() -> tuple[LeftGloveSerialManager, RightGloveSerialManager, AudioBoardManager]:
  audio_board = AudioBoardManager()

  glove_1 = LeftGloveSerialManager(baud_rate=SERIAL_BAUD_RATE)
  glove_2 = LeftGloveSerialManager(baud_rate=SERIAL_BAUD_RATE)

  result_1, hand_1, glove_1 = glove_1.connect(num_retries=5, exclude_ports='COM7')
  result_2, hand_2, glove_2 = glove_2.connect(num_retries=5, exclude_ports='COM7')

  

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

def teensy_calibrate(left_glove: LeftGloveSerialManager, right_glove: RightGloveSerialManager, audio_board: AudioBoardManager):
  """
  Assumes connected to the gloves and audio board
  function returns when flush command is received
  """

  if not left_glove.is_connected() or not right_glove.is_connected() or not audio_board.is_connected():
    raise ValueError("Not connected to the gloves and audio board")

  print("swag is swag")
  left_glove.conn.reset_input_buffer()

  # entering calibation process
  time.sleep(1)
  
  while True:
    # read byte from left_glove
    # blocks until a byte is received
    command = left_glove.receive_byte()
    
    #flush will break the loop
    if command == VoiceCommand.FLUSH.value:
      logger.warning("Flush command received")
      break
    
    # print command
    if command is not None:
      print(f"command is {command}")

      match command:
        
        # LEARNING MODE CONFIRM VOICE COMMAND RECEIVED
        case VoiceCommand.LEARNING_MODE_CONFIRM.value:
          left_glove._play_mode = PlayingMode.LEARNING_MODE
          right_glove._play_mode = PlayingMode.LEARNING_MODE

          logger.info("Learning mode confirm voice command received")
          
          # send learning mode confirm byte to right glove
          right_glove.send_byte(PlayingMode.LEARNING_MODE.value)
          

          # send learning mode confirm voice command to audio board
          audio_board.play_voice_command(VoiceCommand.LEARNING_MODE_CONFIRM)


        # FREEPLAY MODE CONFIRM VOICE COMMAND RECEIVED
        case VoiceCommand.FREEPLAY_MODE_CONFIRM.value:
          left_glove._play_mode = PlayingMode.FREEPLAY_MODE
          right_glove._play_mode = PlayingMode.FREEPLAY_MODE

          logger.info("Freeplay mode confirm voice command received")
          
          # send freeplay mode confirm byte to right glove
          right_glove.send_byte(PlayingMode.FREEPLAY_MODE.value)
          
          # send freeplay mode confirm voice command to audio board
          audio_board.play_voice_command(VoiceCommand.FREEPLAY_MODE_CONFIRM)
          
        # Any other voice command byte received
        case _:
          command_enum = VoiceCommand(command)
          audio_board.play_voice_command(command_enum)

      time.sleep(0.2)
    else:
      print("No command received")
      time.sleep(0.2)
    
  return left_glove, right_glove, audio_board


if __name__ == "__main__":
  left_glove, right_glove, audio_board = teensy_connect()

  teensy_calibrate(left_glove, right_glove, audio_board)
  # if check mode
  #code for freeplay
  #code for learning mode

  while True:
    if left_glove._play_mode == PlayingMode.FreePlayMode:
      # if right glove is in the correct mode change to free play mode ???
      #right_glove.change_mode(PlayingMode.FREEPLAY_MODE)
      while True:
        # get responses from gloves
        left_glove.get_responses()
        right_glove.get_responses()
        # check if reesponse is a button interupt to exit free play mode
        #break
        time.sleep(0.1) #sleep needed 
    elif left_glove._play_mode == PlayingMode.LEARNING_MODE:
      pass

      


  

#gloves_connect()

# while True:
#     time.sleep(1)
#     glove_1.get_responses()
#     glove_2.get_responses()

# glove_2.start()