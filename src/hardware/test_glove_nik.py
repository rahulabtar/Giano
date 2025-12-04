from serial_manager import LeftGloveSerialManager, RightGloveSerialManager, AudioBoardManager
from protocols import PlayingMode, Hand, VoiceCommand
from src.core.constants import LEFT_PORT, RIGHT_PORT, SERIAL_BAUD_RATE
import time

num_gloves = 1


# MAY NEED THREADSAFETY

def teensy_connect():
    # Clear in case trying again
    #it's alwasy gonna be left_glove
    global audio_board, left_glove  # Declare as global to modify the global variables


    glove = None
    audio_board = None

    # result = audio_board.connect(exclude_ports=[LEFT_PORT, RIGHT_PORT]) no connect method for audioboard

    # Initialize and connect to the glove
    left_glove = LeftGloveSerialManager(baud_rate=SERIAL_BAUD_RATE, port = '/dev/cu.usbmodem179425001')
    left_glove.audio_board = AudioBoardManager(baud_rate=SERIAL_BAUD_RATE)
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
    if num_gloves == 2 and glove.hand == Hand.LEFT:
        return False  # Example condition for hand mismatch
    
    return True



if __name__ == "__main__":
  while True:
    if teensy_connect():
        break
    else:
        left_glove = None  # Reset the glove instance if connection fails

  print("swag")
  print("me")

  # Receive and process voice commands
  for message in ["Welcome message", "Mode Select Buttons Message", "Learning Mode or Freeplay Mode"]:
    command = left_glove.recieve_voice_command()
    print(f"command is {command} ({message})")
    audio_board.note_on(command - 1)



  # if (command == VoiceCommand.LEARNING_MODE_SELECTED):
  #   for glove in gloves: glove.set_playing_mode(PlayingMode.LEARNING_MODE)
  # elif (command == VoiceCommand.FREEPLAY_MODE_SELECTED):
  #   for glove in gloves: glove.set_playing_mode(PlayingMode.FREEPLAY_MODE)
  

  


#gloves_connect()

# while True:
#     time.sleep(1)
#     glove_1.get_responses()
#     glove_2.get_responses()

# glove_2.start()