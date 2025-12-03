"""
Protocol definitions and message handling for hardware communication.

Defines binary protocols for:
- Glove controller: [motor_id, midi_note, action]
- Audio board: [command, note, velocity]
"""

import struct
from enum import Enum, IntEnum
from typing import Optional, Tuple, Dict, Any
import numpy as np
import mido

class Hand(IntEnum):
    LEFT = 0
    RIGHT = 1

class SensorValue(IntEnum):
    Pressed=0
    Released=1


# TODO: make sure I need to have this
class SensorNumberLeft(IntEnum):
    Thumb=0
    Index=1
    Middle=2
    Ring=3
    Pinky=4

class PlayMode(Enum):
    LEARNING_MODE = 0
    FREEPLAY_MODE = 1

"""The finger instruction set will be sent back to Python
from the Teensy once the velostat has detected a finger press.
Python will forward this information to the audio Teensy to play the note.
"""


class LearningModeGloveInstructionSet:
    fingerNumber: np.uint8
    midiNote: np.uint8  
    commandCode: np.uint8
    distanceToNote: np.float32

# Matched to firmware
class FreeplayModeGloveInstructionSet:
    hand: Hand
    sensorValue: SensorValue
    sensorNumber: SensorNumberLeft


"""The octave instruction set will be sent back to Python
from the Teensy once the hand position has been detected.
Python will forward this information to the audio Teensy to play the note.
Octave instruction set is only needed for learning mode.
"""
class OctaveInstructionSet:
    handPosition: tuple[float, float]




# TODO: Discuss w 
class ActionCode(IntEnum):
    """Action codes for glove controller haptic feedback."""
    HOVER_GUIDE = 0          # Gentle vibration to guide finger to key
    PRESS_CORRECT = 1        # Strong positive feedback for correct key
    PRESS_INCORRECT = 2      # Warning vibration for incorrect key
    RELEASE = 3              # Stop vibration on key release
    NO_KEY = 4               # No key detected under finger
    ERROR = 5                # Error state







class GloveProtocolFreeplayMode:
    """Protocol handler for glove controller communication."""
    
    # 3-byte message: [fingerInstructionSet]
    PACK_FORMAT = 'BBB'  # 3 unsigned bytes
    MESSAGE_SIZE = 3
    

    @staticmethod
    def pack(instruction_set: FreeplayModeGloveInstructionSet) -> bytes:
        pass
    
    @staticmethod
    def unpack(message: bytes) -> FreeplayModeGloveInstructionSet:
        message_tuple = struct.unpack(GloveProtocolFreeplayMode.PACK_FORMAT, message)
        
        return FreeplayModeGloveInstructionSet(
            hand=message_tuple[0],
            sensorValue=message_tuple[1],
            sensorNumber=message_tuple[2]
        )



class GloveProtocolLearningMode:
    """Protocol handler for glove controller communication."""
    
    # 3-byte message: [fingerInstructionSet]
    PACK_FORMAT = 'BBB'  # 3 unsigned bytes
    MESSAGE_SIZE = 3
    

    # TODO: FIX THIS FUNCTION
    @staticmethod
    def pack(motor_id: int, midi_note: int, action: int) -> bytes:
        """
        Pack glove command into binary message.
        
        Args:
            fingerInstructionSet: FingerInstructionSet object
            midi_note: MIDI note number (0-127)
            action: Action code from ActionCode enum
            
        Returns:
            3-byte binary message
            
        Raises:
            ValueError: If values are out of range
        """
        if not (0 <= motor_id <= 4):
            raise ValueError(f"motor_id must be 0-4, got {motor_id}")
        if not (0 <= midi_note <= 127):
            raise ValueError(f"midi_note must be 0-127, got {midi_note}")
        if not (0 <= action <= 127):
            raise ValueError(f"action must be 0-127, got {action}")
        
        return struct.pack(GloveProtocol.PACK_FORMAT, motor_id, midi_note, action)
    
    # TODO: WRITE THIS FUNCTION
    @staticmethod
    def unpack(message: bytes) -> Optional[Tuple[int, int, int]]:
        """
        Unpack glove message from binary data.
        
        Args:
            message: 3-byte binary message
            
        Returns:
            Tuple of (motor_id, midi_note, action) or None if invalid
        """
        if len(message) != GloveProtocol.MESSAGE_SIZE:
            return None
        
        try:
            motor_id, midi_note, action = struct.unpack(GloveProtocol.PACK_FORMAT, message)
            return (motor_id, midi_note, action)
        except struct.error:
            return None
    
    @staticmethod
    def validate_response(motor_id: int, midi_note: int, action: int) -> bool:
        """
        Validate response from glove controller.
        
        Args:
            motor_id, midi_note, action: Response values
            
        Returns:
            True if response is valid
        """
        # Add validation logic for responses (e.g., ACK, error codes)
        return 0 <= motor_id <= 4 and 0 <= midi_note <= 127 and 0 <= action <= 5


# ENUM FOR VOICE COMMANDS COMING FROM THE GLOVE CONTROLLER AND GOING TO THE AUDIO TEENSY
class VoiceCommand(IntEnum):
    WELCOME_MUSIC = 16
    WELCOME_TEXT = 17
    MODE_SELECT_BUTTONS = 18
    FREEPLAY_MODE_CONFIRM = 19
    LEARNING_MODE_SELECTED = 20
    CALIB_VELO_NO_PRESS = 21
    CALIB_SOFT_PRESS = 22
    CALIB_HARD_PRESS = 23
    HOW_TO_CHANGE_MODE = 24
    CONFIRM_SONG = 25
    CONFIRM_SELECTION = 26
    DEBUG = 254
    INVALID = 255

class AudioProtocol:
    
    def __init__(self, output = 'Teensy MIDI:Teensy MIDI MIDI 1 24:0'):
        self.out = mido.open_output(output)

    def note_on(self, note: int, velocity: int = 100):
        if (note < 60): return #values less than 60 are reserved for voice commands
        if note > 127:
            raise ValueError(f"Invalid MIDI note value: {note}. MIDI notes must be 0-127.")
        velocity = 127 if velocity > 127 else velocity
        velocity = 0 if velocity < 0 else velocity
        self.out.send(mido.Message('note_on', note=note, velocity=velocity, channel=0))
    
    def note_off(self, note: int, velocity: int = 0):
        if (note < 60): return #values less than 60 are reserved for voice commands
        if note > 127:
            raise ValueError(f"Invalid MIDI note value: {note}. MIDI notes must be 0-127.")
        self.out.send(mido.Message('note_off', note=note, velocity=velocity, channel=0))
    
    def play_voice_command(self, command: VoiceCommand):
        """
        Play a voice command by sending it as a MIDI note.
        
        WARNING: Values 254 (DEBUG) and 255 (INVALID) are outside the standard
        MIDI note range (0-127). These values will NOT work correctly with
        standard MIDI protocol. If your Teensy firmware requires these exact
        values, you may need to:
        1. Use raw serial communication instead of MIDI
        2. Map these to valid MIDI values (e.g., use Control Change messages)
        3. Use a different MIDI message type that supports these values
        
        For now, values > 127 will raise a ValueError.
        """
        note_value = command.value
        
        # MIDI note values are strictly 0-127
        if note_value > 127:
            raise ValueError(
                f"VoiceCommand {command.name} has value {note_value} which exceeds "
                f"MIDI note range (0-127). Cannot send via standard MIDI protocol. "
                f"Consider using raw serial or a different communication method."
            )
        
        self.out.send(mido.Message('note_on', note=note_value, velocity=127, channel=0))
        self.out.send(mido.Message('note_off', note=note_value, velocity=0, channel=0))




