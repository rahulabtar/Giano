"""
Protocol definitions and message handling for hardware communication.

Defines binary protocols for:
- Glove controller: [motor_id, midi_note, action]
- Audio board: [command, note, velocity]
"""

import struct
from enum import Enum, IntEnum
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import mido

# TODO: add const for finger colors
# TODO: flush as first byte to indicate mode change

#-------------------------------- GLOVE COLORS --------------------------------



#-------------------------------- COMMON SHARED --------------------------------

# mode change command could be flush

# Defines the "hand byte" during connection
class Hand(IntEnum):
    AUDIO = 253
    LEFT =  254
    RIGHT = 255

# TODO: change on firmware
class SensorValue(IntEnum):
    Pressed =  7
    Released = 8



SensorNumberLeft = {
    0: "Thumb",
    1: "Pointer",
    2: "Middle",
    3: "Ring",
    4: "Pinky"
}

# TODO: ADD ON FIRMWARE
class PlayingMode(Enum):
    LEARNING_MODE = 5
    FREEPLAY_MODE = 6

"""The finger instruction set will be sent back to Python
from the Teensy once the velostat has detected a finger press.
Python will forward this information to the audio Teensy to play the note.
"""


class LearningModeGloveInstructionSet:
    fingerNumber: np.uint8
    midiNote: np.uint8  
    commandCode: np.uint8
    distanceToNote: np.float32

# Matched to firmware, ordered
# this should be a uint_8
@dataclass
class FreeplayModeGloveInstructionSet:
    hand: Hand
    fingerIndex: int
    velocity: int
    sensorValue: SensorValue


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
    PACK_FORMAT = 'BBBB'  # 4 unsigned bytes
    MESSAGE_SIZE = 4
    

    @staticmethod
    def pack(instruction_set: FreeplayModeGloveInstructionSet) -> bytes:
        pass
    
    @staticmethod
    def unpack(message: list[bytes] | tuple[bytes, ...]) -> FreeplayModeGloveInstructionSet:
        """
        Unpack a 4-byte freeplay message into a FreeplayModeGloveInstructionSet.
        
        Expects a list/tuple of exactly 4 single-byte messages (each bytes, length >= 1).
        """
        if not isinstance(message, (list, tuple)):
            raise TypeError(f"Expected list/tuple of 4 bytes, got {type(message)}")
        if len(message) != 4:
            raise ValueError(f"Expected 4 single-byte messages, got {len(message)}")
        
        # Combine the first byte of each element; if any element is empty, use 0x00.
        combined = b"".join([(m[:1] if len(m) > 0 else b"\x00") for m in message])
        
        if len(combined) != GloveProtocolFreeplayMode.MESSAGE_SIZE:
            raise ValueError(f"Expected {GloveProtocolFreeplayMode.MESSAGE_SIZE} bytes after combining, got {len(combined)}")
        
        message_tuple = struct.unpack(GloveProtocolFreeplayMode.PACK_FORMAT, combined)
        
        return FreeplayModeGloveInstructionSet(
            hand=Hand(message_tuple[0]),
            fingerIndex=int(message_tuple[1]),  # Store as integer index (0-4)
            velocity=int(message_tuple[2]),
            sensorValue=SensorValue(int(message_tuple[3]))
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



#-------------------------------- AUDIO --------------------------------

# ENUM FOR VOICE COMMANDS COMING FROM THE GLOVE CONTROLLER AND GOING TO THE AUDIO TEENSY
class VoiceCommand(IntEnum):
    #music played on boot
    WELCOME_MUSIC = 16
    #text played on boot
    WELCOME_TEXT = 17
    # missing
    MODE_SELECT_BUTTONS = 18
    SELECT_SONG = 19
    # confirmation of freeplay mode selection
    FREEPLAY_MODE_CONFIRM = 20
    LEARNING_MODE_CONFIRM = 21
    CALIB_VELO_NO_PRESS = 22
    CALIB_SOFT_PRESS = 23
    CALIB_HARD_PRESS = 24
    HOW_TO_CHANGE_MODE = 25
    HOW_TO_RESET_SONG = 26
    FLUSH = 27
    DEBUG = 28
    INVALID = 29
    CALIBRATING = 30
    CALIBRATE_SINE_WAVE = 31
    CALIBRATION_FAILED = 32
    CALIBRATION_SUCCESS = 33
    FIX_POSTURE = 34




