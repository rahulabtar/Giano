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


# TODO: Add MIDI control codes
class MIDIControl(IntEnum):
    """MIDI command codes."""
    NOTE_OFF = 0x80
    NOTE_ON = 0x90
    CONTROL_CHANGE = 0xB0


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

class voice_command(IntEnum):
    "BOOTED" = 0
    "WELCOME" = 1
    "SELSONG" = 2
    "LEARNMOD" = 3
    "HOWRS" = 4
    "HOWMOD" = 5   
    "FREEMOD" = 6
    "DEBUG" = 7
    "CONFSONG" = 8
    "CONFIRM" = 9

class AudioProtocol:
    
    def __init__(self, output = 'Teensy MIDI:Teensy MIDI MIDI 1 24:0'):
        self.out = mido.open_output(output)

    def note_on(self, note: int, velocity: int = 100):
        if (note < 60): return #values less than 60 are reserved for voice commands
        velocity = 127 if velocity > 127 else velocity
        velocity = 0 if velocity < 0 else velocity
        self.out.send(mido.Message('note_on', note=note, velocity=velocity, channel=0))
    
    def note_off(self, note: int, velocity: int = 0):
        if (note < 60): return #values less than 60 are reserved for voice commands
        self.out.send(mido.Message('note_off', note=note, velocity=velocity, channel=0))
    
    def play_voice_command(self, command: voice_command):
        self.out.send(mido.Message('note_on', note=command.value, velocity=127, channel=0))
        self.out.send(mido.Message('note_off', note=command.value, velocity=0, channel=0))


def map_mediapipe_to_motor(mediapipe_finger_id: int) -> int:
    """
    Map MediaPipe finger ID to motor ID.
    
    MediaPipe finger IDs: 4=thumb, 8=index, 12=middle, 16=ring, 20=pinky
    Motor IDs: 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky
    
    Args:
        mediapipe_finger_id: MediaPipe landmark ID
        
    Returns:
        Motor ID (0-4)
    """
    mapping = {4: 0, 8: 1, 12: 2, 16: 3, 20: 4}
    return mapping.get(mediapipe_finger_id, -1)

