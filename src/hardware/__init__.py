"""Hardware communication modules for glove controller and audio board."""

from .serial_manager import (
    LeftGloveSerialManager,
    RightGloveSerialManager,
    AudioBoardManager,
    BaseSerialManager
)
from .protocols import (
    GloveProtocolFreeplayMode,
    GloveProtocolLearningMode,
    PlayingMode,
    Hand,
    SensorValue,
    VoiceCommand,
    ActionCode,
    
)

__all__ = [
    'LeftGloveSerialManager',
    'RightGloveSerialManager',
    'AudioBoardManager',
    'BaseSerialManager',
    'GloveProtocolFreeplayMode',
    'GloveProtocolLearningMode',
    'PlayingMode',
    'Hand',
    'SensorValue',
    'VoiceCommand',
    'ActionCode',
    'MIDIControl',
    'map_mediapipe_to_motor',
    'AudioProtocol',
    'ActionCode',
    
]

