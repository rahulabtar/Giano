"""Hardware communication modules for glove controller and audio board."""

from .serial_manager import (
    SerialManager,
    GloveSerialManager,
    AudioSerialManager,
    BaseSerialManager
)
from .protocols import (
    GloveProtocol,
    AudioProtocol,
    ActionCode,
    MIDIControl,
    map_mediapipe_to_motor
)

__all__ = [
    'SerialManager',
    'GloveSerialManager',
    'AudioSerialManager',
    'BaseSerialManager',
    'GloveProtocol',
    'AudioProtocol',
    'ActionCode',
    'MIDIControl',
    'map_mediapipe_to_motor'
]

