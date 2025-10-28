"""Hardware communication modules for glove controller and audio board."""

from .serial_manager import SerialManager
from .protocols import (
    GloveProtocol,
    AudioProtocol,
    ActionCode,
    MIDIControl,
    map_mediapipe_to_motor
)

__all__ = [
    'SerialManager',
    'GloveProtocol',
    'AudioProtocol',
    'ActionCode',
    'MIDIControl',
    'map_mediapipe_to_motor'
]

