"""
Common type definitions for the Giano system.
"""

from typing import NamedTuple, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np




# MIDI note representation
@dataclass  
class MidiNote:
    note: int           # MIDI note number (0-127)
    velocity: int       # Note velocity (0-127) 
    duration: float     # Duration in seconds
    start_time: float   # Start time in seconds
    finger: Optional[int] = None  # Suggested fingering (1-5)
    hand: Optional[str] = None    # "Left" or "Right"

