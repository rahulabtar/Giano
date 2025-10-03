"""
Common type definitions for the Giano system.
"""

from typing import NamedTuple, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

# 3D point representation
class Point3D(NamedTuple):
    x: float
    y: float 
    z: float

# Hand landmark data
@dataclass
class HandLandmark:
    position: Point3D
    visibility: float
    presence: float

@dataclass
class HandData:
    landmarks: List[HandLandmark]
    handedness: str  # "Left" or "Right"
    confidence: float

# MIDI note representation
@dataclass  
class MidiNote:
    note: int           # MIDI note number (0-127)
    velocity: int       # Note velocity (0-127) 
    duration: float     # Duration in seconds
    start_time: float   # Start time in seconds
    finger: Optional[int] = None  # Suggested fingering (1-5)
    hand: Optional[str] = None    # "Left" or "Right"

# System state
@dataclass
class SystemState:
    hands_detected: List[HandData]
    current_notes: List[MidiNote]
    piano_pose: Optional[np.ndarray] = None  # 4x4 transformation matrix
    timestamp: float = 0.0