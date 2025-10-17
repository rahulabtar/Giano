"""
Piano key layout configuration for hardcoded coordinate detection.
Defines the boundaries and properties of piano keys for all three keyboard sheets.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class KeyInfo:
    """Information about a piano key."""
    name: str
    midi_note: int
    is_black: bool
    u_min: float
    u_max: float
    v_min: float
    v_max: float

# Define key layouts for all three keyboards
KEYBOARD_LAYOUTS = {
    1: {  # Keys1: C3-B3 (Markers 10, 11)
        'octave': 3,
        'marker_ids': [10, 11],
        'white_keys': [
            KeyInfo('C3', 48, False, 0.000, 0.143, 0.2, 1.0),
            KeyInfo('D3', 50, False, 0.143, 0.286, 0.2, 1.0),
            KeyInfo('E3', 52, False, 0.286, 0.429, 0.2, 1.0),
            KeyInfo('F3', 53, False, 0.429, 0.571, 0.2, 1.0),
            KeyInfo('G3', 55, False, 0.571, 0.714, 0.2, 1.0),
            KeyInfo('A3', 57, False, 0.714, 0.857, 0.2, 1.0),
            KeyInfo('B3', 59, False, 0.857, 1.000, 0.2, 1.0),
        ],
        'black_keys': [
            KeyInfo('C#3', 49, True, 0.100, 0.186, 0.0, 0.5),
            KeyInfo('D#3', 51, True, 0.243, 0.329, 0.0, 0.5),
            KeyInfo('F#3', 54, True, 0.529, 0.615, 0.0, 0.5),
            KeyInfo('G#3', 56, True, 0.671, 0.757, 0.0, 0.5),
            KeyInfo('A#3', 58, True, 0.814, 0.900, 0.0, 0.5),
        ]
    },
    2: {  # Keys2: C4-B4 (Markers 12, 13)
        'octave': 4,
        'marker_ids': [12, 13],
        'white_keys': [
            KeyInfo('C4', 60, False, 0.000, 0.143, 0.2, 1.0),
            KeyInfo('D4', 62, False, 0.143, 0.286, 0.2, 1.0),
            KeyInfo('E4', 64, False, 0.286, 0.429, 0.2, 1.0),
            KeyInfo('F4', 65, False, 0.429, 0.571, 0.2, 1.0),
            KeyInfo('G4', 67, False, 0.571, 0.714, 0.2, 1.0),
            KeyInfo('A4', 69, False, 0.714, 0.857, 0.2, 1.0),
            KeyInfo('B4', 71, False, 0.857, 1.000, 0.2, 1.0),
        ],
        'black_keys': [
            KeyInfo('C#4', 61, True, 0.100, 0.186, 0.0, 0.5),
            KeyInfo('D#4', 63, True, 0.243, 0.329, 0.0, 0.5),
            KeyInfo('F#4', 66, True, 0.529, 0.615, 0.0, 0.5),
            KeyInfo('G#4', 68, True, 0.671, 0.757, 0.0, 0.5),
            KeyInfo('A#4', 70, True, 0.814, 0.900, 0.0, 0.5),
        ]
    },
    3: {  # Keys3: C5-B5 (Markers 14, 15)
        'octave': 5,
        'marker_ids': [14, 15],
        'white_keys': [
            KeyInfo('C5', 72, False, 0.000, 0.143, 0.2, 1.0),
            KeyInfo('D5', 74, False, 0.143, 0.286, 0.2, 1.0),
            KeyInfo('E5', 76, False, 0.286, 0.429, 0.2, 1.0),
            KeyInfo('F5', 77, False, 0.429, 0.571, 0.2, 1.0),
            KeyInfo('G5', 79, False, 0.571, 0.714, 0.2, 1.0),
            KeyInfo('A5', 81, False, 0.714, 0.857, 0.2, 1.0),
            KeyInfo('B5', 83, False, 0.857, 1.000, 0.2, 1.0),
        ],
        'black_keys': [
            KeyInfo('C#5', 73, True, 0.100, 0.186, 0.0, 0.5),
            KeyInfo('D#5', 75, True, 0.243, 0.329, 0.0, 0.5),
            KeyInfo('F#5', 78, True, 0.529, 0.615, 0.0, 0.5),
            KeyInfo('G#5', 80, True, 0.671, 0.757, 0.0, 0.5),
            KeyInfo('A#5', 82, True, 0.814, 0.900, 0.0, 0.5),
        ]
    }
}

def get_keyboard_layout(keyboard_id: int) -> Dict:
    """Get the layout configuration for a specific keyboard."""
    if keyboard_id not in KEYBOARD_LAYOUTS:
        raise ValueError(f"Invalid keyboard ID: {keyboard_id}. Must be 1, 2, or 3.")
    return KEYBOARD_LAYOUTS[keyboard_id]

def get_keyboard_id_from_markers(detected_marker_ids: List[int]) -> Optional[int]:
    """
    Determine which keyboard is being used based on detected marker IDs.
    
    Args:
        detected_marker_ids: List of detected ArUco marker IDs
        
    Returns:
        Keyboard ID (1, 2, or 3) or None if no match found
    """
    for keyboard_id, layout in KEYBOARD_LAYOUTS.items():
        marker_ids = layout['marker_ids']
        # Check if at least one of the expected markers is detected
        if any(marker_id in detected_marker_ids for marker_id in marker_ids):
            return keyboard_id
    return None

def get_all_keys_for_keyboard(keyboard_id: int) -> List[KeyInfo]:
    """Get all keys (white and black) for a specific keyboard."""
    layout = get_keyboard_layout(keyboard_id)
    return layout['white_keys'] + layout['black_keys']
