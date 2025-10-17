"""
Piano key detection system using hardcoded coordinate boundaries.
Detects which piano key a finger is hovering over based on normalized ArUco coordinates.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from ..core.piano_config import get_keyboard_layout, get_all_keys_for_keyboard, KeyInfo

class PianoKeyDetector:
    """
    Detects piano keys using hardcoded coordinate boundaries.
    Fast and reliable for consistent piano layouts.
    """
    
    def __init__(self, keyboard_id: int):
        """
        Initialize the piano key detector for a specific keyboard.
        
        Args:
            keyboard_id: Keyboard number (1, 2, or 3)
        """
        self.keyboard_id = keyboard_id
        self.layout = get_keyboard_layout(keyboard_id)
        self.all_keys = get_all_keys_for_keyboard(keyboard_id)
        
        # Create lookup structures for faster detection
        self._build_key_lookup()
    
    def _build_key_lookup(self):
        """Build internal lookup structures for efficient key detection."""
        # Sort keys by priority: black keys first (they're on top), then white keys
        self.black_keys = [key for key in self.all_keys if key.is_black]
        self.white_keys = [key for key in self.all_keys if not key.is_black]
    
    def get_key_at_position(self, u: float, v: float) -> Optional[Dict]:
        """
        Get the piano key at the specified normalized ArUco coordinates.
        
        Args:
            u: Normalized horizontal coordinate (0-1, left to right)
            v: Normalized vertical coordinate (0-1, top to bottom)
            
        Returns:
            Dictionary with key information or None if no key found:
            {
                'key_name': 'C4',
                'midi_note': 60,
                'is_black': False,
                'keyboard_id': 2
            }
        """
        # Check if position is within the piano area
        if not self._is_within_piano_area(u, v):
            return None
        
        # Check black keys first (they're positioned above white keys)
        for key in self.black_keys:
            if self._is_within_key_bounds(u, v, key):
                return {
                    'key_name': key.name,
                    'midi_note': key.midi_note,
                    'is_black': key.is_black,
                    'keyboard_id': self.keyboard_id
                }
        
        # Check white keys
        for key in self.white_keys:
            if self._is_within_key_bounds(u, v, key):
                return {
                    'key_name': key.name,
                    'midi_note': key.midi_note,
                    'is_black': key.is_black,
                    'keyboard_id': self.keyboard_id
                }
        
        return None
    
    def _is_within_piano_area(self, u: float, v: float) -> bool:
        """
        Check if coordinates are within the general piano area.
        
        Args:
            u: Normalized horizontal coordinate
            v: Normalized vertical coordinate
            
        Returns:
            True if within piano area, False otherwise
        """
        # Piano area: full width (0-1), from v=0.2 to v=1.0
        return 0.0 <= u <= 1.0 and 0.2 <= v <= 1.0
    
    def _is_within_key_bounds(self, u: float, v: float, key: KeyInfo) -> bool:
        """
        Check if coordinates are within a specific key's boundaries.
        
        Args:
            u: Normalized horizontal coordinate
            v: Normalized vertical coordinate
            key: KeyInfo object with boundaries
            
        Returns:
            True if within key bounds, False otherwise
        """
        return (key.u_min <= u <= key.u_max and 
                key.v_min <= v <= key.v_max)
    
    def detect_finger_keys(self, finger_positions: Dict[int, Tuple[float, float]]) -> Dict[int, Optional[Dict]]:
        """
        Detect keys for multiple finger positions.
        
        Args:
            finger_positions: Dictionary mapping finger_id to (u, v) coordinates
            
        Returns:
            Dictionary mapping finger_id to key information (or None)
        """
        results = {}
        for finger_id, (u, v) in finger_positions.items():
            results[finger_id] = self.get_key_at_position(u, v)
        return results
    
    def get_keyboard_info(self) -> Dict:
        """
        Get information about the current keyboard configuration.
        
        Returns:
            Dictionary with keyboard information
        """
        return {
            'keyboard_id': self.keyboard_id,
            'octave': self.layout['octave'],
            'marker_ids': self.layout['marker_ids'],
            'total_keys': len(self.all_keys),
            'white_keys': len(self.white_keys),
            'black_keys': len(self.black_keys)
        }
    
    def get_all_key_boundaries(self) -> List[Dict]:
        """
        Get boundaries for all keys in the current keyboard.
        
        Returns:
            List of dictionaries with key boundary information
        """
        boundaries = []
        for key in self.all_keys:
            boundaries.append({
                'name': key.name,
                'midi_note': key.midi_note,
                'is_black': key.is_black,
                'bounds': {
                    'u_min': key.u_min,
                    'u_max': key.u_max,
                    'v_min': key.v_min,
                    'v_max': key.v_max
                }
            })
        return boundaries
