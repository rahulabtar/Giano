"""
Finger state tracking for piano key detection.

Tracks finger positions over time, detects state changes, and provides
debounced updates to prevent jitter and redundant serial communication.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum
import time


class FingerState(IntEnum):
    """State of a finger relative to piano keys."""
    NO_KEY = 0      # No key detected
    HOVERING = 1    # Over a key but not pressed
    PRESSED = 2     # Key is being pressed (detected by glove)
    CORRECT = 3     # Correct key pressed
    INCORRECT = 4   # Incorrect key pressed


@dataclass
class FingerInfo:
    """Information about a single finger."""
    motor_id: int           # Motor ID (0-4)
    key_name: Optional[str] # Piano key name (e.g., "C4")
    midi_note: Optional[int] # MIDI note number
    u: float                 # Normalized X coordinate
    v: float                 # Normalized Y coordinate
    state: FingerState       # Current state
    last_update: float      # Timestamp of last update
    last_key: Optional[str] # Last detected key (for debouncing)


class FingerStateTracker:
    """
    Tracks finger states and detects changes to minimize redundant communication.
    """
    
    def __init__(self, debounce_time: float = 0.05):
        """
        Initialize finger state tracker.
        
        Args:
            debounce_time: Minimum time between state changes (seconds)
        """
        self.debounce_time = debounce_time
        self.fingers: Dict[int, FingerInfo] = {}
        
        # MediaPipe finger IDs to motor IDs
        self.mediapipe_to_motor = {4: 0, 8: 1, 12: 2, 16: 3, 20: 4}
    
    def update_fingers(self, finger_keys: Dict[int, Optional[Dict]]) -> Dict[int, Dict]:
        """
        Update finger states from detected keys.
        
        Args:
            finger_keys: Dictionary mapping MediaPipe finger ID to key info
            
        Returns:
            Dictionary of changed states {motor_id: state_info}
        """
        current_time = time.time()
        changes = {}
        
        for mediapipe_id, key_info in finger_keys.items():
            motor_id = self.mediapipe_to_motor.get(mediapipe_id, -1)
            if motor_id == -1:
                continue  # Unknown finger
            
            # Get existing finger info or create new
            if motor_id not in self.fingers:
                self.fingers[motor_id] = FingerInfo(
                    motor_id=motor_id,
                    key_name=None,
                    midi_note=None,
                    u=0.0,
                    v=0.0,
                    state=FingerState.NO_KEY,
                    last_update=current_time,
                    last_key=None
                )
            
            finger = self.fingers[motor_id]
            
            # Check if enough time has passed for debouncing
            time_since_update = current_time - finger.last_update
            if time_since_update < self.debounce_time:
                continue  # Skip debounce period
            
            # Determine new state and key
            new_key_name = key_info.get('key_name') if key_info else None
            new_midi_note = key_info.get('midi_note') if key_info else None
            
            # Detect state change
            key_changed = (new_key_name != finger.last_key)
            
            if key_changed:
                if new_key_name is None:
                    new_state = FingerState.NO_KEY
                else:
                    new_state = FingerState.HOVERING  # CV only detects hover, not press
                
                # Update finger info
                finger.key_name = new_key_name
                finger.midi_note = new_midi_note
                finger.state = new_state
                finger.last_update = current_time
                finger.last_key = new_key_name
                
                # Record change
                changes[motor_id] = {
                    'state': new_state,
                    'key_name': new_key_name,
                    'midi_note': new_midi_note
                }
        
        # Check for fingers that were removed (no longer in finger_keys)
        for motor_id, finger in list(self.fingers.items()):
            # Map motor_id back to mediapipe_id
            mediapipe_ids = [k for k, v in self.mediapipe_to_motor.items() if v == motor_id]
            
            if mediapipe_ids and mediapipe_ids[0] not in finger_keys:
                # Finger disappeared - update to NO_KEY state
                if finger.state != FingerState.NO_KEY:
                    finger.state = FingerState.NO_KEY
                    finger.key_name = None
                    finger.midi_note = None
                    finger.last_update = current_time
                    finger.last_key = None
                    
                    changes[motor_id] = {
                        'state': FingerState.NO_KEY,
                        'key_name': None,
                        'midi_note': None
                    }
        
        return changes
    
    def get_finger_state(self, motor_id: int) -> Optional[FingerInfo]:
        """Get current state of a finger by motor ID."""
        return self.fingers.get(motor_id)
    
    def get_all_states(self) -> Dict[int, FingerInfo]:
        """Get all current finger states."""
        return self.fingers.copy()
    
    def update_finger_state(self, motor_id: int, state: FingerState):
        """
        Manually update finger state (e.g., when glove detects press).
        
        Args:
            motor_id: Motor ID (0-4)
            state: New state (PRESSED, CORRECT, INCORRECT)
        """
        if motor_id in self.fingers:
            self.fingers[motor_id].state = state
            self.fingers[motor_id].last_update = time.time()
    
    def reset(self):
        """Reset all finger states."""
        self.fingers.clear()
    
    def get_state_summary(self) -> str:
        """Get human-readable summary of finger states."""
        if not self.fingers:
            return "No fingers detected"
        
        lines = []
        for motor_id in sorted(self.fingers.keys()):
            finger = self.fingers[motor_id]
            state_name = finger.state.name if finger.state else "UNKNOWN"
            key_str = f"{finger.key_name} ({finger.midi_note})" if finger.key_name else "None"
            lines.append(f"Motor {motor_id}: {state_name} - {key_str}")
        
        return "\n".join(lines)

