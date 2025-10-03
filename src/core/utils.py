"""
Shared utility functions for the Giano system.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

def get_project_root() -> Path:
    """Get the absolute path to the project root directory."""
    return Path(__file__).parent.parent.parent

def load_config(config_name: str) -> Dict[str, Any]:
    """Load configuration from config directory."""
    config_path = get_project_root() / "config" / f"{config_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config_name: str, config_data: Dict[str, Any]) -> None:
    """Save configuration to config directory."""
    config_path = get_project_root() / "config" / f"{config_name}.json"
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

def get_asset_path(asset_name: str) -> Path:
    """Get absolute path to an asset file."""
    return get_project_root() / "assets" / asset_name

def midi_note_to_frequency(note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))

def velocity_to_amplitude(velocity: int, scaling_factor: int = 5) -> float:
    """Convert MIDI velocity to audio amplitude (0.0 - 1.0)."""
    if scaling_factor > 10:
        scaling_factor = 10
    return np.log(1.0 + scaling_factor * velocity / 127.0) / np.log(1.0 + scaling_factor)

def ensure_directory_exists(path: Path) -> None:
    """Ensure a directory exists, create if it doesn't."""
    path.mkdir(parents=True, exist_ok=True)