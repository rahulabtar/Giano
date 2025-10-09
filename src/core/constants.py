"""
Core constants used throughout the Giano system.
"""

import os
from dataclasses import dataclass

# Physical constants
PI = 3.1415926
IN_TO_METERS = 0.0254

# ArUco marker configuration and enums
PAGE_DPI = 300
MARKER_SIZE = 1.2
CORNER_OFFSET = 0.25

# Def common paper sizes
@dataclass
class PaperSize:
    width: float
    height: float
    
class PAPER_SIZES:
    LETTER = PaperSize(8.5, 11.0)
    A4 = PaperSize(8.27, 11.69)
    A3 = PaperSize(8.5, 14.0)

MARKER_IDS = [10,11,12,13,14,15]

# Hand tracking parameters
MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5

# MIDI constants
MIDDLE_C = 60
VELOCITY_SCALE = 127

# Communication protocols
SERIAL_BAUD_RATE = 115200
DEFAULT_CAMERA_ID = 0

# Get the project root directory (two levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# File paths (absolute paths from project root)
HAND_MODEL_PATH = os.path.join(PROJECT_ROOT, "assets", "models", "hand_landmarker.task")
CAMERA_CALIBRATION_PATH = os.path.join(PROJECT_ROOT, "assets", "calibration", "camera_calibration.npz")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")