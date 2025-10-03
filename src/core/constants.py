"""
Core constants used throughout the Giano system.
"""

# Physical constants
PI = 3.1415926
IN_TO_METERS = 0.0254

# ArUco marker configuration and enums
PAGE_DPI = 300
MARKER_SIZE = 1
CORNER_OFFSET = 0.25
PAPER_SIZES = {"LETTER": (8.5, 11.0)}
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

# File paths (relative to project root)
HAND_MODEL_PATH = "assets/models/hand_landmarker.task"
CAMERA_CALIBRATION_PATH = "assets/calibration/camera_calibration.npz"
CONFIG_DIR = "config"
ASSETS_DIR = "assets"