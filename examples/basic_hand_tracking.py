"""
Basic hand tracking example using the reorganized Giano system.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import cv2 as cv
import numpy as np
from computer_vision.hand_tracking import HandTracker
from core.constants import DEFAULT_CAMERA_ID, MAX_HANDS
from core.utils import load_config, get_asset_path

def main():
    """Run basic hand tracking demonstration."""
    
    print("Starting basic hand tracking example...")
    
    # Initialize hand tracker
    try:
        # Load camera configuration
        camera_config = load_config("camera")
        camera_id = camera_config.get("camera_id", DEFAULT_CAMERA_ID)
    except FileNotFoundError:
        print("Camera config not found, using defaults")
        camera_id = DEFAULT_CAMERA_ID
    
    # Initialize video capture
    cap = cv.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    # Initialize hand tracker
    hand_tracker = HandTracker(max_hands=MAX_HANDS)
    
    print("Hand tracking started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process frame for hand detection
        frame_with_hands = hand_tracker.hands_finder(frame, draw=True)
        
        # Get hand position data
        hands_data = hand_tracker.position_finder(frame)
        
        # Display hand information
        if hands_data:
            for i, hand_data in enumerate(hands_data):
                cv.putText(frame_with_hands, 
                          f"Hand {i+1}: {hand_data['handedness']}", 
                          (10, 30 + i*30), 
                          cv.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 255, 0), 2)
        
        # Display frame
        cv.imshow('Giano Hand Tracking', frame_with_hands)
        
        # Check for quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv.destroyAllWindows()
    print("Hand tracking stopped.")

if __name__ == "__main__":
    main()