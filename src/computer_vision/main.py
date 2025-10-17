# Add project root to Python path for direct execution
from operator import eq
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2 as cv
import mediapipe as mp
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


import numpy as np
import time as time
import math
import matplotlib.pyplot as plt


# Import ArUco system from the pose_tracker module
try:
    from src.computer_vision.aruco_pose_tracker import ArucoPoseTracker
    from src.computer_vision.aruco_polygon_detector import ArucoPolygonDetector
    from src.computer_vision.hand_tracking import HandTracker 
    from src.computer_vision.finger_aruco_tracker import FingerArucoTracker
    from src.computer_vision.piano_key_detector import PianoKeyDetector
    from src.core.constants import MARKER_SIZE, MARKER_IDS, HAND_MODEL_PATH, CAMERA_CALIBRATION_PATH, IN_TO_METERS, PI
    from src.core.piano_config import get_keyboard_id_from_markers

except ImportError:
    print(f"Could not import user stuff")
    raise


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH, delegate='GPU'), 
    running_mode='LIVE_STREAM'
)


def main():
    # Choose which camera to use. Default for computer onboard webcam is 0
    # available_cams = test_available_cams(2)

    # Choose the second one because that SHOULD be the webcam on laptop
     # attempt to open the calibration file as a z
    try:
        calib_npz = np.load(CAMERA_CALIBRATION_PATH)
        
    except(OSError): 
        print("Calibration file not found!")
    except:
        print("Issue with reading calibration file")
    
    camera_matrix = calib_npz["camera_matrix"]
    dist_coeffs = calib_npz["dist_coeffs"]

    cap = cv.VideoCapture(0)
    
    tracker = HandTracker()
    aruco_pose_tracker = ArucoPoseTracker()
    aruco_polygon = ArucoPolygonDetector(camera_matrix, dist_coeffs)
    finger_aruco = FingerArucoTracker()
    piano_detector = None  # Will be initialized when keyboard is detected
    
    # Configure pose filtering - you can experiment with these settings
    aruco_pose_tracker.configure_pose_filtering(
        enable_filtering=False,         # Enable similarity filtering
        adaptive_thresholds=False,      # Use adaptive thresholds  
        debug_output=False,              # Set to True to see filtering decisions
        enforce_z_axis_out=False,       # Force Z-axis to always point toward camera
        enable_moving_average=True,     # Enable TRUE LTI moving average filter
        filter_window_size=20            # Number of samples to average (higher = smoother)
    )

   

    
    # FPS monitor setup
    prev_time = 0
    i = 0
    poses_list = []
    while True:
        success, image = cap.read()
        if not success:
            print("Could not take picture!")
            break
        h, w, c = image.shape
        
        # find aruco markers and FIFO marker poses list logic
        if len(poses_list) > 0: last_poses = poses_list.pop()
        else: last_poses = None
        poses = aruco_pose_tracker.get_marker_poses(image, 
                camera_matrix,
                dist_coeffs,
                marker_size_meters=MARKER_SIZE*IN_TO_METERS, 
                last_poses=last_poses)
        poses_list.append(poses)
        # for pose in poses:
            # image = cv.drawFrameAxes(image, camera_matrix, dist_coeffs, pose['rvec'], pose['tvec'], 0.1, 10)
        
        
        # Aruco polygon
        marker_centers_2d = aruco_polygon.get_marker_polygon(MARKER_IDS, poses, image, MARKER_SIZE)
        image = aruco_polygon.draw_box(image, marker_centers_2d=marker_centers_2d)
        if not np.array_equal(marker_centers_2d, [0,0,0,0]):
            image = finger_aruco.transform_image_to_birdseye(image, marker_centers_2d)
        # Determine which keyboard is being used and initialize piano detector
        if poses and piano_detector is None:
            detected_marker_ids = [pose['id'] for pose in poses]
            keyboard_id = get_keyboard_id_from_markers(detected_marker_ids)
            if keyboard_id is not None:
                piano_detector = PianoKeyDetector(keyboard_id)
                print(f"Initialized piano detector for keyboard {keyboard_id}")
        cv.threshold()
        i+=1
        if i >= 30:
            for j, pose in enumerate(poses): 
                #print(f"Pose {j}: {pose}")
                pass
            i=0

        # find hands, return drawn image
        # HAND FINDER PART
        image = tracker.hands_finder(image)
        lm_list, absolute_pos = tracker.position_finder(image, hand_no = 0, draw=False)
        """
        # Piano key detection
        if piano_detector is not None and not np.array_equal(marker_centers_2d, [0,0,0,0]):
            finger_keys = finger_aruco.get_finger_keys(lm_list, marker_centers_2d, piano_detector)
            image = finger_aruco.draw_finger_keys(image, finger_keys, lm_list)
            
        else:
            # Fallback to original coordinate display
            for landmark in lm_list:
                lm_id, x_px, y_px = landmark[0], landmark[1], landmark[2]
                # fingertip IDs
                if lm_id in [4,8,12,16,20]:
                    aruco_coords = finger_aruco.transform_finger_to_aruco_space((x_px,y_px), marker_centers_2d)
                    if aruco_coords is not None:
                        print(f"Finger {lm_id}: ArUco coords {aruco_coords}")
        """
        
        for landmark in lm_list:
                lm_id, x_px, y_px = landmark[0], landmark[1], landmark[2]
                # fingertip IDs
                if lm_id in [4,8,12,16,20]:
                    aruco_coords = finger_aruco.transform_finger_to_aruco_space((x_px,y_px), marker_centers_2d)
                    if aruco_coords is not None:
                        print(f"Finger {lm_id}: ArUco coords {aruco_coords}")


       

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Draw FPS on top right corner
        fps_text = f"FPS: {int(fps)}"
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (0, 255, 0)
        thickness = 2
        text_size, _ = cv.getTextSize(fps_text, font, font_scale, thickness)
        text_x = w - text_size[0] - 10
        text_y = 30
        cv.putText(image, fps_text, (text_x, text_y), font, font_scale, color, thickness)

        cv.imshow("Video", image)

        # Exit the loop if the 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    #closing everything properly !!! DO NOT REMOVE
    cap.release()
    cv.destroyAllWindows()
    
def test_available_cams(num_cams: int) -> list:
    """Test available cameras.
    
    Args:
        num_cams: the number of camera indices to test. Fewer will be faster.    
    """
    available_cams = []
    for i in range(num_cams):
        test_cap = cv.VideoCapture(i)
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"Camera {i}: {w}x{h} resolution")
                available_cams.append(i)
            test_cap.release()
        else:
            print(f"Camera {i} not available")
    return available_cams



if __name__ == "__main__":
    main()