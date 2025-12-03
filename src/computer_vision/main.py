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
    from src.computer_vision.piano_calibration import PianoCalibration
    from src.computer_vision.aruco_pose_tracker import ArucoPoseTracker
    from src.computer_vision.aruco_polygon_detector import ArucoPolygonDetector
    from src.computer_vision.aruco_pose_tracker import TrackingMode
    from src.computer_vision.hand_tracking import HandTracker 
    from src.computer_vision.finger_aruco_tracker import FingerArucoTracker
    from src.computer_vision.finger_state_tracker import FingerStateTracker, FingerState
    from src.hardware.serial_manager import SerialManager
    from src.hardware.protocols import ActionCode
    from src.core.constants import MARKER_SIZE, MARKER_IDS, HAND_MODEL_PATH, CAMERA_CALIBRATION_PATH, IN_TO_METERS, PI
    from src.core.utils import name_from_midi
except ImportError as e:
    print(f"Could not import user stuff: {e}")
    raise


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH, delegate='GPU'), 
    running_mode='LIVE_STREAM'
)


def resize_for_display(image: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
    """
    Resize an image for display while maintaining aspect ratio.
    
    Args:
        image: Input image (BGR or grayscale)
        max_width: Maximum display width in pixels (default: 1280)
        max_height: Maximum display height in pixels (default: 720)
        
    Returns:
        Resized image that fits within max_width x max_height
    """
    if image is None:
        return image
    
    h, w = image.shape[:2]
    
    # Calculate scaling factor to fit within max dimensions
    scale_w = max_width / w if w > max_width else 1.0
    scale_h = max_height / h if h > max_height else 1.0
    scale = min(scale_w, scale_h)
    
    # Only resize if scaling is needed
    if scale < 1.0:
        new_width = int(w * scale)
        new_height = int(h * scale)
        return cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
    
    return image


def main():
    # Choose which camera to use. Default for computer onboard webcam is 0
    # available_cams = test_available_cams(2)

    # Choose the second one because that SHOULD be the webcam on laptop
     # attempt to open the calibration file as a z
    try:
        calib_npz = np.load(CAMERA_CALIBRATION_PATH)
        camera_matrix = calib_npz["camera_matrix"]
        dist_coeffs = calib_npz["dist_coeffs"]
    except(OSError): 
        print("Calibration file not found!")
        return
    except:
        print("Issue with reading calibration file")
        return
    
    

    # # Initialize serial communication manager
    # print("Initializing serial communication...")
    # serial_manager = SerialManager(auto_connect=True)
    # time.sleep(1)  # Give time for connections to establish
    
    # # Check connection status
    # connections = serial_manager.is_connected()
    # print(f"Connection status - Glove: {connections['glove']}, Audio: {connections['audio']}")
    
    # # Start serial manager
    # serial_manager.start()

    # Let user select camera
    # camera_id = list_and_select_camera(max_cameras=10)
    # TODO: change hardcoding
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {0}")
        return
    # initialize piano calibration
    # initialize aruco stuff
    tracker = HandTracker()
    aruco_pose_tracker = ArucoPoseTracker(camera_matrix, dist_coeffs, mode = TrackingMode.STATIC)
    finger_aruco = FingerArucoTracker(camera_matrix, dist_coeffs,
                                      image_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))),
                                      correct_camera_distortion = True)

    finger_tracker = FingerStateTracker(debounce_time=0.05)
    
    piano_calibrator = PianoCalibration(camera_matrix, 
                                        dist_coeffs, 
                                        aruco_pose_tracker, 
                                        finger_aruco, 
                                        MARKER_IDS, 
                                        MARKER_SIZE*IN_TO_METERS,
                                        )
    
    # Configure pose filtering - you can experiment with these settings
    aruco_pose_tracker.configure_pose_filtering(
        enable_filtering=False,         # Enable similarity filtering
        adaptive_thresholds=False,      # Use adaptive thresholds  
        debug_output=False,              # Set to True to see filtering decisions
        enforce_z_axis_out=False,       # Force Z-axis to always point toward camera
        enable_moving_average=True,     # Enable TRUE LTI moving average filter
        filter_window_size=20            # Number of samples to average (higher = smoother)
    )
    
    # Configure adaptive preprocessing for handling high intensity changes (direct light, etc.)
    # This helps detect ArUco markers even when some regions are overexposed
    aruco_pose_tracker.configure_adaptive_preprocessing(
        enable=True,                    # Enable adaptive preprocessing
        use_clahe=True,                 # Use CLAHE for local contrast enhancement
        use_adaptive_threshold=False,    # Adaptive thresholding (usually too aggressive for ArUco)
        use_histogram_equalization=False, # Global histogram equalization (can be too aggressive)
        use_gamma_correction=True,       # Gamma correction for brightness adjustment
        try_multiple_strategies=True,    # Try multiple strategies if detection fails
        clahe_clip_limit=2.0,           # CLAHE clip limit (higher = more contrast)
        clahe_tile_size=(8, 8),         # CLAHE tile size (smaller = more local adaptation)
        gamma_correction_value=1.5      # Gamma value (>1.0 = brighter, <1.0 = darker)
    )

   

    
    # FPS monitor setup
    prev_time = 0
    poses_list = []

    status, piano_calibration_result = piano_calibrator.get_piano_calibration(cap, debug_mode=False)
    if status == 2:
        print("Calibration cancelled by user")
        return
    
    finger_aruco.set_keyboard_map(piano_calibration_result['labeled_keys'])
    
    frame_count = 0
    # MAIN LOOP
    while True:
        success, image = cap.read()
        if not success:
            print("Could not take picture!")
            break
        h, w, c = image.shape
        
        
        # find aruco markers and FIFO marker poses list logic
        # TODO: look into Kalman filter for marker tracking
        if frame_count % 30 == 0:
            if len(poses_list) > 0: last_poses = poses_list.pop()
            else: last_poses = None
            poses = aruco_pose_tracker.get_marker_poses(
                image=image, 
                marker_size_meters=MARKER_SIZE*IN_TO_METERS, 
                last_poses=last_poses
                )
            poses_list.append(poses)
            # for pose in poses:
                # image = cv.drawFrameAxes(image, camera_matrix, dist_coeffs, pose['rvec'], pose['tvec'], 0.1, 10)
            
            # Aruco polygon
            success, _ = finger_aruco.get_marker_polygon(MARKER_IDS, poses)
            image = finger_aruco.draw_box(image)

      
            
            frame_count+=1
            if frame_count >= 30:
                frame_count=0

        
        

        # Piano key detection and serial communication
        # if piano_detector is not None and not np.array_equal(aruco_polygon.polygon, [0,0,0,0]):
        #     # Get finger-key mappings
        #     finger_keys = finger_aruco.get_finger_keys(lm_list, aruco_polygon.polygon, piano_detector)
            
        #     # Update finger state tracker and get changes
        #     changes = finger_tracker.update_fingers(finger_keys)
            
        #     # Check current connection status
        #     connections = serial_manager.is_connected()
            
        #     # Send glove commands for changed states
        #     if connections['glove']:
        #         for motor_id, state_info in changes.items():
        #             midi_note = state_info.get('midi_note', 0)
        #             action = state_info.get('state', FingerState.NO_KEY).value
                    
        #             # Send command to glove
        #             serial_manager.send_glove_command(
        #                 motor_id=motor_id,
        #                 midi_note=midi_note if midi_note else 0,
        #                 action=action
        #             )
                    
        #             # Debug output
        #             if state_info.get('key_name'):
        #                 print(f"Motor {motor_id}: {state_info['key_name']} (MIDI {midi_note}) - Action: {action}")
            
        #     # Draw finger keys on image
        #     image = finger_aruco.draw_finger_keys(image, finger_keys, lm_list)
        
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
        
        # Draw serial connection status (check dynamically)
        # connections = serial_manager.is_connected()
        # glove_status = "G" if connections['glove'] else "g"
        # audio_status = "A" if connections['audio'] else "a"
        # status_text = f"{glove_status}{audio_status}"
        # cv.putText(image, status_text, (10, 30), font, font_scale, (255, 255, 0), thickness)
        # find hands, return drawn image
        # HAND FINDER PART
        image = tracker.hands_finder(image, draw=False)
        lm_list, absolute_pos = tracker.position_finder(image, hand_no = 0, draw=False)

        finger_keys = {}
        for landmark in lm_list:
            lm_id, x_px, y_px = landmark[0], landmark[1], landmark[2]
            if lm_id in [4,8,12,16,20] and (x_px != 0 and y_px != 0):
                midi_note = finger_aruco.find_closest_key(x_px, y_px)
                midi_note = finger_aruco.find_closest_key_with_full_correction(x_px, y_px)
                name, is_black_from_midi = name_from_midi(midi_note)
                distance = finger_aruco.measure_distance_to_key(x_px, y_px, midi_note)
                if lm_id == 4:
                    distance_to_keys = {}
                    
                    for key in finger_aruco.keyboard_map:
                        distance_to_keys[key['name']] = finger_aruco.measure_distance_to_key(x_px, y_px, key['midi_note'])
                    print(f"Distance to keys: {distance_to_keys}")
                    print(f"Finger {lm_id}: Regular image coords {x_px, y_px}, MIDI note {name}, Distance {distance}")
                finger_keys[lm_id] = midi_note
        
        if len(lm_list) > 0:
            # birdseye_image = finger_aruco.draw_birdseye_keys(image, lm_list, finger_keys)
            birdseye_image = finger_aruco.transform_image_to_orthographic_plane(image, poses, plane_extent_meters=(2.0, 0.6), output_size=(1280, 720))
            # Resize for display if image is too large
            # display_image = resize_for_display(birdseye_image, max_width=1280, max_height=720)
            cv.imshow("Birdseye", birdseye_image)
            cv.imshow("Original", image)
        else:
            cv.imshow("Video", image)

        # Exit the loop if the 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    #closing everything properly !!! DO NOT REMOVE
    print("Shutting down...")
    # serial_manager.stop()
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


def list_and_select_camera(max_cameras: int = 10) -> int:
    """
    List available cameras with details and let user select one.
    
    Args:
        max_cameras: Maximum number of camera indices to test
        
    Returns:
        Selected camera index, or 0 if none selected
    """
    print("\n=== Scanning for available cameras ===")
    available_cameras = []
    
    for i in range(max_cameras):
        cap = cv.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                
                # Try to get backend name (if available)
                backend = cap.getBackendName()
                
                # Try to get additional properties
                fps = cap.get(cv.CAP_PROP_FPS)
                
                camera_info = {
                    'index': i,
                    'width': w,
                    'height': h,
                    'backend': backend,
                    'fps': fps
                }
                available_cameras.append(camera_info)
                
                print(f"  [{i}] {w}x{h} @ {fps:.1f}fps (Backend: {backend})")
            cap.release()
    
    if not available_cameras:
        print("No cameras found! Using default camera 0.")
        return 0
    
    print(f"\nFound {len(available_cameras)} available camera(s).")
    
    # Let user select
    while True:
        try:
            selection = input(f"Select camera (0-{len(available_cameras)-1}) or press Enter for default (0): ").strip()
            
            if selection == "":
                # Use first available camera
                selected_index = available_cameras[0]['index']
                print(f"Using default camera {selected_index}")
                return selected_index
            
            camera_num = int(selection)
            
            # Find camera with this index
            for cam_info in available_cameras:
                if cam_info['index'] == camera_num:
                    print(f"Selected camera {camera_num}: {cam_info['width']}x{cam_info['height']}")
                    return camera_num
            
            print(f"Camera {camera_num} not available. Please select from the list above.")
            
        except ValueError:
            print("Invalid input. Please enter a number or press Enter for default.")
        except KeyboardInterrupt:
            print("\nCancelled. Using default camera 0.")
            return 0



if __name__ == "__main__":
    main()