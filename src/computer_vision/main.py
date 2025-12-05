# TODO: for piano calibration, I can just poll the color similar to the way it gets them in color_tracker.py


# Add project root to Python path for direct execution
from operator import eq
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2 as cv


import numpy as np
import time as time
import math
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ArUco system from the pose_tracker module
try:
    from src.computer_vision.piano_calibration import PianoCalibration
    from src.computer_vision.aruco_pose_tracker import ArucoPoseTracker
    from src.computer_vision.aruco_polygon_detector import ArucoPolygonDetector
    from src.computer_vision.aruco_pose_tracker import TrackingMode
    from src.computer_vision.color_tracker import ColorTracker, GLOVE_COLORS
    from src.computer_vision.finger_aruco_tracker import FingerArucoTracker
    from src.hardware.serial_manager import LeftGloveSerialManager, RightGloveSerialManager, AudioBoardManager
    from src.hardware.serial_main import teensy_connect, teensy_calibrate
    from src.hardware.protocols import ActionCode, PlayingMode, Hand, SensorValue, VoiceCommand, GloveProtocolFreeplayMode, SensorNumberLeft
    from src.core.constants import MARKER_SIZE, MARKER_IDS, HAND_MODEL_PATH, CAMERA_CALIBRATION_PATH, IN_TO_METERS, PI
    from src.core.utils import name_from_midi
except ImportError as e:
    print(f"Could not import user stuff: {e}")
    raise





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
    #first thing we will do is connect to the serial port
    left_glove, right_glove, audio_board = teensy_connect()
    if not left_glove.is_connected() or not right_glove.is_connected() or not audio_board.is_connected():
        raise ValueError("Not connected to the gloves and audio board")
    
    logger.info("Connected to the gloves and audio board")
    # enter calibration process
    left_glove, right_glove, audio_board = teensy_calibrate(left_glove, right_glove, audio_board)
    
    if left_glove._play_mode == PlayingMode.LEARNING_MODE:
        logger.info("Left glove is in learning mode")
    
    elif left_glove._play_mode == PlayingMode.FREEPLAY_MODE:
        logger.info("Left glove is in freeplay mode")
    
    else:
        raise ValueError("Invalid mode")

    left_glove.start()
    right_glove.start()
    # sets the global playing mode
    playing_mode = left_glove._play_mode
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
    
    

    
    # Let user select camera
    # camera_id = list_and_select_camera(max_cameras=10)
    # TODO: change hardcoding
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {0}")
        return
    # initialize piano calibration
    # initialize aruco stuff
    tracker = ColorTracker(min_area=1000,max_area=50000, max_number_of_colors=5, max_trackers_per_color=2, max_total_trackers=10)

    aruco_pose_tracker = ArucoPoseTracker(camera_matrix, dist_coeffs, mode = TrackingMode.STATIC)
    finger_aruco = FingerArucoTracker(camera_matrix, dist_coeffs,
                                      image_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))),
                                      correct_camera_distortion = True)

    
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

    status, piano_calibration_result = piano_calibrator.get_piano_calibration(cap, debug_mode=False, start_midi=48, expected_keys=53)
    if status == 2:
        print("Calibration cancelled by user")
        return
    
    finger_aruco.set_keyboard_map(piano_calibration_result['labeled_keys'])
    
    frame_count = 0
    
    # Track active MIDI notes per hand and finger/color
    # Structure: {hand: {color: midi_number}}
    active_notes = {"left": {}, "right": {}}
    
    # ================================================ MAIN LOOP ================================================
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

        
        # ============================= HAND FINDER PART =============================
        birdseye_image = finger_aruco.transform_image_to_birdseye(image)
      
        tracked_image = tracker.process_frame(birdseye_image, draw_contours=True, draw_colors=True, draw_centroids=False)
        # we'll get both of these
        tracked_boxes = tracker.get_tracked_boxes('all')

        # get the centroids of all colors
        tracked_centroids = tracker.get_centroids('all')
        
        
        if playing_mode == PlayingMode.FREEPLAY_MODE:
        
        # get the responses from the gloves
            left_glove_responses = left_glove.get_from_recv_queue()
            
            right_glove_responses = right_glove.get_from_recv_queue()
            

            if left_glove_responses:
                logger.info(f"Left glove responses: {left_glove_responses}")
            # Process left glove responses: expect 4 single-byte messages -> list of 4 bytes
            if len(left_glove_responses) >= 4:
                left_glove_instruction_set = GloveProtocolFreeplayMode.unpack(left_glove_responses[:4])
                logger.info(f"Unpacked left glove instruction: {left_glove_instruction_set}")
                
                # Get finger name from index (fingerIndex is 0-4)
                finger_index = int(left_glove_instruction_set.fingerIndex)
                if finger_index in SensorNumberLeft:
                    color_to_find = SensorNumberLeft[finger_index]
                else:
                    logger.warning(f"Invalid finger index: {finger_index}")
                    color_to_find = None
            else:
                logger.debug(f"Waiting for 4 left glove responses (have {len(left_glove_responses)})")
                left_glove_instruction_set = None
                color_to_find = None

            if left_glove_instruction_set and color_to_find:
                if left_glove_instruction_set.sensorValue == SensorValue.Pressed:
                    logger.info(f"Left glove finger: {color_to_find} pressed")

                    tracked_boxes = tracker.get_tracked_boxes(color_to_find)
                    
                    for box in tracked_boxes.get(color_to_find, []):
                        x, y, w, h = box
                        midi_number = finger_aruco.find_closest_key(x + w/2, y + h, method='centroid')
                        midi_note_name = name_from_midi(midi_number)
                        velocity = left_glove_instruction_set.velocity

                        audio_board.note_on(midi_number, velocity)
                        # Store the MIDI note for this finger/color so we can turn it off later
                        active_notes["left"][color_to_find] = midi_number
                        logger.info(f"Playing note {midi_note_name} with velocity {velocity}")
                else:
                    # Retrieve the stored MIDI note for this finger/color and turn it off
                    if color_to_find in active_notes["left"]:
                        midi_number = active_notes["left"][color_to_find]
                        audio_board.note_off(midi_number)
                        del active_notes["left"][color_to_find]  # Remove from tracking
                        midi_note_name = name_from_midi(midi_number)
                        logger.info(f"Left glove finger: {color_to_find} released, stopped note {midi_note_name}")
                    else:
                        logger.warning(f"No active note found for left glove finger: {color_to_find}")

            

            # Process right glove responses: expect 4 single-byte messages -> list of 4 bytes
            if right_glove_responses:
                logger.info(f"Right glove responses: {right_glove_responses}")
            if len(right_glove_responses) >= 4:
                right_glove_instruction_set = GloveProtocolFreeplayMode.unpack(right_glove_responses[:4])
                logger.info(f"Unpacked right glove instruction: {right_glove_instruction_set}")
                
                # Get finger name from index (fingerIndex is 0-4)
                finger_index = int(right_glove_instruction_set.fingerIndex)
                if finger_index in SensorNumberLeft:  # Assuming same mapping for right hand
                    color_to_find_right = SensorNumberLeft[finger_index]
                else:
                    logger.warning(f"Invalid finger index: {finger_index}")
                    color_to_find_right = None
            else:
                logger.debug(f"Waiting for 4 right glove responses (have {len(right_glove_responses)})")
                right_glove_instruction_set = None
                color_to_find_right = None

            if right_glove_instruction_set and color_to_find_right:
                if right_glove_instruction_set.sensorValue == SensorValue.Pressed:
                    logger.info(f"Right glove finger: {color_to_find_right} pressed")

                    tracked_boxes_right = tracker.get_tracked_boxes(color_to_find_right)
                    
                    for box in tracked_boxes_right.get(color_to_find_right, []):
                        x, y, w, h = box
                        midi_number = finger_aruco.find_closest_key(x + w/2, y + h, method='centroid')
                        midi_note_name = name_from_midi(midi_number)
                        velocity = right_glove_instruction_set.velocity

                        audio_board.note_on(midi_number, velocity)
                        time.sleep(0.1)
                        # Store the MIDI note for this finger/color so we can turn it off later
                        active_notes["right"][color_to_find_right] = midi_number
                        logger.info(f"Playing note {midi_note_name} with velocity {velocity}")
                else:
                    # Retrieve the stored MIDI note for this finger/color and turn it off
                    if color_to_find_right in active_notes["right"]:
                        midi_number = active_notes["right"][color_to_find_right]
                        audio_board.note_off(midi_number)
                        del active_notes["right"][color_to_find_right]  # Remove from tracking
                        midi_note_name = name_from_midi(midi_number)
                        logger.info(f"Right glove finger: {color_to_find_right} released, stopped note {midi_note_name}")
                    else:
                        logger.warning(f"No active note found for right glove finger: {color_to_find_right}")
            

                
            # tracked_boxes = tracker.get_tracked_boxes('all')
            # # find the closest key for each bbox
            # for color_name, bboxes in tracked_boxes.items():
            #     color_tracking_points = []
            #     if len(bboxes) > 1:
            #         logger.warning(f"Multiple bboxes found for color {color_name}")
            #         continue
                
            #     for i, bbox in enumerate(bboxes):
            #         x, y, w, h = bbox
            #         # get the closest point to the edge of the finger
            #         tracking_point = int(x + w/2), int(y + h)
            #         midi_number = finger_aruco.find_closest_key(tracking_point[0], tracking_point[1], method='centroid')
                    
            #         # this might break
            #         tracked_image = cv.drawMarker(tracked_image, tracking_point, (0, 255, 0), cv.MARKER_CROSS, 10)
            #         midi_note_name = name_from_midi(midi_number)
            #         print(f"Color {color_name} bbox {i}: {x}, {y}, {w}, {h}: {midi_note_name}")
            #         color_tracking_points.append((tracking_point[0], tracking_point[1], midi_number))
                
            #     # sort by x coordinate, because the keyboard is backwards, assume the leftmost x is the righthand
            #     # TODO: see if the kalman filter tracks the hand
            #     sorted_color_tracking_points = sorted(color_tracking_points, key=lambda x: x[0])
            
            
            


        cv.imshow("Tracked", tracked_image)


        # find hands, return drawn image
        
        
        # birdseye_image = finger_aruco.draw_birdseye_keys(image, lm_list, finger_keys)

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


            # birdseye_image = finger_aruco.transform_image_to_orthographic_plane(image, poses, plane_extent_meters=(2.0, 0.6), output_size=(1280, 720))
            # Resize for display if image is too large
            # display_image = resize_for_display(birdseye_image, max_width=1280, max_height=720)
        cv.imshow("Birdseye", birdseye_image)
        cv.imshow("Original", image)

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