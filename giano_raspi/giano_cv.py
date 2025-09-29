import cv2 as cv
import mediapipe as mp
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


import numpy as np
import time as time
import os
import math
import matplotlib.pyplot as plt
import sys

# Add giano_utils directory to Python path
UTILS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'giano_utils'))
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

# Import ArUco system from giano_utils
try:
    from giano_aruco import ArucoMarkerSystem
    from aruco_add_to_sheet import MARKER_SIZE
except ImportError:
    print(f"Could not import from {UTILS_DIR}")
    raise

PI = 3.1415926
IN_TO_METERS = 0.0254

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task', delegate='GPU'),
)

class HandTracker():
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, model_complexity=1, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.model_complexity = model_complexity
        self.track_con = track_con
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity,
                                        self.detection_con, self.track_con)
        self.mp_draw = mp.solutions.drawing_utils

    # find hands and draw landmarks on image
    def hands_finder(self, image, draw=True):
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(image, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return image
    
    # find position of landmarks
    def position_finder(self, image, hand_no=0, draw=False):
        lm_list = []
        lm_absolute = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]
            
            for id, lm in enumerate(hand.landmark):
                #get height and width of the webcam view
                h, w, c = image.shape
                
                #return the x and y pixel locations for each landmark
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                lm_absolute.append([id, lm.x, lm.y])
                
                #custom circles
                if draw:
                    cv.circle(image, (cx, cy), 10, (math.sin(PI*lm.x)*255, -1*math.sqrt(math.pow(lm.x, 2)*math.pow(lm.y, 2)), lm.y*255), cv.FILLED)  
        else:
            lm_list = [[j,0,0] for j in range(20)]
            lm_absolute = lm_list
            
        #conditional way to find hand2 based on errors passed when it isn't shown on camera   
        try:
            hand2 = self.results.multi_hand_landmarks[1]
            for id2, lm2 in enumerate(hand2.landmark):
                cx, cy = int(lm2.x * w), int(lm2.y * h)
                lm_list.append([id2, cx, cy])
                lm_absolute.append([id2, lm.x, lm.y])
                
                if draw:
                    cv.circle(image, (cx, cy), 10, (math.sin(PI*lm2.x)*255, -math.sqrt(math.pow(lm2.x, 2)*math.pow(lm2.y, 2)), lm2.y*255), cv.FILLED)
        except:
            pass
 
        return lm_list, lm_absolute


def main():
    # Choose which camera to use. Default for computer onboard webcam is 0
    # available_cams = test_available_cams(2)

    # Choose the second one because that SHOULD be the webcam on laptop
    cap = cv.VideoCapture(0)
    
    tracker = HandTracker()
    aruco_sys = ArucoMarkerSystem()

    # attempt to open the calibration file
    try:
        calib_npz = np.load("camera_calibration.npz")
        
    except(OSError): 
        print("Calibration file not found!")
    except:
        print("Issue with reading calibration file")
    
    camera_matrix = calib_npz["camera_matrix"]
    dist_coeffs = calib_npz["dist_coeffs"]

    
    # FPS monitor setup
    prev_time = 0
    i = 0
    while True:
        success, image = cap.read()
        if not success:
            print("Could not take picture!")
            break
        h, w, c = image.shape
        
        # find aruco markers
        poses = aruco_sys.get_marker_poses(image, camera_matrix, dist_coeffs, marker_size_meters=MARKER_SIZE*IN_TO_METERS)
        aruco_sys.detect_markers(image)
        for pose in poses:
            image = cv.drawFrameAxes(image, camera_matrix, dist_coeffs, pose['rvec'], pose['tvec'], 0.1, 10)
        
        i+=1
        if i >= 30:
            for j, pose in enumerate(poses): 
                print(f"Pose {j}: {pose}")
            i=0

        # find hands, return drawn image
        # HAND FINDER PART
        # image = tracker.hands_finder(image)
        # lm_list, absolute_pos = tracker.position_finder(image, hand_no = 0, draw=False)
        # print(lm_list)

       

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