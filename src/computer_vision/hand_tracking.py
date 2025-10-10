import cv2 as cv
from src.core.constants import PI, IN_TO_METERS, ASSETS_DIR, HAND_MODEL_PATH
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult


import numpy as np
import time as time
import os
import math
import matplotlib.pyplot as plt
PI = 3.1415926
IN_TO_METERS = 0.0254


class HandTracker():
    """ Basically a wrapper for MediaPipe Hands hand pose tracking solution."""
    def __init__(self, num_hands=2, hand_detection_confidence=0.5, hand_presence_confidence=0.5, tracking_confidence=0.5):
        
        
        # Initialize the hand landmarker model
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH, delegate='GPU')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=num_hands,
            min_hand_detection_confidence=hand_detection_confidence,
            min_hand_presence_confidence=hand_presence_confidence,
            min_tracking_confidence=tracking_confidence,
            result_callback=save_result)

        self.detector = vision.HandLandmarker.create_from_options(options)
        self.detection_result = None
    # find hands and draw landmarks on image
    def hands_finder(self, image, draw=True):
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(image, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return image
    
    def detect(self, image, hand_no=0, draw=False):
        """
        Detect hands
        """
        lm_list = []
        lm_absolute = []
        image = cv.flip(image, 1)
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        self.detector.detect_async(mp_image, time.time_ns() // 1_000_000)
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
    
    def save_result(self, result: vision.HandLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):

        self.detection_result = result
        
def run(model: str, num_hands: int,
    min_hand_detection_confidence: float,
    min_hand_presence_confidence: float, min_tracking_confidence: float,
    camera_id: int, width: int, height: int) -> None:
    """Continuously run inference on images acquired from the camera.

    Args:
        model: Name of the hand landmarker model bundle.
        num_hands: Max number of hands that can be detected by the landmarker.
        min_hand_detection_confidence: The minimum confidence score for hand
        detection to be considered successful.
        min_hand_presence_confidence: The minimum confidence score of hand
        presence score in the hand landmark detection.
        min_tracking_confidence: The minimum confidence score for the hand
        tracking to be considered successful.
        camera_id: The camera id to be passed to OpenCV.
        width: The width of the frame captured from the camera.
        height: The height of the frame captured from the camera.
    """


    # Visualization parameters
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

def save_result(result: vision.HandLandmarkerResult,
                unused_output_image: mp.Image, timestamp_ms: int):
    global FPS, COUNTER, START_TIME, DETECTION_RESULT

    # Calculate the FPS
    if COUNTER % fps_avg_frame_count == 0:
        FPS = fps_avg_frame_count / (time.time() - START_TIME)
        START_TIME = time.time()

    DETECTION_RESULT = result
    COUNTER += 1



    # TODO: Important
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Run hand landmarker using the model.
    detector.detect_async(mp_image, time.time_ns() // 1_000_000)

    # Show the FPS

    
    # Landmark visualization parameters.
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

    if DETECTION_RESULT:
        # Draw landmarks and indicate handedness.
        for idx in range(len(DETECTION_RESULT.hand_landmarks)):
            hand_landmarks = DETECTION_RESULT.hand_landmarks[idx]
            handedness = DETECTION_RESULT.handedness[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                                z=landmark.z) for landmark
                in hand_landmarks
            ])
            mp_drawing.draw_landmarks(
                current_frame,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = current_frame.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(current_frame, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS,
                        cv2.LINE_AA)

    cv2.imshow('hand_landmarker', current_frame)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
        break

detector.close()