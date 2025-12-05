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

    
    
    def preprocess_for_gloves(self, image):
        """
        Remap black/grey glove colors to skin tones to help MediaPipe detection.
        
        Args:
            image: Input BGR image
        """
        image = image.copy()
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        
        # Black and grey: any hue, low saturation, dark to medium brightness (V: 0-200)
        # This captures both black (V: 0-50) and grey (V: 50-200) gloves
        lower_glove = np.array([0, 0, 0])
        upper_glove = np.array([179, 50, 200])
        
        mask = cv.inRange(hsv, lower_glove, upper_glove)
        cv.imshow('glove mask', mask)
        skin_h, skin_s, skin_v = 15, 80, 200  # Skin tone HSV
        hsv[mask > 0] = [skin_h, skin_s, skin_v]
        # Convert back to BGR for proper display (cv.imshow expects BGR)
        processed_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imshow('processed (hsv->bgr)', processed_bgr)
        return processed_bgr

        
    # find hands and draw landmarks on image
    def hands_finder(self, image, draw=True) -> np.ndarray:
        
        image = self.preprocess_for_gloves(image)
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