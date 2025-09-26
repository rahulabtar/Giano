#!/usr/bin/env python3
"""
Simple test script for recogninzing AruCo markers.
This script demonstrates basic marker detection.
"""

import cv2 as cv
import numpy as np
from giano_aruco import ArucoMarkerSystem

# Initialize ArUco system
aruco_system = ArucoMarkerSystem()
aruco_detector = cv.aruco.ArucoDetector(aruco_system.dictionary, aruco_system.detector_params)
cap = cv.VideoCapture(0)  # 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect markers
    corners, ids, rejected = aruco_detector.detectMarkers(frame)

    # Draw detected markers
    if ids is not None:
        cv.aruco.drawDetectedMarkers(frame, corners, ids)

    # Display the resulting frame
    cv.imshow('AruCo Marker Detection', frame)

    # Break the loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break