import cv2 as cv
import numpy as np
from typing import List
try:
    from .aruco_pose_tracker import ArucoPoseTracker
except ImportError:
    from aruco_pose_tracker import ArucoPoseTracker


class ArucoPolygonDetector:
    """Utility class for detecting polygons formed by ArUco markers."""
    
    def __init__(self, camera_matrix:np.ndarray, dist_coeffs: np.ndarray):
        """
        Initialize with a pose tracker instance.
        
        Args:
            camera_matrix: The camera projection matrix from calibration
            dist_coeffs: The distortion coefficients from calibration
        """

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.polygon = None

    def sort_markers(self, ordered_ids: List[int], poses: List[dict]):
        found_markers = []
        # Match found markers with user provided IDs
        for pose in poses:
            if pose['id'] in ordered_ids:
                position = pose['tvec'].flatten()
                found_markers.append((position[0], position[1], pose))
        
        # Sort by y (top to bottom), then by x (left to right)
        found_markers.sort(key=lambda p: (p[1], p[0]))
        
        # Top two markers (lowest y values)
        top_markers = found_markers[:2]
        bottom_markers = found_markers[2:]
        
        # Sort top markers by x (left to right)
        top_markers.sort(key=lambda p: p[0])
        # Sort bottom markers by x (right to left for clockwise order)
        bottom_markers.sort(key=lambda p: -p[0])

        ordered = [
            top_markers[0][2],    # top-left
            top_markers[1][2],    # top-right  
            bottom_markers[0][2], # bottom-right
            bottom_markers[1][2]  # bottom-left
        ]

        return ordered

    def get_marker_polygon(self, ordered_ids: List[int], poses: List, image: np.ndarray, marker_size_meters: float = 0.05) -> List:
        """
        Get the polygon created by the ArUco markers with the given IDs.
        
        Args:
            ordered_ids: List of marker IDs in the desired order
            image: Input image
            camera_matrix: Camera calibration matrix
            dist_coeffs: Camera distortion coefficients
            marker_size_meters: Size of markers in meters
            
        Returns:
            List of 2D points forming the polygon, or [0,0,0,0] if insufficient markers
        """
        
        if len(poses) < 4:
            # print('fewer than 4 markers found! Returning blank')
            return [0, 0, 0, 0]

        ordered = self.sort_markers(ordered_ids, poses)
        

        marker_centers_2d = []
        for marker in ordered:
            # Project 3D marker center (0,0,0 in marker coords) to 2D
            center_3d = np.array([[0, 0, 0]], dtype=np.float32)
            center_2d, _ = cv.projectPoints(
                center_3d, marker['rvec'], marker['tvec'], 
                self.camera_matrix, self.dist_coeffs
            )
            
            center_2d = center_2d.reshape(-1, 2)[0]
            marker_centers_2d.append(center_2d)
        
        self.polygon = marker_centers_2d
        return marker_centers_2d
    
    def draw_box(self, image: np.ndarray, marker_centers_2d: List):
        #If no marker centers detected, don't draw on the image
        if np.array_equal(marker_centers_2d, [0,0,0,0]):
            return image
        points = np.array(marker_centers_2d, dtype=np.int32)
        new_image = cv.polylines(image, [points], True, (0, 255, 0), 2)
        
        return new_image
    
    