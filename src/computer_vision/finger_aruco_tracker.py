import numpy as np
import cv2 as cv
from typing import List


class FingerArucoTracker():
    def __init__(self) -> None:
        self.fingertip_ids = [4,8,12,16,20]

    def transform_finger_to_aruco_space(self, finger_pixel_coords, aruco_polygon_2d:List, use_internal=False):
        """
        Transform finger pixel coordinates to ArUco polygon coordinate space.
        
        Args:
            finger_pixel_coords: (x, y) pixel coordinates of finger
            aruco_polygon_2d: List of 4 2D points defining the ArUco polygon corners
                            [top-left, top-right, bottom-right, bottom-left]
        
        Returns:
            (u, v) coordinates in ArUco space where (0,0) is top-left, (1,1) is bottom-right
        """
        if np.array_equal(aruco_polygon_2d, [0,0,0,0]) or len(aruco_polygon_2d) != 4:
            #TODO: Remove after debugging
            print('No polygon exists!')
            return None
    
        # Define the ArUco polygon corners (in pixel space)
        src_points = np.array(aruco_polygon_2d, dtype=np.float32)

        # Define the target coordinate space (normalized 0-1 rectangle)
        dst_points = np.array([
            [0, 0],    # top-left
            [1, 0],    # top-right  
            [1, 1],    # bottom-right
            [0, 1]     # bottom-left
        ], dtype=np.float32)

        # Calculate homography transformation matrix
        homography_matrix = cv.getPerspectiveTransform(src_points, dst_points)

        finger_point = np.array([[finger_pixel_coords]], dtype=np.float32)
        transformed_point = cv.perspectiveTransform(finger_point, homography_matrix)

        return transformed_point[0][0]
    
    def draw_finger_positions(self, image, finger_positions, hand_landmarks:List):
        """Draw finger positions and coordinates on the image."""
        for landmark in hand_landmarks:
            lm_id, x_px, y_px = landmark[0], landmark[1], landmark[2]
            
            if lm_id in finger_positions:
                u, v = finger_positions[lm_id]
                
                # Draw finger position
                cv.circle(image, (x_px, y_px), 8, (255, 0, 0), -1)
                
                # Draw coordinates
                coord_text = f"({u:.2f}, {v:.2f})"
                cv.putText(image, coord_text, (x_px + 10, y_px - 10),
                         cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                         
        return image    