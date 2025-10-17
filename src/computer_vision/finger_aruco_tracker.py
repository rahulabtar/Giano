import numpy as np
import cv2 as cv
from typing import List, Dict, Optional, Tuple


class FingerArucoTracker():
    def __init__(self) -> None:
        self.fingertip_ids = [4,8,12,16,20]

    def transform_finger_to_aruco_space(self, finger_pixel_coords:tuple, aruco_polygon_2d:List, use_internal=False):
        """
        Transform finger pixel coordinates to ArUco polygon coordinate space.
        
        Args:
            finger_pixel_coords: tuple of (x, y) pixel coordinates of finger
            aruco_polygon_2d: List of 4 2D points defining the ArUco polygon corners
                            [top-left, top-right, bottom-right, bottom-left]
        
        Returns:
            (u, v) coordinates in ArUco space where (0,0) is top-left, (1,1) is bottom-right
        """

        # return none if no polygon has been detected
        if np.array_equal(aruco_polygon_2d, [0,0,0,0]) or len(aruco_polygon_2d) != 4:
            #TODO: Remove after debugging
            print('No polygon exists!')
            return None

        # return none if the finger coordinates are 0 as well
        if finger_pixel_coords == (0,0):
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
    
    def get_finger_keys(self, hand_landmarks: List, aruco_polygon_2d: List, 
                       piano_detector) -> Dict[int, Optional[Dict]]:
        """
        Get piano keys for all fingertip positions.
        
        Args:
            hand_landmarks: List of hand landmark data from MediaPipe
            aruco_polygon_2d: List of 4 2D points defining the ArUco polygon corners
            piano_detector: PianoKeyDetector instance
            
        Returns:
            Dictionary mapping finger_id to key information (or None)
        """
        finger_positions = {}
        
        # Get normalized positions for all fingertips
        for landmark in hand_landmarks:
            lm_id, x_px, y_px = landmark[0], landmark[1], landmark[2]
            
            if lm_id in self.fingertip_ids:
                aruco_coords = self.transform_finger_to_aruco_space(
                    (x_px, y_px), aruco_polygon_2d)
                
                if aruco_coords is not None:
                    finger_positions[lm_id] = aruco_coords
        
        # Detect keys for all finger positions
        return piano_detector.detect_finger_keys(finger_positions)
    
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
    
    def draw_finger_keys(self, image, finger_keys, hand_landmarks: List):
        """
        Draw finger positions with piano key information on the image.
        
        Args:
            image: Input image
            finger_keys: Dictionary mapping finger_id to key information
            hand_landmarks: List of hand landmark data
            
        Returns:
            Image with finger positions and key labels drawn
        """
        for landmark in hand_landmarks:
            lm_id, x_px, y_px = landmark[0], landmark[1], landmark[2]
            
            if lm_id in finger_keys and finger_keys[lm_id] is not None:
                key_info = finger_keys[lm_id]
                
                # Choose color based on key type
                color = (0, 0, 255) if key_info['is_black'] else (0, 255, 0)
                
                # Draw finger position
                cv.circle(image, (x_px, y_px), 8, color, -1)
                
                # Draw key name
                key_text = key_info['key_name']
                cv.putText(image, key_text, (x_px + 10, y_px - 10),
                         cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            elif lm_id in self.fingertip_ids:
                # Draw fingertip even if no key detected
                cv.circle(image, (x_px, y_px), 8, (128, 128, 128), -1)
                         
        return image    

    def transform_image_to_birdseye(self, image:np.ndarray, aruco_polygon_2d:List, output_size:Optional[tuple]=None) -> np.ndarray:
        """
        Transform the camera view to show the piano surface from directly above.
        
        Args:
            image: Input camera image
            aruco_polygon_2d: List of 4 corner points of the ArUco polygon
            
        Returns:
            Warped image showing piano from bird's-eye view
        """
        if len(aruco_polygon_2d) != 4 or np.array_equal(aruco_polygon_2d, [0,0,0,0]):
            return image
        
        # Define source points (the 4 corners of the ArUco polygon in camera view)
        src_points = np.array(aruco_polygon_2d, dtype=np.float32)
        
        # Define destination points (a rectangle in the output image)
        # You can adjust the width/height to control the output resolution
        if output_size:
            output_width, output_height = output_size
        else:
            h,w,_ = image.shape
            output_width = w
            output_height = h
        
        dst_points = np.array([
            [0, 0],                           # top-left
            [output_width, 0],                # top-right
            [output_width, output_height],    # bottom-right
            [0, output_height]                # bottom-left
        ], dtype=np.float32)
        
        # Calculate the perspective transformation matrix
        perspective_matrix = cv.getPerspectiveTransform(src_points, dst_points)
        
        # Apply the transformation
        warped_image = cv.warpPerspective(image, perspective_matrix, 
                                        (output_width, output_height))
        
        return warped_image