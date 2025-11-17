import numpy as np
import cv2 as cv
from typing import List, Dict, Optional, Tuple


class FingerArucoTracker():
    def __init__(self, output_size:tuple =(640,480)) -> None:
        """
        Args:
            output_size: (width, height of the output size for birdseye transform)
        """
        self.fingertip_ids = [4,8,12,16,20]
        self.output_size = output_size

        output_width, output_height = output_size
        # array of destination points for birdseye transform
        self.dst_points = np.array([
            [0, 0],                           # top-left
            [output_width, 0],                # top-right
            [output_width, output_height],    # bottom-right
            [0, output_height]                # bottom-left
        ], dtype=np.float32)

    @property
    def output_size(self)->tuple:
        """
        Get the output size for birdseye transform.
        """
        return self.__output_size

    @output_size.setter
    def output_size(self, output_size:tuple):
        """
        Set the output size for birdseye transform.
        Args:
            output_size: (width, height of the output size for birdseye transform)
        """
        if output_size is not None:
            if len(output_size) != 2:
                raise ValueError("Output size must be a tuple of (width, height)")
            if output_size[0] <= 0 or output_size[1] <= 0:
                raise ValueError("Output size must be greater than 0")
            if output_size[0] % 2 != 0 or output_size[1] % 2 != 0:
                raise ValueError("Output size must be even")
            if output_size[0] > 1920 or output_size[1] > 1080:
                raise ValueError("Output size must be less than or equal to 1920x1080")
        self.__output_size = output_size

    def get_output_size(self)->tuple:
        """
        Get the output size for birdseye transform.
        """
        return self.__output_size

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
        if aruco_polygon_2d is None:
            return None
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


    
    

