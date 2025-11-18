import numpy as np
import cv2 as cv
from typing import List, Dict, Optional, Tuple
from src.computer_vision.aruco_polygon_detector import ArucoPolygonDetector
from src.core.utils import name_from_midi

class FingerArucoTracker(ArucoPolygonDetector):
    def __init__(self, 
                 camera_matrix:np.ndarray, 
                 dist_coeffs: np.ndarray, 
                 output_size:tuple =(640,480)
                 ):
        """
        Args:
            camera_matrix: The camera projection matrix from calibration
            dist_coeffs: The distortion coefficients from calibration
            output_size: (width, height of the output size for birdseye transform)
        """
        super().__init__(camera_matrix, dist_coeffs, output_size)
        self.fingertip_ids = [4,8,12,16,20]
        self.keyboard_map = None

    def set_keyboard_map(self, labeled_keys: List[Dict]):
        """
        Set the keyboard map for the finger aruco tracker.
        """
        for i,key in enumerate(labeled_keys):
            closed_contour = self._ensure_contour_closed(key['contour'])
            if self._is_contour_closed(closed_contour):
                labeled_keys[i]['contour'] = closed_contour
            else:
                print("Contour is not closed")
                return False
        self.keyboard_map = labeled_keys

    
    def get_finger_keys(self, hand_landmarks: List, 
                       ) -> Dict[int, Optional[int]]:
        """
        Get piano keys for all fingertip positions.
        
        Args:
            hand_landmarks: List of hand landmark data from MediaPipe
            
        Returns:
            Dictionary mapping finger_id to key information (or None)
        """
        finger_keys = {}
        
        # Get normalized positions for all fingertips
        for landmark in hand_landmarks:
            lm_id, x_px, y_px = landmark[0], landmark[1], landmark[2]
            
            if lm_id in self.fingertip_ids:
                if x_px != 0 and y_px != 0:

                    midi_note = self.find_closest_key(x_px, y_px)
                    if midi_note is not None:
                        finger_keys[lm_id] = midi_note

        return finger_keys
    
    def find_closest_key(self, x_px: float, y_px: float) -> Optional[Dict]:
        """
        Find the closest key to a given (x_px, y_px) point in image space.
        
        Args:
            x_px: X coordinate in image space (pixels)
            y_px: Y coordinate in image space (pixels)
            
        Returns:
            MIDI note number of the closest key, 
            or None if no key is found
        """
        if self.keyboard_map is None:
            return None
        
        # Transform point from image space to birdseye space
        aruco_coords = self.transform_point_from_image_to_birdseye((x_px, y_px))
        point = np.array([aruco_coords[0], aruco_coords[1]])
        
        closest_midi_note = None
        min_distance = float('inf')


        for key in self.keyboard_map:
            result = cv.pointPolygonTest(key['contour'], point, measureDist=False)
            if result > 0:
                return key['midi_note']
            
            distance = np.linalg.norm(point - key['centroid'])
            if distance < min_distance:
                min_distance = distance
                closest_midi_note = key['midi_note']
        
        return closest_midi_note
    
    def measure_distance_to_key(self, x_px: float, y_px: float, midi_note: int) -> Optional[float]:
        """
        Measure the distance to a given key in image space.
        """
        if self.keyboard_map is None:
            return None
        
        for key in self.keyboard_map:
            if key['midi_note'] == midi_note:
                aruco_x, aruco_y = self.transform_point_from_image_to_birdseye((x_px, y_px))
                return np.linalg.norm(np.array([aruco_x, aruco_y]) - np.array(key['centroid']))
    
    
    def _is_contour_closed(self, contour: np.ndarray) -> bool:
        """
        Check if a contour is closed (first and last points are the same).
        
        Args:
            contour: Contour array with shape (N, 1, 2)
            
        Returns:
            True if contour is closed, False otherwise
        """
        if len(contour) < 3:
            return False
        first_point = contour[0, 0]
        last_point = contour[-1, 0]
        return np.allclose(first_point, last_point, atol=1e-6)
    
    def _ensure_contour_closed(self, contour: np.ndarray) -> np.ndarray:
        """
        Ensure a contour is closed by adding the first point at the end if needed.
        
        Args:
            contour: Contour array with shape (N, 1, 2)
            
        Returns:
            Closed contour array
        """
        if self._is_contour_closed(contour):
            return contour
        
        # Add first point at the end to close the contour
        first_point = contour[0:1, :, :].copy()
        closed_contour = np.vstack([contour, first_point])
        return closed_contour
    


    def _map_coords_to_uncropped(self, coords_or_contour, y_offset: int):
        """
        Map coordinates or contours from cropped image space back to uncropped image space.
        
        Args:
            coords_or_contour: Can be:
                - Tuple (x, y) for a single point
                - NumPy array with shape (N, 1, 2) for contour/polygon
                - Tuple (x, y, w, h) for bbox
            y_offset: Y offset to add (pixels cropped from top)
            
        Returns:
            Same type as input but with y coordinates adjusted
        """
        if isinstance(coords_or_contour, tuple):
            if len(coords_or_contour) == 2:
                # Single point (x, y)
                x, y = coords_or_contour
                return (x, y + y_offset)
            elif len(coords_or_contour) == 4:
                # Bbox (x, y, w, h)
                x, y, w, h = coords_or_contour
                return (x, y + y_offset, w, h)
        elif isinstance(coords_or_contour, np.ndarray):
            # Contour or polygon array (N, 1, 2)
            mapped = coords_or_contour.copy()
            mapped[:, 0, 1] += y_offset  # Add offset to all y coordinates
            return mapped
        
        return coords_or_contour


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
    
    

    
    

